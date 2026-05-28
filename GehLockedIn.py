import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import time
import os
import tensorflow as tf

# --- SEITEN KONFIGURATION ---
st.set_page_config(page_title="AI Focus Assistant", layout="centered")

# --- CUSTOM CSS FÜR DESIGN (Dynamische Farben) ---
if "app_state" not in st.session_state: 
    st.session_state.app_state = "normal" # Mögliche Zustände: normal, alert, pause

def update_design():
    if st.session_state.app_state == "alert":
        # Eine feurige Mischung aus Orange und Rot (#FF4B2B)
        color_theme = """
        <style>
        .stApp { background-color: #FF4B2B; transition: background-color 0.5s ease; }
        .box { background-color: white; padding: 30px; border-radius: 15px; text-align: center; box-shadow: 0px 4px 15px rgba(0,0,0,0.3); }
        .box h1 { color: #FF4B2B; font-weight: bold; font-size: 42px; margin: 0; }
        </style>
        """
    elif st.session_state.app_state == "pause":
        # Dunkellila (#2E1A47) wie gewünscht
        color_theme = """
        <style>
        .stApp { background-color: #2E1A47; transition: background-color 0.5s ease; }
        .box { background-color: white; padding: 30px; border-radius: 15px; text-align: center; }
        .box h1 { color: #2E1A47; font-weight: bold; font-size: 42px; margin: 0; }
        </style>
        """
    else:
        # Standard Streamlit Design für die Arbeitsphase
        color_theme = "<style>.box { display: none; }</style>"
    st.markdown(color_theme, unsafe_allow_html=True)

# --- MODELL LADEN ---
@st.cache_resource
def load_keras_model():
    model_path = "model.h5"
    label_path = "labels.txt"
    
    if os.path.exists(model_path) and os.path.exists(label_path):
        model = tf.keras.models.load_model(model_path, compile=False)
        with open(label_path, "r") as f:
            labels = [line.strip().split(" ")[1] for line in f.readlines()]
        return model, labels
    return None, None

model, labels = load_keras_model()

# --- INITIALISIERUNG DES SESSION STATES ---
if "focus_duration" not in st.session_state:
    st.session_state.focus_duration = 45 * 60
    st.session_state.pause_duration = 10 * 60
    st.session_state.focus_time_left = 45 * 60
    st.session_state.distracted_start_time = None
    st.session_state.locked_in_start_time = None
    st.session_state.last_update = time.time()
    st.session_state.is_running = False

# --- UI EINSTELLUNGEN (Sidebar) ---
st.sidebar.title("⏰ Timer Einstellungen")
focus_mins = st.sidebar.slider("Fokus-Zeit (Minuten)", 45, 60, 45)
pause_mins = st.sidebar.slider("Pausen-Zeit (Minuten)", 10, 15, 10)

if st.sidebar.button("Ziele & Timer festlegen / Reset"):
    st.session_state.focus_duration = focus_mins * 60
    st.session_state.focus_time_left = focus_mins * 60
    st.session_state.pause_duration = pause_mins * 60
    st.session_state.app_state = "normal"
    st.session_state.distracted_start_time = None
    st.session_state.locked_in_start_time = None
    st.session_state.is_running = True
    st.session_state.last_update = time.time()
    st.rerun()

# --- HINTERGRUND-SOUNDS ---
def play_alert_sound():
    # Einbindung eines warnenden Alarm-Sounds per HTML
    sound_html = """
    <audio autoplay loop>
    <source src="https://www.soundjay.com/buttons/sounds/button-4.mp3" type="audio/mp3">
    </audio>
    """
    st.markdown(sound_html, unsafe_allow_html=True)

# --- VIDEO VERARBEITUNGS-KLASSE ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.current_prediction = "Unbekannt"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if model is not None:
            # Bild für Teachable Machine Modell vorbereiten (224x224)
            resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            normalized = (np.asarray(resized, dtype=np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized
            
            # Vorhersage treffen
            prediction = model.predict(data, verbose=0)
            index = np.argmax(prediction)
            self.current_prediction = labels[index]
            
            # Status-Text live ins Kamerabild zeichnen
            color = (0, 255, 0) if "locked" in self.current_prediction.lower() else (0, 0, 255)
            cv2.putText(img, f"KI Status: {self.current_prediction}", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
        return frame

# --- HAUPT-ANWENDUNG ---
st.title("🧠 KI Fokus & Anti-Ablenkungs-Coach")

if model is None:
    st.error("⚠️ Bitte stelle sicher, dass 'model.h5' and 'labels.txt' im selben GitHub-Ordner liegen!")
else:
    # Webcam Streamer starten
    ctx = webrtc_streamer(key="focus-streamer", video_transformer_factory=VideoProcessor)

    # Platzhalter für die Benutzeroberfläche
    timer_placeholder = st.empty()
    alert_placeholder = st.empty()

    # App-Schleife für die Timer-Logik
    if ctx.video_transformer and st.session_state.is_running:
        while ctx.state.playing:
            current_time = time.time()
            dt = current_time - st.session_state.last_update
            st.session_state.last_update = current_time
            
            # Zustand aus dem Video-Stream holen
            ki_status = ctx.video_transformer.current_prediction.lower()
            is_distracted = "abgelenkt" in ki_status or "distracted" in ki_status
            is_locked_in = "locked" in ki_status or "fokus" in ki_status

            # --- LOGIK FÜR DEN 2-MINUTEN TIMER ---
            if is_distracted:
                st.session_state.locked_in_start_time = None # Locked-In Counter löschen
                if st.session_state.distracted_start_time is None:
                    st.session_state.distracted_start_time = current_time
                
                # Wie lange schon abgelenkt?
                elapsed_distraction = current_time - st.session_state.distracted_start_time
                
                if elapsed_distraction >= 120: # 2 Minuten durchgehend abgelenkt
                    st.session_state.app_state = "alert"
                    st.session_state.focus_time_left = st.session_state.focus_duration # Fokus-Timer komplett resetten
            
            elif is_locked_in:
                if st.session_state.distracted_start_time is not None:
                    if st.session_state.locked_in_start_time is None:
                        st.session_state.locked_in_start_time = current_time
                    
                    # Wenn man 30 Sekunden am Stück wieder fokussiert ist, resettet sich der Ablenkungstimer
                    if current_time - st.session_state.locked_in_start_time >= 30:
                        st.session_state.distracted_start_time = None
                        st.session_state.locked_in_start_time = None
                        if st.session_state.app_state == "alert":
                            st.session_state.app_state = "normal"
            
            # --- LOGIK FÜR DEN FOKUS TIMER ---
            # Fokus-Timer läuft nur weiter, wenn der Nutzer arbeitet und KEIN Alarm aktiv ist
            if st.session_state.app_state == "normal":
                st.session_state.focus_time_left -= dt
                if st.session_state.focus_time_left <= 0:
                    st.session_state.app_state = "pause"

            # --- UI RENDERING ---
            update_design()
            
            # Timer-Anzeige formatieren
            mins, secs = divmod(int(max(0, st.session_state.focus_time_left)), 60)
            
            with timer_placeholder.container():
                if st.session_state.app_state == "normal":
                    st.metric(label="⏱️ Verbleibende Fokus-Zeit", value=f"{mins:02d}:{secs:02d}")
                    if st.session_state.distracted_start_time is not None:
                        verbleibende_warnzeit = 120 - int(time.time() - st.session_state.distracted_start_time)
                        st.warning(f"⚠️ Warnung: Du wirkst abgelenkt! Alarm in {max(0, verbleibende_warnzeit)} Sekunden.")
                
            # Alarm/Pause Boxen anzeigen
            if st.session_state.app_state == "alert":
                with alert_placeholder.container():
                    st.markdown('<div class="box"><h1>DU BIST ABGELENKT. ARBEITE WEITER!</h1></div>', unsafe_allow_html=True)
                    play_alert_sound()
                    
            elif st.session_state.app_state == "pause":
                with alert_placeholder.container():
                    st.markdown('<div class="box"><h1>MACH EINE PAUSE!</h1></div>', unsafe_allow_html=True)
            else:
                alert_placeholder.empty()

            time.sleep(0.5) # CPU schonen
            st.rerun()  
