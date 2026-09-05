import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import time
from pathlib import Path
from src.utils.config import get
from src.utils.landmarks import extract_landmarks, normalize_landmarks, draw_landmarks

MODEL_PATH = get('model.model_path')
CLASSES = get('data.classes')
CONF_THRESH = get('inference.confidence_threshold')
COOLDOWN = get('inference.cooldown_frames')
MAX_TEXT = get('inference.max_text_length')
SHOW_LM = get('ui.show_landmarks')
SHOW_CONF = get('ui.show_confidence')

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def init_tts():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    return engine

model = load_model()
tts_engine = init_tts()

st.set_page_config(page_title=get('ui.window_title'), layout="wide")

st.title("ISL / ASL Sign Language Translator")

col1, col2 = st.columns([3, 1])

with col1:
    frame_placeholder = st.empty()

with col2:
    st.subheader("Recognized Text")
    text_box = st.empty()
    conf_bar = st.empty()
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        speak_btn = st.button("Speak", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("Clear", use_container_width=True)

if 'recognized_text' not in st.session_state:
    st.session_state.recognized_text = ""
if 'last_pred' not in st.session_state:
    st.session_state.last_pred = None
if 'cooldown' not in st.session_state:
    st.session_state.cooldown = 0
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = True

def speak_text(text):
    if text.strip():
        tts_engine.say(text)
        tts_engine.runAndWait()

def process_frame(frame):
    landmarks = extract_landmarks(frame)
    pred_class = None
    confidence = 0.0
    
    if landmarks is not None:
        landmarks = normalize_landmarks(landmarks)
        landmarks = landmarks.reshape(1, -1)
        preds = model.predict(landmarks, verbose=0)[0]
        pred_idx = np.argmax(preds)
        confidence = preds[pred_idx]
        
        if confidence >= CONF_THRESH:
            pred_class = CLASSES[pred_idx]
    
    return pred_class, confidence, landmarks

cap = cv2.VideoCapture(0)

while st.session_state.run_camera:
    ret, frame = cap.read()
    if not ret:
        st.error("Cannot access camera")
        break
    
    frame = cv2.flip(frame, 1)
    pred_class, confidence, landmarks = process_frame(frame)
    
    if SHOW_LM and landmarks is not None:
        frame = draw_landmarks(frame, landmarks)
    
    if pred_class and st.session_state.cooldown == 0:
        if pred_class == "space":
            st.session_state.recognized_text += " "
        elif pred_class == "delete":
            st.session_state.recognized_text = st.session_state.recognized_text[:-1]
        elif pred_class != "nothing":
            st.session_state.recognized_text += pred_class
        
        if len(st.session_state.recognized_text) > MAX_TEXT:
            st.session_state.recognized_text = st.session_state.recognized_text[-MAX_TEXT:]
        
        st.session_state.last_pred = pred_class
        st.session_state.cooldown = COOLDOWN
    
    if st.session_state.cooldown > 0:
        st.session_state.cooldown -= 1
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    
    text_box.text_area("", value=st.session_state.recognized_text, height=100, disabled=True, label_visibility="collapsed")
    
    if SHOW_CONF and st.session_state.last_pred:
        conf_bar.progress(float(confidence), text=f"{st.session_state.last_pred}: {confidence:.2f}")
    
    if speak_btn:
        speak_text(st.session_state.recognized_text)
    
    if clear_btn:
        st.session_state.recognized_text = ""
        st.session_state.last_pred = None
    
    time.sleep(0.03)

cap.release()