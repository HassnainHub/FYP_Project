import os
# Keras version compatibility fix
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
import tempfile

# 1. Page Configuration
st.set_page_config(page_title="Urdu SignSpeak", layout="wide")

# 2. Custom CSS for exact UI match
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    div.stButton > button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
        height: 3.5em;
        background-color: #28a745;
        color: white;
    }
    .reset-btn > div > button {
        background-color: #6c757d !important;
    }
    .urdu-box {
        background: white;
        border-radius: 20px;
        padding: 60px;
        border: 1px solid #ddd;
        text-align: center;
        min-height: 220px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 70px !important;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        color: #333;
    }
    .video-label { text-align: center; font-weight: bold; margin-bottom: 15px; font-size: 20px; }
    </style>
    """, unsafe_allow_html=True)

# 3. Complete Mapping (Original)
urdu_labels = {
    'aaj': 'آج', 'aath': 'آٹھ', 'ahista': 'آہستہ', 'anywalakal': 'آنے والا کل',
    'behtreen': 'بہترین', 'btana': 'بتانا', 'bukhar': 'بخار', 'bus': 'بس',
    'car': 'کار', 'char': 'char', 'chawal': 'چاول', 
    'chay': 'چھ', 'chaye': 'چائے', 'chini': 'چینی', 'dard': 'درد', 'das': 'دس',
    'dawai': 'دوائی', 'dekhna': 'دیکھنا', 'do': 'دو', 'dobara': 'دوبارہ',
    'doctor': 'ڈاکٹر', 'doodh': 'دودھ', 'dost': 'دوست', 'ek': 'ایک',
    'emergency': 'ایمرجنسی', 'ghalat': 'غلط', 'ghanta': 'گھنٹہ', 'gosht': 'gosht',
    'hafta': 'ہفتہ', 'intezar': 'انتظار', 'kal': 'کل', 'likhna': 'لکھنا',
    'mahina': 'مہینہ', 'mask': 'ماسک', 'minute': 'منٹ', 'no': 'نہیں',
    'paanch': 'پانچ', 'parhna': 'پڑھنا', 'raasta': 'راستہ', 'roti': 'روٹی',
    'saat': 'سات', 'sabzi': 'سبزی', 'sahih': 'صحیح', 'samajhna': 'سمجھنا',
    'stop': 'سٹاپ', 'sunna': 'سننا', 'tabdeel': 'تبدیل', 'teen': 'تین',
    'tez': 'تیز', 'ticket': 'ٹکٹ'
}

# 4. Model Loading
@st.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model('SignSpeak_FYP1_v1.h5', compile=False, safe_mode=False)
        label_map = np.load('master_label_map.npy', allow_pickle=True).item()
        return model, label_map, None
    except Exception as e:
        return None, None, str(e)

model, label_map, load_error = load_my_model()

# 5. Header Section
st.title("SignSpeak")
st.write("Upload or record a video of a sign language gesture.")

# 6. Main UI Layout
col_btns, col_vid, col_pred = st.columns([0.7, 2, 1.3])

with col_btns:
    st.write("### ")
    st.write("### ")
    predict_clicked = st.button("Predict")
    st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
    if st.button("Reset ↻"):
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

final_output = "..."

with col_vid:
    st.markdown("<div class='video-label'>VIDEO INPUT</div>", unsafe_allow_html=True)
    # File Uploader
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'], label_visibility="collapsed")
    
    video_path = None
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.video(uploaded_file)
    
    # Record button (Visual only for Cloud, functional for local)
    if st.button("🔴 Record Video"):
        st.info("Record feature requires local environment or WebRTC. Use 'Upload Video' for now.")

# 7. Prediction Logic Execution
if predict_clicked and video_path:
    if model is not None:
        with st.spinner("Processing Gesture..."):
            cap = cv2.VideoCapture(video_path)
            frames_list = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (224, 224))
                frames_list.append(resized)
            cap.release()

            if len(frames_list) >= 8:
                idx = np.linspace(0, len(frames_list) - 1, 8, dtype=int)
                input_frames = [frames_list[i] for i in idx]
                final_input = np.array(input_frames).reshape(1, 8, 224, 224, 1) / 255.0
                
                # Model Prediction
                preds = model.predict(final_input)
                inv_label_map = {v: k for k, v in label_map.items()}
                eng_word = inv_label_map[np.argmax(preds)]
                final_output = urdu_labels.get(eng_word, eng_word)
                st.success(f"Detected: {eng_word}")
            else:
                st.error("Video too short. Need at least 8 frames.")

with col_pred:
    st.markdown("<div class='video-label'>PREDICTION</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='urdu-box'>{final_output}</div>", unsafe_allow_html=True)

# 8. Footer Status
st.write("---")
st.markdown("**STATUS**")
if load_error:
    st.error(f"Model Error: {load_error}")
elif uploaded_file:
    st.info("Video loaded successfully. Ready for prediction.")