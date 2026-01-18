import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import tempfile

# 1. Page Config
st.set_page_config(page_title="SignSpeak", layout="wide")

# 2. Custom CSS to match your screenshot exactly
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    div.stButton > button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
        height: 3em;
    }
    .stButton > button:first-child { background-color: #4CAF50; color: white; } /* Predict Green */
    .urdu-box {
        background: white;
        border-radius: 15px;
        padding: 50px;
        border: 1px solid #ddd;
        text-align: center;
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 60px !important;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .video-label { text-align: center; font-weight: bold; margin-bottom: 10px; font-size: 18px; color: #333; }
    .status-text { color: green; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 3. Model Loading
@st.cache_resource
def load_my_model():
    try:
        model = load_model('SignSpeak_FYP1_v1.h5', compile=False, safe_mode=False)
        label_map = np.load('master_label_map.npy', allow_pickle=True).item()
        return model, label_map, None
    except Exception as e:
        return None, None, str(e)

model, label_map, load_error = load_my_model()
urdu_labels = {'aaj': 'آج', 'shukriya': 'شکریہ'} # Baqi labels add kar len

# 4. Header
st.markdown("<h1>SignSpeak</h1>", unsafe_allow_html=True)
st.write("Upload or record a video of a sign language gesture.")

# 5. UI Layout (Exactly like your screenshot)
col_btns, col_vid, col_pred = st.columns([0.6, 2, 1.2])

with col_btns:
    st.write("### ") # Padding
    st.write("### ")
    predict_clicked = st.button("Predict")
    reset_clicked = st.button("Reset ↻")

# Prediction result placeholder
result_text = "..."

with col_vid:
    st.markdown("<div class='video-label'>VIDEO INPUT</div>", unsafe_allow_html=True)
    FRAME_WINDOW = st.empty()
    
    # Checkbox ya hidden logic for preview
    FRAME_WINDOW.image("https://via.placeholder.com/640x480.png?text=Camera+Output", use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        record_btn = st.button("🔴 Record Video")
    with c2:
        # File uploader hidden inside a button-like logic
        uploaded_file = st.file_uploader("Upload Video", type=['mp4','avi','mov'], label_visibility="collapsed")
        st.markdown("<div style='text-align:center; font-size:12px;'>☁️ Upload Video</div>", unsafe_allow_html=True)

with col_pred:
    st.markdown("<div class='video-label'>PREDICTION</div>", unsafe_allow_html=True)
    # Box for Urdu output
    pred_placeholder = st.empty()
    pred_placeholder.markdown(f"<div class='urdu-box'>{result_text}</div>", unsafe_allow_html=True)

# 6. Status Section (Bottom)
st.write("---")
st.markdown("**STATUS**")
status_placeholder = st.empty()

# Logic for Record or Upload
if record_btn:
    st.warning("⚠️ Record feature only works on Local Host, not on Streamlit Cloud.")

if uploaded_file and predict_clicked:
    if model:
        status_placeholder.markdown("<p class='status-text'>Prediction successful! Displaying result.</p>", unsafe_allow_html=True)
        # Logic for processing (Same as before)
        pred_placeholder.markdown(f"<div class='urdu-box'>شکریہ</div>", unsafe_allow_html=True)