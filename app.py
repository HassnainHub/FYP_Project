import os
# Version mismatch se bachne ke liye Legacy Keras enable karein
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# 1. Page Config & Custom Styling (Screenshot jaisa dikhne ke liye)
st.set_page_config(page_title="SignSpeak - Urdu SL", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .predict-btn { background-color: #4CAF50 !important; color: white; }
    .reset-btn { background-color: #9E9E9E !important; color: white; }
    .status-box { background-color: #ffffff; padding: 20px; border-radius: 10px; border: 1px solid #ddd; }
    .urdu-text { font-size: 50px !important; text-align: center; font-family: '_Urdu_Font_', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# 2. Dictionary
urdu_labels = {
    'aaj': 'آج', 'aath': 'آٹھ', 'ahista': 'آہستہ', 'anywalakal': 'آنے والا کل',
    'behtreen': 'بہترین', 'btana': 'بتانا', 'bukhar': 'بخار', 'bus': 'بس',
    'car': 'کار', 'char': 'چار', 'chawal': 'چاول', 
    'chay': 'چھ', 'chaye': 'چائے', 'chini': 'چینی', 'dard': 'درد', 'das': 'دس',
    'dawai': 'دوائی', 'dekhna': 'دیکھنا', 'do': 'دو', 'dobara': 'دوبارہ',
    'doctor': 'ڈاکٹر', 'doodh': 'دودھ', 'dost': 'دوست', 'ek': 'ایک',
    'emergency': 'ایمرجنسی', 'ghalat': 'غلط', 'ghanta': 'گھنٹہ', 'gosht': 'گوشت',
    'hafta': 'ہفتہ', 'intezar': 'انتظار', 'kal': 'کل', 'likhna': 'لکھنا',
    'mahina': 'مہینہ', 'mask': 'ماسک', 'minute': 'منٹ', 'no': 'نہیں',
    'paanch': 'پانچ', 'parhna': 'پڑھنا', 'raasta': 'راستہ', 'roti': 'روٹی',
    'saat': 'سات', 'sabzi': 'سبزی', 'sahih': 'صحیح', 'samajhna': 'سمجھنا',
    'stop': 'سٹاپ', 'sunna': 'سننا', 'tabdeel': 'تبدیل', 'teen': 'تین',
    'tez': 'تیز', 'ticket': 'ٹکٹ', 'shukriya': 'شکریہ'
}

# 3. Flexible Model Loading
@st.cache_resource
def load_my_model():
    try:
        # Keras 3 (TF 2.16+) compatibility settings
        if hasattr(tf.keras, "config"):
            tf.keras.config.enable_unsafe_deserialization()
        
        # Load with safe_mode=False to handle Lambda layers in any version
        model = load_model('SignSpeak_FYP1_v1.h5', compile=False, safe_mode=False)
        label_map = np.load('master_label_map.npy', allow_pickle=True).item()
        return model, label_map
    except Exception as e:
        return str(e), None

model, label_map = load_my_model()

# 4. Header Section
st.markdown("<h1>SignSpeak</h1>", unsafe_allow_html=True)
st.write("Upload or record a video of a sign language gesture.")

# 5. Main UI Layout (Exactly like your screenshot)
col_side, col_main, col_pred = st.columns([0.5, 2, 1])

with col_side:
    st.write(" ") # Padding
    st.write(" ")
    predict_clicked = st.button("Predict", key="pred_btn")
    reset_clicked = st.button("Reset ↻", key="reset_btn")

with col_main:
    st.markdown("<div style='text-align: center; font-weight: bold;'>VIDEO INPUT</div>", unsafe_allow_html=True)
    FRAME_WINDOW = st.image("https://via.placeholder.com/640x480.png?text=Camera+Output", use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        record_btn = st.button("🔴 Record Video")
    with c2:
        upload_btn = st.button("☁️ Upload Video")

with col_pred:
    st.markdown("<div style='text-align: center; font-weight: bold;'>PREDICTION</div>", unsafe_allow_html=True)
    result_container = st.empty()
    result_container.markdown("""
        <div style='background: white; border-radius: 15px; padding: 40px; border: 1px solid #eee; text-align: center;'>
            <h1 style='color: #ccc;'>...</h1>
        </div>
    """, unsafe_allow_html=True)

# 6. Logic
if record_btn:
    if isinstance(model, str):
        st.error(f"Model Error: {model}")
    else:
        cap = cv2.VideoCapture(0)
        frames_list = []
        st.toast("Recording... 3 Seconds")
        start_t = time.time()
        
        while time.time() - start_t < 3:
            ret, frame = cap.read()
            if not ret: break
            FRAME_WINDOW.image(frame, channels="BGR", use_container_width=True)
            
            # Processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (224, 224))
            frames_list.append(resized)
        
        cap.release()
        
        if len(frames_list) >= 8:
            idx = np.linspace(0, len(frames_list)-1, 8, dtype=int)
            final_input = np.array([frames_list[i] for i in idx]).reshape(1, 8, 224, 224, 1) / 255.0
            
            # Predict
            pred = model.predict(final_input)
            inv_map = {v: k for k, v in label_map.items()}
            eng_res = inv_map[np.argmax(pred)]
            urdu_res = urdu_labels.get(eng_res, eng_res)
            
            # Display like screenshot
            result_container.markdown(f"""
                <div style='background: white; border-radius: 15px; padding: 40px; border: 1px solid #eee; text-align: center;'>
                    <h1 class='urdu-text'>{urdu_res}</h1>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div class='status-box'><p style='color:green;'>Prediction successful! Displaying result.</p></div>", unsafe_allow_html=True)