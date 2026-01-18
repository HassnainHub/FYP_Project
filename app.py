import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# 1. Page Config
st.set_page_config(page_title="Urdu SignSpeak", layout="wide")

# Updated Dictionary (Corrected mapping)
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
    'tez': 'تیز', 'ticket': 'ٹکٹ'
}
# 3. Model Loading Logic (Fixing Lambda & NameError)
@st.cache_resource
def load_my_model():
    try:
        # Keras 3 compatibility fix
        if hasattr(tf.keras, "config"):
            tf.keras.config.enable_unsafe_deserialization()
        
        # Loading Model & Label Map
        model = load_model('SignSpeak_FYP1_v1.h5', compile=False, safe_mode=False)
        label_map = np.load('master_label_map.npy', allow_pickle=True).item()
        return model, label_map
    except Exception as e:
        # Screen par error dikhane ke liye
        return str(e), None

# Initialization
res = load_my_model()

# Check if model loaded or returned error string
if isinstance(res[0], str):
    st.error(f"❌ Model Load Error: {res[0]}")
    model, label_map = None, None
else:
    model, label_map = res

# 4. UI Elements
st.title("🇵🇰 Urdu Sign Language Translator")

col_vid, col_res = st.columns([2, 1])

with col_vid:
    st.write("### Camera Input")
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    
    if run:
        if model is None:
            st.error("Model load nahi ho saka. File check karein.")
        else:
            camera = cv2.VideoCapture(0)
            st.info("Recording for 3 seconds...")
            frames_list = []
            start_time = time.time()
            
            while (time.time() - start_time) < 3:
                ret, frame = camera.read()
                if not ret: break
                
                FRAME_WINDOW.image(frame, channels="BGR")
                
                # Pre-processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (224, 224))
                frames_list.append(resized)
                
            camera.release()
            st.success("Processing...")

            # 5. Prediction Logic
            if len(frames_list) >= 8:
                idx = np.linspace(0, len(frames_list) - 1, 8, dtype=int)
                input_frames = [frames_list[i] for i in idx]
                final_input = np.array(input_frames).reshape(1, 8, 224, 224, 1) / 255.0
                
                # Prediction
                inv_label_map = {v: k for k, v in label_map.items()}
                pred = model.predict(final_input)
                eng_word = inv_label_map[np.argmax(pred)]
                urdu_word = urdu_labels.get(eng_word, "N/A")
                
                with col_res:
                    st.write("### Prediction Result")
                    st.success(f"Urdu: {urdu_word}")
                    st.write(f"English: {eng_word}")
                    st.write(f"Confidence: {np.max(pred)*100:.2f}%")
            else:
                st.warning("Kafi frames nahi milay. Dobara koshish karein.")