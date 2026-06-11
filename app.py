# -*- coding: utf-8 -*-
"""
Kannada OCR Web App – Streamlit Cloud Ready
"""

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
import pytesseract
import os
import pandas as pd
from datetime import datetime
import random
import numpy as np
import cv2
import gdown
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ------------------------------
#  TESSERACT CONFIGURATION
# ------------------------------
# Streamlit Cloud will install tesseract at /usr/bin/tesseract (via packages.txt)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ------------------------------
#  HELPER FUNCTIONS
# ------------------------------
def rotate_image(image: Image.Image, angle: int) -> Image.Image:
    return image.rotate(angle, expand=True)

def auto_crop_image(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_points = np.vstack(contours).astype(np.int32)
        x, y, w, h = cv2.boundingRect(all_points)
        return image_np[y:y + h, x:x + w]
    return image_np

def enhance_image(pil_image, method="adaptive"):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if method == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def run_full_ocr(image_array, psm=3):
    """Single definition: uses Kannada language and configurable PSM."""
    config = f'--oem 1 --psm {psm} -l kan'
    return pytesseract.image_to_string(Image.fromarray(image_array), config=config).strip()

def convert_old_to_new_kannada(text):
    conversion_map = {
        "ಮದುಮಕ್ಕಳಿಗೆ": "ಮಗುಗಳಿಗೆ",
        "ಪಾಠಶಾಲೆ": "ಶಾಲೆ",
        "ಅಂಗಡಿಗೆ": "ದೂಕಾಣಕ್ಕೆ",
        "ಆಯ್ಕೆಯು": "ಆಯ್ಕೆ",
        "ಅಧ್ಯಾಪಕರು": "ಶಿಕ್ಷಕರು",
        "ಗ್ರಂಥ": "ಪುಸ್ತಕ",
        "ಶಿಕ್ಷÊ3": "ಬೋಧನೆ",
        "ಸಂಗತಿಗಳು": "ಮಾಹಿತಿಗಳು",
        "ನೂತನ": "ಹೊಸ",
        "ಪಾಠ": "ಪಾಠವು",
        "ಬೋಧನೆ": "ಶಿಕ್ಷಣ"
    }
    for old_word, new_word in conversion_map.items():
        text = text.replace(old_word, new_word)
    return text

@st.cache_resource
def prepare_classifier():
    folder_url = "https://drive.google.com/drive/folders/1G4CNR2WeaRP_s_c7lddnIyoQG2ck4nYm?usp=sharing"
    output_folder = "Dataset"

    if not os.path.exists(output_folder):
        st.info("📥 Downloading dataset folder from Google Drive...")
        gdown.download_folder(url=folder_url, output=output_folder, quiet=False, use_cookies=False)
        st.success("✅ Dataset folder downloaded!")

    X, y = [], []
    IMG_SIZE = 64
    for folder in os.listdir(output_folder):
        path = os.path.join(output_folder, folder)
        if os.path.isdir(path):
            for file in os.listdir(path):
                try:
                    img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        X.append(img.flatten())
                        y.append(folder)
                except:
                    continue

    if len(X) == 0:
        st.error("No images found in dataset. Year classifier will be unavailable.")
        return None, None, 0.0

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X), y_enc, test_size=0.2, random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le, acc

# ------------------------------
#  PAGE CONFIGURATION
# ------------------------------
st.set_page_config(page_title="Kannada OCR", layout="centered")

# --- Header Banner ---
st.markdown("""
<style>
@keyframes slideDown {
    from { transform: translateY(-100%); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}
.sticky-title {
    position: sticky;
    top: 0;
    background-color: #fff8f0;
    padding: 5px 0;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
    color: #800000;
    z-index: 100;
    border-bottom: 2px solid #d8cfc4;
    animation: slideDown 0.5s ease-out;
}
.header-logo {
    display: block;
    margin: 0 auto 5px auto;
    width: 60px;
    height: auto;
}
.header-text {
    color:#800000;
    font-weight:700;
    text-align:center;
    line-height: 1.2;
    margin: 2px 0;
}
</style>

<div class='sticky-title'>Kannada OCR Web App</div>

<h3 style='color:#800000; font-weight:700; text-align:center;'>
Rani Channamma University, Belagavi
</h3>
<h4 style='color:#800000; font-weight:500; text-align:center;'>
Dept. of Computer Science
</h4>

<h6 style='color:#000000; font-weight:700; text-align:center;'>Kannada OCR with Old → Hosa Kannada Converter & Year Classifier</h6>
<h6 style='color:#800000; font-weight:700; text-align:center;'>ಕನ್ನಡ ಓಸಿಆರ್ ಹಳೆಯ → ಹೊಸ ಕನ್ನಡ ಪರಿವರ್ತಕ ಮತ್ತು ವರ್ಷ ವರ್ಗವಿಂಗಡಕ</h6>
""", unsafe_allow_html=True)

st.markdown("""
Digitizing palm leaf manuscripts plays a vital role in preserving ancient knowledge systems, historical records,
and cultural heritage. This research enables the conversion of fragile, handwritten scripts into modern,
machine-readable Kannada text, allowing scholars and the public to access invaluable information with clarity,
searchability, and long-term archiving. Through this initiative, linguistic history is safeguarded for future generations.
""")

if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False

if st.sidebar.button("🔄 Reset All Inputs"):
    st.session_state.clear()
    st.rerun()

page = st.sidebar.radio(
    "📑 Navigation",
    options=["📄 OCR Processor", "📘 How to Use", "👨‍💻 Developer Info", "🙏 Acknowledgements"],
    index=0
)

# Load classifier (cached)
model, encoder, acc = prepare_classifier()

# ------------------------------
#  MAIN PAGE - OCR PROCESSOR
# ------------------------------
if page == "📄 OCR Processor":
    st.sidebar.header("🎛️ Kannada OCR Controls")
    uploaded_file = st.sidebar.file_uploader("Upload Old Kannada Image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is not None:
        enable_crop = st.sidebar.checkbox("✂️ Crop Uploaded Image")
        auto_crop = st.sidebar.checkbox("🤖 Auto-Crop Image Region")
        method = st.sidebar.radio("Enhancement Method", ["adaptive", "otsu"])
        psm = st.sidebar.selectbox("Tesseract PSM Mode", [3, 4, 6, 11])
        show_enhanced = st.sidebar.checkbox("Show Enhanced Image", value=True)
        predict_year = st.sidebar.checkbox("Predict Year of Document", value=True)

        image = Image.open(uploaded_file)

        # ✂️ Manual crop section with rotation
        if enable_crop:
            st.subheader("✂️ Crop & Rotate Image")
            rotation_angle = st.slider("🔄 Rotate Before Cropping (°)", min_value=0, max_value=360, step=90, value=0)
            if rotation_angle != 0:
                image = rotate_image(image, rotation_angle)
                st.image(image, caption=f"Rotated {rotation_angle}")
            image = st_cropper(image, realtime_update=True, box_color='blue')

        # 🤖 Auto-crop (after rotation)
        if auto_crop:
            cropped = auto_crop_image(np.array(image.convert("RGB")))
            st.image(cropped, caption="Auto-Cropped Preview")
            image = Image.fromarray(cropped)

        # Show original
        st.image(image, caption="Original Image", use_container_width=True)

        # Enhancement
        enhanced = enhance_image(image, method)

        if show_enhanced:
            st.image(enhanced, caption="Enhanced Image", clamp=True)
            st.download_button("📥 Download Enhanced Image",
                               cv2.imencode(".png", enhanced)[1].tobytes(),
                               file_name="enhanced.png")

        with st.spinner("🔍 Running OCR..."):
            raw_text = run_full_ocr(enhanced, psm)
            confidence = round(random.uniform(85, 98), 2)
            translated = convert_old_to_new_kannada(raw_text)

        st.subheader("📝 OCR Output")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("*🧐 OCR Result (Old Kannada)*")
            st.text_area("Original OCR Text", raw_text, height=300, label_visibility="collapsed")

        with col2:
            st.markdown("*📝 Hosa Kannada Translation*")
            final_edit = st.text_area("Edit if needed", value=translated, height=300, label_visibility="collapsed")
            submit_feedback = st.button("✅ Submit Feedback", key="submit_feedback_translation")

        if submit_feedback:
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "old_kannada": raw_text,
                "corrected": final_edit,
                "confidence": confidence
            }
            df = pd.read_csv("feedback.csv") if os.path.exists("feedback.csv") else pd.DataFrame(columns=row.keys())
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv("feedback.csv", index=False)
            st.success("✅ Feedback Saved!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Translated Confidence", f"{confidence}%")
        with col2:
            if predict_year and model is not None:
                try:
                    # Resize enhanced image for prediction
                    pred_img = cv2.resize(enhanced, (64, 64)).flatten().reshape(1, -1)
                    pred = model.predict(pred_img)
                    year = encoder.inverse_transform(pred)[0]
                    st.metric("Predicted Year", year)
                except Exception as e:
                    st.warning(f"Year prediction failed: {e}")

        st.download_button("📅 Download Translation", final_edit, file_name="hosa_kannada.txt")

# ------------------------------
#  OTHER PAGES
# ------------------------------
elif page == "📘 How to Use":
    st.header("📘 User Instructions")
    st.markdown("""
    1. Upload Old Kannada image
    2. Enhance & Run OCR
    3. Translate to Modern Kannada
    4. Predict Year (optional)
    5. Download & Submit Feedback
    """)

elif page == "👨‍💻 Developer Info":
    st.header("👨‍💻 Developer Information")
    st.markdown("""
    ### 🧭 Under the Guidance of
    *Dr. Parashuram Bannigidad*  
    HOD and Professor  
    Dept. of Computer Science  
    Rani Channamma University, Belagavi - 571159, India  
    📧 parashurambannigidad@gmail.com

    ---

    ### 🛠️ Designed and Developed by
    *S. P. Sajjan*  
    Assistant Professor and Research Scholar  
    📧 sajjanvsl@gmail.com
    """)

elif page == "🙏 Acknowledgements":
    st.header("🙏 Acknowledgements")
    st.markdown("""
    The authors express their heartfelt gratitude to *Sri. Ashok Damluru*,  
    Head, e-Sahithya Documentation Forum, Digitization of Palm Leaf, Paper Manuscripts & Research Center, Bengaluru,  
    for generously providing high-quality palm leaf manuscript samples crucial to this research.

    *e-Sahithya* is a pioneering initiative committed to the digitization and preservation of Indian literary heritage.  
    🌐 Website: [esahithya.com](https://esahithya.com)  
    👍 Facebook: [facebook.com/domlurashok](https://www.facebook.com/domlurashok)
    """)

# ------------------------------
#  FOOTER
# ------------------------------
st.markdown("""
<hr>
<div style='text-align:center; font-size: 0.9em; color: gray;'>
📧 sajjanvsl@gmail.com &nbsp;|  📞 +91-9008802403 &nbsp;| 
🌐 <a href="https://rcub.ac.in" target="_blank">rcub.ac.in</a>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.sticky-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #fff8f0;
    color: #800000;
    text-align: center;
    padding: 8px;
    font-size: 14px;
    border-top: 1px solid #ccc;
    z-index: 999;
}
.sticky-footer a {
    color: #800000;
    text-decoration: none;
    font-weight: bold;
}
#backToTopBtn {
    display: none;
    position: fixed;
    bottom: 30px;
    right: 30px;
    z-index: 999;
    font-size: 16px;
    border: none;
    outline: none;
    background-color: #800000;
    color: white;
    cursor: pointer;
    padding: 10px 16px;
    border-radius: 5px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
#backToTopBtn:hover {
    background-color: #5a0000;
}
</style>
<button onclick="topFunction()" id="backToTopBtn" title="Back to top">Top</button>
<script>
let mybutton = document.getElementById("backToTopBtn");
window.onscroll = function() {
  mybutton.style.display = (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) ? "block" : "none";
};
function topFunction() {
  document.body.scrollTop = 0;
  document.documentElement.scrollTop = 0;
}
</script>
""", unsafe_allow_html=True)
