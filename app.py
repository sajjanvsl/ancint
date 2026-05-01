import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import zipfile
import os
import tensorflow as tf
from tensorflow.keras import layers, Model

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def apply_hist_eq(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def apply_clahe(img, clip_limit, grid_size):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def apply_sharpen(img, strength):
    kernel = np.array([[0, -1, 0],
                       [-1, 4 + strength, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(img, -1, kernel)
    return sharp

def apply_denoise(img, h, template_window, search_window):
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, template_window, search_window)

def adaptive_binarization(img, window_size=25, k=0.2, r=128):
    """Niblack‑style adaptive thresholding (Sauvola variant)"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Integral image for fast mean / std computation
    integral = cv2.integral(gray.astype(np.float32))
    height, width = gray.shape
    binary = np.zeros((height, width), dtype=np.uint8)
    pad = window_size // 2

    for i in range(height):
        for j in range(width):
            # Window boundaries
            y1 = max(0, i - pad)
            y2 = min(height, i + pad + 1)
            x1 = max(0, j - pad)
            x2 = min(width, j + pad + 1)

            # Mean and standard deviation via integral
            area = (y2 - y1) * (x2 - x1)
            sum_pixels = (integral[y2, x2] - integral[y1, x2] -
                          integral[y2, x1] + integral[y1, x1])
            mean = sum_pixels / area

            if area > 1:
                sq_sum = np.sum(gray[y1:y2, x1:x2] ** 2)
                variance = (sq_sum / area) - (mean ** 2)
                std = np.sqrt(max(variance, 0))
            else:
                std = 0

            # Sauvola threshold formula
            threshold = mean * (1 + k * ((std / r) - 1))
            binary[i, j] = 255 if gray[i, j] > threshold else 0

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# --------------------------------------------------
# U‑Net Model Definition
# --------------------------------------------------
def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)
    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    # Bottleneck
    b1 = conv_block(p4, 1024)
    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)
    model = Model(inputs, outputs)
    return model

# Pre‑trained model loader (if file exists)
def load_unet_model(model_path='unet_model.h5'):
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        return model
    else:
        st.warning("Pre‑trained model file not found. Using built‑in U‑Net (untrained).")
        return build_unet()

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Ancient Script Enhancer", layout="wide")
st.markdown("""
<div style='background-color:#2E4057; padding:10px; border-radius:10px; text-align:center; margin-bottom:20px'>
    <h3 style='color:white; margin:0'>Govt. First Grade College for Women, Jamkhandi</h3>
    <p style='color:#E0E0E0; margin:0'>Dept. of Computer Science and Application</p>
</div>
""", unsafe_allow_html=True)

st.title("Ancient Script Enhancement Tool")
menu = st.sidebar.radio("Navigate", ["Enhancement", "Dashboard", "About"])

if "results" not in st.session_state:
    st.session_state["results"] = {}
if "comparison_images" not in st.session_state:
    st.session_state["comparison_images"] = []

# =========================
# ENHANCEMENT PAGE
# =========================
if menu == "Enhancement":
    st.header("Enhance an Ancient Script Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if original is None:
            st.error("Cannot read image.")
            st.stop()
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Original Image")
            st.image(original_rgb, use_container_width=True)

        with col2:
            st.subheader("Enhancement Settings")
            method = st.selectbox("Method", [
                "Histogram Equalization", "CLAHE", "Sharpening", "Denoising",
                "Adaptive Binarization", "U‑Net Binarization"
            ])

            # Method‑specific parameters
            if method == "CLAHE":
                clip_limit = st.slider("Clip limit", 1.0, 5.0, 2.0)
                grid_size = st.slider("Tile grid size", 4, 16, 8)
            elif method == "Sharpening":
                strength = st.slider("Sharpening strength", 0.5, 3.0, 1.0)
            elif method == "Denoising":
                h = st.slider("Denoising strength", 3, 15, 10)
                template_window = st.slider("Template window", 5, 15, 7)
                search_window = st.slider("Search window", 11, 25, 21)
            elif method == "Adaptive Binarization":
                window = st.slider("Window size (odd)", 15, 51, 25, 2)
                # Ensure odd
                if window % 2 == 0: window += 1
                k = st.slider("k factor", 0.1, 0.5, 0.2, 0.01)
                r = st.slider("r (dynamic range)", 64, 192, 128)
            # U‑Net uses default fixed parameters

            contrast_val = st.slider("Contrast", 0.5, 2.0, 1.0)
            brightness_val = st.slider("Brightness", -50, 50, 0)

            if st.button("Enhance Now", type="primary"):
                with st.spinner("Enhancing..."):
                    try:
                        if method == "Histogram Equalization":
                            enhanced = apply_hist_eq(original)
                        elif method == "CLAHE":
                            enhanced = apply_clahe(original, clip_limit, grid_size)
                        elif method == "Sharpening":
                            enhanced = apply_sharpen(original, strength)
                        elif method == "Denoising":
                            enhanced = apply_denoise(original, h, template_window, search_window)
                        elif method == "Adaptive Binarization":
                            enhanced = adaptive_binarization(original, window, k, r)
                        else:  # U‑Net Binarization
                            # Prepare model (load or build)
                            if 'unet_model' not in st.session_state:
                                with st.spinner("Loading U‑Net model..."):
                                    st.session_state.unet_model = load_unet_model()
                            model = st.session_state.unet_model
                            # Preprocess image
                            img_resized = cv2.resize(original, (256, 256))
                            img_norm = img_resized.astype(np.float32) / 255.0
                            img_batch = np.expand_dims(img_norm, axis=0)
                            pred = model.predict(img_batch)[0, :, :, 0]
                            # Resize mask back to original size
                            mask = (pred > 0.5).astype(np.uint8) * 255
                            mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
                            binary = np.stack([mask, mask, mask], axis=2)
                            enhanced = binary.astype(np.uint8)
                            st.info("U‑Net model applied (untrained demo). For best results, provide a trained .h5 model.")

                        # Brightness/Contrast
                        enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast_val, beta=brightness_val)

                        # Accuracy score (contrast improvement)
                        def contrast(img):
                            if len(img.shape) == 3:
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            else:
                                gray = img
                            return gray.std()
                        c_orig = contrast(original)
                        c_enh = contrast(enhanced)
                        acc = max(0.0, min(100.0, (c_enh - c_orig) / c_orig * 100 if c_orig != 0 else 50.0))
                        st.session_state["results"][method] = acc
                        st.session_state["comparison_images"].append((original.copy(), enhanced.copy(), method))

                        st.success(f"Enhancement successful! Accuracy: {acc:.2f}%")
                        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                        st.image(enhanced_rgb, use_container_width=True)

                        # Download button
                        _, buffer = cv2.imencode('.png', enhanced)
                        st.download_button("Download Enhanced Image", data=buffer.tobytes(),
                                           file_name=f"{method.replace(' ', '_')}_enhanced.png")

                    except Exception as e:
                        st.error(f"Enhancement failed: {str(e)}")

    # Batch processing (optional)
    with st.expander("Batch Processing (multiple images)"):
        batch_files = st.file_uploader("Upload several images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if batch_files:
            batch_method = st.selectbox("Method for batch", [
                "Histogram Equalization", "CLAHE", "Sharpening", "Denoising",
                "Adaptive Binarization", "U‑Net Binarization"
            ], key="batch_method")
            if st.button("Process Batch"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for idx, f in enumerate(batch_files):
                        bytes_data = np.asarray(bytearray(f.read()), dtype=np.uint8)
                        img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        if batch_method == "Histogram Equalization":
                            enh = apply_hist_eq(img)
                        elif batch_method == "CLAHE":
                            enh = apply_clahe(img, 2.0, 8)
                        elif batch_method == "Sharpening":
                            enh = apply_sharpen(img, 1.0)
                        elif batch_method == "Denoising":
                            enh = apply_denoise(img, 10, 7, 21)
                        elif batch_method == "Adaptive Binarization":
                            enh = adaptive_binarization(img, 25, 0.2, 128)
                        else:
                            # U‑Net
                            if 'unet_model' not in st.session_state:
                                st.session_state.unet_model = load_unet_model()
                            model = st.session_state.unet_model
                            img_resized = cv2.resize(img, (256, 256))
                            img_norm = img_resized.astype(np.float32) / 255.0
                            pred = model.predict(np.expand_dims(img_norm, axis=0))[0, :, :, 0]
                            mask = (pred > 0.5).astype(np.uint8) * 255
                            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                            enh = np.stack([mask, mask, mask], axis=2).astype(np.uint8)
                        _, buf = cv2.imencode('.png', enh)
                        zf.writestr(f"enhanced_{idx}.png", buf.tobytes())
                st.download_button("Download All as ZIP", data=zip_buffer.getvalue(),
                                   file_name="batch_enhanced.zip", mime="application/zip")
                st.success(f"Processed {len(batch_files)} images.")

# =========================
# DASHBOARD PAGE
# =========================
elif menu == "Dashboard":
    st.header("Results Dashboard")
    if not st.session_state["results"] and not st.session_state["comparison_images"]:
        st.warning("No enhancement results yet. Go to the Enhancement page.")
    else:
        if st.session_state["results"]:
            df = pd.DataFrame(list(st.session_state["results"].items()), columns=["Method", "Accuracy"])
            df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
            best_method = df.iloc[0]["Method"]
            best_score = df.iloc[0]["Accuracy"]
            st.success(f"Best Method: {best_method} ({best_score:.2f}%)")
            st.bar_chart(df.set_index("Method"))
            st.dataframe(df, use_container_width=True)

        if st.session_state["comparison_images"]:
            st.subheader("Recent Comparisons")
            for orig, enh, meth in st.session_state["comparison_images"][-5:]:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Original", width=250)
                with col_b:
                    st.image(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB), caption=meth, width=250)

        if st.button("Clear History"):
            st.session_state["results"] = {}
            st.session_state["comparison_images"] = []
            st.rerun()

# =========================
# ABOUT PAGE
# =========================
elif menu == "About":
    st.header("Project Information")
    st.markdown("""
    **Student Name:** Parvati S Savalagi  
    **College:** Govt. First Grade College for Women, Jamkhandi  
    **Department:** Computer Science and Application  

    **Features:**
    - Traditional: Histogram Equalization, CLAHE, Sharpening, Denoising
    - Adaptive binarization (Sauvola method using integral images)
    - U‑Net deep learning binarization (trainable with your own data)
    - Batch processing, download, dashboard comparisons

    **Technology:** Streamlit, OpenCV, NumPy, Pandas, Matplotlib, TensorFlow/Keras.
    """)
