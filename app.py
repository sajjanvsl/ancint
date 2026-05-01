import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import zipfile

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def apply_hist_eq(img):
    """Global histogram equalization"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def apply_clahe(img, clip_limit, grid_size):
    """CLAHE adaptive equalization"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def apply_sharpen(img, strength):
    """Sharpen using custom kernel"""
    kernel = np.array([[0, -1, 0],
                       [-1, 4 + strength, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(img, -1, kernel)
    return sharp

def apply_denoise(img, h, template_window, search_window):
    """Non-local means denoising"""
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, template_window, search_window)

def apply_binarization(img, method_type, threshold_val=127, block_size=11, constant=2, adaptive_method='gaussian'):
    """
    Convert image to binary (black and white).
    method_type: 'simple', 'otsu', 'adaptive'
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    if method_type == 'simple':
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
    elif method_type == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # adaptive
        if adaptive_method == 'gaussian':
            adaptive_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            adaptive_type = cv2.ADAPTIVE_THRESH_MEAN_C
        binary = cv2.adaptiveThreshold(gray, 255, adaptive_type, cv2.THRESH_BINARY, block_size, constant)
    
    # Convert back to 3-channel BGR for consistency
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def adjust_brightness_contrast(img, alpha, beta):
    """alpha = contrast, beta = brightness"""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def compute_accuracy(img_original, img_enhanced):
    """Contrast-based 'accuracy' (higher is better)"""
    def contrast(img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return gray.std()
    c_orig = contrast(img_original)
    c_enh = contrast(img_enhanced)
    if c_orig == 0:
        return 50.0
    improvement = (c_enh - c_orig) / c_orig * 100
    return max(0.0, min(100.0, improvement))

def get_histogram(img):
    """Compute grayscale histogram"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Ancient Script Enhancer", layout="wide")

# College Header
st.markdown("""
<div style='background-color:#2E4057; padding:10px; border-radius:10px; text-align:center; margin-bottom:20px'>
    <h3 style='color:white; margin:0'>Govt. First Grade College for Women, Jamkhandi</h3>
    <p style='color:#E0E0E0; margin:0'>Dept. of Computer Science and Application</p>
</div>
""", unsafe_allow_html=True)

st.title("Ancient Script Enhancement Tool")

menu = st.sidebar.radio("Navigate", ["Enhancement", "Dashboard", "About"])

# Session state
if "results" not in st.session_state:
    st.session_state["results"] = {}
if "comparison_images" not in st.session_state:
    st.session_state["comparison_images"] = []  # each: (original, enhanced, method)

# =========================
# ENHANCEMENT PAGE
# =========================
if menu == "Enhancement":
    st.header("Enhance an Ancient Script Image")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if original is None:
            st.error("Cannot read image. Please try another file.")
            st.stop()

        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Original Image")
            st.image(original_rgb, use_container_width=True)
            # Histogram
            hist = get_histogram(original)
            fig_hist, ax = plt.subplots()
            ax.plot(hist, color='black')
            ax.set_title("Original Histogram")
            ax.set_xlabel("Pixel intensity")
            ax.set_ylabel("Frequency")
            st.pyplot(fig_hist)

        with col2:
            st.subheader("Enhancement Settings")
            method = st.selectbox("Method", 
                                  ["Histogram Equalization", "CLAHE", "Sharpening", "Denoising", "Binarization"])

            # Method-specific parameters
            if method == "CLAHE":
                clip_limit = st.slider("Clip limit", 1.0, 5.0, 2.0, 0.1)
                grid_size = st.slider("Tile grid size", 4, 16, 8, 2)
            elif method == "Sharpening":
                strength = st.slider("Sharpening strength", 0.5, 3.0, 1.0, 0.1)
            elif method == "Denoising":
                h = st.slider("Denoising strength (h)", 3, 15, 10, 1)
                template_window = st.slider("Template window size", 5, 15, 7, 2)
                search_window = st.slider("Search window size", 11, 25, 21, 2)
            elif method == "Binarization":
                binary_type = st.selectbox("Threshold type", ["Simple", "Otsu", "Adaptive"])
                if binary_type == "Simple":
                    threshold_val = st.slider("Threshold value", 0, 255, 127, 1)
                elif binary_type == "Adaptive":
                    adaptive_method = st.selectbox("Adaptive method", ["Gaussian", "Mean"])
                    block_size = st.slider("Block size (odd)", 3, 31, 11, 2)
                    # ensure block_size is odd
                    if block_size % 2 == 0:
                        block_size += 1
                    constant = st.slider("Constant C", 0, 20, 2, 1)
                # For Otsu, no extra params
                threshold_val_local = threshold_val if binary_type == "Simple" else 0

            # Brightness / Contrast (only for non-binarization methods? but can apply after)
            contrast_val = st.slider("Contrast", 0.5, 2.0, 1.0, 0.01)
            brightness_val = st.slider("Brightness", -50, 50, 0, 1)

            if st.button("Enhance Now", type="primary"):
                with st.spinner("Enhancing..."):
                    try:
                        # Apply selected method
                        if method == "Histogram Equalization":
                            enhanced = apply_hist_eq(original)
                        elif method == "CLAHE":
                            enhanced = apply_clahe(original, clip_limit, grid_size)
                        elif method == "Sharpening":
                            enhanced = apply_sharpen(original, strength)
                        elif method == "Denoising":
                            enhanced = apply_denoise(original, h, template_window, search_window)
                        elif method == "Binarization":
                            if binary_type == "Simple":
                                enhanced = apply_binarization(original, 'simple', threshold_val=threshold_val)
                            elif binary_type == "Otsu":
                                enhanced = apply_binarization(original, 'otsu')
                            else:  # Adaptive
                                enhanced = apply_binarization(original, 'adaptive', 
                                                              block_size=block_size, 
                                                              constant=constant,
                                                              adaptive_method=adaptive_method.lower())
                        else:
                            enhanced = original.copy()

                        # Fine-tuning (skip for binarization? but can still adjust contrast/brightness on binary image)
                        enhanced = adjust_brightness_contrast(enhanced, contrast_val, brightness_val)

                        # Compute accuracy
                        acc = compute_accuracy(original, enhanced)
                        st.session_state["results"][method] = acc
                        st.session_state["comparison_images"].append((original.copy(), enhanced.copy(), method))

                        st.success(f"Enhancement successful! Accuracy (contrast improvement): {acc:.2f}%")

                        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                        st.image(enhanced_rgb, caption=f"Enhanced ({method})", use_container_width=True)

                        # Enhanced histogram
                        hist_enh = get_histogram(enhanced)
                        fig_hist2, ax2 = plt.subplots()
                        ax2.plot(hist_enh, color='red')
                        ax2.set_title("Enhanced Histogram")
                        ax2.set_xlabel("Pixel intensity")
                        ax2.set_ylabel("Frequency")
                        st.pyplot(fig_hist2)

                        # Download button
                        _, buffer = cv2.imencode('.png', enhanced)
                        st.download_button(
                            label="Download Enhanced Image",
                            data=buffer.tobytes(),
                            file_name=f"{method.replace(' ', '_')}_enhanced.png",
                            mime="image/png"
                        )

                    except Exception as e:
                        st.error(f"Enhancement failed: {str(e)}")

    # Batch processing (with binarization)
    with st.expander("Batch Processing (multiple images)"):
        batch_files = st.file_uploader("Upload several images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if batch_files:
            batch_method = st.selectbox("Method for batch", 
                                        ["Histogram Equalization", "CLAHE", "Sharpening", "Denoising", "Binarization"],
                                        key="batch_method")
            if batch_method == "Binarization":
                batch_binary_type = st.selectbox("Threshold type (batch)", ["Simple", "Otsu", "Adaptive"], key="batch_binary")
                # Simplifying: use fixed parameters for batch (to avoid too many sliders)
            if st.button("Process Batch"):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for idx, file in enumerate(batch_files):
                        bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
                        img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
                        if img is None:
                            continue
                        # Apply selected method with default parameters
                        if batch_method == "Histogram Equalization":
                            enh = apply_hist_eq(img)
                        elif batch_method == "CLAHE":
                            enh = apply_clahe(img, 2.0, 8)
                        elif batch_method == "Sharpening":
                            enh = apply_sharpen(img, 1.0)
                        elif batch_method == "Denoising":
                            enh = apply_denoise(img, 10, 7, 21)
                        elif batch_method == "Binarization":
                            if batch_binary_type == "Simple":
                                enh = apply_binarization(img, 'simple', threshold_val=127)
                            elif batch_binary_type == "Otsu":
                                enh = apply_binarization(img, 'otsu')
                            else:
                                enh = apply_binarization(img, 'adaptive', block_size=11, constant=2, adaptive_method='gaussian')
                        else:
                            enh = img
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
        st.warning("No enhancement results yet. Go to the Enhancement page and run some methods.")
    else:
        # Accuracy ranking
        if st.session_state["results"]:
            df = pd.DataFrame(list(st.session_state["results"].items()), columns=["Method", "Accuracy"])
            df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
            best_method = df.iloc[0]["Method"]
            best_score = df.iloc[0]["Accuracy"]
            st.success(f"Best Method: {best_method} ({best_score:.2f}%)")

            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(df["Method"], df["Accuracy"], color='skyblue')
            ax.set_xlabel("Method")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Enhancement Accuracy Comparison")
            ax.set_ylim(0, 105)
            for bar, val in zip(bars, df["Accuracy"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.1f}", ha='center', va='bottom', fontsize=10)
            st.pyplot(fig)
            st.dataframe(df, use_container_width=True)

        # Side-by-side comparison history
        if st.session_state["comparison_images"]:
            st.subheader("Recent Comparisons")
            for orig, enh, meth in st.session_state["comparison_images"][-5:]:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Original", width=250)
                with col_b:
                    st.image(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB), caption=f"{meth}", width=250)

        # Clear history button
        if st.button("Clear All History"):
            st.session_state["results"] = {}
            st.session_state["comparison_images"] = []
            st.rerun()

# =========================
# ABOUT PAGE
# =========================
elif menu == "About":
    st.header("Project Information")
    st.markdown("""
    ### Ancient Script Enhancement Tool

    **Student Name:** Parvati S Savalagi  
    **College:** Govt. First Grade College for Women, Jamkhandi  
    **Department:** Computer Science and Application  

    **Features:**
    - Five enhancement methods: Histogram Equalization, CLAHE, Sharpening, Denoising, **Binarization**
    - Binarization options: Simple threshold, Otsu's method, Adaptive threshold
    - Adjustable parameters for each method
    - Brightness & Contrast fine-tuning
    - Histogram visualisation (original vs enhanced)
    - Accuracy metric based on contrast improvement
    - Batch processing with ZIP download
    - Results dashboard with method ranking
    - Side-by-side comparison history

    **Technology Stack:** Streamlit, OpenCV, NumPy, Pandas, Matplotlib, Pillow
    """)
