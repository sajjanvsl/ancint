# -*- coding: utf-8 -*-
"""
Kannada OCR Web App – Streamlit Cloud Ready
Auto-detects Tesseract installation.
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
from gdown.exceptions import FileURLRetrievalError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import shutil

# ------------------------------
#  TESSERACT AUTO-CONFIGURATION
# ------------------------------
# Try to find tesseract binary automatically
tesseract_path = shutil.which("tesseract")
if tesseract_path is None:
    # Fallback to common paths (for Streamlit Cloud after installation)
    possible_paths = ["/usr/bin/tesseract", "/app/.apt/usr/bin/tesseract"]
    for p in possible_paths:
        if os.path.exists(p):
            tesseract_path = p
            break

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    st.error("""
    ❌ **Tesseract OCR is not installed on this server.**  
    To fix this, create a file named `packages.txt` in the **same folder as app.py** with the following content:
