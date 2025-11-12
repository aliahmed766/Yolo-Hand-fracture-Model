import streamlit as st
import sys
import os

# ✅ Fix OpenCV issues on Streamlit Cloud
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import cv2
cv2.setNumThreads(1)
sys.modules['cv2'] = cv2

from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Hand Fracture Detection", layout="centered")

st.title("🦴 Hand Fracture Detection | YOLOv8")
st.write("Upload an X-ray hand image to detect fractures using an AI model.")

# ✅ Load model safely
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

uploaded_file = st.file_uploader("Upload Hand X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    # ✅ Read image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ✅ Convert to numpy
    img_array = np.array(img)

    st.write("🔍 Detecting fractures...")

    # ✅ Run inference
    try:
        results = model(img_array)
        result_img = results[0].plot()   # YOLO renders bounding boxes

        st.image(result_img, caption="Detection Result", use_column_width=True)
        st.success("✅ Detection Complete")

    except Exception as e:
        st.error(f"Error during detection: {e}")

else:
    st.info("Please upload an image to begin.")
