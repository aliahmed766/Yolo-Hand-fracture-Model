import streamlit as st
import sys
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ✅ Check Python version (debug helper)
st.write("🧩 Python version:", sys.version)

# ✅ Page title
st.title("🖐️ Hand Fracture Detection using YOLOv8")
st.write("Upload an X-ray image of a hand to detect fractures using your trained YOLO model.")

# ✅ Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Ensure 'best.pt' is in the same folder
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("❌ Failed to load YOLO model. Make sure 'best.pt' is in the app directory.")
    st.stop()

# ✅ File uploader
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="🩻 Uploaded Image", use_column_width=True)

    # Run YOLO detection
    st.write("🔍 Detecting fractures...")
    results = model.predict(image, conf=0.25)

    # Convert result to image for display
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="✅ Detection Results", use_column_width=True)

    # Show detection labels and confidence
    st.subheader("📊 Detection Summary")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls] if model.names else f"Class {cls}"
        st.write(f"**{label}** – Confidence: {conf:.2f}")

else:
    st.info("📤 Please upload an image to begin detection.")
