import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import cv2

# -----------------------------
# Streamlit App Title
# -----------------------------
st.set_page_config(page_title="Hand Fracture Detection", page_icon="🩻", layout="centered")
st.title("🩻 Hand Fracture Detection using YOLOv8")
st.write("Upload a hand X-ray image below, and the model will detect any fractures.")

# -----------------------------
# Load YOLOv8 Model
# -----------------------------
model_path = "best.pt"  # Ensure best.pt is in same folder as app.py
try:
    model = YOLO(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -----------------------------
# File Upload Section
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload a Hand X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    # Temporary save
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    # -----------------------------
    # Prediction
    # -----------------------------
    st.write("🔍 Detecting fractures... Please wait.")
    results = model.predict(source=image_path, conf=0.5)
    result_image = results[0].plot()  # Draw detections

    # Convert BGR (OpenCV) to RGB (PIL)
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Display result
    st.image(result_rgb, caption="Detection Result", use_container_width=True)
    st.success("✅ Detection complete!")

    # Option to download
    result_pil = Image.fromarray(result_rgb)
    st.download_button(
        label="💾 Download Labeled Image",
        data=cv2.imencode('.jpg', cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))[1].tobytes(),
        file_name="fracture_detected.jpg",
        mime="image/jpeg"
    )
