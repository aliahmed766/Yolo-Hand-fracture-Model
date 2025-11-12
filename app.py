import sys
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

# --- Debug info (can remove later) ---
st.write("Python version:", sys.version)

# --- Streamlit page setup ---
st.set_page_config(page_title="Hand Fracture Detection", page_icon="🖐️", layout="centered")
st.title("🦴 Hand Fracture Detection using YOLO")

# --- Load YOLO model ---
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # Make sure best.pt is in the same repo
    return model

try:
    model = load_model()
    st.success("YOLO model loaded successfully ✅")
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

# --- Image upload section ---
uploaded_file = st.file_uploader("Upload a hand X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temporary file (needed for YOLO inference)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        temp_path = temp_file.name

    # --- Run YOLO detection ---
    st.write("🔍 Detecting fractures...")
    results = model(temp_path)

    # --- Process and display results ---
    for r in results:
        # Render YOLO output with bounding boxes
        annotated_frame = r.plot()  # returns a numpy array (BGR)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st.image(annotated_frame, caption="Detection Result", use_column_width=True)

    # --- Show detection summary ---
    detections = results[0].boxes
    if len(detections) > 0:
        st.success(f"✅ {len(detections)} fracture(s) detected.")
    else:
        st.warning("❌ No fractures detected.")
else:
    st.info("Please upload an image to begin.")
