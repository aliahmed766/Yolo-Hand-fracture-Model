import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Hand Fracture Detection", layout="centered")
st.title("🖐 Hand Fracture Detection using YOLO")

# --- Load YOLO Model ---
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # make sure best.pt is in the same folder
    return model

model = load_model()

st.write("Model loaded successfully!")

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run prediction
    results = model.predict(image_np, imgsz=640)
    
    # Draw boxes on image
    annotated_frame = results[0].plot()
    annotated_image = Image.fromarray(annotated_frame)
    
    st.image(annotated_image, caption="Prediction", use_column_width=True)

    # Show class names and confidence
    st.write("Predictions:")
    for r in results:
        for box in r.boxes:
            cls = r.names[int(box.cls[0])]
            conf = float(box.conf[0])
            st.write(f"- {cls} : {conf:.2f}")
