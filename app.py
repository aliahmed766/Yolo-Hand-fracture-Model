import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# --- Page Setup ---
st.set_page_config(page_title="Hand Fracture Detection", layout="centered")
st.title("🖐 Hand Fracture Detection App")
st.write("Upload an X-ray image and the app will detect fractures.")

# --- Load YOLO Model ---
@st.cache_resource
def load_model():
    # Make sure best.pt is in the same folder
    model = YOLO("best.pt")
    return model

model = load_model()
st.success("Model loaded successfully!")

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open the image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # --- Run Prediction ---
    with st.spinner("Detecting fractures..."):
        results = model.predict(image_np, imgsz=640)
    
    # Annotate image with bounding boxes
    annotated_image = results[0].plot()  # draws boxes
    annotated_image = Image.fromarray(annotated_image)
    
    # Display prediction
    st.image(annotated_image, caption="Predicted Fractures", use_column_width=True)
    
    # Show class names and confidence
    st.subheader("Detected Fractures")
    for r in results:
        if hasattr(r, 'boxes'):
            for box in r.boxes:
                cls = r.names[int(box.cls[0])]
                conf = float(box.conf[0])
                st.write(f"- {cls} : {conf:.2f}")
        else:
            st.write("No fractures detected.")
