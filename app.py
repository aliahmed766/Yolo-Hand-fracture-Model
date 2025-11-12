import streamlit as st
from PIL import Image
import yaml
from ultralytics import YOLO
import numpy as np

# -----------------------
# Load class names from data.yaml
# -----------------------
with open("data.yaml", "r") as file:
    data = yaml.safe_load(file)
class_names = data['names']  # ['Fracture']

# -----------------------
# Load YOLO model
# -----------------------
model = YOLO("best.pt")  # Make sure your trained weights are saved as best.pt

# -----------------------
# Streamlit App
# -----------------------
st.title("Hand Fracture Detection")
st.write("Upload an X-ray image to detect fractures.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict with YOLO
    results = model.predict(np.array(image))
    
    # Loop through results and display
    for result in results:
        boxes = result.boxes
        if len(boxes) > 0:
            st.success("Fracture detected!")
            for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                x1, y1, x2, y2 = map(int, box)
                label = class_names[int(cls_id)]
                st.write(f"Class: {label}, Confidence: {conf:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")
            
            # Show image with boxes
            annotated_image = result.plot()
            st.image(annotated_image, caption="Detected Fractures", use_column_width=True)
        else:
            st.info("No fracture detected in this image.")
