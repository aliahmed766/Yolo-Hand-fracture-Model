import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load model
@st.cache_resource
def load_model():
    model = YOLO("models/best.pt")   # <-- make sure best.pt is inside /models
    return model

st.title("🦴 Fracture Detection App")
st.write("Upload an X-ray image, and the model will detect any fractures.")

model = load_model()

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    with st.spinner("Detecting fracture..."):
        results = model.predict(img_array)[0]

    # Draw bounding boxes
    annotated = results.plot()  # YOLO returns image with boxes

    st.subheader("✅ Detection Result")
    st.image(annotated, caption="Detected Fracture", use_column_width=True)

    st.subheader("📄 Raw Predictions")
    st.write(results.boxes.data.cpu().numpy())
