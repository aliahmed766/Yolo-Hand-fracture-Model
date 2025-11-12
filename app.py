import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLO model from root folder
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # ✅ Your model file in root

st.title("🦴 Fracture Detection App")
st.write("Upload an X-ray image, and the model will detect any fractures.")

model = load_model()

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    with st.spinner("Detecting fracture..."):
        result = model.predict(img_array)[0]

    annotated = result.plot()

    st.subheader("✅ Detection Result")
    st.image(annotated, caption="Detected Fracture", use_column_width=True)

    st.subheader("📄 Raw Model Output")
    st.write(result.boxes.data.cpu().numpy())
