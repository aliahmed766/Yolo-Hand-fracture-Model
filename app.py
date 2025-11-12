import streamlit as st
from PIL import Image
import numpy as np
import torch

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path='models/best.pt',   # your model file
        force_reload=False
    )
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
        results = model(img_array)

    st.subheader("✅ Detection Result")
    output_image = results.render()[0]
    st.image(output_image, caption="Detected Fracture", use_column_width=True)

    st.subheader("📄 Raw Model Output")
    st.write(results.pandas().xyxy[0])
