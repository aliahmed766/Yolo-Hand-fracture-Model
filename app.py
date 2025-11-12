import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Hand Fracture Detection", layout="centered")

st.title("🦴 Hand Fracture Detection | YOLOv8")
st.write("Upload an X-ray hand image to detect fractures using an AI model.")

# ✅ Load YOLO model
@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"Model failed to load: {e}")
        return None

model = load_model()

# ✅ Upload image
uploaded_file = st.file_uploader("Upload Hand X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img)

    st.write("🔍 Detecting fractures...")

    try:
        # ✅ Run inference
        results = model.predict(img_array)

        # ✅ Use PIL result (no cv2 needed)
        result_img = Image.fromarray(results[0].plot())

        st.image(result_img, caption="Detection Result", use_column_width=True)
        st.success("✅ Detection Complete")

    except Exception as e:
        st.error(f"Error during detection: {e}")

else:
    st.info("Please upload an image to begin.")
