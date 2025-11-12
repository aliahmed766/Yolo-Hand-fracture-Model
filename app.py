import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import torch

# ✅ FIX FOR PYTORCH 2.6 – allow YOLO model loading
from ultralytics.nn.tasks import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])

# Streamlit page config
st.set_page_config(
    page_title="Hand Fracture Detection",
    page_icon="🦴",
    layout="wide"
)

# UI styling
st.markdown("""
<style>
    .main { padding: 2rem; }
</style>
""", unsafe_allow_html=True)

# Session state
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.model_loaded = False


@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path, task="detect")   # ✅ SAFE LOAD
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    model_file = st.file_uploader("Upload best.pt model", type=["pt"])

    confidence = st.slider(
        "Confidence Threshold", 0.0, 1.0, 0.25, 0.05
    )

    iou = st.slider(
        "IOU Threshold", 0.0, 1.0, 0.45, 0.05
    )


# ✅ Load uploaded model or best.pt from root
if model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name

    with st.spinner("Loading model..."):
        st.session_state.model = load_model(model_path)
        st.session_state.model_loaded = True

else:
    if not st.session_state.model_loaded and os.path.exists("best.pt"):
        with st.spinner("Loading best.pt..."):
            st.session_state.model = load_model("best.pt")
            st.session_state.model_loaded = True


st.title("🦴 Hand Fracture Detection System")
st.markdown("<p>Upload an X-ray to detect fractures.</p>",
            unsafe_allow_html=True)

if not st.session_state.model_loaded:
    st.warning("⚠️ Please upload your model or place best.pt in root.")
    st.stop()

# Image upload
image_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if image_file:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Detect Fracture"):
        with st.spinner("Detecting..."):
            img_array = np.array(image)

            results = st.session_state.model.predict(
                img_array, conf=confidence, iou=iou
            )

            annotated = results[0].plot()[:, :, ::-1]  # BGR → RGB
            st.image(annotated, caption="Detection Result", use_container_width=True)

            import io
            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="PNG")

            st.download_button(
                "📥 Download result",
                buf.getvalue(),
                file_name="fracture_result.png",
                mime="image/png"
            )
