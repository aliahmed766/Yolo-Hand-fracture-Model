import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Hand Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        border-color: #FF6B6B;
    }
    .upload-text {
        text-align: center;
        color: #666;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    h1 {
        color: #FF4B4B;
        text-align: center;
        padding-bottom: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None

# Title
st.title("🦴 Hand Fracture Detection System")
st.markdown("""
    <p style='text-align: center; color: #666; font-size: 1.1rem;'>
    Upload an X-ray image to detect and locate hand fractures using AI
    </p>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("1. Load Model")
    model_file = st.file_uploader(
        "Upload your best.pt model file",
        type=['pt'],
        help="Upload the trained YOLO model file"
    )

    st.subheader("2. Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.25, 0.05
    )

    iou_threshold = st.slider(
        "IOU Threshold",
        0.0, 1.0, 0.45, 0.05
    )

    st.markdown("---")
    st.subheader("ℹ️ About")
    st.info("""
    1. Upload your model file (best.pt)  
    2. Upload an X-ray image  
    3. Detect fractures  
    """)

@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Model file upload
if model_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name

    with st.spinner("Loading model..."):
        st.session_state.model = load_model(model_path)
        if st.session_state.model:
            st.session_state.model_loaded = True
            st.sidebar.success("✅ Model loaded!")

else:
    if os.path.exists("best.pt") and not st.session_state.model_loaded:
        with st.spinner("Loading model from best.pt..."):
            st.session_state.model = load_model("best.pt")
            if st.session_state.model:
                st.session_state.model_loaded = True
                st.sidebar.success("✅ Model loaded from best.pt")

# Main content
if not st.session_state.model_loaded:
    st.warning("⚠️ Please upload your model (best.pt).")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📤 Upload X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose Image",
            type=['jpg', 'jpeg', 'png']
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original X-ray", use_container_width=True)

            if st.button("🔍 Detect Fractures"):
                with st.spinner("Analyzing..."):
                    img = np.array(image)
                    results = st.session_state.model.predict(
                        img,
                        conf=confidence_threshold,
                        iou=iou_threshold,
                        verbose=False
                    )
                    st.session_state.results = results
                    st.session_state.detection_done = True

    with col2:
        st.subheader("📊 Detection Results")

        if st.session_state.get("detection_done"):
            results = st.session_state.results

            annotated_img = results[0].plot()[:, :, ::-1]
            st.image(annotated_img, caption="Detected Fractures", use_container_width=True)

            num = len(results[0].boxes)

            if num > 0:
                st.success(f"✅ {num} fracture(s) detected")

                for i, box in enumerate(results[0].boxes):
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = results[0].names[cls]

                    with st.expander(f"Detection #{i+1}: {name}"):
                        st.metric("Confidence", f"{conf:.2%}")
                        xyxy = box.xyxy[0].cpu().numpy()
                        st.caption(f"Coords: {xyxy.tolist()}")

            else:
                st.info("No fractures detected.")

            # Download button
            from io import BytesIO
            buf = BytesIO()
            Image.fromarray(annotated_img).save(buf, format='PNG')
            st.download_button(
                "📥 Download Annotated Image",
                data=buf.getvalue(),
                file_name="fracture_result.png",
                mime="image/png"
            )

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align:center;color:#666;font-size:0.9rem;'>
Hand Fracture Detection | YOLOv8 + Streamlit
</p>
""", unsafe_allow_html=True)
