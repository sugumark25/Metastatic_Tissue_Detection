import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import gdown  # pip install gdown if you host model on Google Drive

# --- Altair Fix for Streamlit ---
try:
    from altair.vegalite.v4.api import Chart
    from altair.vegalite.v4.schema.core import Facet
except ImportError:
    try:
        from altair.api import Chart
        from altair.schema.core import Facet
    except ImportError:
        pass  # Streamlit cloud handles Altair installation separately

# --- Page Config ---
st.set_page_config(page_title="Metastatic Tissue Detector", layout="centered")
st.title("🔬 Metastatic Tissue Detection")
st.write("Upload a histopathology image (lymph node scans) to detect metastatic tissue.")

# --- Model Path and Download ---
MODEL_PATH = "models/model.pth"
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # Replace with your actual Google Drive file ID

@st.cache_resource
def load_model():
    """Load PyTorch model from local or download if missing"""
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    try:
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Image Preprocessing ---
def preprocess_image(image: Image.Image):
    """Resize and normalize the image for the PyTorch model"""
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # Adjust to your model input size
        transforms.ToTensor(),         # Scale pixels to [0,1]
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# --- Sidebar Upload ---
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader(
    "Choose a tissue scan...", type=["jpg", "png", "jpeg", "tif"]
)

# --- Main Prediction ---
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Tissue Scan', use_column_width=True)

    if st.button("Run Analysis"):
        if model is not None:
            with st.spinner("Analyzing cells..."):
                input_tensor = preprocess_image(image)
                with torch.no_grad():
                    output = model(input_tensor)
                    # Binary classification logic
                    if output.shape[1] == 1:  # Sigmoid
                        confidence = torch.sigmoid(output)[0][0].item()
                    else:  # Softmax with 2 outputs
                        probs = torch.softmax(output, dim=1)
                        confidence = probs[0][1].item()  # metastatic class

                st.divider()
                if confidence > 0.5:
                    st.error(f"**Result: Metastatic Tissue Detected**")
                    st.warning(f"Confidence Score: {confidence:.2%}")
                else:
                    st.success(f"**Result: Healthy Tissue (Negative)**")
                    st.info(f"Confidence Score: {(1 - confidence):.2%}")
        else:
            st.error("Model not loaded. Please check the model path.")
else:
    st.info("Please upload an image file to begin analysis.")
