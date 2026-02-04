# --- Altair Compatibility Fix for Streamlit ---
# This must be at the VERY TOP of the file, before any other imports
try:
    from altair.vegalite.v4.api import Chart
    from altair.vegalite.v4.schema.core import Facet
except ImportError:
    # For newer versions of Altair (v6+)
    try:
        from altair.api import Chart
        from altair.schema.core import Facet
        # Apply monkey patch to streamlit's altair module
        import sys
        import types
        
        # Create a fake v4 module that redirects to the new location
        class FakeV4Module:
            api = types.ModuleType('altair.vegalite.v4.api')
            schema = types.ModuleType('altair.vegalite.v4.schema')
            
        FakeV4Module.api.Chart = Chart
        FakeV4Module.schema.core = types.ModuleType('altair.vegalite.v4.schema.core')
        FakeV4Module.schema.core.Facet = Facet
        
        # Check if we need to patch sys.modules
        if 'altair.vegalite.v4' not in sys.modules:
            sys.modules['altair.vegalite.v4'] = FakeV4Module()
            sys.modules['altair.vegalite.v4.api'] = FakeV4Module.api
            sys.modules['altair.vegalite.v4.schema.core'] = FakeV4Module.schema.core
            
    except ImportError:
        pass  # Altair might not be installed yet, Streamlit Cloud will handle it

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import gdown  # pip install gdown if you host model on Google Drive

# --- Page Config ---
st.set_page_config(page_title="Metastatic Tissue Detector", layout="centered")
st.title("🔬 Metastatic Tissue Detection")
st.write("Upload a histopathology image (lymph node scans) to detect metastatic tissue.")

# --- Model Path and Download ---
MODEL_PATH = "models/model.pth"
# Replace with your actual Google Drive file ID
# MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  

@st.cache_resource
def load_model():
    """Load PyTorch model from local or download if missing"""
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Model not found at {MODEL_PATH}. Please ensure the model file exists.")
        return None
    
    try:
        # Load the model
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Provide more detailed error info
        if "pickle" in str(e).lower():
            st.info("Tip: The model file might be corrupted or saved in an incompatible format.")
        return None

# Load the model
model = load_model()

# --- Image Preprocessing ---
def preprocess_image(image: Image.Image):
    """Resize and normalize the image for the PyTorch model"""
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # Adjust to your model input size
        transforms.ToTensor(),         # Scale pixels to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# --- Sidebar Upload ---
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader(
    "Choose a tissue scan...", 
    type=["jpg", "png", "jpeg", "tif", "tiff", "bmp"],
    help="Upload histopathology images for analysis"
)

# Add some example images or instructions
with st.sidebar.expander("ℹ️ Instructions"):
    st.write("""
    1. Upload a histopathology image
    2. Click 'Run Analysis'
    3. View the detection results
    4. For best results, use clear tissue scan images
    """)

# --- Main Prediction ---
if uploaded_file is not None:
    # Display uploaded image
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Tissue Scan', use_column_width=True)
        
        # Show image info
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Format:** {image.format or 'Unknown'}")
        with col2:
            st.write(f"**Size:** {image.size[0]}×{image.size[1]} pixels")
        
        if st.button("Run Analysis", type="primary"):
            if model is not None:
                with st.spinner("Analyzing tissue structure..."):
                    try:
                        # Preprocess image
                        input_tensor = preprocess_image(image)
                        
                        # Make prediction
                        with torch.no_grad():
                            output = model(input_tensor)
                            
                            # Handle different model output formats
                            if output.shape[1] == 1:  # Sigmoid output
                                confidence = torch.sigmoid(output)[0][0].item()
                            else:  # Softmax with 2+ outputs
                                probs = torch.softmax(output, dim=1)
                                if probs.shape[1] >= 2:
                                    confidence = probs[0][1].item()  # metastatic class
                                else:
                                    confidence = probs[0][0].item()
                        
                        # Display results
                        st.divider()
                        st.subheader("🔍 Analysis Results")
                        
                        # Create columns for results
                        result_col, conf_col = st.columns(2)
                        
                        with result_col:
                            if confidence > 0.5:
                                st.error("**Result: Metastatic Tissue Detected**")
                                st.write("The model has detected potential metastatic tissue.")
                            else:
                                st.success("**Result: Healthy Tissue (Negative)**")
                                st.write("No metastatic tissue detected.")
                        
                        with conf_col:
                            # Display confidence with progress bar
                            conf_value = confidence if confidence > 0.5 else (1 - confidence)
                            st.metric("Confidence", f"{conf_value:.2%}")
                            st.progress(conf_value)
                        
                        # Add interpretation guidance
                        with st.expander("📊 Interpretation Notes"):
                            st.write("""
                            **Confidence Score Interpretation:**
                            - **> 80%**: High confidence in prediction
                            - **60-80%**: Moderate confidence
                            - **50-60%**: Low confidence, consider review
                            
                            **Disclaimer:** This tool is for research/assistive purposes only. 
                            Always consult with a certified pathologist for clinical diagnosis.
                            """)
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        st.info("Please try with a different image or check the image format.")
            else:
                st.error("Model is not available. Please check the model file.")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        st.info("Please upload a valid image file.")
else:
    # Display welcome/instruction message
    st.info("👈 Please upload an image file from the sidebar to begin analysis.")
    
    # Add example section or instructions
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Supported Formats:
        - JPEG, PNG
        - TIFF, BMP
        - Color or grayscale
        """)
    
    with col2:
        st.markdown("""
        ### Recommended:
        - Clear tissue scans
        - Proper lighting
        - High resolution
        """)

# Add footer
st.divider()
st.caption("Metastatic Tissue Detection System | For Research Use Only")
