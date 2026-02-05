import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import os
import sys

# Allow imports from backend/
sys.path.append("backend")

from model import get_model
from dataset import get_transforms

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Metastatic Image Detection",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Metastatic Image Detection")
st.write("Histopathology Image Classification")
st.write("Classes: **Normal vs Cancer**")

# ================= DEVICE =================
device = "cpu"

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pth")

# ================= LOAD TRANSFORMS =================
_, eval_transform = get_transforms()
CLASS_NAMES = ["Normal", "Cancer"]

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ model.pth not found in models/")
        st.stop()

    model = get_model()
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


model = load_model()
st.success("✅ Model loaded successfully")

# ================= IMAGE UPLOAD =================
uploaded_file = st.file_uploader(
    "Upload histopathology image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if image.width < 224 or image.height < 224:
        st.error("❌ Image too small (min 224×224)")
        st.stop()

    if st.button("🔍 Predict"):
        with st.spinner("Running inference..."):
            tensor = eval_transform(image).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)[0]

        pred = CLASS_NAMES[int(probs.argmax())]

        st.subheader("🧠 Prediction")
        st.write(f"**Result:** {pred}")
        st.write(f"**Confidence:** {float(probs.max()):.4f}")

        st.write("### Probabilities")
        st.json({
            "Normal": round(float(probs[0]), 4),
            "Cancer": round(float(probs[1]), 4)
        })
