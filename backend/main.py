
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from PIL import Image
import io
import os

from model import get_model
from dataset import get_transforms

# ============ FASTAPI APP SETUP ============
app = FastAPI(
    title="Medical Image Classification API",
    description="Binary classification of histopathology images: Normal vs Metastatic",
    version="1.0.0"
)

# ============ CORS CONFIGURATION ============
# Allow requests from frontend (different origin)
# In production: restrict to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (development)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ============ DEVICE CONFIGURATION ============
# Force CPU on Render free tier (no GPU)
# For local development: use torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# ============ GLOBAL VARIABLES ============
model = None  # Will be loaded on startup
model_path = "../models/model.pth"  # Path to saved model weights

_, eval_transform = get_transforms()  # Evaluation transforms (no augmentation)
CLASS_NAMES = ["Normal", "Cancer"]  # Human-readable class labels


@app.on_event("startup")
def load_model():

    global model
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"❌ Model weights not found at {model_path}\n"
            f"   Please ensure models/model.pth is committed to GitHub"
        )
    
    # Build model architecture
    model = get_model()
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move to device and set evaluation mode
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")


@app.get("/")
def read_root():
    
    return {
        "message": "🏥 Medical Image Classification API running",
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
   

    # ========== VALIDATION ==========
    # Check file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"❌ Invalid file type: {file.content_type}. Expected image/*"
        )

    # ========== IMAGE LOADING ==========
    # Read image bytes and convert to PIL Image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"❌ Failed to load image: {str(e)}"
        )

    # ========== IMAGE VALIDATION ==========
    # Check minimum size (ResNet-34 requires 224x224)
    if image.width < 224 or image.height < 224:
        raise HTTPException(
            status_code=400,
            detail=f"❌ Image too small: {image.width}×{image.height}. Min required: 224×224"
        )

    # ========== PREPROCESSING ==========
    # Apply evaluation transforms:
    # - Resize to 224×224
    # - Normalize using ImageNet statistics
    tensor = eval_transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # ========== INFERENCE ==========
    # Run model prediction
    try:
        with torch.no_grad():  # Disable gradient computation
            logits = model(tensor)  # Model output (unnormalized scores)
            probs = F.softmax(logits, dim=1)[0]  # Convert to probabilities [0-1]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"❌ Prediction failed: {str(e)}"
        )

    # ========== RESULT FORMATTING ==========
    # Extract probabilities for each class
    normal_prob = float(probs[0])    # Probability of Normal class
    cancer_prob = float(probs[1])    # Probability of Metastatic class
    
    # Determine final prediction (argmax)
    prediction = CLASS_NAMES[int(probs.argmax())]

    # ========== RESPONSE ==========
    return {
        "prediction": prediction,
        "confidence": float(probs.max()),  # Confidence of prediction
        "probabilities": {
            "Normal": round(normal_prob, 4),
            "Cancer": round(cancer_prob, 4)
        }
    }
