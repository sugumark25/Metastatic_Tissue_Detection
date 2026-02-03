from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from PIL import Image
import io
import os

from model import get_model
from dataset import get_transforms

app = FastAPI(title="Medical Image Classification API")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Force CPU on Render free tier
device = "cpu"

model = None
model_path = "../models/model.pth"

_, eval_transform = get_transforms()
CLASS_NAMES = ["Normal", "Cancer"]

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(model_path):
        raise RuntimeError("Model weights not found. Commit models/model.pth to GitHub!")
    model = get_model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")

@app.get("/")
def read_root():
    return {"message": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Validate size
    if image.width < 224 or image.height < 224:
        raise HTTPException(status_code=400, detail="Image too small")

    # Transform
    tensor = eval_transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    normal_prob = float(probs[0])
    cancer_prob = float(probs[1])
    prediction = CLASS_NAMES[int(probs.argmax())]

    return {
        "prediction": prediction,
        "probabilities": {
            "Normal": normal_prob,
            "Cancer": cancer_prob
        }
    }
