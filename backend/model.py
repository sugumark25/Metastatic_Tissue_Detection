
import torch.nn as nn
from torchvision import models


def get_model(num_classes=2):
   
    # Load pretrained ResNet-34 from torchvision
    # weights=None means random initialization (alternative: weights=ResNet34_Weights.IMAGENET1K_V1)
    model = models.resnet34(weights=None)
    

    # ❄️ FREEZE all parameters in early layers
    # These layers learn general image features (edges, textures, colors)
    # We don't want to change these - keep them as initialized
    print("\n❄️  Freezing early layers (Conv1, Layer1, Layer2, Layer3)...")
    for param in model.parameters():
        param.requires_grad = False  # Stop gradient updates

    # 🔥 UNFREEZE layer4 (last residual block)
    # This layer is closest to the classification task
    # Allow it to learn cancer-specific features
    print("🔥 Unfreezing Layer4 for cancer-specific feature learning...")
    for param in model.layer4.parameters():
        param.requires_grad = True  # Allow gradient updates

    # Modify final classification layer
    # ResNet-34 outputs 512 features → we need 2 class probabilities
    num_ftrs = model.fc.in_features  # 512
    print(f"\n🔧 Replacing final FC layer: {num_ftrs} → {num_classes}")
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    print("\n✅ Model architecture ready:")
    print(f"   • Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   • Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   • Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}\n")

    return model
