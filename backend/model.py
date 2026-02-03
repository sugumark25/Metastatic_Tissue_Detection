import torch.nn as nn
from torchvision import models

def get_model(num_classes=2):
    model = models.resnet34(weights=None)  # 🚫 NO pretrained weights

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Modify last layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model
