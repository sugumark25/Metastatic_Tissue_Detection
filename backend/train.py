import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from dataset import CancerDataset, get_transforms
from model import get_model

def train():
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 15

    DATA_DIR = "../data" if os.path.exists("../data/labels.csv") else "data"
    MODEL_SAVE_PATH = "../models/model.pth" if DATA_DIR.startswith("..") else "models/model.pth"

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU NOT Detected. Using CPU. Ensure you have installed PyTorch with CUDA support.")
    
    print("Using device:", device)

    train_transform, eval_transform = get_transforms()

    train_dataset = CancerDataset(DATA_DIR, transform=train_transform, split="train")
    val_dataset = CancerDataset(DATA_DIR, transform=eval_transform, split="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = get_model()
    model.to(device)

    # Class weights
    weights = torch.tensor([1.0, 130908 / 89117]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_val_loss = np.inf

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _,  predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1), acc=100 * correct_train / total_train)

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1} | Train loss: {running_loss/len(train_loader):.4f} | Train Acc: {100*correct_train/total_train:.2f}% | Val loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("✅ Saved best model.")

    print("Training complete.")

if __name__ == "__main__":
    train()
