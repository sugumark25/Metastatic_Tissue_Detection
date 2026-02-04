
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
    
    # ============ HYPERPARAMETERS ============
    BATCH_SIZE = 32       # Number of images per gradient step
    LR = 1e-4             # Learning rate (conservative for fine-tuning)
    EPOCHS = 15           # Number of training iterations over full dataset

    # ============ PATH CONFIGURATION ============
    # Adaptive path handling for different working directories
    DATA_DIR = "../data" if os.path.exists("../data/labels.csv") else "data"
    MODEL_SAVE_PATH = "../models/model.pth" if DATA_DIR.startswith("..") else "models/model.pth"

    # ============ DEVICE SELECTION ============
    # GPU if available (faster training), else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Capability: {torch.cuda.get_device_capability(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU NOT Detected. Using CPU.")
        print("   ℹ️ For faster training, install PyTorch with CUDA:")
        print("      pip install torch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")
    
    print(f"\n📍 Using device: {device}")

    # ============ DATA LOADING ============
    print("\n📂 Loading datasets...")
    train_transform, eval_transform = get_transforms()

    # Create train and validation datasets
    # Split: 70% train, 15% val, 15% test
    train_dataset = CancerDataset(DATA_DIR, transform=train_transform, split="train")
    val_dataset = CancerDataset(DATA_DIR, transform=eval_transform, split="val")

    # Create data loaders (batching, shuffling)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"✅ Train samples: {len(train_dataset):,}")
    print(f"✅ Validation samples: {len(val_dataset):,}")
    print(f"✅ Batches per epoch: {len(train_loader)}")

    # ============ MODEL SETUP ============
    print("\n🧠 Initializing model...")
    model = get_model()
    model.to(device)  # Move model to GPU/CPU

    # ============ LOSS FUNCTION (WEIGHTED FOR CLASS IMBALANCE) ============
    # Class weights: penalize minority class misclassification more
    # Class 0 (Normal): weight = 1.0 (baseline)
    # Class 1 (Metastatic): weight = 130908/89117 ≈ 1.47 (minority)
    # This makes the loss 1.47x larger when misclassifying metastatic cases
    weights = torch.tensor([1.0, 130908 / 89117]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"✅ Loss function: Weighted CrossEntropyLoss")
    print(f"   Class weights: Normal={weights[0]:.2f}, Metastatic={weights[1]:.2f}")

    # ============ OPTIMIZER ============
    # Adam: Adaptive learning rate optimization
    # Only optimize parameters that require gradients (layer4 + FC)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    print(f"✅ Optimizer: Adam (lr={LR})")

    # ============ LEARNING RATE SCHEDULER ============
    # Reduce learning rate if validation loss doesn't improve
    # patience=2: wait 2 epochs before reducing
    # factor=0.5: multiply LR by 0.5 when triggered
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    print(f"✅ Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)")

    # ============ TRAINING LOOP ============
    best_val_loss = np.inf
    print(f"\n{'='*80}")
    print(f"🎓 Starting training for {EPOCHS} epochs...")
    print(f"{'='*80}\n")

    for epoch in range(EPOCHS):
        # ========== TRAINING PHASE ==========
        model.train()  # Enable dropout and batch norm updates
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for inputs, labels in pbar:
            # Move data to device (GPU/CPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()  # Clear previous gradients
            outputs = model(inputs)  # Predict
            loss = criterion(outputs, labels)  # Compute loss

            # Backward pass
            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters

            # Compute training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update metrics
            running_loss += loss.item()
            avg_loss = running_loss / (pbar.n + 1)
            avg_acc = 100 * correct_train / total_train
            
            # Update progress bar
            pbar.set_postfix(loss=avg_loss, acc=f"{avg_acc:.2f}%")

        # ========== VALIDATION PHASE ==========
        model.eval()  # Disable dropout and batch norm updates
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():  # Disable gradient computation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # ========== EPOCH SUMMARY ==========
        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # Learning rate scheduling
        scheduler.step(val_loss)

        # Print epoch results
        print(f"Epoch {epoch+1:2d}/{EPOCHS} │ "
              f"Train Loss: {train_loss:.4f} │ Train Acc: {train_acc:.2f}% │ "
              f"Val Loss: {val_loss:.4f} │ Val Acc: {val_acc:.2f}%")

        # ========== CHECKPOINTING ==========
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Best model saved (val_loss: {val_loss:.4f})")

    # ============ TRAINING COMPLETE ============
    print(f"\n{'='*80}")
    print("🎉 Training complete!")
    print(f"{'='*80}")
    print(f"✅ Best model saved to: {MODEL_SAVE_PATH}")
    print(f"📊 Final validation loss: {best_val_loss:.4f}")
    print(f"\nNext steps:")
    print(f"  1. Start API: uvicorn main:app --reload")


if __name__ == "__main__":
    """
    Main entry point for training.
    
    Usage:
        python train.py
    """
