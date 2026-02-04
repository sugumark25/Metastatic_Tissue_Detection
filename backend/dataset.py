

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random


class CancerDataset(Dataset):
   
    def __init__(self, data_dir, transform=None, split="train"):
        
        self.data_dir = data_dir
        self.transform = transform

        # Path to image directory and labels file
        images_dir = os.path.join(data_dir, "data_sample")
        labels_path = os.path.join(data_dir, "labels.csv")

        # Load CSV: columns [id, label] where label ∈ {0, 1}
        # 0 = Normal/Non-Metastatic
        # 1 = Metastatic
        df = pd.read_csv(labels_path)
        df["filename"] = df["id"].astype(str) + ".tif"

        # Build list of (image_path, label) tuples, filtering missing files
        self.samples = []
        for _, row in df.iterrows():
            img_path = os.path.join(images_dir, row["filename"])
            if os.path.exists(img_path):
                self.samples.append((img_path, int(row["label"])))

        # Shuffle with fixed seed for reproducibility
        # ✅ Ensures same split every run
        random.seed(42)
        random.shuffle(self.samples)

        # Deterministic train-val-test split
        # 70% training, 15% validation, 15% test
        total = len(self.samples)
        train_end = int(0.7 * total)   # indices 0 to 70%
        val_end = int(0.85 * total)    # indices 70% to 85%
        # test: indices 85% to 100%

        if split == "train":
            self.samples = self.samples[:train_end]
        elif split == "val":
            self.samples = self.samples[train_end:val_end]
        else:  # test
            self.samples = self.samples[val_end:]
        
        print(f"✅ Loaded {len(self.samples)} {split} samples from {data_dir}")

    def __len__(self):
        """Returns total number of samples in this split."""
        return len(self.samples)

    def __getitem__(self, idx):
       
        path, label = self.samples[idx]
        
        # Load image: convert to RGB (may be grayscale originally)
        img = Image.open(path).convert("RGB")
        
        # Apply transform if provided (resize, normalize, augmentation)
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_transforms():
   
    # ImageNet normalization statistics
    # Used for pre-trained ResNet from torchvision
    mean = [0.485, 0.456, 0.406]  # R, G, B mean values
    std = [0.229, 0.224, 0.225]   # R, G, B standard deviation

    # TRAINING TRANSFORMS: Include augmentation
    # Goal: Increase dataset diversity, prevent overfitting
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),           # Standardize size for model
        transforms.RandomHorizontalFlip(),       # Tissue orientation invariance
        transforms.RandomVerticalFlip(),         # Tissue orientation invariance
        transforms.RandomRotation(15),           # Handle rotated tissue patches
        transforms.ColorJitter(0.3, 0.3, 0.3),  # Resist staining variations
        transforms.ToTensor(),                   # Convert to tensor [0, 1]
        transforms.Normalize(mean, std)          # Apply ImageNet normalization
    ])

    # EVALUATION TRANSFORMS: No augmentation (deterministic)
    # Goal: Consistent, reproducible evaluation
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),           # Standardize size for model
        transforms.ToTensor(),                   # Convert to tensor [0, 1]
        transforms.Normalize(mean, std)          # Apply ImageNet normalization
    ])

    return train_tf, eval_tf
