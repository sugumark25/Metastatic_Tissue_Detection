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

        images_dir = os.path.join(data_dir, "data_sample")
        labels_path = os.path.join(data_dir, "labels.csv")

        df = pd.read_csv(labels_path)
        df["filename"] = df["id"].astype(str) + ".tif"

        self.samples = []
        for _, row in df.iterrows():
            img_path = os.path.join(images_dir, row["filename"])
            if os.path.exists(img_path):
                self.samples.append((img_path, int(row["label"])))

        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(self.samples)

        # Deterministic split
        total = len(self.samples)
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)

        if split == "train":
            self.samples = self.samples[:train_end]
        elif split == "val":
            self.samples = self.samples[train_end:val_end]
        else:
            self.samples = self.samples[val_end:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3,0.3,0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])

    return train_tf, eval_tf
