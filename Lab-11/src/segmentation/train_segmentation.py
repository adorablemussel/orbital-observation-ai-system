from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from src.segmentation.segmentation_dataset import SyntheticSegmentationDataset
from src.segmentation.unet_model import SmallUNet

IMAGE_DIR = Path("data/segmentation/images")
MASK_DIR = Path("data/segmentation/masks")
MODEL_PATH = Path("models/small_unet.pt")
REPORT_PATH = Path("reports/segmentation_report.txt")
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 4
RANDOM_SEED = 42

def create_dataloaders():
    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    dataset = SyntheticSegmentationDataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        transform=transform
    )

    train_size = int(
        0.8 * len(dataset)
    )

    test_size = (
        len(dataset)
        - train_size
    )

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(
        RANDOM_SEED
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("=== Segmentation DataLoaders ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    images, masks = next(
    iter(train_loader)
    )
    print(f"Batch image shape: {images.shape}")
    print(f"Batch mask shape: {masks.shape}")
    return train_loader, test_loader

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_model(
    model,
    train_loader,
    device
):
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
model.parameters(),
lr=LEARNING_RATE
)
model.train()
for epoch in range(EPOCHS):
total_loss = 0.0
for images, masks in train_loader:
images = images.to(device)
masks = masks.to(device)
optimizer.zero_grad()
outputs = model(images)
