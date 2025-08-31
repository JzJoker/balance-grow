# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HairSegDataset, get_default_transforms
from model import get_model
from utils import mask_accuracy  # optional helper
import warnings
warnings.filterwarnings("ignore", message="NVIDIA GeForce RTX 5070 with CUDA capability")
# -------------------------
# Settings
# -------------------------
images_root = "data/images"
masks_root = "data/masks"
batch_size = 4
num_epochs = 10
learning_rate = 1e-4
num_workers = 2  # keep low on Windows to avoid issues

# Force CPU usage
device = torch.device("cpu")

# -------------------------
# Training function
# -------------------------
def train():
    # Dataset and Dataloader
    train_dataset = HairSegDataset(
        images_root, 
        masks_root, 
        transform=get_default_transforms()
    )    
    train_dataset.images = train_dataset.images[:500]

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )

    # Model, Loss, Optimizer
    model = get_model(num_classes=3, encoder_name="resnet34", pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Optional: mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        acc_total = 0.0

        for images, masks in train_loader:
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.long)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            # Optional accuracy function
            if 'mask_accuracy' in globals():
                acc_total += mask_accuracy(outputs, masks)

        avg_loss = epoch_loss / len(train_loader)
        avg_acc = acc_total / len(train_loader) if acc_total else 0
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Acc: {avg_acc:.4f}")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/hair_seg_model.pth")
    print("Training finished. Model saved to checkpoints/hair_seg_model.pth")

# -------------------------
# Windows multiprocessing fix
# -------------------------
if __name__ == "__main__":
    # Required for Windows to avoid "RuntimeError: spawn"
    torch.multiprocessing.freeze_support()
    train()
