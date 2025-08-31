import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HairSegDataset, get_default_transforms
from model import get_model
from utils import mask_accuracy
from utils import mask_iou
import warnings
warnings.filterwarnings("ignore", message="NVIDIA GeForce RTX 5070 with CUDA capability")


# ------------------------
# Custom Dice / IoU Loss
# ------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] (logits from model)
        targets: [B, H, W] (ground truth mask)
        """
        # Convert logits -> probabilities
        inputs = torch.softmax(inputs, dim=1)
        
        # One-hot encode ground truth
        num_classes = inputs.shape[1]
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).permute(0,3,1,2).float()
        
        # Dice coefficient per class
        dims = (0,2,3)
        intersection = torch.sum(inputs * targets_onehot, dims)
        cardinality = torch.sum(inputs + targets_onehot, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Dice loss = 1 - mean dice
        return 1 - dice_score.mean()

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
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss()

    def combined_loss(outputs, targets, alpha=0.5, beta=0.5):
        """
        Combine CrossEntropy and Dice loss
        alpha: weight for CE
        beta: weight for Dice
        """
        return alpha * ce_loss(outputs, targets) + beta * dice_loss(outputs, targets)

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
                    loss = combined_loss(outputs, masks, alpha=0.5, beta=0.5)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = combined_loss(outputs, masks, alpha=0.5, beta=0.5)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            if 'mask_iou' in globals():
                acc_total += mask_iou(outputs, masks)

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
