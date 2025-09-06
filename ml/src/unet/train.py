import sys,os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import HairSegDataset, get_default_transforms
from src.unet.model import get_model
from src.unet.utils import mask_accuracy, mask_dice
from src.unet.utils import mask_iou

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
images_root = "../../data/images"
masks_root = "../../data/masks"
batch_size = 10
num_epochs = 10
learning_rate = 1e-4
num_workers = 2  # keep low on Windows to avoid issues

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
    train_dataset.images = train_dataset.images

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
        iou_total = 0.0
        dice_total = 0.0

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

            with torch.no_grad():
                if 'mask_iou' in globals():
                    iou_total += mask_iou(outputs, masks)
                if 'mask_dice' in globals():
                    dice_total += mask_dice(outputs, masks)

        avg_loss = epoch_loss / len(train_loader)
        avg_iou = iou_total / len(train_loader) if iou_total else 0
        avg_dice = dice_total / len(train_loader) if dice_total else 0
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Dice: {avg_dice:.4f} - IoU: {avg_iou:.4f}")

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
