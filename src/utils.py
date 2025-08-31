import torch
import numpy as np

def mask_accuracy(preds, masks):
    """
    Compute pixel-wise accuracy
    preds: model output logits [B, C, H, W]
    masks: ground truth [B, H, W]
    """
    preds_labels = torch.argmax(preds, dim=1)
    correct = (preds_labels == masks).float()
    return correct.mean().item()

def mask_iou(outputs, targets, num_classes=3):
    """
    outputs: [B, C, H, W] logits
    targets: [B, H, W] ground truth
    Returns mean IoU over batch
    """
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)  # [B,H,W]
        iou_per_class = []

        for cls in range(num_classes):
            intersection = ((preds == cls) & (targets == cls)).sum(dim=(1,2)).float()
            union = ((preds == cls) | (targets == cls)).sum(dim=(1,2)).float()
            iou_cls = (intersection / (union + 1e-6))  # avoid division by 0
            iou_per_class.append(iou_cls)

        iou_per_class = torch.stack(iou_per_class, dim=1)  # [B, C]
        mean_iou = iou_per_class.mean(dim=1)  # mean over classes per image
        return mean_iou.mean().item()  # mean over batch

def mask_dice(outputs, targets, num_classes=3, smooth=1e-6):
    """
    outputs: [B, C, H, W] logits
    targets: [B, H, W] ground truth
    Returns mean Dice score over batch
    """
    with torch.no_grad():
        probs = torch.softmax(outputs, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).permute(0,3,1,2).float()

        intersection = (probs * targets_onehot).sum(dim=(2,3))  # [B,C]
        cardinality = (probs + targets_onehot).sum(dim=(2,3))    # [B,C]
        dice = (2 * intersection + smooth) / (cardinality + smooth)  # [B,C]
        return dice.mean(dim=1).mean().item()  # mean over classes then batch

def one_hot_encode(mask, num_classes=3):
    """
    Convert mask HxW to one-hot CxHxW
    """
    one_hot = np.zeros((num_classes, mask.shape[0], mask.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        one_hot[c] = (mask == c).astype(np.float32)
    return one_hot
