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

def mask_iou(preds, masks, num_classes=3, eps=1e-6):
    """
    Compute mean IoU (Intersection over Union) across classes.
    
    preds: model output logits [B, C, H, W]
    masks: ground truth [B, H, W]
    num_classes: number of classes (default=3: background, skin, hair)
    eps: small constant to avoid division by zero
    
    Returns:
        mean IoU over all classes (float)
    """
    preds_labels = torch.argmax(preds, dim=1)  # [B, H, W]

    ious = []
    for c in range(num_classes):
        pred_c = (preds_labels == c).float()
        mask_c = (masks == c).float()

        intersection = (pred_c * mask_c).sum()
        union = pred_c.sum() + mask_c.sum() - intersection

        if union > 0:
            iou = (intersection + eps) / (union + eps)
            ious.append(iou.item())

    if len(ious) == 0:
        return 0.0
    return sum(ious) / len(ious) 

def one_hot_encode(mask, num_classes=3):
    """
    Convert mask HxW to one-hot CxHxW
    """
    one_hot = np.zeros((num_classes, mask.shape[0], mask.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        one_hot[c] = (mask == c).astype(np.float32)
    return one_hot
