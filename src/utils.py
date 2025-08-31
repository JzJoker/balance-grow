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


def one_hot_encode(mask, num_classes=3):
    """
    Convert mask HxW to one-hot CxHxW
    """
    one_hot = np.zeros((num_classes, mask.shape[0], mask.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        one_hot[c] = (mask == c).astype(np.float32)
    return one_hot
