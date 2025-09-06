import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def get_model(num_classes=3, encoder_name='resnet34', pretrained=True):
    """
    Returns a segmentation model (U-Net) for hair, eyebrow, and skin segmentation.
    num_classes: 3 (background, skin, hair)
    encoder_name: backbone for the U-Net
    pretrained: use ImageNet weights for encoder
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights='imagenet' if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None 
    )
    return model
