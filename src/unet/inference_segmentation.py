# inference_segmentation_matplotlib.py
import torch
import sys,os
import cv2
import numpy as np
import argparse
from model import get_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import get_default_transforms
import matplotlib.pyplot as plt
import os


# -------------------------
# Constants
# -------------------------
MODEL_PATH = "../../checkpoints/hair_seg_model.pth"
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map class indices to BGR colors (OpenCV uses BGR)
CLASS_COLORS = {
    0: [0, 0, 0],       # Background = black
    1: [0, 255, 0],     # Skin = green
    2: [0, 0, 255],     # Hair = red
}

# -------------------------
# Load model
# -------------------------
def load_model():
    model = get_model(num_classes=NUM_CLASSES, encoder_name="resnet34", pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# -------------------------
# Preprocess image using training transforms
# -------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = get_default_transforms()
    augmented = transform(image=img)
    img_tensor = augmented['image'].float().unsqueeze(0).to(DEVICE)
    return img_tensor, img  # original image for overlay

# -------------------------
# Postprocess mask
# -------------------------
def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in CLASS_COLORS.items():
        color_mask[mask == cls_idx] = color
    return color_mask

# -------------------------
# Inference
# -------------------------
def predict(model, img_tensor, orig_img_shape):
    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        pred_mask_resized = cv2.resize(pred_mask, (orig_img_shape.shape[1], orig_img_shape.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
        color_mask = mask_to_color(pred_mask_resized)
    return color_mask

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--flip", type=bool, help="Bool flip image horizontally")
    parser.add_argument("--out", type=str, default="output.png", help="Path to save output overlay")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay transparency (0.0–1.0)")
    args = parser.parse_args()

    model = load_model()
    img_tensor, orig_img = preprocess_image(args.img)
    color_mask = predict(model, img_tensor, orig_img)

    # Create mask boolean
    mask_bool = np.any(color_mask != [0, 0, 0], axis=2)

    # Semi-transparent overlay
    alpha = args.alpha

    # Flip horizontally (over y-axis)
    if args.flip:
        orig_img = np.flip(orig_img, axis=1)

    overlay_img = orig_img.copy()
    overlay_img[mask_bool] = cv2.addWeighted(orig_img[mask_bool], 1 - alpha, color_mask[mask_bool], alpha, 0)

    # Display overlay with Matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay_img)
    plt.axis('off')
    plt.title("Segmentation Overlay (Flipped)")
    plt.show()
