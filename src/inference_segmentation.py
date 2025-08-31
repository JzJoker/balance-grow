# inference_segmentation.py
import torch
import cv2
import numpy as np
import argparse
from model import get_model
from dataset import get_default_transforms
from PIL import Image

# -------------------------
# Constants
# -------------------------
MODEL_PATH = "../checkpoints/hair_seg_model.pth"
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Map class indices to BGR colors (OpenCV uses BGR)
CLASS_COLORS = {
    0: [0, 0, 0],       # Background = black
    1: [0, 0, 255],     # Hair = red
    2: [0, 255, 0],     # Skin = green
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
    import cv2
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Albumentations expects RGB

    transform = get_default_transforms()  # Albumentations transform
    augmented = transform(image=img)
    img_tensor = augmented['image']  # this is already a torch.Tensor if you used ToTensorV2

    # Ensure float and add batch dimension
    img_tensor = img_tensor.float().unsqueeze(0).to(DEVICE)

    return img_tensor, img  # original image as np.array for overlay


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
        # Resize mask to original image size
        pred_mask_resized = cv2.resize(pred_mask, (orig_img_shape.shape[1], orig_img_shape.shape[0]), interpolation=cv2.INTER_NEAREST)
        color_mask = mask_to_color(pred_mask_resized)
    return color_mask

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--out", type=str, default="output.png", help="Path to save output overlay")
    args = parser.parse_args()

    model = load_model()
    img_tensor, orig_img = preprocess_image(args.img)
    color_mask = predict(model, img_tensor, orig_img)

    # Debug prints
    print("Predicted mask shape:", color_mask.shape)
    print("Unique colors in mask:", np.unique(color_mask.reshape(-1, 3), axis=0))
    mask_bool = np.any(color_mask != [0, 0, 0], axis=2)
    print("Number of pixels to overlay:", np.sum(mask_bool))

    if np.sum(mask_bool) == 0:
        print("Warning: No hair or skin pixels detected. Check your model predictions!")

    # Hard overlay
    hard_overlay = orig_img.copy()
    hard_overlay[mask_bool] = color_mask[mask_bool]

    cv2.imwrite(args.out, hard_overlay)
    print(f"Saved hard overlay segmentation result to {args.out}")
