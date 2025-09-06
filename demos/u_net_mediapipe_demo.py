import sys, os
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import ImageSegmenter, ImageSegmenterOptions
from mediapipe.tasks.python.vision import RunningMode
from mediapipe import Image as mpImage
from mediapipe import ImageFormat

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import get_default_transforms


# ------------------------
# U-Net model setup (YOUR MODEL)
# ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAIR_CLASS = 2  # adjust if your dataset uses different label

# Load your trained model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.unet.model import get_model
model = get_model(num_classes=3)
model.load_state_dict(torch.load("../ml/checkpoints/hair_seg_model.pth", map_location=DEVICE))
model.eval().to(DEVICE)

# Preprocessing
transform = get_default_transforms()

def predict_unet_mask(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed = transform(image=img)
    tensor = transformed["image"].float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
    return cv2.resize(pred.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)



# ------------------------
# MediaPipe Hair Segmenter setup
# ------------------------
mp_options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path="hair_segmenter.tflite"),
    running_mode=RunningMode.IMAGE,
    output_category_mask=True
)
mp_segmenter = ImageSegmenter.create_from_options(mp_options)


def predict_mp_mask(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_frame = mpImage(ImageFormat.SRGB, rgb)
    result = mp_segmenter.segment(mp_frame)
    mask = result.category_mask.numpy_view()
    # MediaPipe hair = 1
    return cv2.resize((mask == 1).astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

# ------------------------
# Overlay helper
# ------------------------
def overlay_hair(frame, mask, color=(0, 0, 255), alpha=0.4):
    overlay = frame.copy()
    overlay[mask == 1] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

# ------------------------
# Main loop
# ------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get masks
    unet_mask = predict_unet_mask(frame)
    mp_mask = predict_mp_mask(frame)

    # Apply overlays
    unet_overlay = overlay_hair(frame, (unet_mask == HAIR_CLASS).astype(np.uint8))
    mp_overlay = overlay_hair(frame, mp_mask)

    # Stack side by side
    combined = np.hstack((unet_overlay, mp_overlay))

    # Labels
    cv2.putText(combined, "U-Net", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "MediaPipe", (frame.shape[1] + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hair Segmentation: U-Net vs MediaPipe", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
