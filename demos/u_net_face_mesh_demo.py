# realtime_eyebrow_hair_rays.py
import os, sys, time
import cv2
import numpy as np
import torch
import argparse

# --- Your project imports ---
# Assumes this file is in eg. src/ and model/dataset are one level up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.unet.model import get_model
from src.dataset import get_default_transforms

# -------------------------
# Constants / Config
# -------------------------
MODEL_PATH = "../ml/checkpoints/hair_seg_model.pth"
NUM_CLASSES = 3
HAIR_CLASS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

# Eyebrow landmark indices (MediaPipe 468-landmark topology)
# These are commonly used "upper eyebrow ridge" points. Tweak if you want denser sampling.
RIGHT_EYEBROW_IDX = [70, 63, 105, 66, 107]     # subject's right eyebrow
LEFT_EYEBROW_IDX  = [336, 296, 334, 293, 300]  # subject's left eyebrow

# -------------------------
# Model loading
# -------------------------
def load_model():
    model = get_model(num_classes=NUM_CLASSES, encoder_name="resnet34", pretrained=False)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# -------------------------
# Segmentation helpers
# -------------------------
def preprocess_frame_for_model(frame_bgr):
    # to RGB because your transforms expect RGB
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    transform = get_default_transforms()  # should resize/normalize without random aug
    out = transform(image=img_rgb)
    img_tensor = out["image"].float().unsqueeze(0).to(DEVICE)
    return img_tensor

@torch.no_grad()
def predict_mask(model, frame_bgr, out_size):
    """
    Returns label mask (H,W) with ints in [0..NUM_CLASSES-1], resized to out_size (W,H).
    """
    tin = preprocess_frame_for_model(frame_bgr)
    logits = model(tin)
    pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    # Resize label mask with nearest neighbor to match original frame
    mask_resized = cv2.resize(pred, out_size, interpolation=cv2.INTER_NEAREST)
    return mask_resized  # (H,W)

# -------------------------
# Face Mesh helpers
# -------------------------
def landmarks_to_pixels(landmarks, w, h):
    """
    Convert normalized (x,y) landmarks to integer pixel coords.
    Returns list of (x,y) tuples.
    """
    pts = []
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        pts.append((x, y))
    return pts

def select_eyebrow_points(landmarks, w, h):
    pts = landmarks_to_pixels(landmarks, w, h)
    # Extract only the eyebrow points we care about
    left = [pts[i] for i in LEFT_EYEBROW_IDX]
    right = [pts[i] for i in RIGHT_EYEBROW_IDX]
    return left, right

# -------------------------
# Ray casting (vertical up)
# -------------------------
def cast_vertical_ray_up(start_xy, hair_mask):
    """
    start_xy: (x0, y0) in pixel coords within hair_mask frame size
    hair_mask: (H,W) label mask with HAIR_CLASS marking hair
    Returns (x0, y_stop) where the ray first hits hair, or top if none found.
    """
    x0, y0 = start_xy
    H, W = hair_mask.shape
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))

    # Scan upward
    for y in range(y0, -1, -1):
        if hair_mask[y, x0] == HAIR_CLASS:
            return x0, y
    return x0, 0  # no hair above -> stop at top

# -------------------------
# Main realtime loop
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--draw-step", type=int, default=1, help="Cast every Nth eyebrow point")
    parser.add_argument("--seg-interval", type=int, default=1, help="Run segmentation every N frames")
    parser.add_argument("--max-width", type=int, default=960, help="Resize frame for speed (keeps aspect)")
    parser.add_argument("--line-thickness", type=int, default=2)
    parser.add_argument("--circle-radius", type=int, default=3)
    parser.add_argument("--show-fps", action="store_true")
    args = parser.parse_args()

    # Load model
    model = load_model()
    if DEVICE.type == "cuda":
        try:
            model.half()  # optional speed-up on GPU if your model supports fp16
        except Exception:
            pass

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Failed to open camera")
        return

    # FaceMesh config: refine_landmarks improves around eyes/eyebrows slightly
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_count = 0
    last_mask = None
    last_time = time.time()
    fps_hist = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Resize for consistent processing speed
            h0, w0 = frame.shape[:2]
            if w0 > args.max_width:
                scale = args.max_width / w0
                frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)

            H, W = frame.shape[:2]

            # Run segmentation at chosen interval; reuse last mask otherwise
            if frame_count % args.seg_interval == 0 or last_mask is None:
                mask = predict_mask(model, frame, out_size=(W, H))
                last_mask = mask
            else:
                mask = last_mask

            # --- NEW: Hair mask overlay ---
            hair_mask = (mask == HAIR_CLASS).astype(np.uint8)  # 1 where hair
            overlay = frame.copy()
            overlay[hair_mask == 1] = (0, 0, 255)  # red overlay
            alpha = 0.4  # 0 = invisible, 1 = fully red
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


            # Face Mesh
            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                landmarks = res.multi_face_landmarks[0].landmark
                left_pts, right_pts = select_eyebrow_points(landmarks, W, H)

                # Optionally subsample eyebrow points
                draw_pts = left_pts[::args.draw_step] + right_pts[::args.draw_step]

                # Draw rays
                for (x0, y0) in draw_pts:
                    x_stop, y_stop = cast_vertical_ray_up((x0, y0), mask)
                    # line from eyebrow point to stop
                    cv2.line(frame, (x0, y0), (x_stop, y_stop), (0, 255, 255), args.line_thickness)
                    # mark stop
                    cv2.circle(frame, (x_stop, y_stop), args.circle_radius, (0, 255, 255), -1)

                # Optional: draw eyebrow points
                for (x, y) in draw_pts:
                    cv2.circle(frame, (x, y), 2, (255, 0, 255), -1)

            # FPS display
            if args.show_fps:
                now = time.time()
                dt = now - last_time
                last_time = now
                fps = 1.0 / dt if dt > 0 else 0.0
                fps_hist.append(fps)
                if len(fps_hist) > 30:
                    fps_hist.pop(0)
                cv2.putText(frame, f"FPS: {np.mean(fps_hist):.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Eyebrow rays to hair", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()

if __name__ == "__main__":
    # Workaround for MKL/OpenMP duplicate errors on some Windows setups
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
