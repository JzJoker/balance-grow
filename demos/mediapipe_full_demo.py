# realtime_eyebrow_hair_rays.py
import os, sys, time
import cv2
import numpy as np
import argparse
import mediapipe as mp

# -------------------------
# Constants / Config
# -------------------------
MODEL_PATH = "hair_segmenter.tflite"
HAIR_THRESHOLD = 0.5  # probability cutoff for hair mask

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
RIGHT_EYEBROW_IDX = [70, 63, 105, 66, 107]     # subject's right eyebrow
LEFT_EYEBROW_IDX  = [336, 296, 334, 293, 300]  # subject's left eyebrow

# MediaPipe Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------------------------
# Segmentation helpers
# -------------------------
def load_segmenter():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageSegmenterOptions(base_options=base_options)
    return vision.ImageSegmenter.create_from_options(options)

def predict_mask(segmenter, frame_bgr, out_size):
    """
    Returns binary hair mask (H,W) resized to frame size.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    seg_result = segmenter.segment(mp_image)

    if seg_result.category_mask is not None:
        mask = seg_result.category_mask.numpy_view()
    else:
        conf_mask = seg_result.confidence_masks[0].numpy_view()  # float [0..1]
        # Invert: treat (1 - background_prob) as hair_prob
        hair_prob = 1.0 - conf_mask
        mask = (hair_prob > 0.5).astype(np.uint8)

    mask_resized = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)
    return mask_resized

# -------------------------
# Face Mesh helpers
# -------------------------
def landmarks_to_pixels(landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

def select_eyebrow_points(landmarks, w, h):
    pts = landmarks_to_pixels(landmarks, w, h)
    left = [pts[i] for i in LEFT_EYEBROW_IDX]
    right = [pts[i] for i in RIGHT_EYEBROW_IDX]
    return left, right

# Landmarks for orientation check
NOSE_IDX = 1
LEFT_CHEEK_IDX = 234
RIGHT_CHEEK_IDX = 454

def get_head_orientation(landmarks, w, h, threshold=0.03):
    """
    Returns one of: 'straight', 'left', 'right'
    based on relative cheek-to-nose horizontal distances.
    """
    nose = landmarks[NOSE_IDX]
    left_cheek = landmarks[LEFT_CHEEK_IDX]
    right_cheek = landmarks[RIGHT_CHEEK_IDX]

    # Normalized x positions
    nose_x = nose.x
    left_x = left_cheek.x
    right_x = right_cheek.x

    # Distances from nose
    dist_left = abs(nose_x - left_x)
    dist_right = abs(right_x - nose_x)

    # Compare
    diff = dist_left - dist_right
    if diff > threshold:
        return "right"   # user turned right
    elif diff < -threshold:
        return "left"    # user turned left
    else:
        return "straight"



# -------------------------
# Ray casting (vertical up)
# -------------------------
def cast_vertical_ray_up(start_xy, hair_mask):
    x0, y0 = start_xy
    H, W = hair_mask.shape
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))

    for y in range(y0, -1, -1):
        if hair_mask[y, x0] == 1:  # hair pixel
            return x0, y
    return x0, 0


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

    # Load hair segmenter
    segmenter = load_segmenter()

    # Open camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Failed to open camera")
        return

    # Face Mesh
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
            
            frame = cv2.flip(frame, 1)
            h0, w0 = frame.shape[:2]
            if w0 > args.max_width:
                scale = args.max_width / w0
                frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)

            H, W = frame.shape[:2]

            # Run segmentation every N frames
            if frame_count % args.seg_interval == 0 or last_mask is None:
                mask = predict_mask(segmenter, frame, (W, H))
                last_mask = mask
            else:
                mask = last_mask

            # Overlay
            overlay = frame.copy()
            overlay[mask == 1] = (0, 0, 255)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            # Face Mesh
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                landmarks = res.multi_face_landmarks[0].landmark
                left_pts, right_pts = select_eyebrow_points(landmarks, W, H)
                orientation = get_head_orientation(landmarks, W, H)
                if orientation == "left":
                    cv2.putText(frame, "Turn your head slightly right", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                elif orientation == "right":
                    cv2.putText(frame, "Turn your head slightly left", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "Good: look straight", (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                draw_pts = left_pts[::args.draw_step] + right_pts[::args.draw_step]

                for (x0, y0) in draw_pts:
                    x_stop, y_stop = cast_vertical_ray_up((x0, y0), mask)
                    cv2.line(frame, (x0, y0), (x_stop, y_stop), (0, 255, 255), args.line_thickness)
                    cv2.circle(frame, (x_stop, y_stop), args.circle_radius, (0, 255, 255), -1)

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
            if key == 27 or key == ord("q"):
                break

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
