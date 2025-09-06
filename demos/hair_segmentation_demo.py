import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
# model_selection=1 is general (0 is landscape), 1 is better for portrait/hair
hair_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hair_segmentation.process(rgb_frame)

    # Get mask (values between 0 and 1)
    mask = results.segmentation_mask

    # Threshold mask to binary
    _, binary_mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    binary_mask = binary_mask.astype(np.uint8)

    # Optional: colorize hair (blue overlay)
    hair_overlay = np.zeros_like(frame)
    hair_overlay[:, :, 0] = 255  # Blue channel

    # Apply mask
    hair_only = cv2.bitwise_and(hair_overlay, hair_overlay, mask=binary_mask)
    output = cv2.addWeighted(frame, 1, hair_only, 0.7, 0)

    cv2.imshow("MediaPipe Hair Segmentation", output)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
