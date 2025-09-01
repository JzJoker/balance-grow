import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.8
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Smoothing factor for landmarks (0 = no smoothing, 1 = no update)
SMOOTHING_FACTOR = 0.6
prev_landmarks = None

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process frame
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert landmarks to numpy array
            current_landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            )

            # Apply smoothing
            if prev_landmarks is not None:
                smoothed_landmarks = (
                    SMOOTHING_FACTOR * prev_landmarks
                    + (1 - SMOOTHING_FACTOR) * current_landmarks
                )
            else:
                smoothed_landmarks = current_landmarks

            prev_landmarks = smoothed_landmarks

            # Update face_landmarks for drawing
            for i, lm in enumerate(face_landmarks.landmark):
                lm.x, lm.y, lm.z = smoothed_landmarks[i]

            # Draw landmarks
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )

    cv2.imshow('Smoothed MediaPipe Face Mesh', image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
