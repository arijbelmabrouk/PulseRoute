import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Get screen resolution for maximum window size
screen_w, screen_h = 1920, 1080  # adjust if needed

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    cv2.namedWindow("Landmarks", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Landmarks", screen_w, screen_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]

        # Scale up frame to fill screen
        scale   = min(screen_w / frame_w, screen_h / frame_h)
        new_w   = int(frame_w * scale)
        new_h   = int(frame_h * scale)
        display = cv2.resize(frame, (new_w, new_h))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            for idx, lm in enumerate(landmarks):
                # Scale landmark coordinates to display size
                x = int(lm.x * new_w)
                y = int(lm.y * new_h)

                # Draw dot
                cv2.circle(display, (x, y), 3,
                           (0, 255, 255), -1)

                # Draw ID — white text, small but readable
                cv2.putText(
                    display,
                    str(idx),
                    (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28,
                    (255, 255, 255),
                    1
                )

        cv2.putText(
            display,
            f"Total landmarks: 468  |  Press Q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.imshow("Landmarks", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()