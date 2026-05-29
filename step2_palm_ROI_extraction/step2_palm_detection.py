import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ─────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Landmark IDs for palm ROI sub-regions
# Revised based on anatomical analysis of MediaPipe hand map
# Verified against landmark image — your corrections applied
THENAR_LANDMARKS       = [0, 1, 2, 5]       # thumb base fleshy mound
HYPOTHENAR_LANDMARKS   = [0, 13, 17]         # pinky base fleshy mound
CENTRAL_PALM_LANDMARKS = [0, 5, 9, 13, 17]  # central palm area


def get_landmark_pixels(landmarks, indices, frame_w, frame_h):
    """
    Convert normalized MediaPipe landmark coordinates
    to actual pixel coordinates for given indices.

    Input:
        landmarks — MediaPipe landmark list
        indices   — list of landmark IDs you want
        frame_w   — frame width in pixels
        frame_h   — frame height in pixels

    Output:
        list of (x, y) pixel coordinate tuples
    """
    points = []
    for idx in indices:
        lm = landmarks[idx]
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        points.append((x, y))
    return points


def draw_roi(frame, points, color, alpha=0.3, label=None):
    """
    Draw a filled semi-transparent polygon over the ROI
    and outline it with a solid border.

    Input:
        frame  — BGR frame to draw on
        points — list of (x,y) polygon vertices
        color  — BGR color tuple
        alpha  — transparency of fill
        label  — optional text label on region

    Output:
        frame with ROI drawn on it
    """
    if len(points) < 3:
        return frame

    pts = np.array(points, dtype=np.int32)

    # Filled semi-transparent polygon
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Solid outline
    cv2.polylines(frame, [pts], isClosed=True,
                  color=color, thickness=2)

    # Label at centroid of polygon
    if label:
        cx = int(np.mean([p[0] for p in points]))
        cy = int(np.mean([p[1] for p in points]))
        cv2.putText(frame, label, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1)

    return frame


def draw_all_landmarks(frame, landmarks, frame_w, frame_h):
    """
    Draw all 21 hand landmarks as dots with their ID numbers.
    Used during development to visually verify which landmark
    ID maps to which anatomical point on your specific hand.
    Remove or disable in final production code.

    Input:
        frame      — BGR frame
        landmarks  — MediaPipe hand landmark list
        frame_w    — frame width
        frame_h    — frame height
    """
    for idx, lm in enumerate(landmarks):
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)

        # Draw dot
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

        # Draw landmark ID number next to dot
        cv2.putText(frame, str(idx), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1)

    return frame


def is_palm_facing_camera(landmarks, frame_w, frame_h):
    """
    Detect whether palm or back of hand faces the camera.
    Only palm-facing gives good rPPG signal.
    Back of hand has tendons and worse perfusion.

    Method: checks if middle finger base (9) is above
    wrist (0) in frame — indicates upright hand position
    with palm likely visible.

    Output:
        True  — palm likely facing camera
        False — back of hand likely facing camera
    """
    wrist_y  = landmarks[0].y  * frame_h
    middle_y = landmarks[9].y  * frame_h

    # Middle finger base above wrist = hand upright = palm visible
    return middle_y < wrist_y


def calculate_palm_confidence(hand_results, frame_w, frame_h):
    """
    Estimate confidence that palm ROI is suitable for rPPG.
    Combines landmark visibility with palm orientation check.

    Output:
        confidence float between 0 and 1
    """
    if not hand_results.multi_hand_landmarks:
        return 0.0

    landmarks = hand_results.multi_hand_landmarks[0].landmark

    # Check visibility of key palm landmarks
    key_ids = [0, 1, 5, 9, 13, 17]
    visibilities = [landmarks[idx].visibility for idx in key_ids]
    visibility_score = float(np.mean(visibilities))

    # Check palm orientation
    palm_facing      = is_palm_facing_camera(landmarks, frame_w, frame_h)
    orientation_score = 1.0 if palm_facing else 0.3

    # Weighted combination
    confidence = visibility_score * 0.6 + orientation_score * 0.4

    return round(confidence, 2)


def run_palm_detection():
    """
    Main function — opens camera, runs MediaPipe Hands
    in real time, highlights all three palm ROI sub-regions.
    Draws all 21 landmarks with ID numbers for visual verification.
    Press Q to quit.
    """

    # ── Camera setup ──────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Camera not detected")
        return

    # Warm up camera — discard first 30 frames
    print("Warming up camera...")
    for _ in range(30):
        cap.read()

    # Measure actual fps over 5 seconds
    print("Measuring fps...")
    frame_count = 0
    start = time.time()
    while time.time() - start < 5:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
    actual_fps = frame_count / 5
    print(f"Camera ready @ {actual_fps:.1f} fps")

    # Adaptive rolling fps window — always 1 second of history
    timestamps = deque(maxlen=max(int(actual_fps), 2))

    # ── MediaPipe Hands ────────────────────────────────
    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,            # 0=fast 1=balanced 2=accurate
        min_detection_confidence=0.7,  # high — need clear palm
        min_tracking_confidence=0.5
    ) as hands:

        print("Palm detection running — press Q to quit")
        print("Show your PALM to the camera")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Rolling fps calculation
            timestamps.append(time.time())
            if len(timestamps) >= 2:
                rolling_fps = (len(timestamps) - 1) / (
                    timestamps[-1] - timestamps[0]
                )
            else:
                rolling_fps = actual_fps

            frame_h, frame_w = frame.shape[:2]

            # Convert BGR → RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run MediaPipe Hands — returns landmark coordinates
            results = hands.process(frame_rgb)

            # Calculate confidence score
            confidence = calculate_palm_confidence(
                results, frame_w, frame_h
            )

            # ── Draw ROIs if hand detected ─────────────
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0].landmark

                # Step 1 — draw all 21 landmarks with IDs
                # so you can visually verify positions
                frame = draw_all_landmarks(
                    frame, landmarks, frame_w, frame_h
                )

                # Step 2 — get pixel coordinates for each sub-region
                thenar_pts = get_landmark_pixels(
                    landmarks, THENAR_LANDMARKS, frame_w, frame_h
                )
                hypothenar_pts = get_landmark_pixels(
                    landmarks, HYPOTHENAR_LANDMARKS, frame_w, frame_h
                )
                central_pts = get_landmark_pixels(
                    landmarks, CENTRAL_PALM_LANDMARKS, frame_w, frame_h
                )

                # Step 3 — draw each sub-region
                # Green  = thenar      (best signal — start here)
                # Blue   = hypothenar  (good signal)
                # Yellow = central     (variable signal)
                frame = draw_roi(
                    frame, thenar_pts,
                    color=(0, 255, 0),
                    alpha=0.3,
                    label="Thenar"
                )
                frame = draw_roi(
                    frame, hypothenar_pts,
                    color=(255, 0, 0),
                    alpha=0.3,
                    label="Hypo"
                )
                frame = draw_roi(
                    frame, central_pts,
                    color=(0, 255, 255),
                    alpha=0.3,
                    label="Central"
                )

                # Palm orientation check
                palm_facing = is_palm_facing_camera(
                    landmarks, frame_w, frame_h
                )
                orientation_text  = (
                    "Palm facing camera ✓" if palm_facing
                    else "Show PALM not back of hand"
                )
                orientation_color = (
                    (0, 255, 0) if palm_facing
                    else (0, 0, 255)
                )

                status       = "Hand detected"
                status_color = (0, 255, 0)

            else:
                orientation_text  = "No hand detected"
                orientation_color = (0, 0, 255)
                status            = "Show your palm to the camera"
                status_color      = (0, 0, 255)

            # ── HUD overlay ───────────────────────────
            # FPS
            cv2.putText(frame,
                        f"FPS: {rolling_fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            # Detection status
            cv2.putText(frame, status,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, status_color, 2)

            # Confidence score — orange if low, green if good
            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if confidence > 0.7 else (0, 165, 255),
                2
            )

            # Palm orientation
            cv2.putText(frame, orientation_text,
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, orientation_color, 2)

            # Region color legend at bottom
            cv2.putText(
                frame,
                "GREEN=Thenar  BLUE=Hypothenar  YELLOW=Central",
                (10, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1
            )

            cv2.imshow("Palm Detection — ROI", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Palm detection stopped")


# ─────────────────────────────────────────
if __name__ == "__main__":
    run_palm_detection()