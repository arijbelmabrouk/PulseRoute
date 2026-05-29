import cv2
import sys
import os
import mediapipe as mp
import numpy as np
import time
from collections import deque
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from step1_video_capture.step1_video_captureV3 import initialize_camera

# ─────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────
mp_hands = mp.solutions.hands

# Landmark IDs for palm ROI sub-regions
# Non-overlapping — share only boundary edges
# Thenar     : thumb side  (0, 1, 2, 5)
# Central    : middle palm (0, 5, 9, 13)
# Hypothenar : pinky side  (0, 13, 17)
THENAR_LANDMARKS       = [0, 1, 2, 5]
CENTRAL_PALM_LANDMARKS = [0, 5, 9, 13]
HYPOTHENAR_LANDMARKS   = [0, 13, 17]

# Run heavy evaluations every N frames to preserve fps
EVAL_EVERY_N_FRAMES = 10


# ─────────────────────────────────────────
# ROI utilities
# ─────────────────────────────────────────

def get_landmark_pixels(landmarks, indices, frame_w, frame_h):
    """
    Convert normalized MediaPipe landmark coordinates
    to actual pixel coordinates for given indices.

    Input:
        landmarks — MediaPipe landmark list
        indices   — list of landmark IDs
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


def get_combined_mask(all_region_pts, frame_h, frame_w):
    """
    Build combined binary mask for all ROI regions
    in one single operation instead of three separate
    mask creations and a clip merge.

    Input:
        all_region_pts — list of polygon point lists
                         [thenar_pts, central_pts, hypothenar_pts]
        frame_h        — frame height
        frame_w        — frame width

    Output:
        binary mask array of shape (frame_h, frame_w)
        pixels inside any region = 1, outside all = 0
    """
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for points in all_region_pts:
        if len(points) >= 3:
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def get_roi_area(mask):
    """
    Count number of pixels inside ROI mask.

    Input:
        mask — binary mask array

    Output:
        integer pixel count
    """
    return int(np.sum(mask))


def draw_roi(frame, points, color, alpha=0.3, label=None):
    """
    Draw filled semi-transparent polygon over ROI
    with solid border outline.

    Input:
        frame  — BGR frame to draw on
        points — list of (x,y) polygon vertices
        color  — BGR color tuple
        alpha  — fill transparency
        label  — optional text label at centroid

    Output:
        frame with ROI drawn on it
    """
    if len(points) < 3:
        return frame

    pts = np.array(points, dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True,
                  color=color, thickness=2)

    if label:
        cx = int(np.mean([p[0] for p in points]))
        cy = int(np.mean([p[1] for p in points]))
        cv2.putText(frame, label, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1)
    return frame


def draw_all_landmarks(frame, landmarks, frame_w, frame_h):
    """
    Draw all 21 hand landmarks as dots with ID numbers.
    Used during development to verify landmark positions.
    """
    for idx, lm in enumerate(landmarks):
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
        cv2.putText(frame, str(idx), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 255), 1)
    return frame


# ─────────────────────────────────────────
# Palm orientation — handedness aware
# ─────────────────────────────────────────

def is_palm_facing_camera(landmarks, handedness):
    """
    Detect palm vs back of hand using wrist geometry.
    Handedness-aware — applies different cross product
    condition for left vs right hand.

    MediaPipe labels from its own perspective (mirrored):
        'Left'  label = your RIGHT hand
        'Right' label = your LEFT  hand

    Cross product sign flips between left and right hand
    so different conditions are applied for each.

    Verified on user setup:
        Left hand  (MediaPipe 'Right'): cross > 0 = palm
        Right hand (MediaPipe 'Left') : cross < 0 = palm

    Input:
        landmarks  — MediaPipe hand landmark list
        handedness — MediaPipe handedness result object

    Output:
        palm_facing — True if palm facing camera
        cross       — raw cross product value
    """
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    index = np.array([landmarks[5].x, landmarks[5].y])
    pinky = np.array([landmarks[17].x, landmarks[17].y])

    v1    = index - wrist
    v2    = pinky - wrist
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    hand_label = handedness.classification[0].label

    if hand_label == "Right":
        # MediaPipe 'Right' = your LEFT hand
        # Verified: cross > 0 = palm facing
        palm_facing = cross > 0
    else:
        # MediaPipe 'Left' = your RIGHT hand
        # Opposite condition
        palm_facing = cross < 0

    return palm_facing, cross


# ─────────────────────────────────────────
# Confidence score
# ─────────────────────────────────────────

def calculate_palm_confidence(hand_results, frame_w, frame_h):
    """
    Confidence that palm ROI is suitable for rPPG.

    Three signals:
        Score 1 — palm orientation     weight 0.5
        Score 2 — landmarks in bounds  weight 0.3
        Score 3 — MediaPipe confidence weight 0.2

    Output:
        confidence float 0.0 to 1.0
    """
    if not hand_results.multi_hand_landmarks:
        return 0.0

    landmarks  = hand_results.multi_hand_landmarks[0].landmark
    handedness = hand_results.multi_handedness[0]

    palm_facing, _    = is_palm_facing_camera(landmarks, handedness)
    orientation_score = 1.0 if palm_facing else 0.0

    key_ids   = [0, 1, 2, 5, 9, 13, 17]
    in_bounds = sum(
        1 for idx in key_ids
        if 0.0 < landmarks[idx].x < 1.0
        and 0.0 < landmarks[idx].y < 1.0
    )
    bounds_score    = in_bounds / len(key_ids)
    detection_score = handedness.classification[0].score

    confidence = (
        orientation_score * 0.5 +
        bounds_score      * 0.3 +
        detection_score   * 0.2
    )
    return round(confidence, 2)


# ─────────────────────────────────────────
# Test 1 — Landmark stability
# ─────────────────────────────────────────

def calculate_roi_stability(roi_area_history):
    """
    Measure ROI stability via coefficient of variation
    of ROI pixel area across recent frames.

    Input:
        roi_area_history — deque of recent ROI pixel counts

    Output:
        stability — score 0.0 to 1.0
        raw_cv    — raw coefficient of variation
    """
    if len(roi_area_history) < 5:
        return 1.0, 0.0

    areas     = np.array(list(roi_area_history))
    mean_area = np.mean(areas)

    if mean_area == 0:
        return 0.0, 1.0

    cv        = np.std(areas) / mean_area
    stability = max(0.0, 1.0 - (cv / 0.30))

    return round(stability, 2), round(cv, 3)


# ─────────────────────────────────────────
# Test 2 — Skin tone estimate
# ─────────────────────────────────────────

def estimate_skin_tone(frame, mask):
    """
    Fitzpatrick type estimate using ITA formula on
    palm pixels in CIELab color space.

        ITA >  55  → FST I-II   (Very Light)
        28 – 55    → FST III    (Light-Medium)
        10 – 28    → FST IV     (Medium)
       -30 – 10    → FST V      (Medium-Dark)
        ITA < -30  → FST VI     (Dark)

    Input:
        frame — BGR frame
        mask  — binary ROI mask

    Output:
        fitzpatrick — string label
        ita_angle   — raw ITA value
    """
    if np.sum(mask) == 0:
        return "Unknown", 0.0

    roi_pixels = frame[mask == 1]
    if len(roi_pixels) == 0:
        return "Unknown", 0.0

    mean_bgr     = np.mean(roi_pixels, axis=0)
    mean_bgr_img = np.uint8([[mean_bgr]])

    lab = cv2.cvtColor(mean_bgr_img, cv2.COLOR_BGR2Lab)
    L   = float(lab[0, 0, 0]) * 100.0 / 255.0
    b   = float(lab[0, 0, 2]) - 128.0

    ita = np.degrees(np.arctan((L - 50.0) / b)) if b != 0 else 90.0

    if ita > 55:
        fitzpatrick = "FST I-II (Very Light)"
    elif ita > 28:
        fitzpatrick = "FST III (Light-Medium)"
    elif ita > 10:
        fitzpatrick = "FST IV (Medium)"
    elif ita > -30:
        fitzpatrick = "FST V (Medium-Dark)"
    else:
        fitzpatrick = "FST VI (Dark)"

    return fitzpatrick, round(ita, 1)


# ─────────────────────────────────────────
# Test 3 — Lighting assessment
# ─────────────────────────────────────────

def assess_lighting(frame, mask):
    """
    Assess lighting quality inside ROI.

    Checks:
        1. Mean brightness  — too dark or too bright
        2. Overexposure     — saturated pixels > 240
        3. Uniformity       — shadows / uneven light

    Input:
        frame — BGR frame
        mask  — binary ROI mask

    Output:
        status          — string classification
        brightness      — mean pixel brightness 0-255
        uniformity      — coefficient of variation
        overexposed_pct — fraction of saturated pixels
    """
    if np.sum(mask) == 0:
        return "No ROI", 0.0, 0.0, 0.0

    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_pixels = gray[mask == 1].astype(float)

    if len(roi_pixels) == 0:
        return "No ROI", 0.0, 0.0, 0.0

    brightness      = float(np.mean(roi_pixels))
    uniformity      = float(np.std(roi_pixels)) / brightness \
                      if brightness > 0 else 1.0
    overexposed_pct = float(
        np.sum(roi_pixels > 240) / len(roi_pixels)
    )

    if brightness < 40:
        status = "Too Dark"
    elif overexposed_pct > 0.2:
        status = "Overexposed"
    elif uniformity > 0.4:
        status = "Uneven Lighting"
    elif 80 <= brightness <= 200:
        status = "Good"
    else:
        status = "Acceptable"

    return (
        status,
        round(brightness, 1),
        round(uniformity, 3),
        round(overexposed_pct, 3)
    )


# ─────────────────────────────────────────
# Test 4 — Motion detection
# ─────────────────────────────────────────

def calculate_motion(current_landmarks,
                     previous_landmarks,
                     frame_w, frame_h):
    """
    Mean landmark displacement in pixels between frames.

    Input:
        current_landmarks  — landmarks from current frame
        previous_landmarks — landmarks from previous frame
        frame_w, frame_h   — frame dimensions

    Output:
        motion_pixels — mean displacement
        motion_status — string classification
    """
    if previous_landmarks is None:
        return 0.0, "Stable"

    key_ids       = [0, 1, 5, 9, 13, 17]
    displacements = []

    for idx in key_ids:
        curr = current_landmarks[idx]
        prev = previous_landmarks[idx]
        dx   = (curr.x - prev.x) * frame_w
        dy   = (curr.y - prev.y) * frame_h
        displacements.append(np.sqrt(dx**2 + dy**2))

    motion_pixels = float(np.mean(displacements))

    if motion_pixels < 2.0:
        status = "Stable"
    elif motion_pixels < 5.0:
        status = "Slight Motion"
    elif motion_pixels < 15.0:
        status = "Motion Detected"
    else:
        status = "Too Much Motion"

    return round(motion_pixels, 2), status


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def run_palm_detection(cap, actual_fps):
    """
    Runs MediaPipe Hands detection on frames from camera.
    Highlights three non-overlapping palm ROI regions
    only when palm correctly faces camera.

    Works correctly for both left and right hands.

    Input:
        cap        — OpenCV VideoCapture object from step1
        actual_fps — measured fps from step1

    Test 1 (stability) and Test 4 (motion) run every frame.
    Test 2 (skin tone) and Test 3 (lighting) run every
    10 frames only — cached between evaluations.

    Press Q to quit.
    """

    timestamps         = deque(maxlen=max(int(actual_fps), 2))
    roi_area_history   = deque(maxlen=int(actual_fps * 2))
    previous_landmarks = None
    frame_counter      = 0

    # Cached values for heavy evaluations
    cached_fitzpatrick     = "Unknown"
    cached_ita             = 0.0
    cached_lighting_status = "No ROI"
    cached_brightness      = 0.0

    # ── MediaPipe Hands ────────────────────────────────
    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        print("Palm detection running — press Q to quit")
        print("Show your PALM to the camera")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_counter += 1

            # Rolling fps
            timestamps.append(time.time())
            if len(timestamps) >= 2:
                rolling_fps = (len(timestamps) - 1) / (
                    timestamps[-1] - timestamps[0]
                )
            else:
                rolling_fps = actual_fps

            frame_h, frame_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = hands.process(frame_rgb)

            confidence = calculate_palm_confidence(
                results, frame_w, frame_h
            )

            # Per-frame defaults
            stability     = 1.0
            stability_cv  = 0.0
            motion_pixels = 0.0
            motion_status = "Stable"

            # ── Process if hand detected ───────────────
            if results.multi_hand_landmarks:
                landmarks  = results.multi_hand_landmarks[0].landmark
                handedness = results.multi_handedness[0]

                # Pass handedness to orientation check
                palm_facing, cross = is_palm_facing_camera(
                    landmarks, handedness
                )

                frame = draw_all_landmarks(
                    frame, landmarks, frame_w, frame_h
                )

                motion_pixels, motion_status = calculate_motion(
                    landmarks, previous_landmarks,
                    frame_w, frame_h
                )
                previous_landmarks = landmarks

                if palm_facing:
                    thenar_pts = get_landmark_pixels(
                        landmarks, THENAR_LANDMARKS,
                        frame_w, frame_h
                    )
                    central_pts = get_landmark_pixels(
                        landmarks, CENTRAL_PALM_LANDMARKS,
                        frame_w, frame_h
                    )
                    hypothenar_pts = get_landmark_pixels(
                        landmarks, HYPOTHENAR_LANDMARKS,
                        frame_w, frame_h
                    )

                    combined_mask = get_combined_mask(
                        [thenar_pts, central_pts, hypothenar_pts],
                        frame_h, frame_w
                    )

                    total_area = get_roi_area(combined_mask)
                    roi_area_history.append(total_area)
                    stability, stability_cv = calculate_roi_stability(
                        roi_area_history
                    )

                    if frame_counter % EVAL_EVERY_N_FRAMES == 0:
                        cached_fitzpatrick, cached_ita = \
                            estimate_skin_tone(frame, combined_mask)
                        (cached_lighting_status,
                         cached_brightness,
                         _,
                         _) = assess_lighting(frame, combined_mask)

                    frame = draw_roi(
                        frame, thenar_pts,
                        color=(0, 255, 0),
                        alpha=0.3, label="Thenar"
                    )
                    frame = draw_roi(
                        frame, central_pts,
                        color=(0, 255, 255),
                        alpha=0.3, label="Central"
                    )
                    frame = draw_roi(
                        frame, hypothenar_pts,
                        color=(255, 0, 0),
                        alpha=0.3, label="Hypo"
                    )

                    # Show hand label + cross for verification
                    hand_label = handedness.classification[0].label
                    cv2.putText(
                        frame,
                        f"{hand_label} | Cross: {cross:.4f}",
                        (frame_w - 260, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2
                    )

                    orientation_text  = "Palm facing camera - GOOD"
                    orientation_color = (0, 255, 0)

                else:
                    hand_label = handedness.classification[0].label
                    cv2.putText(
                        frame,
                        f"{hand_label} | Cross: {cross:.4f}",
                        (frame_w - 260, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2
                    )
                    previous_landmarks     = None
                    roi_area_history.clear()
                    cached_fitzpatrick     = "Unknown"
                    cached_ita             = 0.0
                    cached_lighting_status = "No ROI"
                    cached_brightness      = 0.0
                    orientation_text       = "Please show your PALM"
                    orientation_color      = (0, 0, 255)

                status_text  = "Hand detected"
                status_color = (0, 255, 0)

            else:
                previous_landmarks     = None
                roi_area_history.clear()
                cached_fitzpatrick     = "Unknown"
                cached_ita             = 0.0
                cached_lighting_status = "No ROI"
                cached_brightness      = 0.0
                orientation_text       = "No hand detected"
                orientation_color      = (0, 0, 255)
                status_text            = "Show your palm to the camera"
                status_color           = (0, 0, 255)

            # ── HUD ───────────────────────────────────
            y = [30]

            def put(text, color=(0, 255, 0), scale=0.6):
                cv2.putText(frame, text, (10, y[0]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, 2)
                y[0] += 28

            put(f"FPS: {rolling_fps:.1f}")
            put(status_text, status_color)
            put(
                f"Confidence: {confidence:.2f}",
                (0, 255, 0) if confidence > 0.7
                else (0, 165, 255)
            )
            put(orientation_text, orientation_color)

            y[0] += 6

            put(
                f"[T1] Stability: {stability:.2f}"
                f"  CV={stability_cv:.3f}",
                (0, 255, 0) if stability > 0.85
                else (0, 165, 255)
            )
            put(
                f"[T2] Skin: {cached_fitzpatrick}"
                f"  ITA={cached_ita}",
                (0, 255, 255)
            )
            put(
                f"[T3] Light: {cached_lighting_status}"
                f"  Bright={cached_brightness:.0f}",
                (0, 255, 0) if cached_lighting_status == "Good"
                else (0, 165, 255)
            )
            put(
                f"[T4] Motion: {motion_status}"
                f"  {motion_pixels:.1f}px",
                (0, 255, 0) if motion_status == "Stable"
                else (0, 0, 255)
            )

            cv2.putText(
                frame,
                "GREEN=Thenar  YELLOW=Central  BLUE=Hypothenar",
                (10, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1
            )

            cv2.imshow("Palm Detection - ROI", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Palm detection stopped")


# ─────────────────────────────────────────
#if __name__ == "__main__":
#    cap, actual_fps = initialize_camera(camera_index=0)
#    if cap is not None:
#        run_palm_detection(cap, actual_fps)