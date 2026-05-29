import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

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


def get_roi_mask(points, frame_h, frame_w):
    """
    Create binary mask from polygon points.
    Pixels inside polygon = 1, outside = 0.

    Input:
        points  — list of (x,y) polygon vertices
        frame_h — frame height
        frame_w — frame width

    Output:
        binary mask array of shape (frame_h, frame_w)
    """
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
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
# Palm orientation — mirror independent
# ─────────────────────────────────────────

def is_palm_facing_camera(landmarks):
    """
    Detect palm vs back of hand using wrist geometry.
    Mirror-independent — does not rely on handedness label.

    Uses cross product of two vectors from the wrist:
        v1 = wrist → index base  (landmark 5)
        v2 = wrist → pinky base  (landmark 17)

    The sign of the cross product z-component flips
    when the hand rotates from palm-facing to back-facing.

    We also print the raw cross value during development
    so you can verify which sign = palm on your setup.

    Output:
        True  — palm facing camera
        False — back of hand facing camera
        cross — raw cross product value for verification
    """
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    index = np.array([landmarks[5].x, landmarks[5].y])
    pinky = np.array([landmarks[17].x, landmarks[17].y])

    # Vectors from wrist
    v1 = index - wrist
    v2 = pinky - wrist

    # Cross product z-component
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    # Print for verification — remove once confirmed
    print(f"Cross: {cross:.4f}  "
          f"({'PALM' if cross > 0 else 'BACK'})", end="\r")

    return cross > 0, cross


# ─────────────────────────────────────────
# Confidence score
# ─────────────────────────────────────────

def calculate_palm_confidence(hand_results, frame_w, frame_h):
    """
    Confidence that palm ROI is suitable for rPPG.

    Three signals (MediaPipe visibility not used —
    not reliably populated for hands):

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

    # Score 1 — orientation
    palm_facing, _ = is_palm_facing_camera(landmarks)
    orientation_score = 1.0 if palm_facing else 0.0

    # Score 2 — key landmarks within frame bounds
    key_ids   = [0, 1, 2, 5, 9, 13, 17]
    in_bounds = sum(
        1 for idx in key_ids
        if 0.0 < landmarks[idx].x < 1.0
        and 0.0 < landmarks[idx].y < 1.0
    )
    bounds_score = in_bounds / len(key_ids)

    # Score 3 — MediaPipe detection confidence
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
    Measure ROI stability across frames using
    coefficient of variation of ROI pixel area.

    Stable ROI = MediaPipe consistently placing
    landmarks on same anatomical points frame to frame.

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

    std_area = np.std(areas)
    cv       = std_area / mean_area

    # CV <= 0.05 = perfect stability
    # CV >= 0.30 = completely unstable
    stability = max(0.0, 1.0 - (cv / 0.30))

    return round(stability, 2), round(cv, 3)


# ─────────────────────────────────────────
# Test 2 — Skin tone estimate
# ─────────────────────────────────────────

def estimate_skin_tone(frame, mask):
    """
    Rough Fitzpatrick type estimate from palm pixels
    using ITA (Individual Typology Angle) formula.

    ITA converts mean L* and b* from CIELab color
    space to an angle correlating with Fitzpatrick type:

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

    if b != 0:
        ita = np.degrees(np.arctan((L - 50.0) / b))
    else:
        ita = 90.0

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

    Three checks:
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
    std_brightness  = float(np.std(roi_pixels))
    uniformity      = std_brightness / brightness if brightness > 0 else 1.0
    overexposed_pct = float(np.sum(roi_pixels > 240) / len(roi_pixels))

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
    Measure hand motion between consecutive frames
    via mean displacement of key landmarks in pixels.

    High motion introduces artifacts into rPPG signal.

    Input:
        current_landmarks  — landmarks from current frame
        previous_landmarks — landmarks from previous frame
        frame_w, frame_h   — frame dimensions

    Output:
        motion_pixels — mean landmark displacement
        motion_status — string classification
    """
    if previous_landmarks is None:
        return 0.0, "Stable"

    key_ids       = [0, 1, 5, 9, 13, 17]
    displacements = []

    for idx in key_ids:
        curr = current_landmarks[idx]
        prev = previous_landmarks[idx]

        dx = (curr.x - prev.x) * frame_w
        dy = (curr.y - prev.y) * frame_h

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

def run_palm_detection():
    """
    Opens camera, runs MediaPipe Hands in real time.
    Highlights three non-overlapping palm ROI regions
    only when palm correctly faces camera.

    Evaluations running in parallel:
        Test 1 — ROI stability  (landmark consistency)
        Test 2 — Skin tone      (ITA Fitzpatrick estimate)
        Test 3 — Lighting       (brightness + uniformity)
        Test 4 — Motion         (landmark displacement)

    Cross product printed to terminal for orientation
    verification — remove print once confirmed correct.

    Press Q to quit.
    """

    # ── Camera setup ──────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Camera not detected")
        return

    print("Warming up camera...")
    for _ in range(30):
        cap.read()

    print("Measuring fps...")
    frame_count = 0
    start = time.time()
    while time.time() - start < 5:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
    actual_fps = frame_count / 5
    print(f"Camera ready @ {actual_fps:.1f} fps")

    timestamps       = deque(maxlen=max(int(actual_fps), 2))
    roi_area_history = deque(maxlen=int(actual_fps * 2))
    previous_landmarks = None

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

            # Default values
            stability       = 1.0
            stability_cv    = 0.0
            fitzpatrick     = "Unknown"
            ita_angle       = 0.0
            lighting_status = "No ROI"
            brightness      = 0.0
            motion_pixels   = 0.0
            motion_status   = "Stable"

            # ── Process if hand detected ───────────────
            if results.multi_hand_landmarks:
                landmarks  = results.multi_hand_landmarks[0].landmark
                handedness = results.multi_handedness[0]

                palm_facing, cross = is_palm_facing_camera(landmarks)

                # Always draw landmark IDs
                frame = draw_all_landmarks(
                    frame, landmarks, frame_w, frame_h
                )

                # Test 4 — motion
                motion_pixels, motion_status = calculate_motion(
                    landmarks, previous_landmarks,
                    frame_w, frame_h
                )
                previous_landmarks = landmarks

                if palm_facing:
                    # Pixel coordinates for each region
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

                    # Combined mask for Tests 2 and 3
                    thenar_mask = get_roi_mask(
                        thenar_pts, frame_h, frame_w
                    )
                    central_mask = get_roi_mask(
                        central_pts, frame_h, frame_w
                    )
                    hypothenar_mask = get_roi_mask(
                        hypothenar_pts, frame_h, frame_w
                    )
                    combined_mask = np.clip(
                        thenar_mask + central_mask + hypothenar_mask,
                        0, 1
                    )

                    # Test 1 — stability
                    total_area = get_roi_area(combined_mask)
                    roi_area_history.append(total_area)
                    stability, stability_cv = calculate_roi_stability(
                        roi_area_history
                    )

                    # Test 2 — skin tone
                    fitzpatrick, ita_angle = estimate_skin_tone(
                        frame, combined_mask
                    )

                    # Test 3 — lighting
                    (lighting_status,
                     brightness,
                     uniformity,
                     overexposed_pct) = assess_lighting(
                        frame, combined_mask
                    )

                    # Draw ROIs
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

                    # Show raw cross value on screen for verification
                    cv2.putText(
                        frame,
                        f"Cross: {cross:.4f}",
                        (frame_w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2
                    )

                    orientation_text  = "Palm facing camera - GOOD"
                    orientation_color = (0, 255, 0)

                else:
                    # Show cross value even when back of hand
                    cv2.putText(
                        frame,
                        f"Cross: {cross:.4f}",
                        (frame_w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2
                    )

                    previous_landmarks = None
                    roi_area_history.clear()
                    orientation_text  = "Please show your PALM"
                    orientation_color = (0, 0, 255)

                status_text  = "Hand detected"
                status_color = (0, 255, 0)

            else:
                previous_landmarks = None
                roi_area_history.clear()
                orientation_text  = "No hand detected"
                orientation_color = (0, 0, 255)
                status_text       = "Show your palm to the camera"
                status_color      = (0, 0, 255)

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
                f"[T2] Skin: {fitzpatrick}"
                f"  ITA={ita_angle}",
                (0, 255, 255)
            )
            put(
                f"[T3] Light: {lighting_status}"
                f"  Bright={brightness:.0f}",
                (0, 255, 0) if lighting_status == "Good"
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
if __name__ == "__main__":
    run_palm_detection()