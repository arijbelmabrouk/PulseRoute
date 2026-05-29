import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ─────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

# Landmark IDs — sourced from:
# 1. MediaPipe official documentation (forehead, nose)
# 2. Wang et al. 2017 POS paper (forehead ROI)
# 3. Community-verified MediaPipe repos (cheeks)
# 4. Visual verification on user face (all regions)
#
# Forehead: confirmed across rPPG literature
# Wang 2017, yarppg, multiple GitHub implementations
FOREHEAD_LANDMARKS = [
    10,               # top center — hairline
    338, 297, 332,    # left arc
    109, 67,  103,    # right arc
    151, 9,   8       # inner upper centerline
]

# Nose: confirmed in MediaPipe official docs
# and TensorFlow.js canonical face model
NOSE_LANDMARKS = [
    168,   # top — between eyes
    6,     # bridge top
    197,   # bridge mid
    195,   # bridge lower
    5,     # bridge bottom
    44,    # right nostril
    2,     # lower base
    274,   # left nostril
    4      # tip
]

# Cheeks: community-verified from MediaPipe
# face mesh topology — not officially documented
# by Google. Verified visually on user face.
# Left cheek from user perspective
LEFT_CHEEK_LANDMARKS = [
    423, 434, 427,   # fleshy center
    323, 361, 288    # cheekbone outer
]

# Right cheek from user perspective
RIGHT_CHEEK_LANDMARKS = [
    203, 214, 207,   # fleshy center
    93,  132, 58     # cheekbone outer
]

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
        x  = int(lm.x * frame_w)
        y  = int(lm.y * frame_h)
        points.append((x, y))
    return points


def get_combined_mask(all_region_pts, frame_h, frame_w):
    """
    Build combined binary mask for all ROI regions
    in one single operation.

    Input:
        all_region_pts — list of polygon point lists
        frame_h        — frame height
        frame_w        — frame width

    Output:
        binary mask array of shape (frame_h, frame_w)
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


def draw_key_landmarks(frame, landmarks, frame_w, frame_h):
    """
    Draw every 10th landmark ID to avoid screen clutter.
    ROI landmarks highlighted in cyan for verification.

    Input:
        frame      — BGR frame
        landmarks  — MediaPipe face landmark list
        frame_w    — frame width
        frame_h    — frame height
    """
    roi_ids = set(
        FOREHEAD_LANDMARKS    +
        LEFT_CHEEK_LANDMARKS  +
        RIGHT_CHEEK_LANDMARKS +
        NOSE_LANDMARKS
    )

    for idx, lm in enumerate(landmarks):
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)

        if idx in roi_ids:
            # ROI landmarks — larger cyan dot with ID
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)
            cv2.putText(frame, str(idx), (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 255, 255), 1)
        elif idx % 10 == 0:
            # Every 10th — small white dot with ID
            cv2.circle(frame, (x, y), 2, (200, 200, 200), -1)
            cv2.putText(frame, str(idx), (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.25, (200, 200, 200), 1)

    return frame


# ─────────────────────────────────────────
# Face visibility check
# ─────────────────────────────────────────

def is_face_visible(landmarks, frame_w, frame_h):
    """
    Check if face is sufficiently visible for rPPG.

    Frontality: nose tip (1) centered between
    left (234) and right (454) face edges.
    Both are confirmed outer face boundary landmarks
    in MediaPipe official topology.

    Input:
        landmarks — MediaPipe face landmark list
        frame_w   — frame width
        frame_h   — frame height

    Output:
        visible        — True if face suitable for rPPG
        frontality     — 0.0 to 1.0
        visibility_pct — fraction of key landmarks in bounds
    """
    key_ids = (
        FOREHEAD_LANDMARKS    +
        LEFT_CHEEK_LANDMARKS  +
        RIGHT_CHEEK_LANDMARKS +
        NOSE_LANDMARKS
    )

    in_bounds = sum(
        1 for idx in key_ids
        if 0.05 < landmarks[idx].x < 0.95
        and 0.05 < landmarks[idx].y < 0.95
    )
    visibility_pct = in_bounds / len(key_ids)

    # 234 = left face edge, 454 = right face edge
    # confirmed in MediaPipe face oval landmarks
    nose_x     = landmarks[1].x
    left_x     = landmarks[234].x
    right_x    = landmarks[454].x
    face_width = abs(right_x - left_x)

    if face_width > 0:
        nose_ratio = (nose_x - left_x) / face_width
        frontality = max(0.0, 1.0 - abs(nose_ratio - 0.5) / 0.3)
    else:
        frontality = 0.0

    visible = visibility_pct > 0.8 and frontality > 0.5

    return visible, round(frontality, 2), round(visibility_pct, 2)


# ─────────────────────────────────────────
# Confidence score
# ─────────────────────────────────────────

def calculate_face_confidence(face_results, frame_w, frame_h):
    """
    Confidence that face ROI is suitable for rPPG.

    Three signals:
        Score 1 — visibility + frontality  weight 0.5
        Score 2 — key landmarks in bounds  weight 0.3
        Score 3 — landmark visibility mean weight 0.2

    Face Mesh reliably populates landmark visibility
    unlike Hands — used here as genuine signal.

    Output:
        confidence float 0.0 to 1.0
    """
    if not face_results.multi_face_landmarks:
        return 0.0

    landmarks = face_results.multi_face_landmarks[0].landmark

    visible, frontality, visibility_pct = is_face_visible(
        landmarks, frame_w, frame_h
    )
    visibility_score = frontality * 0.5 + visibility_pct * 0.5

    key_ids   = FOREHEAD_LANDMARKS + NOSE_LANDMARKS
    in_bounds = sum(
        1 for idx in key_ids
        if 0.0 < landmarks[idx].x < 1.0
        and 0.0 < landmarks[idx].y < 1.0
    )
    bounds_score = in_bounds / len(key_ids)

    # 10 = forehead top, 234/454 = face edges
    # 1 = nose tip — all confirmed stable landmarks
    sample_ids     = [10, 234, 454, 1, 151, 9]
    visibilities   = [landmarks[idx].visibility
                      for idx in sample_ids]
    landmark_score = float(np.mean(visibilities))

    confidence = (
        visibility_score * 0.5 +
        bounds_score     * 0.3 +
        landmark_score   * 0.2
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
    face ROI pixels in CIELab color space.

        ITA >  55  → FST I-II   (Very Light)
        28 – 55    → FST III    (Light-Medium)
        10 – 28    → FST IV     (Medium)
       -30 – 10    → FST V      (Medium-Dark)
        ITA < -30  → FST VI     (Dark)

    FST V-VI triggers palm fallback in full pipeline.

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
    Measure face motion between consecutive frames
    via mean displacement of key landmarks in pixels.

    Uses confirmed stable anchor landmarks:
        10  — forehead top center
        1   — nose tip
        234 — left face edge
        454 — right face edge
        152 — chin bottom

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

    # All confirmed in MediaPipe face oval / keypoints
    key_ids       = [10, 1, 234, 454, 152]
    displacements = []

    for idx in key_ids:
        curr = current_landmarks[idx]
        prev = previous_landmarks[idx]
        dx   = (curr.x - prev.x) * frame_w
        dy   = (curr.y - prev.y) * frame_h
        displacements.append(np.sqrt(dx**2 + dy**2))

    motion_pixels = float(np.mean(displacements))

    if motion_pixels < 1.0:
        status = "Stable"
    elif motion_pixels < 3.0:
        status = "Slight Motion"
    elif motion_pixels < 10.0:
        status = "Motion Detected"
    else:
        status = "Too Much Motion"

    return round(motion_pixels, 2), status


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def run_face_detection():
    """
    Opens camera, runs MediaPipe Face Mesh in real time.

    Regions with source references:
        Green  — Forehead  (Wang 2017, yarppg, confirmed)
        Blue   — Cheeks    (community-verified)
        Yellow — Nose      (MediaPipe official docs)

    Test 1 (stability) and Test 4 (motion) every frame.
    Test 2 (skin tone) and Test 3 (lighting) every
    10 frames — cached between evaluations.

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

    timestamps         = deque(maxlen=max(int(actual_fps), 2))
    roi_area_history   = deque(maxlen=int(actual_fps * 2))
    previous_landmarks = None
    frame_counter      = 0

    cached_fitzpatrick     = "Unknown"
    cached_ita             = 0.0
    cached_lighting_status = "No ROI"
    cached_brightness      = 0.0

    # ── MediaPipe Face Mesh ────────────────────────────
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        print("Face detection running — press Q to quit")
        print("Look directly at the camera")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame_counter += 1

            timestamps.append(time.time())
            if len(timestamps) >= 2:
                rolling_fps = (len(timestamps) - 1) / (
                    timestamps[-1] - timestamps[0]
                )
            else:
                rolling_fps = actual_fps

            frame_h, frame_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = face_mesh.process(frame_rgb)

            confidence = calculate_face_confidence(
                results, frame_w, frame_h
            )

            stability     = 1.0
            stability_cv  = 0.0
            motion_pixels = 0.0
            motion_status = "Stable"
            face_status   = "No face detected"
            face_color    = (0, 0, 255)

            # ── Process if face detected ───────────────
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                visible, frontality, vis_pct = is_face_visible(
                    landmarks, frame_w, frame_h
                )

                frame = draw_key_landmarks(
                    frame, landmarks, frame_w, frame_h
                )

                motion_pixels, motion_status = calculate_motion(
                    landmarks, previous_landmarks,
                    frame_w, frame_h
                )
                previous_landmarks = landmarks

                forehead_pts = get_landmark_pixels(
                    landmarks, FOREHEAD_LANDMARKS,
                    frame_w, frame_h
                )
                left_cheek_pts = get_landmark_pixels(
                    landmarks, LEFT_CHEEK_LANDMARKS,
                    frame_w, frame_h
                )
                right_cheek_pts = get_landmark_pixels(
                    landmarks, RIGHT_CHEEK_LANDMARKS,
                    frame_w, frame_h
                )
                nose_pts = get_landmark_pixels(
                    landmarks, NOSE_LANDMARKS,
                    frame_w, frame_h
                )

                combined_mask = get_combined_mask(
                    [forehead_pts, left_cheek_pts,
                     right_cheek_pts, nose_pts],
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

                # Green  = Forehead  (best signal)
                # Blue   = Cheeks    (good signal)
                # Yellow = Nose      (medium signal)
                frame = draw_roi(
                    frame, forehead_pts,
                    color=(0, 255, 0),
                    alpha=0.3, label="Forehead"
                )
                frame = draw_roi(
                    frame, left_cheek_pts,
                    color=(255, 0, 0),
                    alpha=0.3, label="L.Cheek"
                )
                frame = draw_roi(
                    frame, right_cheek_pts,
                    color=(255, 0, 0),
                    alpha=0.3, label="R.Cheek"
                )
                frame = draw_roi(
                    frame, nose_pts,
                    color=(0, 255, 255),
                    alpha=0.3, label="Nose"
                )

                cv2.putText(
                    frame,
                    f"Frontality: {frontality:.2f}"
                    f"  Vis: {vis_pct:.2f}",
                    (frame_w - 280, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if visible else (0, 165, 255),
                    2
                )

                face_status = (
                    "Face visible - GOOD" if visible
                    else "Move face to center"
                )
                face_color = (
                    (0, 255, 0) if visible
                    else (0, 165, 255)
                )

            else:
                previous_landmarks     = None
                roi_area_history.clear()
                cached_fitzpatrick     = "Unknown"
                cached_ita             = 0.0
                cached_lighting_status = "No ROI"
                cached_brightness      = 0.0

            # ── HUD ───────────────────────────────────
            y = [30]

            def put(text, color=(0, 255, 0), scale=0.6):
                cv2.putText(frame, text, (10, y[0]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, 2)
                y[0] += 28

            put(f"FPS: {rolling_fps:.1f}")
            put(face_status, face_color)
            put(
                f"Confidence: {confidence:.2f}",
                (0, 255, 0) if confidence > 0.7
                else (0, 165, 255)
            )

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
                "GREEN=Forehead  BLUE=Cheeks  YELLOW=Nose",
                (10, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1
            )

            cv2.imshow("Face Detection - ROI", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped")


# ─────────────────────────────────────────
if __name__ == "__main__":
    run_face_detection()