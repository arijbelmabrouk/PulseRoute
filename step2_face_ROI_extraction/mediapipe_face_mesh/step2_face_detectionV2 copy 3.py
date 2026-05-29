import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ─────────────────────────────────────────
# MediaPipe setup
# ─────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh

# ─────────────────────────────────────────
# Landmark IDs — verified from MediaPipe
# face mesh topology map
# ─────────────────────────────────────────

# Forehead — full coverage
# Vertical centerline + upper boundary + mid area
# Keep this as global for visibility/confidence checks
FOREHEAD_LANDMARKS = [
    10, 151,
    109, 67, 103, 54, 21,
    338, 297, 332, 284, 251,
    105, 66, 107,
    334, 296, 336
]
def get_forehead_region(landmarks, frame_w, frame_h):
    """
    Build forehead ROI clipped at eyebrow level.
    Uses convex hull of upper forehead landmarks
    then clips bottom at just above eyebrow line.
    """
    # Upper forehead points only
    upper_ids = [
        10, 151,
        109, 67, 103, 54, 21,
        338, 297, 332, 284, 251
    ]

    pts = []
    for idx in upper_ids:
        lm = landmarks[idx]
        x  = int(lm.x * frame_w)
        y  = int(lm.y * frame_h)
        pts.append((x, y))

    if len(pts) < 3:
        return []

    # Build convex hull of upper forehead
    hull = cv2.convexHull(
        np.array(pts, dtype=np.int32)
    ).reshape(-1, 2)

    # Find eyebrow bottom boundary from actual landmarks
    brow_ids = [105, 66, 107, 334, 296, 336]
    brow_y_values = [
        int(landmarks[idx].y * frame_h)
        for idx in brow_ids
    ]
    # Clip bottom = highest eyebrow point (min y value)
    # with small buffer of 3 pixels above eyebrows
    clip_y = min(brow_y_values) - 3

    # Remove hull points below clip line
    clipped = [p for p in hull if p[1] <= clip_y]

    # Add two bottom corner points at clip line
    # using leftmost and rightmost x of hull
    if len(clipped) >= 2:
        x_vals  = [p[0] for p in hull]
        left_x  = min(x_vals)
        right_x = max(x_vals)
        clipped.append((left_x,  clip_y))
        clipped.append((right_x, clip_y))

    if len(clipped) < 3:
        return hull.tolist()

    # Final convex hull of clipped points
    final = cv2.convexHull(
        np.array(clipped, dtype=np.int32)
    ).reshape(-1, 2)

    return final.tolist()
LEFT_CHEEK_LANDMARKS = [
    346, 347, 348, 349, 350, # upper — below eye
    423, 425, 427, 434,      # center flesh
    361, 288, 397,           # outer cheekbone — extends laterally
    366, 401, 435            # bottom — mouth corner level
]

RIGHT_CHEEK_LANDMARKS = [
    116, 117, 118, 119, 120, # upper — below eye
    206, 207, 214, 213,      # center flesh
    132, 58, 172,            # outer cheekbone — extends laterally
    135, 177, 215            # bottom — mouth corner level
]
# Run heavy evaluations every N frames
EVAL_EVERY_N_FRAMES = 10


# ─────────────────────────────────────────
# ROI utilities
# ─────────────────────────────────────────

def get_landmark_pixels(landmarks, indices, frame_w, frame_h):
    """
    Convert normalized MediaPipe landmark coordinates
    to pixel coordinates.

    Input:
        landmarks — MediaPipe landmark list
        indices   — list of landmark IDs
        frame_w   — frame width
        frame_h   — frame height

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


def get_convex_mask(points, frame_h, frame_w):
    """
    Build binary mask from convex hull of points.
    Always uses convex hull — no holes, no distortion.

    Input:
        points  — list of (x, y) pixel coordinates
        frame_h — frame height
        frame_w — frame width

    Output:
        binary mask array (frame_h, frame_w)
    """
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if len(points) >= 3:
        pts  = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 1)
    return mask


def get_combined_mask(all_region_pts, frame_h, frame_w):
    """
    Build combined binary mask for all ROI regions.
    Each region uses convex hull before fill.

    Input:
        all_region_pts — list of polygon point lists
        frame_h        — frame height
        frame_w        — frame width

    Output:
        binary mask array (frame_h, frame_w)
    """
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for points in all_region_pts:
        if len(points) >= 3:
            pts  = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillPoly(mask, [hull], 1)
    return mask


def get_roi_area(mask):
    return int(np.sum(mask))


def draw_roi(frame, points, color, alpha=0.3, label=None):
    """
    Draw filled semi-transparent convex hull polygon.

    Input:
        frame  — BGR frame
        points — list of (x, y) vertices
        color  — BGR color tuple
        alpha  — fill transparency
        label  — optional label at centroid
    """
    if len(points) < 3:
        return frame

    pts  = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(pts)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [hull], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [hull], isClosed=True,
                  color=color, thickness=2)

    if label:
        cx = int(np.mean(hull[:, 0, 0]))
        cy = int(np.mean(hull[:, 0, 1]))
        cv2.putText(frame, label, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, color, 1)
    return frame


def draw_roi_landmarks(frame, landmarks, frame_w, frame_h):
    """
    Draw ROI landmark dots with IDs.
    Color coded by region:
        Green  = Forehead
        Red    = Left cheek
        Blue   = Right cheek

    Non-ROI landmarks shown as small white dots
    every 10th to avoid clutter.
    """
    forehead_ids   = set(FOREHEAD_LANDMARKS)
    left_cheek_ids = set(LEFT_CHEEK_LANDMARKS)
    right_cheek_ids = set(RIGHT_CHEEK_LANDMARKS)

    for idx, lm in enumerate(landmarks):
        x = int(lm.x * frame_w)
        y = int(lm.y * frame_h)

        if idx in forehead_ids:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(frame, str(idx), (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28, (0, 255, 0), 1)
        elif idx in left_cheek_ids:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(idx), (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28, (0, 0, 255), 1)
        elif idx in right_cheek_ids:
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            cv2.putText(frame, str(idx), (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.28, (255, 0, 0), 1)
        elif idx % 10 == 0:
            cv2.circle(frame, (x, y), 2, (180, 180, 180), -1)
            cv2.putText(frame, str(idx), (x + 2, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.22, (180, 180, 180), 1)

    return frame


# ─────────────────────────────────────────
# Face visibility check
# ─────────────────────────────────────────

def is_face_visible(landmarks, frame_w, frame_h):
    """
    Check face frontality and visibility.

    Frontality: nose tip (1) centered between
    face edges 234 (left) and 454 (right).

    Output:
        visible        — bool
        frontality     — 0.0 to 1.0
        visibility_pct — fraction of ROI landmarks in bounds
    """
    key_ids = (
        FOREHEAD_LANDMARKS +
        LEFT_CHEEK_LANDMARKS +
        RIGHT_CHEEK_LANDMARKS
    )

    in_bounds = sum(
        1 for idx in key_ids
        if 0.05 < landmarks[idx].x < 0.95
        and 0.05 < landmarks[idx].y < 0.95
    )
    visibility_pct = in_bounds / len(key_ids)

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

    Score 1 — visibility + frontality  weight 0.5
    Score 2 — key landmarks in bounds  weight 0.3
    Score 3 — landmark visibility mean weight 0.2
    """
    if not face_results.multi_face_landmarks:
        return 0.0

    landmarks = face_results.multi_face_landmarks[0].landmark

    visible, frontality, visibility_pct = is_face_visible(
        landmarks, frame_w, frame_h
    )
    visibility_score = frontality * 0.5 + visibility_pct * 0.5

    key_ids   = FOREHEAD_LANDMARKS
    in_bounds = sum(
        1 for idx in key_ids
        if 0.0 < landmarks[idx].x < 1.0
        and 0.0 < landmarks[idx].y < 1.0
    )
    bounds_score = in_bounds / len(key_ids)

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
# Evaluation tests
# ─────────────────────────────────────────

def calculate_roi_stability(roi_area_history):
    if len(roi_area_history) < 5:
        return 1.0, 0.0
    areas     = np.array(list(roi_area_history))
    mean_area = np.mean(areas)
    if mean_area == 0:
        return 0.0, 1.0
    cv        = np.std(areas) / mean_area
    stability = max(0.0, 1.0 - (cv / 0.30))
    return round(stability, 2), round(cv, 3)


def estimate_skin_tone(frame, mask):
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
    ita = np.degrees(np.arctan((L - 50.0) / b)) \
          if b != 0 else 90.0
    if ita > 55:
        fst = "FST I-II (Very Light)"
    elif ita > 28:
        fst = "FST III (Light-Medium)"
    elif ita > 10:
        fst = "FST IV (Medium)"
    elif ita > -30:
        fst = "FST V (Medium-Dark)"
    else:
        fst = "FST VI (Dark)"
    return fst, round(ita, 1)


def assess_lighting(frame, mask):
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


def calculate_motion(current_landmarks,
                     previous_landmarks,
                     frame_w, frame_h):
    if previous_landmarks is None:
        return 0.0, "Stable"
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
    Face ROI detection using verified landmark sets.

    Regions:
        Green  — Forehead  (centerline + boundaries)
        Red    — Left cheek  (subject left)
        Blue   — Right cheek (subject right)

    Nose removed — breathing motion + specular
    reflection degrade rPPG signal quality.

    Press Q to quit.
    """

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

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        print("Face detection running — press Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
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

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                visible, frontality, vis_pct = is_face_visible(
                    landmarks, frame_w, frame_h
                )

                # Draw color-coded ROI landmarks
                frame = draw_roi_landmarks(
                    frame, landmarks, frame_w, frame_h
                )

                motion_pixels, motion_status = calculate_motion(
                    landmarks, previous_landmarks,
                    frame_w, frame_h
                )
                previous_landmarks = landmarks

                # ── Build ROIs ─────────────────────────
                forehead_pts = get_landmark_pixels(
                    landmarks, frame_w, frame_h
                )
                left_cheek_pts = get_landmark_pixels(
                    landmarks, LEFT_CHEEK_LANDMARKS,
                    frame_w, frame_h
                )
                right_cheek_pts = get_landmark_pixels(
                    landmarks, RIGHT_CHEEK_LANDMARKS,
                    frame_w, frame_h
                )

                # Combined mask — no nose
                combined_mask = get_combined_mask(
                    [forehead_pts,
                     left_cheek_pts,
                     right_cheek_pts],
                    frame_h, frame_w
                )

                total_area = get_roi_area(combined_mask)
                roi_area_history.append(total_area)
                stability, stability_cv = \
                    calculate_roi_stability(roi_area_history)

                if frame_counter % EVAL_EVERY_N_FRAMES == 0:
                    cached_fitzpatrick, cached_ita = \
                        estimate_skin_tone(frame, combined_mask)
                    (cached_lighting_status,
                     cached_brightness, _, _) = \
                        assess_lighting(frame, combined_mask)

                # Draw filled ROIs
                frame = draw_roi(
                    frame, forehead_pts,
                    color=(0, 255, 0),
                    alpha=0.3, label="Forehead"
                )
                frame = draw_roi(
                    frame, left_cheek_pts,
                    color=(0, 0, 255),
                    alpha=0.3, label="L.Cheek"
                )
                frame = draw_roi(
                    frame, right_cheek_pts,
                    color=(255, 0, 0),
                    alpha=0.3, label="R.Cheek"
                )

                cv2.putText(
                    frame,
                    f"Frontality:{frontality:.2f}"
                    f"  Vis:{vis_pct:.2f}",
                    (frame_w - 260, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0) if visible
                    else (0, 165, 255),
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
                "GREEN=Forehead  RED=L.Cheek  BLUE=R.Cheek",
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