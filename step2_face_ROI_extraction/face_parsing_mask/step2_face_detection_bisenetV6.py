import cv2
import numpy as np
import torch
import time
import os
import sys
from collections import deque

# ─────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add face-parsing.PyTorch repo to path
# so we can import the correct BiSeNet architecture
REPO_DIR = os.path.join(CURRENT_DIR, "face-parsing.PyTorch")
sys.path.insert(0, REPO_DIR)

# Add project root to path
# so we can import step1_video_capture
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))
)
sys.path.insert(0, PROJECT_ROOT)

# BiSeNet pretrained weights
MODEL_PATH = os.path.join(
    CURRENT_DIR, "models", "bisenet_resnet18.pth"
)

# ── Parameters ────────────────────────────
# BiSeNet runs every N frames — cached between
# Higher = better fps, lower = more responsive
EVAL_EVERY_N_FRAMES = 20

# IoU threshold for mask stability
# New mask only applied if overlap with old mask
# drops below this threshold — prevents glitching
# from small BiSeNet inference variance
IOU_UPDATE_THRESHOLD = 0.85

# BiSeNet CelebAMask-HQ label map — 19 classes
LABEL_BACKGROUND = 0
LABEL_SKIN       = 1
LABEL_L_BROW     = 2
LABEL_R_BROW     = 3
LABEL_L_EYE      = 4
LABEL_R_EYE      = 5
LABEL_L_GLASS    = 6
LABEL_R_GLASS    = 7
LABEL_L_EAR      = 8
LABEL_R_EAR      = 9
LABEL_EARRING    = 10
LABEL_NOSE       = 11
LABEL_MOUTH      = 12
LABEL_U_LIP      = 13
LABEL_L_LIP      = 14
LABEL_NECK       = 15
LABEL_NECKLACE   = 16
LABEL_CLOTH      = 17
LABEL_HAIR       = 18
LABEL_HAT        = 19

# Labels excluded from skin ROI
# These are anatomically on the face but not
# suitable for rPPG signal extraction
EXCLUDE_LABELS = [
    LABEL_L_BROW, LABEL_R_BROW,
    LABEL_L_EYE,  LABEL_R_EYE,
    LABEL_L_GLASS,LABEL_R_GLASS,
    LABEL_NOSE,   LABEL_MOUTH,
    LABEL_U_LIP,  LABEL_L_LIP,
    LABEL_HAIR,   LABEL_HAT
]


# ─────────────────────────────────────────
# PHASE 1 — Model loading
# ─────────────────────────────────────────

def load_bisenet(model_path, device):
    """
    PHASE 1: Load BiSeNet with pretrained weights.

    Uses official architecture from
    zllrunning/face-parsing.PyTorch — guarantees
    exact match between architecture and weights.

    Input:
        model_path — path to bisenet_resnet18.pth
        device     — torch device (cpu)

    Output:
        model in eval mode ready for inference
    """
    from model import BiSeNet

    model = BiSeNet(n_classes=19)
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()
    print(f"BiSeNet loaded from {model_path}")
    return model


# ─────────────────────────────────────────
# PHASE 2 — Face parsing inference
# ─────────────────────────────────────────

def parse_face(model, frame_rgb, device):
    """
    PHASE 2: Run BiSeNet inference on one RGB frame.

    BiSeNet takes a face image and returns a pixel-level
    label map — every pixel gets one of 19 class labels
    (skin, hair, left eye, nose, lips etc.)

    This is the core of why BiSeNet outperforms
    landmark-based approaches — it classifies every
    single pixel individually rather than defining
    regions from sparse landmark points.

    Processing steps:
        1. Resize to 512x512 (model input size)
        2. Normalize with ImageNet mean/std
        3. Convert to PyTorch tensor NCHW format
        4. Run inference — model returns 19-channel map
        5. Argmax across channels → one label per pixel
        6. Resize label map back to original frame size

    Input:
        model     — loaded BiSeNet model
        frame_rgb — RGB frame (H, W, 3) uint8
        device    — torch device

    Output:
        parsing   — label map (H, W) uint8
                    each pixel = class label 0-18
    """
    h, w = frame_rgb.shape[:2]

    # Step 1 — resize to model input size
    img = cv2.resize(frame_rgb, (512, 512))

    # Step 2 — normalize with ImageNet statistics
    # BiSeNet was trained on ImageNet-normalized images
    img = img.astype(np.float32) / 255.0
    img -= np.array([0.485, 0.456, 0.406],
                    dtype=np.float32)
    img /= np.array([0.229, 0.224, 0.225],
                    dtype=np.float32)

    # Step 3 — HWC → CHW → NCHW tensor
    tensor = torch.from_numpy(
        img.transpose(2, 0, 1)
    ).unsqueeze(0).to(device)

    # Step 4 — inference (no gradient needed)
    with torch.no_grad():
        out = model(tensor)[0]  # returns tuple — take main output

    # Step 5 — argmax → one label per pixel
    parsing = out.squeeze(0).argmax(0).cpu().numpy()

    # Step 6 — resize back to original frame dimensions
    parsing = cv2.resize(
        parsing.astype(np.uint8),
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )
    return parsing


# ─────────────────────────────────────────
# PHASE 3 — ROI mask construction
# ─────────────────────────────────────────

def build_face_roi_masks(parsing, frame_h, frame_w):
    """
    PHASE 3: Build forehead and cheek ROI masks
    from BiSeNet parsing output.

    This phase converts the pixel-level label map
    into clean binary masks for rPPG signal extraction.

    Processing steps:
        1. Extract skin pixels (label 1 only)
        2. Remove non-rPPG regions (eyes, brows, nose etc.)
        3. Morphological cleanup (fill holes, remove noise)
        4. Find vertical skin extent
        5. Adaptive forehead boundary from eyebrow position
        6. Adaptive cheek boundary from eye position
        7. Adaptive nose strip from nose x-position

    Why adaptive boundaries:
        Fixed percentage splits work only for one face.
        Using detected anatomical positions (eyebrows,
        eyes, nose) makes boundaries work correctly for
        any face size, distance, or individual proportion.

    Input:
        parsing  — BiSeNet label map (H, W)
        frame_h  — frame height
        frame_w  — frame width

    Output:
        forehead_mask — binary mask (H, W)
        cheek_mask    — binary mask (H, W)
        combined_mask — binary mask (H, W)
    """

    # ── Step 1: Base skin mask ─────────────────────
    # Start with all pixels BiSeNet labeled as skin
    skin_mask = (parsing == LABEL_SKIN).astype(np.uint8)

    # ── Step 2: Remove non-rPPG regions ───────────
    # BiSeNet labels eyebrows, eyes, nose, lips as
    # separate classes — remove them from skin mask
    # because they degrade signal quality:
    #   Eyes/brows: no blood vessels for rPPG
    #   Nose: specular reflection + breathing motion
    #   Lips: different optical properties
    #   Hair: blocks light path entirely
    for label in EXCLUDE_LABELS:
        skin_mask[parsing == label] = 0

    # ── Step 3: Morphological cleanup ─────────────
    # MORPH_CLOSE: fills small holes inside skin region
    # MORPH_OPEN:  removes small isolated noise pixels
    kernel    = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
    skin_mask = cv2.morphologyEx(
        skin_mask, cv2.MORPH_CLOSE, kernel
    )
    skin_mask = cv2.morphologyEx(
        skin_mask, cv2.MORPH_OPEN, kernel
    )

    # ── Step 4: Find vertical skin extent ─────────
    skin_rows = np.where(skin_mask.sum(axis=1) > 0)[0]

    if len(skin_rows) == 0:
        empty = np.zeros((frame_h, frame_w), dtype=np.uint8)
        return empty, empty, empty

    skin_top    = skin_rows[0]
    skin_bottom = skin_rows[-1]
    skin_height = skin_bottom - skin_top

    if skin_height == 0:
        empty = np.zeros((frame_h, frame_w), dtype=np.uint8)
        return empty, empty, empty

    # ── Step 5: Adaptive forehead boundary ────────
    # Find actual eyebrow position from parsing map
    # Forehead = skin region above the eyebrows
    # 15% buffer above brow_top ensures full coverage
    brow_pixels = np.where(
        (parsing == LABEL_L_BROW) |
        (parsing == LABEL_R_BROW)
    )
    if len(brow_pixels[0]) > 0:
        brow_top     = int(np.min(brow_pixels[0]))
        forehead_cut = max(
            skin_top,
            brow_top - int(skin_height * 0.05)
        )
    else:
        # Fallback if eyebrows not detected
        forehead_cut = skin_top + int(skin_height * 0.30)

    forehead_mask = skin_mask.copy()
    forehead_mask[forehead_cut:, :] = 0

    # ── Step 6: Adaptive cheek boundary ───────────
    # Find actual eye position from parsing map
    # Cheeks = skin region just below the eyes
    # down to 70% of face height (above jaw/mouth)
    eye_pixels = np.where(
        (parsing == LABEL_L_EYE) |
        (parsing == LABEL_R_EYE)
    )
    if len(eye_pixels[0]) > 0:
        eye_bottom = int(np.max(eye_pixels[0]))
        cheek_top  = eye_bottom + int(skin_height * 0.04)
    else:
        # Fallback if eyes not detected
        cheek_top  = skin_top + int(skin_height * 0.45)

    cheek_bottom = skin_top + int(skin_height * 0.70)

    if cheek_top >= cheek_bottom:
        cheek_top = skin_top + int(skin_height * 0.45)

    cheek_mask = skin_mask.copy()
    cheek_mask[:cheek_top, :]    = 0
    cheek_mask[cheek_bottom:, :] = 0

    # ── Step 7: Adaptive nose strip ───────────────
    # Find actual nose x-position from parsing map
    # Remove center strip so cheeks don't include
    # nose area — strip width = 6% of face width
    nose_pixels = np.where(parsing == LABEL_NOSE)
    if len(nose_pixels[1]) > 0:
        nose_center_x = int(np.mean(nose_pixels[1]))
    else:
        skin_cols     = np.where(
            skin_mask.sum(axis=0) > 0
        )[0]
        nose_center_x = int(np.mean(skin_cols)) \
                        if len(skin_cols) > 0 \
                        else frame_w // 2

    skin_cols_all = np.where(
        skin_mask.sum(axis=0) > 0
    )[0]
    if len(skin_cols_all) > 0:
        face_width = int(
            skin_cols_all[-1] - skin_cols_all[0]
        )
        nose_strip = max(int(face_width * 0.06), 10)
    else:
        nose_strip = frame_w // 14

    cheek_mask[
        :,
        nose_center_x - nose_strip:
        nose_center_x + nose_strip
    ] = 0

    combined_mask = np.clip(
        forehead_mask + cheek_mask, 0, 1
    ).astype(np.uint8)

    return forehead_mask, cheek_mask, combined_mask


# ─────────────────────────────────────────
# PHASE 4 — Mask stability (IoU filter)
# ─────────────────────────────────────────

def should_update_mask(old_mask, new_mask,
                       threshold=IOU_UPDATE_THRESHOLD):
    """
    PHASE 4: Decide whether to apply new mask.

    BiSeNet inference has small random variance —
    each run produces slightly different boundaries
    even with the same face in the same position.
    Without filtering, this causes visible glitching
    every EVAL_EVERY_N_FRAMES frames.

    IoU (Intersection over Union) measures overlap
    between old and new mask:
        IoU = 1.0 → identical masks
        IoU = 0.0 → completely different masks

    Decision rule:
        IoU > threshold → masks too similar to update
                          keep stable old mask
        IoU < threshold → face moved significantly
                          apply new mask

    Input:
        old_mask  — currently displayed mask
        new_mask  — newly computed mask
        threshold — IoU threshold (default 0.85)

    Output:
        True  — apply new mask
        False — keep old mask
    """
    if old_mask is None:
        return True  # always update on first frame

    intersection = np.sum(old_mask & new_mask)
    union        = np.sum(old_mask | new_mask)

    if union == 0:
        return True

    iou = intersection / union
    return iou < threshold


# ─────────────────────────────────────────
# PHASE 5 — Visualization
# ─────────────────────────────────────────

def overlay_mask(frame, mask, color, alpha=0.4):
    """
    PHASE 5: Overlay binary mask on frame.

    Draws filled colored region + contour outline
    so ROI boundaries are clearly visible.

    Input:
        frame  — BGR frame
        mask   — binary mask (H, W)
        color  — BGR color tuple
        alpha  — fill transparency (0=invisible 1=solid)

    Output:
        frame with colored mask overlay
    """
    if mask is None or np.sum(mask) == 0:
        return frame

    overlay          = frame.copy()
    colored          = np.zeros_like(frame)
    colored[mask==1] = color
    cv2.addWeighted(
        colored, alpha, overlay, 1 - alpha, 0, overlay
    )
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def visualize_parsing(parsing, frame_h, frame_w):
    """
    PHASE 5: Colored visualization of all 19 regions.
    Toggle with V key — useful for debugging to verify
    BiSeNet is correctly classifying each face region.

    Input:
        parsing — label map (H, W)
        frame_h — frame height
        frame_w — frame width

    Output:
        colored label map (H, W, 3) BGR
    """
    colors = [
        (0,   0,   0),    # 0  background
        (255, 200, 150),  # 1  skin
        (0,   100, 255),  # 2  left brow
        (0,   100, 255),  # 3  right brow
        (0,   200, 255),  # 4  left eye
        (0,   200, 255),  # 5  right eye
        (200, 200, 0),    # 6  left glass
        (200, 200, 0),    # 7  right glass
        (150, 75,  0),    # 8  left ear
        (150, 75,  0),    # 9  right ear
        (200, 100, 0),    # 10 earring
        (0,   150, 255),  # 11 nose
        (0,   255, 100),  # 12 mouth
        (0,   0,   255),  # 13 upper lip
        (0,   0,   200),  # 14 lower lip
        (200, 200, 200),  # 15 neck
        (150, 150, 150),  # 16 necklace
        (100, 100, 200),  # 17 cloth
        (50,  50,  50),   # 18 hair
    ]

    vis = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    for label, color in enumerate(colors):
        vis[parsing == label] = color
    return vis


# ─────────────────────────────────────────
# PHASE 6 — Evaluation tests
# ─────────────────────────────────────────

def calculate_roi_stability(roi_area_history):
    """
    TEST 1: ROI stability via coefficient of variation
    of ROI pixel area across recent frames.

    A stable ROI is essential for rPPG — if the region
    changes size significantly frame to frame, the RGB
    time-series will have artificial jumps unrelated
    to blood flow changes.

    CV < 0.05  → very stable (stability ~1.0)
    CV > 0.30  → unstable   (stability ~0.0)

    Output:
        stability — 0.0 to 1.0
        raw_cv    — coefficient of variation
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


def estimate_skin_tone(frame, mask):
    """
    TEST 2: Fitzpatrick type via ITA formula.

    ITA (Individual Typology Angle) converts mean
    CIELab L* and b* values from the ROI pixels
    to an angle that correlates with skin tone.

        ITA >  55  → FST I-II   (Very Light)
        28 – 55    → FST III    (Light-Medium)
        10 – 28    → FST IV     (Medium)
       -30 – 10    → FST V      (Medium-Dark)
        ITA < -30  → FST VI     (Dark)

    This is the routing decision point —
    FST V-VI triggers palm fallback in full pipeline.
    """
    if np.sum(mask) == 0:
        return "Unknown", 0.0

    roi_pixels = frame[mask == 1]
    if len(roi_pixels) == 0:
        return "Unknown", 0.0

    mean_bgr     = np.mean(roi_pixels, axis=0)
    mean_bgr_img = np.uint8([[mean_bgr]])
    lab          = cv2.cvtColor(
        mean_bgr_img, cv2.COLOR_BGR2Lab
    )
    L   = float(lab[0, 0, 0]) * 100.0 / 255.0
    b   = float(lab[0, 0, 2]) - 128.0
    ita = np.degrees(
        np.arctan((L - 50.0) / b)
    ) if b != 0 else 90.0

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
    """
    TEST 3: Lighting quality inside ROI.

    Checks three conditions:
        Brightness  < 40    → Too Dark
        Overexposed > 20%   → Overexposed
        Uniformity  > 0.4   → Uneven Lighting
        80 ≤ bright ≤ 200   → Good
        Otherwise           → Acceptable
    """
    if np.sum(mask) == 0:
        return "No ROI", 0.0

    gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_pixels = gray[mask == 1].astype(float)

    if len(roi_pixels) == 0:
        return "No ROI", 0.0

    brightness      = float(np.mean(roi_pixels))
    uniformity      = float(np.std(roi_pixels)) \
                      / brightness if brightness > 0 else 1.0
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

    return status, round(brightness, 1)


def calculate_motion(current_parsing, previous_parsing):
    """
    TEST 4: Face motion via skin centroid displacement.

    Tracks center of mass of the skin region between
    consecutive parsing results. Large displacement
    means the face moved — signal frames captured
    during high motion should be flagged for exclusion
    in Step 3 signal extraction.
    """
    if previous_parsing is None:
        return 0.0, "Stable"

    def centroid(parsing):
        mask = (parsing == LABEL_SKIN).astype(np.uint8)
        m    = cv2.moments(mask)
        if m["m00"] == 0:
            return None
        return (m["m10"] / m["m00"],
                m["m01"] / m["m00"])

    curr_c = centroid(current_parsing)
    prev_c = centroid(previous_parsing)

    if curr_c is None or prev_c is None:
        return 0.0, "Stable"

    dx            = curr_c[0] - prev_c[0]
    dy            = curr_c[1] - prev_c[1]
    motion_pixels = float(np.sqrt(dx**2 + dy**2))

    if motion_pixels < 1.0:
        status = "Stable"
    elif motion_pixels < 3.0:
        status = "Slight Motion"
    elif motion_pixels < 10.0:
        status = "Motion Detected"
    else:
        status = "Too Much Motion"

    return round(motion_pixels, 2), status


def confidence_from_mask(mask):
    """
    Overall confidence from ROI pixel coverage.
    More skin pixels detected = higher confidence
    that the ROI is suitable for rPPG extraction.
    """
    if mask is None:
        return 0.0
    pixel_count   = int(np.sum(mask))
    expected_good = 8000
    if pixel_count == 0:
        return 0.0
    return round(min(1.0, pixel_count / expected_good), 2)


# ─────────────────────────────────────────
# PHASE 7 — Main detection loop
# ─────────────────────────────────────────

def run_face_detection(cap, actual_fps):
    """
    PHASE 7: Main face ROI extraction loop.

    Receives camera from step1_video_captureV3
    and runs the full face ROI extraction pipeline.

    Workflow per frame:
        Every N frames:
            → Run BiSeNet inference (Phase 2)
            → Build ROI masks (Phase 3)
            → Apply IoU stability filter (Phase 4)
            → Update evaluations (Phase 6)
        Every frame:
            → Draw overlays (Phase 5)
            → Calculate T1 stability + T4 motion
            → Display HUD

    Input:
        cap        — OpenCV VideoCapture from step1
        actual_fps — measured fps from step1

    Press V to toggle parsing label visualization.
    Press Q to quit.
    """

    # ── Device ────────────────────────────────────
    device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load BiSeNet ──────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: weights not found at {MODEL_PATH}")
        return

    print("Loading BiSeNet...")
    model = load_bisenet(MODEL_PATH, device)

    # ── State variables ───────────────────────────
    timestamps       = deque(maxlen=max(int(actual_fps), 2))
    roi_area_history = deque(maxlen=int(actual_fps * 2))
    frame_counter    = 0
    show_parsing_vis = False

    # Cached results — reused between BiSeNet runs
    cached_parsing         = None
    cached_forehead_mask   = None
    cached_cheek_mask      = None
    cached_combined_mask   = None
    cached_fitzpatrick     = "Unknown"
    cached_ita             = 0.0
    cached_lighting_status = "No ROI"
    cached_brightness      = 0.0
    previous_parsing       = None

    print("Face detection running — Q to quit, V to toggle parsing view")

    # ── Main loop ─────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_counter += 1

        # Rolling fps calculation
        timestamps.append(time.time())
        if len(timestamps) >= 2:
            rolling_fps = (len(timestamps) - 1) / (
                timestamps[-1] - timestamps[0]
            )
        else:
            rolling_fps = actual_fps

        frame_h, frame_w = frame.shape[:2]

        # BGR → RGB for BiSeNet
        frame_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        # ── BiSeNet inference every N frames ───────
        # Cached between runs to preserve fps
        run_parsing = (
            frame_counter % EVAL_EVERY_N_FRAMES == 0
            or cached_parsing is None
        )

        if run_parsing:
            # Phase 2 — inference
            parsing = parse_face(model, frame_rgb, device)

            # Phase 3 — build masks
            new_forehead, new_cheek, new_combined = \
                build_face_roi_masks(
                    parsing, frame_h, frame_w
                )

            # Phase 4 — IoU stability filter
            # only update if face moved significantly
            if should_update_mask(
                cached_combined_mask, new_combined
            ):
                previous_parsing       = cached_parsing
                cached_parsing         = parsing
                cached_forehead_mask   = new_forehead
                cached_cheek_mask      = new_cheek
                cached_combined_mask   = new_combined

                # Phase 6 — update evaluations
                cached_fitzpatrick, cached_ita = \
                    estimate_skin_tone(
                        frame, new_combined
                    )
                cached_lighting_status, \
                cached_brightness = \
                    assess_lighting(frame, new_combined)

        # ── Per frame evaluations ──────────────────
        # T1 stability — runs every frame
        stability, stability_cv = 1.0, 0.0
        if cached_combined_mask is not None:
            area = int(np.sum(cached_combined_mask))
            roi_area_history.append(area)
            stability, stability_cv = \
                calculate_roi_stability(roi_area_history)

        # T4 motion — runs every frame
        motion_pixels, motion_status = calculate_motion(
            cached_parsing, previous_parsing
        )

        # ── Draw overlays ─────────────────────────
        display = frame.copy()

        face_detected = (
            cached_combined_mask is not None
            and np.sum(cached_combined_mask) > 0
        )

        if face_detected:
            # Green = Forehead (best rPPG signal)
            # Blue  = Cheeks   (good rPPG signal)
            display = overlay_mask(
                display, cached_forehead_mask,
                color=(0, 255, 0), alpha=0.4
            )
            display = overlay_mask(
                display, cached_cheek_mask,
                color=(255, 0, 0), alpha=0.4
            )
            face_status = "Face detected - GOOD"
            face_color  = (0, 255, 0)
        else:
            face_status = "No face detected"
            face_color  = (0, 0, 255)

        # Parsing label visualization (V key toggle)
        if show_parsing_vis and \
           cached_parsing is not None:
            vis = visualize_parsing(
                cached_parsing, frame_h, frame_w
            )
            cv2.imshow("Parsing Labels", vis)
        else:
            try:
                cv2.destroyWindow("Parsing Labels")
            except:
                pass

        # ── HUD ───────────────────────────────────
        y = [30]

        def put(text, color=(0, 255, 0), scale=0.6):
            cv2.putText(display, text, (10, y[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 2)
            y[0] += 28

        put(f"FPS: {rolling_fps:.1f}")
        put(face_status, face_color)
        put(
            f"Confidence: {confidence_from_mask(cached_combined_mask):.2f}",
            (0, 255, 0) if face_detected
            else (0, 0, 255)
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

        frames_until = EVAL_EVERY_N_FRAMES - \
                       (frame_counter % EVAL_EVERY_N_FRAMES)
        cv2.putText(
            display,
            f"Parse in: {frames_until}f",
            (frame_w - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 0), 1
        )

        cv2.putText(
            display,
            "GREEN=Forehead  BLUE=Cheeks  V=parsing view",
            (10, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (255, 255, 255), 1
        )

        cv2.imshow("Face ROI — BiSeNet", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('v'):
            show_parsing_vis = not show_parsing_vis

    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

#if __name__ == "__main__":
#    from step1_video_capture.step1_video_captureV3 \
#       import initialize_camera

#    cap, actual_fps = initialize_camera(camera_index=0)
#    if cap is not None:
#        run_face_detection(cap, actual_fps)