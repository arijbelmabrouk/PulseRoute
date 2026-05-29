import cv2
import numpy as np
import torch
import time
import os
import sys
import threading
from collections import deque

# ─────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR    = os.path.join(CURRENT_DIR, "face-parsing.PyTorch")
sys.path.insert(0, REPO_DIR)

MODEL_PATH = os.path.join(
    CURRENT_DIR, "models", "bisenet_resnet18.pth"
)

# ── Optimization parameters ───────────────
EVAL_EVERY_N_FRAMES = 15
BISENET_INPUT_SIZE  = 256

# BiSeNet CelebAMask-HQ label map
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

EXCLUDE_LABELS = [
    LABEL_L_BROW, LABEL_R_BROW,
    LABEL_L_EYE,  LABEL_R_EYE,
    LABEL_L_GLASS,LABEL_R_GLASS,
    LABEL_NOSE,   LABEL_MOUTH,
    LABEL_U_LIP,  LABEL_L_LIP,
    LABEL_HAIR,   LABEL_HAT
]


# ─────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────

def load_bisenet(model_path, device):
    """
    Load BiSeNet using official architecture
    from zllrunning/face-parsing.PyTorch.

    Input:
        model_path — path to .pth weights file
        device     — torch device

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
# Background inference thread
# ─────────────────────────────────────────
class BiSeNetWorker:
    def __init__(self, model, device):
        self.model          = model
        self.device         = device
        self.latest_parsing = None
        self.result_count   = 0        # increment each new result
        self.last_read      = 0        # last count main loop read
        self.lock           = threading.Lock()
        self._pending_frame = None
        self._frame_lock    = threading.Lock()
        self._running       = True
        self._thread        = threading.Thread(
            target=self._run, daemon=True
        )
        self._thread.start()

    def submit(self, frame_rgb):
        with self._frame_lock:
            self._pending_frame = frame_rgb.copy()

    def get_latest(self):
        with self.lock:
            return self.latest_parsing, self.result_count

    def stop(self):
        self._running = False

    def _run(self):
        while self._running:
            frame_rgb = None
            with self._frame_lock:
                if self._pending_frame is not None:
                    frame_rgb           = self._pending_frame
                    self._pending_frame = None

            if frame_rgb is not None:
                parsing = self._infer(frame_rgb)
                with self.lock:
                    self.latest_parsing = parsing
                    self.result_count  += 1
            else:
                time.sleep(0.005)

    def _infer(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        img = cv2.resize(
            frame_rgb,
            (BISENET_INPUT_SIZE, BISENET_INPUT_SIZE)
        )
        img = img.astype(np.float32) / 255.0
        img -= np.array([0.485, 0.456, 0.406],
                        dtype=np.float32)
        img /= np.array([0.229, 0.224, 0.225],
                        dtype=np.float32)
        tensor = torch.from_numpy(
            img.transpose(2, 0, 1)
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)[0]
        parsing = out.squeeze(0).argmax(0).cpu().numpy()
        parsing = cv2.resize(
            parsing.astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        return parsing
# ─────────────────────────────────────────
# ROI mask builders — fully adaptive
# ─────────────────────────────────────────

def build_face_roi_masks(parsing, frame_h, frame_w):
    """
    Build forehead and cheek ROI masks from
    BiSeNet parsing output.

    Fully adaptive — all boundaries from actual
    detected anatomical positions:

        Forehead bottom → just above detected eyebrows
        Cheek top       → just below detected eyes
        Cheek bottom    → 70% of skin height
        Nose center     → actual detected nose x-position
        Nose strip      → 8% of face width

    Fallbacks used if anatomy not detected.

    Input:
        parsing  — BiSeNet label map (H, W)
        frame_h  — frame height
        frame_w  — frame width

    Output:
        forehead_mask — binary mask (H, W)
        cheek_mask    — binary mask (H, W)
        combined_mask — binary mask (H, W)
    """
    # Base skin mask
    skin_mask = (parsing == LABEL_SKIN).astype(np.uint8)

    for label in EXCLUDE_LABELS:
        skin_mask[parsing == label] = 0

    # Morphological cleanup
    kernel    = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
    skin_mask = cv2.morphologyEx(
        skin_mask, cv2.MORPH_CLOSE, kernel
    )
    skin_mask = cv2.morphologyEx(
        skin_mask, cv2.MORPH_OPEN, kernel
    )

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

    # ── Forehead boundary ──────────────────────────
    # Bottom = just above detected eyebrows
    brow_pixels = np.where(
        (parsing == LABEL_L_BROW) |
        (parsing == LABEL_R_BROW)
    )
    if len(brow_pixels[0]) > 0:
        brow_top     = int(np.min(brow_pixels[0]))
        forehead_cut = max(
            skin_top,
            brow_top - int(skin_height * 0.02)
        )
    else:
        forehead_cut = skin_top + int(skin_height * 0.30)

    forehead_mask = skin_mask.copy()
    forehead_mask[forehead_cut:, :] = 0

    # ── Cheek boundaries ───────────────────────────
    # Top = just below detected eyes
    eye_pixels = np.where(
        (parsing == LABEL_L_EYE) |
        (parsing == LABEL_R_EYE)
    )
    if len(eye_pixels[0]) > 0:
        eye_bottom = int(np.max(eye_pixels[0]))
        cheek_top  = eye_bottom + int(skin_height * 0.04)
    else:
        cheek_top  = skin_top + int(skin_height * 0.45)

    # Bottom = above jaw and mouth
    cheek_bottom = skin_top + int(skin_height * 0.70)

    if cheek_top >= cheek_bottom:
        cheek_top = skin_top + int(skin_height * 0.45)

    cheek_mask = skin_mask.copy()
    cheek_mask[:cheek_top, :]    = 0
    cheek_mask[cheek_bottom:, :] = 0

    # ── Nose center strip ──────────────────────────
    # Center = actual detected nose x-position
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

    # Strip width = 8% of face width
    skin_cols_all = np.where(
        skin_mask.sum(axis=0) > 0
    )[0]
    if len(skin_cols_all) > 0:
        face_width = int(
            skin_cols_all[-1] - skin_cols_all[0]
        )
        nose_strip = max(int(face_width * 0.08), 15)
    else:
        nose_strip = frame_w // 10

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
# Visualization
# ─────────────────────────────────────────

def overlay_mask(frame, mask, color, alpha=0.4):
    """
    Overlay binary mask on frame with color
    and contour outline.

    Input:
        frame  — BGR frame
        mask   — binary mask (H, W)
        color  — BGR color tuple
        alpha  — overlay transparency

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
    Colored visualization of all 19 parsed regions.
    Toggle with V key during runtime.
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
# Evaluation tests
# ─────────────────────────────────────────

def calculate_roi_stability(roi_area_history):
    """
    ROI stability via coefficient of variation
    of pixel area across recent frames.

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
    Fitzpatrick type via ITA formula on CIELab pixels.

        ITA >  55  → FST I-II   (Very Light)
        28 – 55    → FST III    (Light-Medium)
        10 – 28    → FST IV     (Medium)
       -30 – 10    → FST V      (Medium-Dark)
        ITA < -30  → FST VI     (Dark)

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
    Lighting quality inside ROI.
    Checks brightness, overexposure, uniformity.
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
    Face motion via skin centroid displacement
    between consecutive parsing results.
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
    """Confidence from ROI pixel coverage."""
    if mask is None:
        return 0.0
    pixel_count   = int(np.sum(mask))
    expected_good = 8000
    if pixel_count == 0:
        return 0.0
    return round(min(1.0, pixel_count / expected_good), 2)


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def run_face_detection():
    """
    Opens camera, loads BiSeNet, runs face parsing
    with background threading for maximum fps.

    Three fps optimizations:
        1. BiSeNet input reduced to 256x256
        2. Inference every 15 frames
        3. Background thread — main loop never blocks

    Temporal mask smoothing prevents glitching
    when new inference results arrive.

    First frame submitted immediately for fast
    initial result.

    Press V to toggle parsing label visualization.
    Press Q to quit.
    """

    # ── Device ────────────────────────────────────
    device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: weights not found at {MODEL_PATH}")
        return

    print("Loading BiSeNet...")
    model  = load_bisenet(MODEL_PATH, device)
    worker = BiSeNetWorker(model, device)
    print("Background inference worker started")

    # ── Camera ────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Camera not detected")
        worker.stop()
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
    frame_counter    = 0
    show_parsing_vis = False

    # Cache
    cached_parsing         = None
    cached_forehead_mask   = None
    cached_cheek_mask      = None
    cached_combined_mask   = None
    cached_fitzpatrick     = "Unknown"
    cached_ita             = 0.0
    cached_lighting_status = "No ROI"
    cached_brightness      = 0.0
    previous_parsing       = None

    print("Running — Q to quit, V to toggle parsing view")

    try:
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
            frame_rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            # ── Submit frame to worker ─────────────
            # First frame submitted immediately
            # then every N frames
            if frame_counter == 1 or \
               frame_counter % EVAL_EVERY_N_FRAMES == 0:
                worker.submit(frame_rgb)

            # ── Get latest parsing result ──────────
            latest, result_count = worker.get_latest()

            if latest is not None and result_count > worker.last_read:
                # New result available — update immediately
                worker.last_read     = result_count
                previous_parsing     = cached_parsing
                cached_parsing       = latest

                # Direct assignment — no smoothing
                # smoothing caused forehead to disappear
                cached_forehead_mask, \
                cached_cheek_mask, \
                cached_combined_mask = build_face_roi_masks(
                    latest, frame_h, frame_w
                )

                cached_fitzpatrick, cached_ita = \
                    estimate_skin_tone(frame, cached_combined_mask)
                cached_lighting_status, cached_brightness = \
                    assess_lighting(frame, cached_combined_mask)
            # ── Per frame ─────────────────────────
            stability, stability_cv = 1.0, 0.0
            if cached_combined_mask is not None:
                area = int(np.sum(cached_combined_mask))
                roi_area_history.append(area)
                stability, stability_cv = \
                    calculate_roi_stability(roi_area_history)

            motion_pixels, motion_status = calculate_motion(
                cached_parsing, previous_parsing
            )

            # ── Draw ──────────────────────────────
            display = frame.copy()

            face_detected = (
                cached_combined_mask is not None
                and np.sum(cached_combined_mask) > 0
            )

            if face_detected:
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

            # ── HUD ───────────────────────────────
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

    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("Face detection stopped")


# ─────────────────────────────────────────
if __name__ == "__main__":
    run_face_detection()