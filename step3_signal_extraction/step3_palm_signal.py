import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys

# ─────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
PALM_DIR     = os.path.join(
    PROJECT_ROOT, "step2_palm_ROI_extraction"
)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, PALM_DIR)

# ── NO initialize_camera import ───────────
# ── NO __main__ block ─────────────────────
# Camera is opened ONLY by step4_normalization.py
# This file receives cap as a parameter always

from step3_signal_extraction.step3_rgb_signal import (
    run_signal_extraction
)
from step2_palm_ROI_extraction.step2_palm_detectionV5 import (
    mp_hands,
    THENAR_LANDMARKS,
    CENTRAL_PALM_LANDMARKS,
    HYPOTHENAR_LANDMARKS,
    EVAL_EVERY_N_FRAMES,
    get_landmark_pixels,
    get_combined_mask,
    get_roi_area,
    is_palm_facing_camera,
    estimate_skin_tone,
    assess_lighting,
    calculate_roi_stability
)


# ─────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────

class PalmROIState:
    """
    Shared state between Step 2 palm and Step 3.

    Step 2 writes mask and ITA after setup phase.
    Step 3 reads them every frame during recording.

    Fields:
        combined_mask   — all three palm regions combined
        thenar_mask     — thumb side region
        central_mask    — middle palm region
        hypothenar_mask — pinky side region
        ita             — skin tone value from Step 2
        palm_visible    — True if palm was detected
    """
    def __init__(self):
        self.combined_mask   = None
        self.thenar_mask     = None
        self.central_mask    = None
        self.hypothenar_mask = None
        self.ita             = 55.0
        self.palm_visible    = False


# ─────────────────────────────────────────
# Step 2 — Palm ROI setup
# ─────────────────────────────────────────

def run_palm_roi_extraction(cap, actual_fps,
                             state, duration_sec=5):
    """
    Run MediaPipe palm ROI extraction for setup phase.

    Receives cap from Step 1 — does NOT open camera.
    Runs MediaPipe Hands for duration_sec seconds to
    establish stable thenar, central, hypothenar masks.

    Writes result to state object so Step 3 can read it.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — PalmROIState shared object
        duration_sec — how long to run setup (seconds)
    """
    frame_counter = 0
    cached_ita    = 0.0

    print(f"Step 2 Palm: Establishing palm ROI "
          f"({duration_sec}s)...")
    print("Show your PALM to the camera")

    end_time = time.time() + duration_sec

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            frame_h, frame_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results      = hands.process(frame_rgb)
            palm_visible = False

            if results.multi_hand_landmarks:
                landmarks  = results.multi_hand_landmarks[0].landmark
                handedness = results.multi_handedness[0]

                palm_facing, _ = is_palm_facing_camera(
                    landmarks, handedness
                )

                if palm_facing:
                    palm_visible = True

                    # Build landmark point lists
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

                    # Build individual region masks
                    thenar_mask     = np.zeros(
                        (frame_h, frame_w), dtype=np.uint8
                    )
                    central_mask    = np.zeros(
                        (frame_h, frame_w), dtype=np.uint8
                    )
                    hypothenar_mask = np.zeros(
                        (frame_h, frame_w), dtype=np.uint8
                    )

                    for pts, msk in [
                        (thenar_pts,     thenar_mask),
                        (central_pts,    central_mask),
                        (hypothenar_pts, hypothenar_mask)
                    ]:
                        if len(pts) >= 3:
                            cv2.fillPoly(
                                msk,
                                [np.array(pts,
                                          dtype=np.int32)],
                                1
                            )

                    # Combined mask — all three regions
                    combined_mask = get_combined_mask(
                        [thenar_pts, central_pts,
                         hypothenar_pts],
                        frame_h, frame_w
                    )

                    # Update skin tone every N frames
                    if frame_counter % EVAL_EVERY_N_FRAMES == 0:
                        _, cached_ita = estimate_skin_tone(
                            frame, combined_mask
                        )

                    # Write to shared state
                    state.combined_mask   = combined_mask
                    state.thenar_mask     = thenar_mask
                    state.central_mask    = central_mask
                    state.hypothenar_mask = hypothenar_mask
                    state.ita             = cached_ita
                    state.palm_visible    = True

            # Preview display
            display = frame.copy()
            for msk, color in [
                (state.thenar_mask,     (0, 255, 0)),
                (state.central_mask,    (0, 255, 255)),
                (state.hypothenar_mask, (255, 0, 0))
            ]:
                if msk is not None and np.sum(msk) > 0:
                    colored         = np.zeros_like(display)
                    colored[msk==1] = color
                    cv2.addWeighted(
                        colored, 0.4, display,
                        0.6, 0, display
                    )

            remaining = max(0, end_time - time.time())
            cv2.putText(
                display,
                f"Palm ROI setup... {remaining:.1f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if palm_visible
                else (0, 0, 255),
                2
            )
            cv2.putText(
                display,
                f"ITA: {cached_ita:.1f}  "
                f"{'Palm visible' if palm_visible else 'Show palm'}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2
            )
            cv2.putText(
                display,
                "GREEN=Thenar  CYAN=Central  BLUE=Hypothenar",
                (10, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1
            )
            cv2.imshow("Step 2 — Palm ROI", display)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    print(
        f"Step 2 Palm complete — "
        f"ITA: {state.ita:.1f}  "
        f"Palm visible: {state.palm_visible}  "
        f"Mask pixels: "
        f"{np.sum(state.combined_mask) if state.combined_mask is not None else 0}"
    )


# ─────────────────────────────────────────
# Step 3 — RGB signal extraction (palm)
# ─────────────────────────────────────────

def run_palm_signal_extraction(cap, actual_fps, state):
    """
    Run Step 3 RGB signal extraction using palm mask.

    Receives cap from Step 1 and state from Step 2.
    Does NOT open camera.

    The mask from state is locked — MediaPipe already
    established it during setup phase. This function
    just reads pixels from the locked mask for 30s.

    Input:
        cap        — VideoCapture from Step 1
        actual_fps — fps from Step 1
        state      — PalmROIState with mask and ITA

    Output:
        r, g, b        — raw signal arrays (N,)
        fps_measured   — actual fps during recording
        quality_ok     — bool
        issues         — list of problem strings
        thresh_summary — adaptive threshold stats dict
    """
    def get_mask_fn(frame_rgb):
        return (
            state.combined_mask,
            state.thenar_mask,
            state.central_mask
        )

    def get_ita_fn():
        return state.ita

    print("\nStep 3: Extracting RGB signal from palm...")
    return run_signal_extraction(
        cap, actual_fps,
        get_mask_fn, get_ita_fn,
        modality="palm"
    )