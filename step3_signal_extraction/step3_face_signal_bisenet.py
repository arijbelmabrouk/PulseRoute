import cv2
import numpy as np
import torch
import time
import os
import sys

# ─────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FACE_DIR     = os.path.join(
    PROJECT_ROOT,
    "step2_face_ROI_extraction",
    "face_parsing_mask"
)
REPO_DIR = os.path.join(FACE_DIR, "face-parsing.PyTorch")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FACE_DIR)
sys.path.insert(0, REPO_DIR)

# ── NO initialize_camera import ───────────
# ── NO __main__ block ─────────────────────
# Camera is opened ONLY by step4_normalization.py
# This file receives cap as a parameter always

from step3_signal_extraction.step3_rgb_signal import (
    run_signal_extraction
)


# ─────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────

class FaceROIState:
    """
    Shared state between Step 2 BiSeNet and Step 3.

    Step 2 writes mask and ITA after setup phase.
    Step 3 reads them every frame during recording.

    Fields:
        combined_mask — forehead + cheeks binary mask
        forehead_mask — forehead only binary mask
        cheek_mask    — cheeks only binary mask
        ita           — skin tone value from Step 2
    """
    def __init__(self):
        self.combined_mask = None
        self.forehead_mask = None
        self.cheek_mask    = None
        self.ita           = 55.0  # default light skin


# ─────────────────────────────────────────
# Step 2 — BiSeNet face ROI setup
# ─────────────────────────────────────────

def run_face_roi_extraction(cap, actual_fps,
                            state, duration_sec=5):
    """
    Run BiSeNet face ROI extraction for setup phase.

    Receives cap from Step 1 — does NOT open camera.
    Runs BiSeNet for duration_sec seconds to establish
    stable forehead and cheek mask.

    Writes result to state object so Step 3 can read it.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — FaceROIState shared object
        duration_sec — how long to run setup (seconds)
    """
    from step2_face_ROI_extraction.face_parsing_mask\
        .step2_face_detection_bisenetV6 import (
        load_bisenet,
        parse_face,
        build_face_roi_masks,
        should_update_mask,
        estimate_skin_tone,
        EVAL_EVERY_N_FRAMES,
        MODEL_PATH
    )

    device = torch.device("cpu")
    model  = load_bisenet(MODEL_PATH, device)

    frame_counter        = 0
    cached_parsing       = None
    cached_combined_mask = None

    print(f"Step 2 BiSeNet: Establishing face ROI "
          f"({duration_sec}s)...")

    end_time = time.time() + duration_sec

    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        frame_h, frame_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        run_parsing = (
            frame_counter % EVAL_EVERY_N_FRAMES == 0
            or cached_parsing is None
        )

        if run_parsing:
            parsing = parse_face(model, frame_rgb, device)

            new_fore, new_cheek, new_combined = \
                build_face_roi_masks(
                    parsing, frame_h, frame_w
                )

            if should_update_mask(
                cached_combined_mask, new_combined
            ):
                cached_parsing       = parsing
                cached_combined_mask = new_combined

                # Write to shared state
                state.combined_mask = new_combined
                state.forehead_mask = new_fore
                state.cheek_mask    = new_cheek

                _, ita = estimate_skin_tone(
                    frame, new_combined
                )
                state.ita = ita

        # Preview display
        display = frame.copy()
        for msk, color in [
            (state.forehead_mask, (0, 255, 0)),
            (state.cheek_mask,    (255, 0, 0))
        ]:
            if msk is not None and np.sum(msk) > 0:
                colored         = np.zeros_like(display)
                colored[msk==1] = color
                cv2.addWeighted(
                    colored, 0.4, display, 0.6, 0, display
                )

        remaining = max(0, end_time - time.time())
        cv2.putText(
            display,
            f"BiSeNet ROI setup... {remaining:.1f}s",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )
        cv2.putText(
            display,
            f"ITA: {state.ita:.1f}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 2
        )
        cv2.imshow("Step 2 — Face ROI (BiSeNet)", display)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    print(
        f"Step 2 BiSeNet complete — "
        f"ITA: {state.ita:.1f}  "
        f"Mask pixels: "
        f"{np.sum(state.combined_mask) if state.combined_mask is not None else 0}"
    )


# ─────────────────────────────────────────
# Step 3 — RGB signal extraction (face)
# ─────────────────────────────────────────

def run_face_signal_extraction(cap, actual_fps, state):
    """
    Run Step 3 RGB signal extraction using face mask.

    Receives cap from Step 1 and state from Step 2.
    Does NOT open camera.

    The mask from state is locked — BiSeNet already
    established it during setup phase. This function
    just reads pixels from the locked mask for 30s.

    Input:
        cap        — VideoCapture from Step 1
        actual_fps — fps from Step 1
        state      — FaceROIState with mask and ITA

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
            state.forehead_mask,
            state.cheek_mask
        )

    def get_ita_fn():
        return state.ita

    print("\nStep 3: Extracting RGB signal from face...")
    return run_signal_extraction(
        cap, actual_fps,
        get_mask_fn, get_ita_fn,
        modality="face"
    )