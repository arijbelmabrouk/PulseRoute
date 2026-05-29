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

from step3_signal_extraction.step3_rgb_signal import (
    run_signal_extraction,
    extract_rgb_signal,
    MIN_ROI_PIXELS
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
# Extended to 10 seconds:
#   First 5s  → mask establishment (unchanged)
#   Next  5s  → calibration pixel sampling
#               → builds SubjectProfile
# ─────────────────────────────────────────

def run_face_roi_extraction(cap, actual_fps,
                            state, duration_sec=10):
    """
    Run BiSeNet face ROI extraction for setup phase.

    Extended from 5s to 10s:
        Phase 1 (0-5s): BiSeNet establishes stable
            forehead + cheek mask. Identical to
            original behavior.
        Phase 2 (5-10s): Pixel values are sampled
            through the locked mask to build the
            patient's personal SubjectProfile.
            Motion threshold, amplitude target, and
            HR estimate are all derived from these
            5 seconds of the patient's own signal.

    The SubjectProfile means every downstream
    threshold is anchored to THIS patient — not a
    population average. Dark skin, home lighting,
    and weak webcams all get fair thresholds.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — FaceROIState shared object
        duration_sec — total setup duration (default 10s)
                       Phase 1: first 5s
                       Phase 2: remaining seconds

    Returns:
        profile — SubjectProfile built from calibration
                  (also accessible via state for compat)
    """
    # Import here to keep path setup clean
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

    # Import SubjectProfile from project root
    sys.path.insert(0, PROJECT_ROOT)
    from subject_profile import SubjectProfile

    device = torch.device("cpu")
    model  = load_bisenet(MODEL_PATH, device)

    frame_counter        = 0
    cached_parsing       = None
    cached_combined_mask = None

    # ── Phase split ───────────────────────────────
    phase1_duration = min(5, duration_sec // 2)
    phase2_duration = duration_sec - phase1_duration

    # Calibration sample accumulators (Phase 2)
    calib_g = []
    calib_r = []
    calib_b = []

    total_start = time.time()
    phase1_end  = total_start + phase1_duration
    phase2_end  = total_start + duration_sec

    print(f"Step 2: Phase 1 — mask setup "
          f"({phase1_duration}s)...")

    # ── Phase 1: mask establishment ───────────────
    while time.time() < phase1_end:
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
                state.combined_mask  = new_combined
                state.forehead_mask  = new_fore
                state.cheek_mask     = new_cheek

                _, ita = estimate_skin_tone(
                    frame, new_combined
                )
                state.ita = ita

        _draw_setup_hud(frame, state,
                        phase1_end - time.time(),
                        "Phase 1: Establishing mask",
                        (0, 255, 255))

        cv2.imshow("Step 2 — Face ROI (BiSeNet)", frame)
        cv2.waitKey(1)

    print(f"Step 2: Phase 2 — calibration sampling "
          f"({phase2_duration}s)...")

    # ── Phase 2: calibration pixel sampling ───────
    # Mask is now locked. Sample pixel values to
    # build the patient's personal SubjectProfile.
    while time.time() < phase2_end:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        if state.combined_mask is None or \
           np.sum(state.combined_mask) < MIN_ROI_PIXELS:
            continue

        # Extract pixel means through the locked mask
        mean_r, mean_g, mean_b, \
        pixel_count, quality_ratio, _ = \
            extract_rgb_signal(
                frame_rgb,
                state.combined_mask,
                state.ita
            )

        if mean_g > 0 and pixel_count >= MIN_ROI_PIXELS:
            calib_g.append(mean_g)
            calib_r.append(mean_r)
            calib_b.append(mean_b)

        remaining_total = phase2_end - time.time()
        _draw_setup_hud(frame, state,
                        remaining_total,
                        f"Phase 2: Calibrating "
                        f"({len(calib_g)} frames)",
                        (0, 200, 255))

        cv2.imshow("Step 2 — Face ROI (BiSeNet)", frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # ── Build SubjectProfile from calibration data ─
    profile = SubjectProfile()
    profile.build_from_calibration(
        calib_g, calib_r, calib_b,
        ita_value=state.ita,
        fps=actual_fps
    )
    profile.print_profile()

    print(
        f"Step 2 complete — "
        f"ITA: {state.ita:.1f}  "
        f"({profile.fitzpatrick})  "
        f"Mask pixels: "
        f"{np.sum(state.combined_mask) if state.combined_mask is not None else 0}  "
        f"Calibration frames: {len(calib_g)}"
    )

    return profile


def _draw_setup_hud(frame, state,
                    remaining, label, color):
    """Draw setup phase HUD on frame (in-place)."""
    for msk, msk_color in [
        (state.forehead_mask, (0, 255, 0)),
        (state.cheek_mask,    (255, 0, 0))
    ]:
        if msk is not None and np.sum(msk) > 0:
            colored         = np.zeros_like(frame)
            colored[msk==1] = msk_color
            cv2.addWeighted(
                colored, 0.4, frame, 0.6, 0, frame
            )

    cv2.putText(
        frame, label,
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, color, 2
    )
    cv2.putText(
        frame,
        f"Remaining: {max(0, remaining):.1f}s",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, color, 2
    )
    cv2.putText(
        frame,
        f"ITA: {state.ita:.1f}",
        (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 255), 2
    )
    cv2.putText(
        frame,
        "Stay still — calibrating your signal",
        (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (200, 200, 200), 1
    )


# ─────────────────────────────────────────
# Step 3 — RGB signal extraction (face)
# ─────────────────────────────────────────

def run_face_signal_extraction(cap, actual_fps,
                               state,
                               duration_sec=30,
                               profile=None):
    """
    Run Step 3 RGB signal extraction using face mask.

    Now accepts optional SubjectProfile from Step 2.
    If profile is provided, motion rejection uses the
    patient's personal threshold. If not, falls back
    to the original behavior.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — FaceROIState with mask + ITA
        duration_sec — recording length in seconds
        profile      — SubjectProfile from Step 2
                       (None → no motion rejection)

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
        modality="face",
        duration_sec=duration_sec,
        profile=profile
    )