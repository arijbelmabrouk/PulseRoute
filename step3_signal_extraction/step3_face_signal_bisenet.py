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

# Maximum calibration attempts before giving up
MAX_CALIB_ATTEMPTS = 3


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
        self.ita           = 55.0


# ─────────────────────────────────────────
# Step 2 — BiSeNet face ROI setup
# ─────────────────────────────────────────

def run_face_roi_extraction(cap, actual_fps,
                             state, duration_sec=10):
    """
    Run BiSeNet face ROI extraction + calibration.

    Two-phase structure:
        Phase 1 (first half): BiSeNet establishes
            stable forehead + cheek mask.
        Phase 2 (second half): Pixel values sampled
            through locked mask → SubjectProfile built.

    NEW — Calibration quality gate + retry loop:
        After build_from_calibration, profile.validate()
        checks whether the calibration data is
        trustworthy. If not, the patient is shown
        a specific error message and asked to fix the
        problem (reposition, improve lighting, etc).
        The calibration phase retries up to
        MAX_CALIB_ATTEMPTS times before giving up
        and using population defaults.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — FaceROIState shared object
        duration_sec — total setup duration (default 10s)

    Returns:
        profile — SubjectProfile (valid or fallback)
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

    sys.path.insert(0, PROJECT_ROOT)
    from subject_profile import SubjectProfile

    device = torch.device("cpu")
    model  = load_bisenet(MODEL_PATH, device)

    phase1_duration = min(5, duration_sec // 2)
    phase2_duration = duration_sec - phase1_duration

    profile          = None
    attempt          = 0
    mask_established = False

    while attempt < MAX_CALIB_ATTEMPTS:
        attempt += 1

        # ── Phase 1 — mask setup (first attempt only) ─
        if not mask_established:
            print(f"\nStep 2 Phase 1 — Establishing face mask "
                  f"({phase1_duration}s)...")

            frame_counter        = 0
            cached_parsing       = None
            cached_combined_mask = None
            phase1_end           = time.time() + phase1_duration

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
                    parsing = parse_face(
                        model, frame_rgb, device
                    )
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

                _draw_setup_hud(
                    frame, state,
                    phase1_end - time.time(),
                    "Phase 1: Establishing mask",
                    (0, 255, 255)
                )
                cv2.imshow(
                    "Step 2 — Face ROI (BiSeNet)", frame
                )
                cv2.waitKey(1)

            mask_established = True

        # ── Phase 2 — calibration sampling ────────────
        if attempt == 1:
            print(f"\nStep 2 Phase 2 — Calibration sampling "
                  f"({phase2_duration}s)...")
        else:
            print(f"\nStep 2 Phase 2 — Retry {attempt}/"
                  f"{MAX_CALIB_ATTEMPTS} "
                  f"({phase2_duration}s)...")

        calib_g    = []
        calib_r    = []
        calib_b    = []
        phase2_end = time.time() + phase2_duration

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

            mean_r, mean_g, mean_b, \
            pixel_count, quality_ratio, _ = \
                extract_rgb_signal(
                    frame_rgb,
                    state.combined_mask,
                    state.ita
                )

            if mean_g > 0 and \
               pixel_count >= MIN_ROI_PIXELS:
                calib_g.append(mean_g)
                calib_r.append(mean_r)
                calib_b.append(mean_b)

            _draw_setup_hud(
                frame, state,
                phase2_end - time.time(),
                f"Phase 2: Calibrating "
                f"({len(calib_g)} frames)",
                (0, 200, 255)
            )
            cv2.imshow(
                "Step 2 — Face ROI (BiSeNet)", frame
            )
            cv2.waitKey(1)

        # ── Build profile ──────────────────────────
        profile = SubjectProfile()
        profile.build_from_calibration(
            calib_g, calib_r, calib_b,
            ita_value  = state.ita,
            fps        = actual_fps
        )

        # ── Validate ───────────────────────────────
        mask_pixels = int(np.sum(state.combined_mask)) \
                      if state.combined_mask is not None \
                      else 0

        valid, reason = profile.validate(
            mask_pixels=mask_pixels
        )

        if valid:
            break

        print(f"\n  ⚠ Calibration quality check FAILED:")
        print(f"  {reason}")

        if attempt < MAX_CALIB_ATTEMPTS:
            _show_retry_screen(
                cap, reason, attempt, MAX_CALIB_ATTEMPTS
            )
        else:
            print(f"\n  ⚠ All {MAX_CALIB_ATTEMPTS} calibration "
                  f"attempts failed.")
            print(f"  Using population defaults. "
                  f"Results may be less accurate.")
            profile._set_population_defaults()
            profile.ita         = state.ita
            profile.fitzpatrick = profile._ita_to_fitzpatrick(
                state.ita
            )

    cv2.destroyAllWindows()

    profile.print_profile()

    mask_pixels = int(np.sum(state.combined_mask)) \
                  if state.combined_mask is not None else 0
    print(
        f"Step 2 complete — "
        f"ITA: {state.ita:.1f}  "
        f"({profile.fitzpatrick})  "
        f"Mask pixels: {mask_pixels}  "
        f"Valid: {profile.is_valid}"
    )

    return profile


def _show_retry_screen(cap, reason,
                        attempt, max_attempts):
    deadline = time.time() + 4.0

    if "moved" in reason.lower() or \
       "noisy" in reason.lower() or \
       "occluded" in reason.lower():
        instruction = "Please stay STILL during calibration"
    elif "dark" in reason.lower() or \
         "underlit" in reason.lower():
        instruction = "Please improve lighting on your face"
    elif "overexposed" in reason.lower() or \
         "bright" in reason.lower():
        instruction = "Please reduce direct light on your face"
    elif "closer" in reason.lower() or \
         "small" in reason.lower():
        instruction = "Please move CLOSER to the camera"
    elif "covered" in reason.lower() or \
         "flat" in reason.lower():
        instruction = "Please ensure your face is visible"
    else:
        instruction = "Please reposition and try again"

    print(f"  → {instruction}")
    print(f"  Retrying in 4 seconds...")

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret:
            break

        overlay        = frame.copy()
        overlay[:]     = (20, 20, 20)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        remaining = max(0, deadline - time.time())
        h, w      = frame.shape[:2]

        cv2.putText(
            frame,
            f"Calibration attempt {attempt}/{max_attempts} failed",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 80, 255), 2
        )

        words   = reason.split()
        lines   = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 > 55:
                lines.append(current)
                current = word
            else:
                current = (current + " " + word).strip()
        if current:
            lines.append(current)

        y = 95
        for line in lines[:3]:
            cv2.putText(
                frame, line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (180, 180, 255), 1
            )
            y += 28

        cv2.putText(
            frame,
            f"→  {instruction}",
            (20, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 150), 2
        )
        cv2.putText(
            frame,
            f"Retrying in {remaining:.0f}s...",
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (200, 200, 200), 1
        )

        cv2.imshow("Step 2 — Face ROI (BiSeNet)", frame)
        cv2.waitKey(1)


def _draw_setup_hud(frame, state,
                     remaining, label, color):
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
                                profile=None,
                                on_frame=None):   # ← NEW
    """
    Run Step 3 RGB signal extraction using face mask.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — FaceROIState with mask + ITA
        duration_sec — recording length in seconds
        profile      — SubjectProfile from Step 2
        on_frame     — optional callback(frame_bgr, mask)
                       called on every accepted frame
                       for live camera feed publishing

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
        modality     = "face",
        duration_sec = duration_sec,
        profile      = profile,
        on_frame     = on_frame        # ← NEW
    )