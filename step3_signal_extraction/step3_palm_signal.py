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
from subject_profile import SubjectProfile


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
# Step 2b — Palm ROI setup + Calibration
# ─────────────────────────────────────────

def run_palm_roi_extraction(cap, actual_fps,
                             state, duration_sec=10):
    """
    Run MediaPipe palm ROI extraction + calibration.

    Two phases:
        Phase 1 (0 – 5s):  Establish stable palm mask
                            via MediaPipe Hands.
        Phase 2 (5 – 10s): Sample green channel values
                            through locked mask →
                            build SubjectProfile.

    Returns a SubjectProfile calibrated to this patient's
    palm signal characteristics, NOT the face profile.
    Palm baseline_g_std is typically higher (less melanin),
    so motion_threshold, amplitude_target, and routing
    thresholds all need to be re-anchored to the palm.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — PalmROIState shared object
        duration_sec — total duration (default 10s)

    Output:
        palm_profile — SubjectProfile built from palm signal
    """
    mask_phase_sec  = duration_sec / 2.0   # 5s
    calib_phase_sec = duration_sec / 2.0   # 5s

    frame_counter = 0
    cached_ita    = 0.0

    # ── Phase 1: Mask establishment ───────────────
    print(f"\nStep 2b Palm — Phase 1: Establishing palm mask "
          f"({mask_phase_sec:.0f}s)...")
    print("Show your PALM flat to the camera")

    phase1_end = time.time() + mask_phase_sec

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while time.time() < phase1_end:
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
                                [np.array(pts, dtype=np.int32)],
                                1
                            )

                    combined_mask = get_combined_mask(
                        [thenar_pts, central_pts,
                         hypothenar_pts],
                        frame_h, frame_w
                    )

                    if frame_counter % EVAL_EVERY_N_FRAMES == 0:
                        _, cached_ita = estimate_skin_tone(
                            frame, combined_mask
                        )

                    state.combined_mask   = combined_mask
                    state.thenar_mask     = thenar_mask
                    state.central_mask    = central_mask
                    state.hypothenar_mask = hypothenar_mask
                    state.ita             = cached_ita
                    state.palm_visible    = True

            # Phase 1 preview
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

            remaining = max(0, phase1_end - time.time())
            cv2.putText(
                display,
                f"Phase 1 — Palm mask setup... {remaining:.1f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if palm_visible else (0, 0, 255),
                2
            )
            cv2.putText(
                display,
                f"ITA: {cached_ita:.1f}  "
                f"{'Palm detected' if palm_visible else 'Show palm to camera'}",
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
            cv2.imshow("Step 2b — Palm ROI", display)
            cv2.waitKey(1)

    # ── Phase 2: Calibration sampling ─────────────
    # Collect green channel values through locked mask.
    # Mirror of face calibration in step3_face_signal_bisenet.py.
    # The palm's baseline_g_std is higher than face (less melanin),
    # so the SubjectProfile built here will have:
    #   - higher motion_threshold (5× palm baseline, not face)
    #   - higher amplitude_target (anchored to palm signal strength)
    #   - bandpass hint from palm HR estimate
    #   - ITA from palm (lighter than face regardless of skin tone)

    print(f"\nStep 2b Palm — Phase 2: Calibration sampling "
          f"({calib_phase_sec:.0f}s)...")
    print("Hold still — measuring your palm signal baseline")

    g_calib_samples = []
    phase2_end      = time.time() + calib_phase_sec

    if state.combined_mask is None:
        print("  WARNING: No palm mask established — "
              "calibration will use defaults")
        # Return a default profile so the pipeline doesn't crash
        profile = SubjectProfile()
        profile.ita         = state.ita
        profile.fitzpatrick = _ita_to_fitzpatrick(state.ita)
        cv2.destroyAllWindows()
        return profile

    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while time.time() < phase2_end:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            # Keep updating mask if palm still visible
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                landmarks  = results.multi_hand_landmarks[0].landmark
                handedness = results.multi_handedness[0]
                palm_facing, _ = is_palm_facing_camera(
                    landmarks, handedness
                )
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
                        [thenar_pts, central_pts,
                         hypothenar_pts],
                        frame_h, frame_w
                    )
                    if np.sum(combined_mask) > 100:
                        state.combined_mask = combined_mask

            # Sample green channel through locked mask
            mask = state.combined_mask
            if mask is not None and np.sum(mask) > 100:
                green_channel = frame[:, :, 1].astype(np.float32)
                mean_g = float(np.mean(
                    green_channel[mask == 1]
                ))
                g_calib_samples.append(mean_g)

            # Phase 2 preview
            display   = frame.copy()
            remaining = max(0, phase2_end - time.time())
            n_samples = len(g_calib_samples)

            cv2.putText(
                display,
                f"Phase 2 — Calibrating... {remaining:.1f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2
            )
            cv2.putText(
                display,
                f"Samples: {n_samples}  "
                f"Hold still",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2
            )
            cv2.imshow("Step 2b — Palm ROI", display)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    # ── Build SubjectProfile from palm calibration ─
    g_arr = np.array(g_calib_samples, dtype=np.float32)

    palm_profile = SubjectProfile()
    palm_profile.build_from_calibration(
        g_samples = g_arr,
        r_samples = g_arr,   # palm: only green collected, r/b approximate
        b_samples = g_arr,
        ita_value = state.ita,
        fps       = actual_fps
    )

    mask_pixels = int(np.sum(state.combined_mask)) \
        if state.combined_mask is not None else 0

    print(
        f"\nStep 2b Palm complete —"
        f"\n  ITA:              {state.ita:.1f}  "
        f"({palm_profile.fitzpatrick})"
        f"\n  Calibration samples: {len(g_calib_samples)}"
        f"\n  Palm baseline std:   "
        f"{palm_profile.baseline_g_std:.4f}"
        f"\n  Motion threshold:    "
        f"{palm_profile.motion_threshold:.4f}"
        f"\n  Amplitude target:    "
        f"{palm_profile.amplitude_target:.6f}"
        f"\n  HR estimate:         "
        f"{palm_profile.hr_estimate_bpm} BPM"
        f"\n  Mask pixels:         {mask_pixels}"
        f"\n  Profile valid:       {palm_profile.is_valid}"
    )

    return palm_profile


# ─────────────────────────────────────────
# Step 3b — RGB signal extraction (palm)
# ─────────────────────────────────────────

def run_palm_signal_extraction(cap, actual_fps, state,
                                duration_sec=35,
                                profile=None):
    """
    Run Step 3 RGB signal extraction using palm mask.

    Receives cap from Step 1 and state from Step 2b.
    Does NOT open camera.

    The mask from state is locked after setup phase.
    This function reads pixels from the locked mask
    for duration_sec seconds.

    Motion rejection uses palm_profile.motion_threshold —
    calibrated to the palm's own baseline, not the face.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — PalmROIState with mask and ITA
        duration_sec — recording duration (default 35s)
        profile      — SubjectProfile from Step 2b palm

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

    print("\nStep 3b: Extracting RGB signal from palm "
          f"({duration_sec}s)...")

    return run_signal_extraction(
        cap, actual_fps,
        get_mask_fn, get_ita_fn,
        modality    = "palm",
        duration_sec = duration_sec,
        profile      = profile
    )


# ─────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────

def _ita_to_fitzpatrick(ita):
    """Map ITA value to Fitzpatrick type string."""
    if   ita > 55:  return "FST I"
    elif ita > 41:  return "FST II"
    elif ita > 28:  return "FST III"
    elif ita > 10:  return "FST IV"
    elif ita > -30: return "FST V"
    else:           return "FST VI"