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
    is_palm_facing_camera,
    estimate_skin_tone,
)
from subject_profile import SubjectProfile


# ─────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────

class PalmROIState:
    """
    Shared state between Step 2 palm and Step 3.

    Step 2 writes mask and ITA after setup phase.
    Step 3 reads — and now also UPDATES — them
    every frame during recording via live MediaPipe.

    Fields:
        combined_mask   — all three palm regions combined
        thenar_mask     — thumb side region
        central_mask    — middle palm region
        hypothenar_mask — pinky side region
        ita             — skin tone value
        palm_visible    — True if palm detected this frame
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
                             state, duration_sec=10,
                             show_display=True,
                             on_frame=None):
    """
    Run MediaPipe palm ROI extraction + calibration.

    Two phases:
        Phase 1 (first half): Establish stable palm mask.
        Phase 2 (second half): Sample green channel values
            through mask → build SubjectProfile.

    NEW — on_frame callback:
        Optional callback(frame_bgr, combined_mask)
        called every frame with the annotated setup frame.
        Used by run_web.py to stream Step 2b live to the
        patient page — same visuals as the OpenCV window.

        The callback receives the frame AFTER _draw_phase_hud
        has drawn all overlays, so what the patient sees is
        identical to what the OpenCV window shows.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — PalmROIState shared object
        duration_sec — total duration (default 10s)
        show_display — whether to call cv2.imshow
        on_frame     — optional callback(frame_bgr, mask)

    Output:
        palm_profile — SubjectProfile built from palm signal
    """
    mask_phase_sec  = duration_sec / 2.0
    calib_phase_sec = duration_sec / 2.0

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
                    _update_state_from_landmarks(
                        state, landmarks,
                        frame_w, frame_h
                    )
                    if frame_counter % EVAL_EVERY_N_FRAMES == 0 \
                       and state.combined_mask is not None:
                        _, cached_ita = estimate_skin_tone(
                            frame, state.combined_mask
                        )
                        state.ita = cached_ita

            remaining = phase1_end - time.time()
            _draw_phase_hud(
                frame, state, palm_visible, cached_ita,
                max(0, remaining),
                "Phase 1 — Palm mask setup"
            )

            # ── Live feed callback ─────────────────
            if on_frame is not None:
                try:
                    on_frame(frame, state.combined_mask)
                except Exception:
                    pass

            if show_display:
                cv2.imshow("Step 2b — Palm ROI", frame)
                cv2.waitKey(1)

    # ── Phase 2: Calibration sampling ─────────────
    print(f"\nStep 2b Palm — Phase 2: Calibration sampling "
          f"({calib_phase_sec:.0f}s)...")
    print("Hold still — measuring your palm signal baseline")

    g_calib_samples = []
    phase2_end      = time.time() + calib_phase_sec

    if state.combined_mask is None:
        print("  WARNING: No palm mask — using defaults")
        profile             = SubjectProfile()
        profile.ita         = state.ita
        profile.fitzpatrick = _ita_to_fitzpatrick(state.ita)
        if show_display:
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

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                landmarks  = results.multi_hand_landmarks[0].landmark
                handedness = results.multi_handedness[0]
                palm_facing, _ = is_palm_facing_camera(
                    landmarks, handedness
                )
                if palm_facing:
                    _update_state_from_landmarks(
                        state, landmarks, frame_w, frame_h
                    )

            mask = state.combined_mask
            if mask is not None and np.sum(mask) > 100:
                green_channel = frame[:, :, 1].astype(np.float32)
                mean_g = float(np.mean(
                    green_channel[mask == 1]
                ))
                g_calib_samples.append(mean_g)

            remaining = phase2_end - time.time()
            _draw_phase_hud(
                frame, state, True, state.ita,
                max(0, remaining),
                f"Phase 2 — Calibrating "
                f"({len(g_calib_samples)} samples)"
            )

            # ── Live feed callback ─────────────────
            if on_frame is not None:
                try:
                    on_frame(frame, state.combined_mask)
                except Exception:
                    pass

            if show_display:
                cv2.imshow("Step 2b — Palm ROI", frame)
                cv2.waitKey(1)

    if show_display:
        cv2.destroyAllWindows()

    # ── Build SubjectProfile ───────────────────────
    g_arr        = np.array(g_calib_samples, dtype=np.float32)
    palm_profile = SubjectProfile()
    palm_profile.build_from_calibration(
        g_samples = g_arr,
        r_samples = g_arr,
        b_samples = g_arr,
        ita_value = state.ita,
        fps       = actual_fps
    )

    # ── Validate calibration ───────────────────────
    mask_pixels = int(np.sum(state.combined_mask)) \
                  if state.combined_mask is not None else 0
    valid, reason = palm_profile.validate(
        mask_pixels=mask_pixels
    )
    if not valid:
        print(f"\n  ⚠ Palm calibration validation failed:")
        print(f"  {reason}")
        print(f"  Using population defaults.")
        palm_profile._set_population_defaults()
        palm_profile.ita         = state.ita
        palm_profile.fitzpatrick = _ita_to_fitzpatrick(state.ita)

    print(
        f"\nStep 2b Palm complete —"
        f"\n  ITA:                 {state.ita:.1f}  "
        f"({palm_profile.fitzpatrick})"
        f"\n  Calibration samples: {len(g_calib_samples)}"
        f"\n  Profile valid:       {palm_profile.is_valid}"
    )
    if palm_profile.baseline_g_std is not None:
        print(
            f"  Baseline std:        "
            f"{palm_profile.baseline_g_std:.4f}"
            f"\n  Motion threshold:    "
            f"{palm_profile.motion_threshold:.4f}"
            f"\n  Amplitude target:    "
            f"{palm_profile.amplitude_target:.6f}"
        )

    return palm_profile


# ─────────────────────────────────────────
# Step 3b — RGB signal extraction (palm)
# with live mask refresh every frame
# ─────────────────────────────────────────

def run_palm_signal_extraction(cap, actual_fps, state,
                                duration_sec=35,
                                profile=None,
                                on_frame=None,
                                on_progress=None,
                                show_display=True):
    """
    Run Step 3 RGB signal extraction using palm mask.

    KEY CHANGE — live mask refresh every frame:
        The original implementation locked the palm mask
        at setup time. Any hand movement during the 35s
        recording would point the mask at wrong pixels.

        This version opens a MediaPipe Hands session that
        runs alongside the signal extraction. get_mask_fn
        is a closure that:
            1. Runs MediaPipe on the current frame
            2. Updates state.combined_mask if palm detected
            3. Falls back to the last known mask if not

        The patient can move their hand slightly and the
        mask follows. If the palm disappears completely,
        the last valid mask is held until it reappears.
        Motion rejection (from profile) handles frames
        where the palm moved too fast to track cleanly.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — PalmROIState with mask and ITA
        duration_sec — recording duration (default 35s)
        profile      — SubjectProfile from Step 2b palm
        on_frame     — optional callback(frame_bgr, mask)
        on_progress  — optional callback(percent: int)
        show_display — whether to call cv2.imshow

    Output:
        r, g, b        — raw signal arrays (N,)
        fps_measured   — actual fps during recording
        quality_ok     — bool
        issues         — list of problem strings
        thresh_summary — adaptive threshold stats dict
    """
    print(f"\nStep 3b: Extracting RGB signal from palm "
          f"({duration_sec}s) — live mask tracking active...")

    # Open a single MediaPipe session for the full recording.
    # The hands context stays open while run_signal_extraction
    # calls get_mask_fn on every frame.
    hands_context = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,       # complexity 0 = faster,
                                  # important at 30fps with
                                  # signal extraction overhead
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    hands_context.__enter__()

    frame_counter   = [0]   # mutable int in closure

    def get_mask_fn(frame_rgb):
        """
        Called by run_signal_extraction on every frame.

        Runs MediaPipe, updates state mask if palm found,
        falls back to last known mask if not.

        Returns (combined, thenar, central) tuple.
        """
        frame_counter[0] += 1
        # Only run MediaPipe every 3rd frame — reduces CPU load
        # while keeping mask tracking responsive enough
        if frame_counter[0] % 3 == 0:
            results = hands_context.process(frame_rgb)
        else:
            results = None

        if results is not None and results.multi_hand_landmarks:
            landmarks  = results.multi_hand_landmarks[0].landmark
            handedness = results.multi_handedness[0]

            palm_facing, _ = is_palm_facing_camera(
                landmarks, handedness
            )

            if palm_facing:
                h, w = frame_rgb.shape[:2]
                _update_state_from_landmarks(
                    state, landmarks, w, h
                )
                state.palm_visible = True

                # Update ITA every N frames — expensive
                if frame_counter[0] % EVAL_EVERY_N_FRAMES == 0:
                    frame_bgr = cv2.cvtColor(
                        frame_rgb, cv2.COLOR_RGB2BGR
                    )
                    if state.combined_mask is not None:
                        _, new_ita = estimate_skin_tone(
                            frame_bgr, state.combined_mask
                        )
                        state.ita = new_ita
            else:
                state.palm_visible = False
        else:
            state.palm_visible = False

        return (
            state.combined_mask,
            state.thenar_mask,
            state.central_mask
        )

    def get_ita_fn():
        return state.ita

    try:
        result = run_signal_extraction(
            cap, actual_fps,
            get_mask_fn, get_ita_fn,
            modality     = "palm",
            duration_sec = duration_sec,
            profile      = profile,
            on_frame     = on_frame,
            on_progress  = on_progress,
            show_display = show_display
        )
    finally:
        hands_context.__exit__(None, None, None)

    # FPS drop check
    measured_fps = result[3]  # fps_measured is index 3
    if measured_fps < actual_fps * 0.70:
        print(f"\n⚠ FPS drop detected during palm recording:")
        print(f"  Calibration fps: {actual_fps:.1f}")
        print(f"  Recording fps:   {measured_fps:.1f}")
        print(f"  Drop: {((actual_fps - measured_fps)/actual_fps)*100:.0f}%")
        print(f"  HR timing may be slightly less reliable.")
        print(f"  If this persists, close other applications.")

    return result


# ─────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────

def _update_state_from_landmarks(state, landmarks,
                                   frame_w, frame_h):
    """
    Rebuild all palm masks from current landmarks.
    Writes directly to state. Called every frame
    when palm is visible and facing camera.
    """
    thenar_pts = get_landmark_pixels(
        landmarks, THENAR_LANDMARKS, frame_w, frame_h
    )
    central_pts = get_landmark_pixels(
        landmarks, CENTRAL_PALM_LANDMARKS, frame_w, frame_h
    )
    hypothenar_pts = get_landmark_pixels(
        landmarks, HYPOTHENAR_LANDMARKS, frame_w, frame_h
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
        [thenar_pts, central_pts, hypothenar_pts],
        frame_h, frame_w
    )

    # Only update if the new mask is non-empty
    if np.sum(combined_mask) > 100:
        state.combined_mask   = combined_mask
        state.thenar_mask     = thenar_mask
        state.central_mask    = central_mask
        state.hypothenar_mask = hypothenar_mask
        state.palm_visible    = True


def _draw_phase_hud(frame, state, palm_visible,
                     cached_ita, remaining, label):
    """Draw phase HUD overlay."""
    frame_h = frame.shape[0]

    for msk, color in [
        (state.thenar_mask,     (0, 255, 0)),
        (state.central_mask,    (0, 255, 255)),
        (state.hypothenar_mask, (255, 0, 0))
    ]:
        if msk is not None and np.sum(msk) > 0:
            colored         = np.zeros_like(frame)
            colored[msk==1] = color
            cv2.addWeighted(
                colored, 0.4, frame, 0.6, 0, frame
            )

    status_color = (0, 255, 0) if palm_visible \
                   else (0, 0, 255)

    cv2.putText(
        frame, f"{label}  {remaining:.1f}s",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, status_color, 2
    )
    cv2.putText(
        frame,
        f"ITA: {cached_ita:.1f}  "
        f"{'Palm detected' if palm_visible else 'Show palm'}",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 255), 2
    )
    cv2.putText(
        frame,
        "GREEN=Thenar  CYAN=Central  BLUE=Hypothenar",
        (10, frame_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4, (255, 255, 255), 1
    )


def _ita_to_fitzpatrick(ita):
    if   ita > 55:  return "FST I"
    elif ita > 41:  return "FST II"
    elif ita > 28:  return "FST III"
    elif ita > 10:  return "FST IV"
    elif ita > -30: return "FST V"
    else:           return "FST VI"