import cv2
import sys
import os
import mediapipe as mp
import numpy as np
import time
from collections import deque

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

from step1_video_capture.step1_video_captureV3 \
    import initialize_camera
from step3_signal_extraction.step3_rgb_signal import (
    run_signal_extraction,
    RGBSignalBuffer,
    extract_rgb_signal
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
# Shared state — Step 2 writes, Step 3 reads
# ─────────────────────────────────────────

class PalmROIState:
    """
    Shared state between Step 2 palm detection
    and Step 3 signal extraction.

    Step 2 writes mask and ITA every frame
    when palm is correctly facing camera.
    Step 3 reads them every frame during recording.
    """
    def __init__(self):
        self.combined_mask  = None
        self.thenar_mask    = None
        self.central_mask   = None
        self.hypothenar_mask = None
        self.ita            = 55.0
        self.palm_visible   = False


# ─────────────────────────────────────────
# Step 2 — Palm ROI setup phase
# ─────────────────────────────────────────

def run_palm_roi_extraction(cap, actual_fps,
                             state, duration_sec=5):
    """
    Run Step 2 palm ROI extraction for setup phase.

    Runs MediaPipe Hands for duration_sec seconds
    to establish stable palm mask before Step 3 begins.

    Updates state every frame when palm is visible.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — PalmROIState shared object
        duration_sec — setup phase duration
    """
    timestamps       = deque(maxlen=max(int(actual_fps), 2))
    roi_area_history = deque(maxlen=int(actual_fps * 2))
    frame_counter    = 0

    cached_fitzpatrick = "Unknown"
    cached_ita         = 0.0

    print(f"Step 2: Establishing palm ROI "
          f"({duration_sec}s setup)...")
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

            results = hands.process(frame_rgb)

            palm_visible = False

            if results.multi_hand_landmarks:
                landmarks  = results.multi_hand_landmarks[0].landmark
                handedness = results.multi_handedness[0]

                palm_facing, _ = is_palm_facing_camera(
                    landmarks, handedness
                )

                if palm_facing:
                    palm_visible = True

                    # Build individual region masks
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

                    # Individual masks for display
                    thenar_mask = np.zeros(
                        (frame_h, frame_w), dtype=np.uint8
                    )
                    central_mask = np.zeros(
                        (frame_h, frame_w), dtype=np.uint8
                    )
                    hypothenar_mask = np.zeros(
                        (frame_h, frame_w), dtype=np.uint8
                    )

                    if len(thenar_pts) >= 3:
                        cv2.fillPoly(
                            thenar_mask,
                            [np.array(thenar_pts,
                                      dtype=np.int32)], 1
                        )
                    if len(central_pts) >= 3:
                        cv2.fillPoly(
                            central_mask,
                            [np.array(central_pts,
                                      dtype=np.int32)], 1
                        )
                    if len(hypothenar_pts) >= 3:
                        cv2.fillPoly(
                            hypothenar_mask,
                            [np.array(hypothenar_pts,
                                      dtype=np.int32)], 1
                        )

                    # Combined mask
                    combined_mask = get_combined_mask(
                        [thenar_pts, central_pts,
                         hypothenar_pts],
                        frame_h, frame_w
                    )

                    # Skin tone every N frames
                    if frame_counter % EVAL_EVERY_N_FRAMES == 0:
                        cached_fitzpatrick, cached_ita = \
                            estimate_skin_tone(
                                frame, combined_mask
                            )

                    # Write to shared state
                    state.combined_mask   = combined_mask
                    state.thenar_mask     = thenar_mask
                    state.central_mask    = central_mask
                    state.hypothenar_mask = hypothenar_mask
                    state.ita             = cached_ita
                    state.palm_visible    = True

            # Display setup preview
            display = frame.copy()

            if state.thenar_mask is not None:
                colored = np.zeros_like(display)
                colored[state.thenar_mask == 1] = \
                    (0, 255, 0)
                cv2.addWeighted(
                    colored, 0.4, display, 0.6, 0, display
                )
            if state.central_mask is not None:
                colored = np.zeros_like(display)
                colored[state.central_mask == 1] = \
                    (0, 255, 255)
                cv2.addWeighted(
                    colored, 0.4, display, 0.6, 0, display
                )
            if state.hypothenar_mask is not None:
                colored = np.zeros_like(display)
                colored[state.hypothenar_mask == 1] = \
                    (255, 0, 0)
                cv2.addWeighted(
                    colored, 0.4, display, 0.6, 0, display
                )

            remaining = max(0, end_time - time.time())

            cv2.putText(
                display,
                f"Setting up palm ROI... {remaining:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if palm_visible
                else (0, 0, 255),
                2
            )
            cv2.putText(
                display,
                f"ITA: {cached_ita:.1f}  "
                f"{'Palm visible' if palm_visible else 'Show palm'}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2
            )
            cv2.putText(
                display,
                "GREEN=Thenar  CYAN=Central  BLUE=Hypothenar",
                (10, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1
            )

            cv2.imshow("Step 2 — Palm ROI Setup", display)
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    print(
        f"Step 2 complete — "
        f"ITA: {state.ita:.1f}  "
        f"Palm visible: {state.palm_visible}  "
        f"Mask pixels: "
        f"{np.sum(state.combined_mask) if state.combined_mask is not None else 0}"
    )


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1 — Camera ───────────────────────────
    cap, actual_fps = initialize_camera(camera_index=0)
    if cap is None:
        print("Camera failed")
        sys.exit(1)

    # ── Shared state ──────────────────────────────
    state = PalmROIState()

    # ── Step 2 — Establish palm ROI ───────────────
    run_palm_roi_extraction(
        cap, actual_fps, state, duration_sec=5
    )

    if state.combined_mask is None or \
       not state.palm_visible:
        print("ERROR: No palm ROI detected")
        print("Make sure your palm faces the camera")
        cap.release()
        sys.exit(1)

    # ── Build callables for Step 3 ────────────────
    def get_mask_fn(frame_rgb):
        """
        Returns palm masks from shared state.
        Thenar and central passed as sub-regions
        for display — hypothenar included in combined.
        """
        return (
            state.combined_mask,
            state.thenar_mask,      # displayed as green
            state.central_mask      # displayed as blue
        )

    def get_ita_fn():
        """Returns palm ITA from shared state."""
        return state.ita

    # ── Step 3 — RGB Signal Extraction ────────────
    print("\nStep 2 complete — starting Step 3...")
    print("Keep your palm facing the camera")

    r, g, b, fps_measured, \
    quality_ok, issues, thresh_summary = \
        run_signal_extraction(
            cap, actual_fps,
            get_mask_fn,
            get_ita_fn,
            modality="palm"
        )

    # ── Results ───────────────────────────────────
    print(f"\n{'='*40}")
    print(f"STEP 3 PALM RESULTS")
    print(f"{'='*40}")
    print(f"Signal length:  {len(g)} frames")
    print(f"Actual fps:     {fps_measured:.1f}")
    print(f"Green mean:     {np.mean(g):.2f}")
    print(f"Green std:      {np.std(g):.4f}")
    print(f"Quality OK:     {quality_ok}")
    print(f"Mean ITA:       {state.ita:.1f}")
    if thresh_summary:
        print(f"Over threshold:  "
              f"{thresh_summary['overexposure_mean']:.0f}")
        print(f"Under threshold: "
              f"{thresh_summary['underexposure_mean']:.0f}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues — signal ready for POS")

    print(f"\nR signal: min={np.min(r):.2f} "
          f"max={np.max(r):.2f} "
          f"mean={np.mean(r):.2f}")
    print(f"G signal: min={np.min(g):.2f} "
          f"max={np.max(g):.2f} "
          f"mean={np.mean(g):.2f}")
    print(f"B signal: min={np.min(b):.2f} "
          f"max={np.max(b):.2f} "
          f"mean={np.mean(b):.2f}")

    cap.release()