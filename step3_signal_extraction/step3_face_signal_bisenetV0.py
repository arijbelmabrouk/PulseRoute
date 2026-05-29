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

from step1_video_capture.step1_video_captureV3 \
    import initialize_camera
from step3_rgb_signal import (
    run_signal_extraction,
    RGBSignalBuffer,
    extract_rgb_signal
)


# ─────────────────────────────────────────
# Shared state — Step 2 writes, Step 3 reads
# ─────────────────────────────────────────

class FaceROIState:
    """
    Shared state object between Step 2 and Step 3.

    Step 2 face detection writes mask and ITA
    every time BiSeNet updates.
    Step 3 signal extraction reads them every frame.

    This decouples Step 2 and Step 3 completely —
    neither needs to know about the other's internals.
    """
    def __init__(self):
        self.combined_mask = None
        self.forehead_mask = None
        self.cheek_mask    = None
        self.ita           = 55.0  # default light skin


# ─────────────────────────────────────────
# Step 2 — Face ROI extraction
# (condensed — runs BiSeNet and updates state)
# ─────────────────────────────────────────

def run_face_roi_extraction(cap, actual_fps,
                            state, duration_sec=5):
    """
    Run Step 2 face ROI extraction for a short
    setup phase — establishes stable mask before
    Step 3 signal extraction begins.

    Updates state.combined_mask, state.forehead_mask,
    state.cheek_mask, state.ita every BiSeNet cycle.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — FaceROIState shared object
        duration_sec — how long to run setup phase
    """
    # Import BiSeNet components from Step 2
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

    print(f"Step 2: Establishing face ROI "
          f"({duration_sec}s setup)...")

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

                # Write to shared state
                state.combined_mask = new_combined
                state.forehead_mask = new_fore
                state.cheek_mask    = new_cheek

                # Update ITA
                _, ita = estimate_skin_tone(
                    frame, new_combined
                )
                state.ita = ita

        # Show setup preview
        display = frame.copy()
        if state.forehead_mask is not None:
            colored = np.zeros_like(display)
            colored[state.forehead_mask == 1] = \
                (0, 255, 0)
            cv2.addWeighted(
                colored, 0.4, display, 0.6, 0, display
            )
        if state.cheek_mask is not None:
            colored = np.zeros_like(display)
            colored[state.cheek_mask == 1] = \
                (255, 0, 0)
            cv2.addWeighted(
                colored, 0.4, display, 0.6, 0, display
            )

        remaining = max(0, end_time - time.time())
        cv2.putText(
            display,
            f"Setting up ROI... {remaining:.1f}s",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )
        cv2.putText(
            display,
            f"ITA: {state.ita:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 2
        )
        cv2.imshow("Step 2 — Face ROI Setup", display)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    print(f"Step 2 complete — ITA: {state.ita:.1f}  "
          f"Mask pixels: "
          f"{np.sum(state.combined_mask) if state.combined_mask is not None else 0}")


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
    state = FaceROIState()

    # ── Step 2 — Establish face ROI ───────────────
    # Runs BiSeNet for 5 seconds to lock stable mask
    run_face_roi_extraction(
        cap, actual_fps, state, duration_sec=5
    )

    if state.combined_mask is None:
        print("ERROR: No face ROI detected")
        cap.release()
        sys.exit(1)

    # ── Build callables for Step 3 ────────────────
    # get_mask_fn: returns current mask each frame
    # get_ita_fn:  returns current ITA each frame
    # Step 3 calls these every frame during recording

    def get_mask_fn(frame_rgb):
        """
        Returns cached mask from shared state.
        Mask stays locked after setup phase —
        BiSeNet already established stable ROI.
        Returns forehead and cheek separately
        for display in Step 3 progress window.
        """
        return (
            state.combined_mask,
            state.forehead_mask,
            state.cheek_mask
        )

    def get_ita_fn():
        """
        Returns current ITA from shared state.
        Fixed after setup phase — skin tone
        doesn't change during 30s recording.
        """
        return state.ita

    # ── Step 3 — RGB Signal Extraction ────────────
    print("\nStep 2 complete — starting Step 3...")

    r, g, b, fps_measured, \
    quality_ok, issues, thresh_summary = \
        run_signal_extraction(
            cap, actual_fps,
            get_mask_fn,
            get_ita_fn,
            modality="face"
        )

    # ── Results ───────────────────────────────────
    print(f"\n{'='*40}")
    print(f"STEP 3 RESULTS")
    print(f"{'='*40}")
    print(f"Signal length:  {len(g)} frames")
    print(f"Actual fps:     {fps_measured:.1f}")
    print(f"Green mean:     {np.mean(g):.2f}")
    print(f"Green std:      {np.std(g):.4f}")
    print(f"Quality OK:     {quality_ok}")
    print(f"Mean ITA:       {state.ita:.1f}")
    if thresh_summary:
        print(f"Over threshold: "
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