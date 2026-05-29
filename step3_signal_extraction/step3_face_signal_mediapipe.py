import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys
from collections import deque

# ─────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FACE_MESH_DIR = os.path.join(
    PROJECT_ROOT,
    "step2_face_ROI_extraction",
    "mediapipe_face_mesh"
)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FACE_MESH_DIR)

from step1_video_capture.step1_video_captureV3 \
    import initialize_camera
from step3_signal_extraction.step3_rgb_signal import (
    run_signal_extraction
)
from step2_face_ROI_extraction.mediapipe_face_mesh\
    .step2_face_detectionV5 import (
    mp_face_mesh,
    FOREHEAD_LANDMARKS,
    LEFT_CHEEK_LANDMARKS,
    RIGHT_CHEEK_LANDMARKS,
    EVAL_EVERY_N_FRAMES,
    get_landmark_pixels,
    get_combined_mask,
    get_roi_area,
    is_face_visible,
    estimate_skin_tone,
    assess_lighting,
    calculate_roi_stability
)


# ─────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────

class FaceMediaPipeState:
    """
    Shared state between Step 2 MediaPipe face mesh
    and Step 3 signal extraction.

    Step 2 writes mask and ITA every frame
    when face is visible.
    Step 3 reads them every frame during recording.
    """
    def __init__(self):
        self.combined_mask    = None
        self.forehead_mask    = None
        self.left_cheek_mask  = None
        self.right_cheek_mask = None
        self.ita              = 55.0
        self.face_visible     = False


# ─────────────────────────────────────────
# Step 2 — MediaPipe face mesh setup phase
# ─────────────────────────────────────────

def run_mediapipe_face_roi_extraction(
        cap, actual_fps, state, duration_sec=5):
    """
    Run MediaPipe Face Mesh ROI extraction
    for setup phase.

    Runs for duration_sec seconds to establish
    stable face mask before Step 3 begins.

    Updates state every frame when face is visible.

    Input:
        cap          — VideoCapture from Step 1
        actual_fps   — fps from Step 1
        state        — FaceMediaPipeState object
        duration_sec — setup phase duration
    """
    frame_counter = 0
    cached_ita    = 55.0

    print(f"Step 2 MediaPipe: Establishing face ROI "
          f"({duration_sec}s setup)...")
    print("Look directly at the camera")

    end_time = time.time() + duration_sec

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            frame_h, frame_w = frame.shape[:2]
            frame_rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results = face_mesh.process(frame_rgb)

            face_visible = False

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                visible, _, _ = is_face_visible(
                    landmarks, frame_w, frame_h
                )

                if visible:
                    face_visible = True

                    # Build ROI point lists
                    forehead_pts = get_landmark_pixels(
                        landmarks, FOREHEAD_LANDMARKS,
                        frame_w, frame_h
                    )
                    left_pts = get_landmark_pixels(
                        landmarks, LEFT_CHEEK_LANDMARKS,
                        frame_w, frame_h
                    )
                    right_pts = get_landmark_pixels(
                        landmarks, RIGHT_CHEEK_LANDMARKS,
                        frame_w, frame_h
                    )

                    # Build individual masks for display
                    forehead_mask = np.zeros(
                        (frame_h, frame_w), dtype=np.uint8
                    )
                    left_mask = np.zeros(
                        (frame_h, frame_w), dtype=np.uint8
                    )
                    right_mask = np.zeros(
                        (frame_h, frame_w), dtype=np.uint8
                    )

                    import cv2 as _cv2
                    for pts_list, msk in [
                        (forehead_pts, forehead_mask),
                        (left_pts,     left_mask),
                        (right_pts,    right_mask)
                    ]:
                        if len(pts_list) >= 3:
                            pts  = np.array(
                                pts_list, dtype=np.int32
                            )
                            hull = _cv2.convexHull(pts)
                            _cv2.fillPoly(msk, [hull], 1)

                    # Combined mask
                    combined_mask = np.clip(
                        forehead_mask +
                        left_mask +
                        right_mask,
                        0, 1
                    ).astype(np.uint8)

                    # Update skin tone every N frames
                    if frame_counter % EVAL_EVERY_N_FRAMES == 0:
                        _, cached_ita = estimate_skin_tone(
                            frame, combined_mask
                        )

                    # Write to shared state
                    state.combined_mask    = combined_mask
                    state.forehead_mask    = forehead_mask
                    state.left_cheek_mask  = left_mask
                    state.right_cheek_mask = right_mask
                    state.ita              = cached_ita
                    state.face_visible     = True

            # Display setup preview
            display = frame.copy()

            for msk, color in [
                (state.forehead_mask,    (0, 255, 0)),
                (state.left_cheek_mask,  (0, 0, 255)),
                (state.right_cheek_mask, (255, 0, 0))
            ]:
                if msk is not None and np.sum(msk) > 0:
                    colored          = np.zeros_like(display)
                    colored[msk==1]  = color
                    cv2.addWeighted(
                        colored, 0.4, display,
                        0.6, 0, display
                    )

            remaining = max(0, end_time - time.time())
            cv2.putText(
                display,
                f"MediaPipe Face ROI... {remaining:.1f}s",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if face_visible
                else (0, 0, 255),
                2
            )
            cv2.putText(
                display,
                f"ITA: {cached_ita:.1f}  "
                f"{'Face visible' if face_visible else 'Look at camera'}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2
            )
            cv2.putText(
                display,
                "GREEN=Forehead  RED=L.Cheek  BLUE=R.Cheek",
                (10, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1
            )

            cv2.imshow(
                "Step 2 MediaPipe — Face ROI Setup",
                display
            )
            cv2.waitKey(1)

    cv2.destroyAllWindows()
    print(
        f"Step 2 MediaPipe complete — "
        f"ITA: {state.ita:.1f}  "
        f"Face visible: {state.face_visible}  "
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
    state = FaceMediaPipeState()

    # ── Step 2 — Establish face ROI ───────────────
    run_mediapipe_face_roi_extraction(
        cap, actual_fps, state, duration_sec=5
    )

    if state.combined_mask is None or \
       not state.face_visible:
        print("ERROR: No face ROI detected")
        print("Make sure your face is visible")
        cap.release()
        sys.exit(1)

    # ── Build callables for Step 3 ────────────────
    def get_mask_fn(frame_rgb):
        """
        Returns face masks from shared state.
        Forehead as first sub-region (green).
        Left cheek as second (blue in display).
        """
        return (
            state.combined_mask,
            state.forehead_mask,
            state.left_cheek_mask
        )

    def get_ita_fn():
        """Returns face ITA from shared state."""
        return state.ita

    # ── Step 3 — RGB Signal Extraction ────────────
    print("\nStep 2 complete — starting Step 3...")
    print("Keep your face facing the camera")

    r, g, b, fps_measured, \
    quality_ok, issues, thresh_summary = \
        run_signal_extraction(
            cap, actual_fps,
            get_mask_fn,
            get_ita_fn,
            modality="face_mediapipe"
        )

    # ── Results ───────────────────────────────────
    print(f"\n{'='*40}")
    print(f"STEP 3 FACE MEDIAPIPE RESULTS")
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