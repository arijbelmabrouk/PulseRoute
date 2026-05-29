import os
import sys
import numpy as np

# ─────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

FACE_DIR = os.path.join(
    PROJECT_ROOT,
    "step2_face_ROI_extraction",
    "face_parsing_mask"
)
REPO_DIR  = os.path.join(FACE_DIR, "face-parsing.PyTorch")
STEP3_DIR = os.path.join(PROJECT_ROOT, "step3_signal_extraction")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, FACE_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, STEP3_DIR)

# ─────────────────────────────────────────
# Imports — one per step
# ─────────────────────────────────────────

# Step 1
from step1_video_capture.step1_video_captureV3 \
    import initialize_camera

# Step 2 + 3 face
from step3_signal_extraction.step3_face_signal_bisenet import (
    FaceROIState,
    run_face_roi_extraction,
    run_face_signal_extraction
)

# Step 4
from step4_normalization.step4_normalization import (
    normalize_rgb,
    print_normalization_report
)

# Step 5
from step5_pulse_signal_extraction.step5_POS import (
    compute_pos,
    print_pos_report
)

# Step 6
from step6_bandpass_filter.step6_bandpass import (
    apply_bandpass_filter,
    print_filter_report
)

# Step 7
from step7_conversion_time_to_frequency.step7_fft import (
    compute_fft,
    print_fft_report
)

# Step 8
from step8_peak_detection.step8_peaks import (
    find_dominant_peak,
    detect_beat_peaks,
    print_peak_report
)

# Steps 9-12 imported here as they are built
# from step9_hr.step9_hr import compute_heart_rate
# from step10_rr.step10_rr import compute_respiratory_rate
# from step11_snr.step11_snr import compute_snr
# from step12_display.step12_display import display_results

# Palm imports — uncommented after Step 11
# from step3_signal_extraction.step3_palm_signal import (
#     PalmROIState,
#     run_palm_roi_extraction,
#     run_palm_signal_extraction
# )


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("="*50)
    print("rPPG Pipeline")
    print("="*50)

    # ── Step 1 — Camera ───────────────────────────
    print("\nSTEP 1 — Camera initialization")
    cap, actual_fps = initialize_camera(camera_index=0)
    if cap is None:
        print("Camera failed")
        sys.exit(1)

    # ── Step 2 — Face ROI (BiSeNet) ───────────────
    print("\n" + "="*50)
    print("STEP 2 — Face ROI extraction (BiSeNet)")
    print("="*50)

    face_state = FaceROIState()
    run_face_roi_extraction(
        cap, actual_fps, face_state, duration_sec=5
    )

    if face_state.combined_mask is None:
        print("ERROR: No face ROI detected")
        cap.release()
        sys.exit(1)

    # ── Step 3 — RGB signal extraction ────────────
    print("\n" + "="*50)
    print("STEP 3 — RGB signal extraction (face)")
    print("="*50)

    r, g, b, fps_measured, quality_ok, issues, _ = \
        run_face_signal_extraction(
            cap, actual_fps, face_state
        )

    if not quality_ok:
        print("Signal quality issues:")
        for issue in issues:
            print(f"  - {issue}")

    # ── Step 4 — Normalization ────────────────────
    print("\n" + "="*50)
    print("STEP 4 — Normalization")
    print("="*50)

    r_norm, g_norm, b_norm, means, _ = \
        normalize_rgb(r, g, b)

    print_normalization_report(
        r_norm, g_norm, b_norm,
        r, g, b, means, "face"
    )

    # ── Step 5 — POS pulse extraction ─────────────
    print("\n" + "="*50)
    print("STEP 5 — Pulse signal extraction (POS)")
    print("="*50)

    pulse, s1, s2, alpha = compute_pos(
        r_norm, g_norm, b_norm
    )

    print_pos_report(pulse, fps_measured, alpha, s1, s2)

    # ── Step 6 — Bandpass filter ──────────────────
    print("\n" + "="*50)
    print("STEP 6 — Bandpass filter (40-180 BPM)")
    print("="*50)

    filtered = apply_bandpass_filter(
        pulse, fps_measured
    )

    print_filter_report(pulse, filtered, fps_measured)

    # ── Step 7 — FFT ──────────────────────────────
    print("\n" + "="*50)
    print("STEP 7 — FFT (frequency analysis)")
    print("="*50)

    freqs, power, bpm_axis, \
    freq_res_hz, freq_res_bpm = \
        compute_fft(filtered, fps_measured)

    fft_quality_ok, fft_report = print_fft_report(
        freqs, power, bpm_axis,
        freq_res_hz, freq_res_bpm,
        fps_measured
    )

    # ── Step 8 — Peak detection ───────────────────
    print("\n" + "="*50)
    print("STEP 8 — Peak detection")
    print("="*50)

    # Frequency domain — dominant HR peak
    hr_bpm, peak_hz, peak_power, \
    n_peaks, n_harmonics = \
        find_dominant_peak(freqs, power)

    # Time domain — individual beat peaks for HRV
    peak_indices, peak_times, rr_intervals = \
        detect_beat_peaks(filtered, fps_measured)

    peak_quality_ok, peak_report = print_peak_report(
        hr_bpm, peak_hz, peak_power,
        rr_intervals, peak_times,
        n_peaks, n_harmonics
    )

    # ── Steps 9-12 added here as built ────────────
    # hr  = compute_heart_rate(hr_bpm, rr_intervals)
    # rr  = compute_respiratory_rate(g_norm, fps_measured)
    # snr = compute_snr(filtered, fps_measured)
    # display_results(hr, rr, snr, face_state.ita)

    # ── Routing added after Step 11 ───────────────
    # if snr < SNR_THRESHOLD:
    #     print("Face insufficient — switching to palm")

    # ── Summary ───────────────────────────────────
    print(f"\n{'='*50}")
    print(f"PIPELINE — Steps 1-8 complete")
    print(f"{'='*50}")
    print(f"Frames:       {len(filtered)}")
    print(f"FPS:          {fps_measured:.1f}")
    print(f"ITA:          {face_state.ita:.1f}")
    print(f"Heart rate:   {hr_bpm:.1f} BPM")
    print(f"Beats found:  {peak_report.get('n_beats', 0)}")
    print(f"RR mean:      "
          f"{peak_report.get('rr_mean_ms', 0):.1f} ms")
    print(f"RR std:       "
          f"{peak_report.get('rr_std_ms', 0):.1f} ms")
    print(f"\nReady for Step 9 (HR/HRV calculation)")

    cap.release()