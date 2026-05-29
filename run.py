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

from step1_video_capture.step1_video_captureV3 \
    import initialize_camera

from step3_signal_extraction.step3_face_signal_bisenet import (
    FaceROIState,
    run_face_roi_extraction,
    run_face_signal_extraction
)

from step4_normalization.step4_normalization import (
    normalize_rgb,
    print_normalization_report
)

from step5_pulse_signal_extraction.step5_POS import (
    compute_pos,
    print_pos_report
)

from step6_bandpass_filter.step6_bandpass import (
    apply_bandpass_filter,
    print_filter_report
)

from step7_conversion_time_to_frequency.step7_fft import (
    compute_fft,
    print_fft_report
)

from step8_peak_detection.step8_peaks import (
    find_dominant_peak,
    detect_beat_peaks,
    print_peak_report
)

from step9_HR_HRV.step9_hr_hrv import (
    compute_hr_hrv,
    print_hr_hrv_report
)

from step10_respiratory_rate.step10_rr import (
    compute_respiratory_rate,
    apply_notch_filter,
    print_respiratory_report
)

from step11_signal_quality_score.step11_snr import (
    compute_snr_score,
    print_snr_report
)

# Step 12 — uncommented when built
# from step12_display.step12_display import display_results

# Palm routing — added after Step 12 complete
# from step3_signal_extraction.step3_palm_signal import (
#     PalmROIState,
#     run_palm_roi_extraction,
#     run_palm_signal_extraction
# )


# ─────────────────────────────────────────
# Adaptive cutoff helper
# ─────────────────────────────────────────

def get_adaptive_low_hz(rr_valid, rr_hz,
                         rr_confidence):
    """
    Compute adaptive bandpass lower cutoff
    based on detected respiratory frequency.

    Sets cutoff just above breathing frequency
    with 0.1 Hz buffer — replaces fixed 55 BPM.

    If respiratory rate not reliably detected:
    falls back to 0.917 Hz (55 BPM).

    Input:
        rr_valid      — bool from Step 10
        rr_hz         — breathing frequency Hz
        rr_confidence — confidence 0.0-1.0

    Output:
        adaptive_low_hz — lower cutoff in Hz
    """
    if rr_valid and rr_hz is not None \
       and rr_confidence > 0.4:
        low_hz = rr_hz + 0.10
        low_hz = max(low_hz, 0.667)  # min 40 BPM
        return round(low_hz, 4)
    else:
        return 0.917  # fallback 55 BPM


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("="*50)
    print("rPPG Pipeline")
    print("="*50)

    # ── Step 1 ────────────────────────────────────
    print("\nSTEP 1 — Camera initialization")
    cap, actual_fps = initialize_camera(camera_index=0)
    if cap is None:
        print("Camera failed")
        sys.exit(1)

    # ── Step 2 — Face ROI ─────────────────────────
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

    # ── Step 3 ────────────────────────────────────
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

    # ── Step 4 ────────────────────────────────────
    print("\n" + "="*50)
    print("STEP 4 — Normalization")
    print("="*50)

    r_norm, g_norm, b_norm, means, _ = \
        normalize_rgb(r, g, b)

    print_normalization_report(
        r_norm, g_norm, b_norm,
        r, g, b, means, "face"
    )

    # ── Step 10 — Respiratory (before bandpass) ───
    print("\n" + "="*50)
    print("STEP 10 — Respiratory rate")
    print("="*50)

    rr_bpm, rr_hz, rr_confidence, \
    rr_snr, rr_valid = \
        compute_respiratory_rate(g_norm, fps_measured)

    print_respiratory_report(
        rr_bpm, rr_hz, rr_confidence,
        rr_snr, rr_valid
    )

    # ── Step 5 ────────────────────────────────────
    print("\n" + "="*50)
    print("STEP 5 — Pulse signal extraction (POS)")
    print("="*50)

    pulse, s1, s2, alpha = compute_pos(
        r_norm, g_norm, b_norm
    )

    print_pos_report(pulse, fps_measured, alpha, s1, s2)

    # ── Adaptive notch + adaptive cutoff ──────────
    if rr_valid and rr_hz is not None \
       and rr_confidence > 0.4:
        pulse_notched, notch_applied = \
            apply_notch_filter(
                pulse, rr_hz, fps_measured
            )
        if notch_applied:
            print(f"\nAdaptive notch applied at "
                  f"{rr_hz:.4f} Hz "
                  f"({rr_bpm:.1f} BrPM)")
            pulse = pulse_notched
    else:
        print("\nSkipping notch — low RR confidence")

    adaptive_low_hz = get_adaptive_low_hz(
        rr_valid, rr_hz, rr_confidence
    )
    print(f"Adaptive low cutoff: "
          f"{adaptive_low_hz:.3f} Hz "
          f"({adaptive_low_hz*60:.1f} BPM)")

    # ── Step 6 ────────────────────────────────────
    print("\n" + "="*50)
    print("STEP 6 — Bandpass filter (adaptive)")
    print("="*50)

    filtered = apply_bandpass_filter(
        pulse, fps_measured,
        low_hz=adaptive_low_hz
    )

    print_filter_report(pulse, filtered, fps_measured)

    # ── Step 7 ────────────────────────────────────
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

    # ── Step 8 ────────────────────────────────────
    print("\n" + "="*50)
    print("STEP 8 — Peak detection")
    print("="*50)

    hr_bpm, peak_hz, peak_power, \
    n_peaks, n_harmonics = \
        find_dominant_peak(freqs, power)

    peak_indices, peak_times, rr_intervals = \
        detect_beat_peaks(
            filtered, fps_measured,
            hr_bpm_fft=hr_bpm
        )

    peak_quality_ok, peak_report = print_peak_report(
        hr_bpm, peak_hz, peak_power,
        rr_intervals, peak_times,
        n_peaks, n_harmonics
    )

    # ── Step 9 ────────────────────────────────────
    print("\n" + "="*50)
    print("STEP 9 — Heart rate & HRV")
    print("="*50)

    hr_results = compute_hr_hrv(
        hr_bpm_fft   = hr_bpm,
        rr_intervals = rr_intervals,
        peak_times   = peak_times,
        snr_ratio    = fft_report.get('snr_ratio', 0),
        quality_ok   = quality_ok,
        fps          = fps_measured
    )

    print_hr_hrv_report(hr_results)

    # ── Step 11 — SNR score ───────────────────────
    print("\n" + "="*50)
    print("STEP 11 — Signal quality score")
    print("="*50)

    snr_score, snr_db, route_palm, \
    quality_level, snr_report = \
        compute_snr_score(
            filtered, freqs, power,
            hr_bpm, rr_intervals, fps_measured
        )

    print_snr_report(
        snr_score, snr_db,
        route_palm, quality_level, snr_report
    )

    # ── Routing — added after Step 12 complete ────
    # if route_palm:
    #     run_palm_pipeline(cap, actual_fps)

    # ── Step 12 — added next ──────────────────────
    # display_results(
    #     hr_results, rr_bpm, snr_score,
    #     face_state.ita, "face"
    # )

    # ── Final summary ─────────────────────────────
    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE — Steps 1-11")
    print(f"{'='*50}")
    print(f"Modality:      FACE")
    print(f"Heart Rate:    {hr_results['final_hr']} BPM")
    print(f"Respiratory:   "
          f"{rr_bpm if rr_bpm else 'N/A'} BrPM")
    print(f"RMSSD:         {hr_results['rmssd']} ms")
    print(f"HRV:           {hr_results['hrv_overall']}")
    print(f"SNR Score:     {snr_score:.4f} "
          f"({quality_level.upper()})")
    print(f"Confidence:    "
          f"{hr_results['confidence']} "
          f"({hr_results['confidence_level'].upper()})")
    print(f"ITA:           {face_state.ita:.1f}")
    print(f"\nReady for Step 12 (display)")

    cap.release()