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

# ── Step 12 publisher ──────────────────────────────────
from step12_display.publisher import (
    publish_status, publish_step, publish_motion,
    publish_routing, publish_final
)

# ─────────────────────────────────────────
# Adaptive cutoff helper
# ─────────────────────────────────────────

def get_adaptive_low_hz(rr_valid, rr_hz, rr_confidence):
    if rr_valid and rr_hz is not None \
       and rr_confidence > 0.4:
        low_hz = rr_hz + 0.10
        low_hz = max(low_hz, 0.667)
        return round(low_hz, 4)
    else:
        return 0.917


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("="*50)
    print("rPPG Pipeline")
    print("="*50)

    publish_status("starting", 0, "Initializing pipeline")

    # ── Step 1 ────────────────────────────────────
    print("\nSTEP 1 — Camera initialization")
    cap, actual_fps = initialize_camera(camera_index=0)
    if cap is None:
        print("Camera failed")
        sys.exit(1)

    publish_step(1, {"fps": round(actual_fps, 1)})
    publish_status("running", 5, "Camera ready")

    # ── Step 2 ────────────────────────────────────
    print("\n" + "="*50)
    print("STEP 2 — Face ROI extraction + Calibration")
    print("="*50)

    face_state = FaceROIState()
    profile = run_face_roi_extraction(
        cap, actual_fps, face_state,
        duration_sec=10
    )

    if face_state.combined_mask is None:
        print("ERROR: No face ROI detected")
        cap.release()
        sys.exit(1)

    publish_step(2, {
        "ita":           round(face_state.ita, 1),
        "fitzpatrick":   profile.fitzpatrick,
        "profile_valid": profile.is_valid,
        "hr_estimate":   profile.hr_estimate_bpm,
    })
    publish_status("running", 15, "Face detected — calibrating")

    # ── Step 3 ────────────────────────────────────
    print("\n" + "="*50)
    print("STEP 3 — RGB signal extraction (face)")
    print("="*50)

    r, g, b, fps_measured, quality_ok, issues, _ = \
        run_face_signal_extraction(
            cap, actual_fps, face_state,
            duration_sec=35,
            profile=profile
        )

    if not quality_ok:
        print("Signal quality issues:")
        for issue in issues:
            print(f"  - {issue}")

    publish_step(3, {
        "quality_ok": quality_ok,
        "fps":        round(fps_measured, 1),
    })
    publish_status("running", 28, "Signal captured")

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

    publish_step(4, {})
    publish_status("running", 38, "Signal normalized")

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

    publish_step(10, {
        "rr_bpm": round(rr_bpm, 1) if rr_bpm else None,
    })
    publish_status("running", 48,
        f"Respiratory: {rr_bpm:.1f} BrPM" if rr_bpm else "Respiratory measured")

    # ── Step 5 ────────────────────────────────────
    print("\n" + "="*50)
    print("STEP 5 — Pulse signal extraction (POS)")
    print("="*50)

    pulse, s1, s2_val, alpha = compute_pos(
        r_norm, g_norm, b_norm
    )

    print_pos_report(pulse, fps_measured, alpha, s1, s2_val)

    # ── Adaptive notch + adaptive cutoff ──────────
    if rr_valid and rr_hz is not None \
       and rr_confidence > 0.4:
        pulse_notched, notch_applied = \
            apply_notch_filter(pulse, rr_hz, fps_measured)
        if notch_applied:
            print(f"\nAdaptive notch applied at "
                  f"{rr_hz:.4f} Hz ({rr_bpm:.1f} BrPM)")
            pulse = pulse_notched
    else:
        print("\nSkipping notch — low RR confidence")

    adaptive_low_hz = get_adaptive_low_hz(
        rr_valid, rr_hz, rr_confidence
    )
    print(f"Adaptive low cutoff: "
          f"{adaptive_low_hz:.3f} Hz "
          f"({adaptive_low_hz*60:.1f} BPM)")

    publish_step(5, {"pulse_std": round(float(np.std(pulse)), 6)})
    publish_status("running", 55, "Pulse extracted")

    # ── Step 6 ────────────────────────────────────
    print("\n" + "="*50)
    print("STEP 6 — Bandpass filter (adaptive)")
    print("="*50)

    filtered, clip_report = apply_bandpass_filter(
        pulse, fps_measured,
        low_hz=adaptive_low_hz,
        profile=profile
    )

    print_filter_report(pulse, filtered, fps_measured, clip_report)

    publish_step(6, {
        "filtered_std":    round(float(np.std(filtered)), 6),
        "bandpass_low_hz": round(adaptive_low_hz, 3),
    })
    publish_status("running", 62, "Signal filtered")

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

    publish_step(7, {
        "fft_peak_bpm": round(fft_report.get("peak_bpm", 0), 1),
        "snr_ratio":    round(fft_report.get("snr_ratio", 0), 1),
    })
    publish_status("running", 68,
        f"FFT peak: {fft_report.get('peak_bpm', 0):.1f} BPM")

    # ── Step 8 — first pass ───────────────────────
    print("\n" + "="*50)
    print("STEP 8 — Peak detection")
    print("="*50)

    hr_bpm, peak_hz, peak_power, \
    n_peaks, n_harmonics = \
        find_dominant_peak(freqs, power)

    # FFT cross-check
    fft_dominant_bpm = fft_report.get('peak_bpm', None)
    if fft_dominant_bpm is not None and \
       hr_bpm is not None and \
       fft_report.get('snr_ratio', 0) > 5.0:
        if (fft_dominant_bpm - hr_bpm) > 15.0:
            print(f"  Step 8 override: "
                  f"{hr_bpm:.1f} → {fft_dominant_bpm:.1f} BPM "
                  f"(FFT cross-check)")
            hr_bpm  = fft_dominant_bpm
            peak_hz = fft_dominant_bpm / 60.0

    # First pass — signal_quality unknown yet
    peak_indices, peak_times, rr_intervals, rr_tolerance = \
        detect_beat_peaks(
            filtered, fps_measured,
            hr_bpm_fft=hr_bpm,
            profile=profile,
            signal_quality=None
        )

    peak_quality_ok, peak_report = print_peak_report(
        hr_bpm, peak_hz, peak_power,
        rr_intervals, peak_times,
        n_peaks, n_harmonics,
        rr_tolerance=rr_tolerance
    )

    publish_step(8, {
        "hr_bpm_step8": round(hr_bpm, 1) if hr_bpm else None,
        "n_peaks":      len(peak_indices),
    })
    publish_status("running", 75,
        f"Peaks detected: {len(peak_indices)} beats")

    # ── Step 9 — first pass ───────────────────────
    print("\n" + "="*50)
    print("STEP 9 — Heart rate & HRV (pass 1)")
    print("="*50)

    hr_results_pass1 = compute_hr_hrv(
        hr_bpm_fft   = hr_bpm,
        rr_intervals = rr_intervals,
        peak_times   = peak_times,
        snr_ratio    = fft_report.get('snr_ratio', 0),
        quality_ok   = quality_ok,
        fps          = fps_measured,
        profile      = profile
    )

    # ── Step 11 ───────────────────────────────────
    print("\n" + "="*50)
    print("STEP 11 — Signal quality score")
    print("="*50)

    snr_score, snr_db, route_palm, \
    quality_level, snr_report = \
        compute_snr_score(
            filtered, freqs, power,
            hr_bpm, rr_intervals, fps_measured,
            hr_confidence=hr_results_pass1['confidence'],
            profile=profile
        )

    print_snr_report(
        snr_score, snr_db,
        route_palm, quality_level, snr_report
    )

    publish_routing(
        route_palm=route_palm,
        reason=(
            "Signal too weak for HRV"
            if snr_report.get("std_floor_triggered")
            else ("SNR below threshold" if route_palm else "Face accepted")
        ),
        snr_score=snr_score,
        std_floor_triggered=snr_report.get("std_floor_triggered", False),
    )
    publish_status("running", 88, "Quality scored — refining peaks")

    # ── Step 8 — second pass ──────────────────────
    print(f"\nStep 8 second pass — "
          f"refining RR filter with quality={snr_score:.3f}")

    _, _, rr_intervals, rr_tolerance = \
        detect_beat_peaks(
            filtered, fps_measured,
            hr_bpm_fft=hr_bpm,
            profile=profile,
            signal_quality=snr_score
        )

    # ── Step 9 — final pass ───────────────────────
    print("\n" + "="*50)
    print("STEP 9 — Heart rate & HRV (final)")
    print("="*50)

    hr_results = compute_hr_hrv(
        hr_bpm_fft   = hr_bpm,
        rr_intervals = rr_intervals,
        peak_times   = peak_times,
        snr_ratio    = fft_report.get('snr_ratio', 0),
        quality_ok   = quality_ok,
        fps          = fps_measured,
        profile      = profile
    )

    print_hr_hrv_report(hr_results)

    publish_step(9, {
        "hr_bpm":           hr_results.get("final_hr"),
        "rmssd":            hr_results.get("rmssd"),
        "hrv_overall":      hr_results.get("hrv_overall"),
        "confidence":       hr_results.get("confidence"),
        "confidence_level": hr_results.get("confidence_level"),
    })
    publish_status("running", 95,
        f"HR: {hr_results.get('final_hr')} BPM")

    # ── Final summary ─────────────────────────────
    print(f"\n{'='*50}")
    print(f"PIPELINE COMPLETE — Steps 1-11")
    print(f"{'='*50}")
    print(f"Modality:      FACE")
    print(f"Heart Rate:    {hr_results['final_hr']} BPM")
    print(f"Respiratory:   "
          f"{rr_bpm if rr_bpm else 'N/A'} BrPM")
    print(f"RMSSD:         {hr_results['rmssd']} ms"
          f"{'  ⚠ LOW CONFIDENCE' if snr_report.get('std_floor_triggered') else ''}")
    print(f"HRV:           {hr_results['hrv_overall']}")
    print(f"SNR Score:     {snr_score:.4f} "
          f"({quality_level.upper()})")
    print(f"Routing:       "
          f"{'⚠ PALM RECOMMENDED' if route_palm else '✓ FACE ACCEPTED'}"
          f"{' — signal too weak for HRV' if snr_report.get('std_floor_triggered') else ''}")
    print(f"Confidence:    "
          f"{hr_results['confidence']} "
          f"({hr_results['confidence_level'].upper()})")
    print(f"ITA:           {face_state.ita:.1f}  "
          f"({profile.fitzpatrick})")
    print(f"Profile valid: {profile.is_valid}")
    if profile.hr_estimate_bpm:
        print(f"Calib HR est:  {profile.hr_estimate_bpm} BPM")
    print(f"\nReady for Step 12 (display) ✓")

    # ── Publish final results ──────────────────────
    publish_final(
        hr_results=hr_results,
        snr_score=snr_score,
        quality_level=quality_level,
        route_palm=route_palm,
        ita=face_state.ita,
        fitzpatrick=profile.fitzpatrick,
        rr_bpm=rr_bpm,
        snr_report=snr_report,
    )
    publish_status("complete", 100, "Pipeline complete")

    cap.release()
