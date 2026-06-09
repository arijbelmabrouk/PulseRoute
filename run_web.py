import base64
import os
import signal
import sys
import numpy as np

# ─────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PIPELINE_MODE = os.environ.get("PULSEROUTE_MODE", "auto").strip()
PATIENT_ID    = os.environ.get("PULSEROUTE_PATIENT_ID", "").strip()
PATIENT_AGE   = os.environ.get("PULSEROUTE_PATIENT_AGE", "").strip()
HEADLESS      = os.environ.get("PULSEROUTE_HEADLESS", "true").strip().lower() in (
    "1", "true", "yes"
)
SHOW_DISPLAY  = not HEADLESS

_SESSION_WRITTEN = False
_CAP = None
_LOGGER = None

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

import cv2

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

from session_logger import SessionLogger

from step12_display.publisher import (
    publish_status,
    publish_step,
    publish_routing,
    publish_final,
    publish,
    publish_frame,
    publish_recording_progress,
)


# ─────────────────────────────────────────
# Frame publishing callback
# ─────────────────────────────────────────
# Passed to signal extraction functions.
# Called every N frames with a base64-encoded
# annotated JPEG so the patient page shows
# a live camera feed with ROI overlay.

_FRAME_EVERY = 2    # publish every 2 accepted clean frames for smoother live UX
_MAX_FRAME_WIDTH = 480


def make_frame_callback(mask_color=(0, 220, 0)):
    """
    Return a callback function for run_signal_extraction.

    The callback receives (frame_bgr, combined_mask) and:
      1. Draws the mask overlay on the frame
      2. Encodes to JPEG (quality 55 — small payload)
      3. Base64-encodes
      4. Calls publish_frame()

    The _FRAME_EVERY counter throttles the live feed to reduce backend load.
    """
    _counter = [0]

    def on_frame(frame_bgr, combined_mask):
        _counter[0] += 1
        if _counter[0] % _FRAME_EVERY != 0:
            return
        try:
            annotated = frame_bgr.copy()
            mask_for_display = combined_mask

            if annotated.shape[1] > _MAX_FRAME_WIDTH:
                scale = _MAX_FRAME_WIDTH / float(annotated.shape[1])
                new_dim = (
                    _MAX_FRAME_WIDTH,
                    int(annotated.shape[0] * scale)
                )
                annotated = cv2.resize(
                    annotated, new_dim,
                    interpolation=cv2.INTER_AREA
                )
                if combined_mask is not None:
                    mask_for_display = cv2.resize(
                        combined_mask.astype('uint8'),
                        new_dim,
                        interpolation=cv2.INTER_NEAREST
                    )

            if mask_for_display is not None and \
               np.sum(mask_for_display) > 0:
                overlay              = np.zeros_like(annotated)
                overlay[mask_for_display == 1] = mask_color
                cv2.addWeighted(
                    overlay, 0.35,
                    annotated, 0.65,
                    0, annotated
                )
            _, buf = cv2.imencode(
                '.jpg', annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 50]
            )
            b64 = base64.b64encode(buf).decode('utf-8')
            publish_frame(b64)
        except Exception:
            pass  # never crash pipeline for a frame

    return on_frame


def make_recording_progress_callback(modality):
    def on_progress(percent):
        try:
            publish_recording_progress(
                progress=int(percent),
                message=f"Recording {modality} signal",
                active=True,
            )
        except Exception:
            pass
    return on_progress


def _cleanup_and_exit(signum, frame):
    global _SESSION_WRITTEN
    print(f"\nReceived signal {signum}. Cleaning up...")
    if _CAP is not None:
        try:
            _CAP.release()
        except Exception:
            pass
    if _LOGGER is not None and not _SESSION_WRITTEN:
        try:
            _LOGGER.write()
            _SESSION_WRITTEN = True
        except Exception:
            pass
    sys.exit(0)


# ─────────────────────────────────────────
# Failure exit helper
# ─────────────────────────────────────────

def measurement_failed(cap, reason, suggestions=None,
                        logger=None):
    print(f"\n{'='*50}")
    print("MEASUREMENT FAILED")
    print(f"{'='*50}")
    print(f"\n{reason}")

    if suggestions:
        print("\nTo fix this:")
        for s in suggestions:
            print(f"  • {s}")

    print(f"\n{'='*50}")

    publish("measurement_failed", {
        "status":   "failed",
        "progress": 0,
        "message":  reason,
    })

    try:
        publish_recording_progress(
            progress=100,
            message="",
            active=False,
        )
    except Exception:
        pass

    global _SESSION_WRITTEN
    if logger is not None:
        try:
            logger.set_failure(reason)
            logger.write()
            _SESSION_WRITTEN = True
        except Exception:
            pass

    cap.release()
    sys.exit(0)


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
# Steps 4–11 runner
# ─────────────────────────────────────────

def run_steps_4_to_11(r, g, b, fps_measured,
                       quality_ok, profile, modality):
    """
    Run Steps 4–11 on a raw RGB signal.
    Signal-agnostic — used by both face and palm.
    Publishes each step result to the dashboard.
    """

    # ── Step 4 ────────────────────────────────────
    print(f"\n{'='*50}\nSTEP 4 — Normalization ({modality})\n{'='*50}")
    r_norm, g_norm, b_norm, means, _ = normalize_rgb(r, g, b)
    print_normalization_report(r_norm, g_norm, b_norm,
                               r, g, b, means, modality)
    publish_step(4, {"modality": modality})
    publish_status("running", 38, "Signal normalised")

    # ── Step 10 — Respiratory ─────────────────────
    print(f"\n{'='*50}\nSTEP 10 — Respiratory rate ({modality})\n{'='*50}")
    rr_bpm, rr_hz, rr_confidence, rr_snr, rr_valid = \
        compute_respiratory_rate(g_norm, fps_measured)
    print_respiratory_report(rr_bpm, rr_hz, rr_confidence,
                             rr_snr, rr_valid)
    publish_step(10, {
        "rr_bpm":   round(rr_bpm, 1) if rr_bpm else None,
        "modality": modality,
    })
    publish_status("running", 46,
        f"Respiratory: {rr_bpm:.1f} BrPM"
        if rr_bpm else "Respiratory measured")

    # ── Step 5 ────────────────────────────────────
    print(f"\n{'='*50}\nSTEP 5 — POS ({modality})\n{'='*50}")
    pulse, s1, s2, alpha = compute_pos(r_norm, g_norm, b_norm)
    print_pos_report(pulse, fps_measured, alpha, s1, s2)

    if rr_valid and rr_hz is not None and rr_confidence > 0.4:
        pulse_notched, notch_applied = apply_notch_filter(
            pulse, rr_hz, fps_measured
        )
        if notch_applied:
            print(f"\nAdaptive notch applied at "
                  f"{rr_hz:.4f} Hz ({rr_bpm:.1f} BrPM)")
            pulse = pulse_notched
    else:
        print("\nSkipping notch — low RR confidence")

    adaptive_low_hz = get_adaptive_low_hz(
        rr_valid, rr_hz, rr_confidence
    )
    print(f"Adaptive low cutoff: {adaptive_low_hz:.3f} Hz "
          f"({adaptive_low_hz*60:.1f} BPM)")
    publish_step(5, {
        "pulse_std": round(float(np.std(pulse)), 6),
        "modality":  modality,
    })
    publish_status("running", 54, "Pulse extracted")

    # ── Step 6 ────────────────────────────────────
    print(f"\n{'='*50}\nSTEP 6 — Bandpass filter ({modality})\n{'='*50}")
    filtered, clip_report = apply_bandpass_filter(
        pulse, fps_measured,
        low_hz=adaptive_low_hz, profile=profile
    )
    print_filter_report(pulse, filtered, fps_measured, clip_report)
    publish_step(6, {
        "filtered_std":    round(float(np.std(filtered)), 6),
        "bandpass_low_hz": round(adaptive_low_hz, 3),
        "modality":        modality,
    })
    publish_status("running", 60, "Signal filtered")

    # ── Step 7 ────────────────────────────────────
    print(f"\n{'='*50}\nSTEP 7 — FFT ({modality})\n{'='*50}")
    freqs, power, bpm_axis, freq_res_hz, freq_res_bpm = \
        compute_fft(filtered, fps_measured)
    fft_quality_ok, fft_report = print_fft_report(
        freqs, power, bpm_axis,
        freq_res_hz, freq_res_bpm, fps_measured
    )
    publish_step(7, {
        "fft_peak_bpm": round(fft_report.get("peak_bpm", 0), 1),
        "snr_ratio":    round(fft_report.get("snr_ratio", 0), 1),
        "modality":     modality,
    })
    publish_status("running", 66,
        f"FFT peak: {fft_report.get('peak_bpm', 0):.1f} BPM")

    # ── Step 8 — first pass ───────────────────────
    print(f"\n{'='*50}\nSTEP 8 — Peak detection ({modality})\n{'='*50}")
    hr_bpm, peak_hz, peak_power, n_peaks, n_harmonics = \
        find_dominant_peak(freqs, power)

    fft_dominant_bpm = fft_report.get('peak_bpm', None)
    if (fft_dominant_bpm is not None and hr_bpm is not None
            and fft_report.get('snr_ratio', 0) > 5.0
            and (fft_dominant_bpm - hr_bpm) > 15.0):
        print(f"  Step 8 override: {hr_bpm:.1f} → "
              f"{fft_dominant_bpm:.1f} BPM (FFT cross-check)")
        hr_bpm  = fft_dominant_bpm
        peak_hz = fft_dominant_bpm / 60.0

    peak_indices, peak_times, rr_intervals, rr_tolerance = \
        detect_beat_peaks(filtered, fps_measured,
                          hr_bpm_fft=hr_bpm, profile=profile,
                          signal_quality=None)
    peak_quality_ok, peak_report = print_peak_report(
        hr_bpm, peak_hz, peak_power, rr_intervals, peak_times,
        n_peaks, n_harmonics, rr_tolerance=rr_tolerance
    )
    publish_step(8, {
        "hr_bpm_step8": round(hr_bpm, 1) if hr_bpm else None,
        "n_peaks":      len(peak_indices),
        "modality":     modality,
    })
    publish_status("running", 72,
        f"Peaks detected: {len(peak_indices)} beats")

    # ── Step 9 — first pass ───────────────────────
    print(f"\n{'='*50}\nSTEP 9 — HR & HRV ({modality})\n{'='*50}")
    hr_results_pass1 = compute_hr_hrv(
        hr_bpm_fft=hr_bpm, rr_intervals=rr_intervals,
        peak_times=peak_times,
        snr_ratio=fft_report.get('snr_ratio', 0),
        quality_ok=quality_ok, fps=fps_measured,
        profile=profile
    )

    # ── Step 11 ───────────────────────────────────
    print(f"\n{'='*50}\nSTEP 11 — Signal quality ({modality})\n{'='*50}")
    snr_score, snr_db, route_palm, quality_level, snr_report = \
        compute_snr_score(
            filtered, freqs, power,
            hr_bpm, rr_intervals, fps_measured,
            hr_confidence=hr_results_pass1['confidence'],
            profile=profile,
            hr_reliable=hr_results_pass1.get('hr_reliable', True)
        )
    print_snr_report(snr_score, snr_db, route_palm,
                     quality_level, snr_report)
    publish_routing(
        route_palm=route_palm,
        reason=(
            "Signal too weak for HRV"
            if snr_report.get("std_floor_triggered")
            else ("SNR below threshold" if route_palm
                  else "Face accepted")
        ),
        snr_score=snr_score,
        std_floor_triggered=snr_report.get(
            "std_floor_triggered", False),
    )
    publish_status("running", 82,
                   "Quality scored — refining peaks")

    # ── Step 8 — second pass ──────────────────────
    print(f"\nStep 8 second pass — quality={snr_score:.3f}")
    _, _, rr_intervals, rr_tolerance = detect_beat_peaks(
        filtered, fps_measured,
        hr_bpm_fft=hr_bpm, profile=profile,
        signal_quality=snr_score
    )

    # ── Step 9 — final pass ───────────────────────
    hr_results = compute_hr_hrv(
        hr_bpm_fft=hr_bpm, rr_intervals=rr_intervals,
        peak_times=peak_times,
        snr_ratio=fft_report.get('snr_ratio', 0),
        quality_ok=quality_ok, fps=fps_measured,
        profile=profile
    )
    print_hr_hrv_report(hr_results)

    publish_step(9, {
        "hr_bpm":           hr_results.get("final_hr"),
        "rmssd":            hr_results.get("rmssd"),
        "hrv_overall":      hr_results.get("hrv_overall"),
        "hrv_available":    hr_results.get("hrv_available", False),
        "hrv_fps_message":  hr_results.get("hrv_fps_message", ""),
        "confidence":       hr_results.get("confidence"),
        "confidence_level": hr_results.get("confidence_level"),
        "hr_reliable":      hr_results.get("hr_reliable", True),
        "hr_agreement":     hr_results.get("hr_agreement"),
        "modality":         modality,
    })
    publish_status("running", 90,
        f"HR: {hr_results.get('final_hr')} BPM")

    return (hr_results, rr_bpm, snr_score, snr_db,
            route_palm, quality_level, snr_report)


# ─────────────────────────────────────────
# Summary printer helpers
# ─────────────────────────────────────────

def print_rmssd_lines(hr_results, snr_report):
    if hr_results.get('hrv_available', False):
        rmssd_line = (
            f"{hr_results['rmssd']} ms"
            if hr_results.get('rmssd') is not None
            else "Insufficient beats"
        )
        if snr_report.get('std_floor_triggered'):
            rmssd_line += "  ⚠ LOW CONFIDENCE"
        if hr_results.get('hrv_confidence') == "low":
            rmssd_line += "  ⚠ LOW CONFIDENCE (60fps)"
        print(f"RMSSD:         {rmssd_line}")
        print(f"HRV:           {hr_results.get('hrv_overall', '—')}")
    else:
        print(f"RMSSD:         ✗ Not reported")
        msg = hr_results.get('hrv_fps_message',
                             'Requires 60fps+ camera')
        if not msg.endswith(')'):
            msg += ')'
        print(f"HRV:           {msg}")


def print_hr_line(hr_results):
    hr_flag = ""
    if not hr_results.get('hr_reliable', True):
        hr_flag = (
            f"  ⚠ LOW CONFIDENCE "
            f"(FFT/RR disagreement "
            f"±{hr_results.get('hr_agreement', '?')} BPM)"
        )
    print(f"Heart Rate:    {hr_results['final_hr']} BPM{hr_flag}")


# ─────────────────────────────────────────
# Palm pipeline runner
# ─────────────────────────────────────────

def run_palm_pipeline(cap, actual_fps,
                       face_state=None, face_profile=None,
                       palm_mode=False, logger=None,
                       show_display=True):
    """Run full palm pipeline: Step 2b → 3b → 4–11."""
    from step3_signal_extraction.step3_palm_signal import (
        PalmROIState,
        run_palm_roi_extraction,
        run_palm_signal_extraction,
    )

    if palm_mode:
        print(f"\n{'='*50}\nPALM MODE — Direct palm measurement\n{'='*50}")
        publish_status("running", 10, "Palm mode — show your palm")
    else:
        print(f"\n{'='*50}\nROUTING TO PALM\n{'='*50}")
        publish_status("running", 55, "Routing to palm measurement")

    print("Please show your palm flat to the camera.")
    publish("modality_switch", {
        "modality":  "palm",
        "palm_mode": palm_mode,
    })

    # ── Step 2b ───────────────────────────────────
    print(f"\n{'='*50}\nSTEP 2b — Palm ROI + Calibration\n{'='*50}")
    palm_state   = PalmROIState()
    palm_profile = run_palm_roi_extraction(
        cap, actual_fps, palm_state,
        duration_sec=10,
        show_display=show_display
    )

    if palm_state.combined_mask is None:
        measurement_failed(
            cap,
            reason=(
                "Palm not detected during setup. "
                "The system could not establish a "
                "measurement region on your palm."
            ),
            suggestions=[
                "Hold your palm open and flat, facing the camera",
                "Keep your hand centered in the frame",
                "Ensure your palm is well-lit",
                "Move your hand closer to the camera",
                "Try your other hand if the problem persists",
            ],
            logger=logger
        )

    publish_step(2, {
        "ita":           round(palm_state.ita, 1),
        "fitzpatrick":   palm_profile.fitzpatrick,
        "profile_valid": palm_profile.is_valid,
        "hr_estimate":   palm_profile.hr_estimate_bpm,
        "modality":      "palm",
    })
    if logger is not None:
        logger.set_profile(palm_profile)

    # ── Step 3b ───────────────────────────────────
    print(f"\n{'='*50}\nSTEP 3b — Palm RGB signal\n{'='*50}")

    publish_recording_progress(
        progress=0,
        message="Recording palm signal",
        active=True,
    )

    palm_frame_cb = make_frame_callback(
        mask_color=(0, 200, 255)   # cyan for palm
    )
    palm_progress_cb = make_recording_progress_callback('palm')

    r, g, b, fps_measured, quality_ok, issues, _ = \
        run_palm_signal_extraction(
            cap, actual_fps, palm_state,
            duration_sec=35, profile=palm_profile,
            on_frame=palm_frame_cb,
            on_progress=palm_progress_cb,
            show_display=show_display
        )

    try:
        publish_recording_progress(
            progress=100,
            message="",
            active=False,
        )
    except Exception:
        pass

    if logger is not None:
        logger.set_fps(fps_measured)

    if not quality_ok:
        print("Palm signal quality issues:")
        for issue in issues:
            print(f"  - {issue}")

    publish_step(3, {
        "quality_ok": quality_ok,
        "fps":        round(fps_measured, 1),
        "modality":   "palm",
    })
    publish_status("running", 30 if palm_mode else 70,
                   "Palm signal captured")

    # ── Steps 4–11 (palm) ─────────────────────────
    (hr_results, rr_bpm, snr_score, snr_db,
     route_palm_again, quality_level, snr_report) = \
        run_steps_4_to_11(
            r, g, b, fps_measured,
            quality_ok, palm_profile, modality="palm"
        )

    if logger is not None:
        logger.set_palm_results(
            hr_results=hr_results,
            snr_score=snr_score,
            rr_bpm=rr_bpm,
        )

    return (hr_results, rr_bpm, snr_score, snr_db,
            route_palm_again, quality_level, snr_report,
            palm_state, palm_profile)


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("="*50)
    print("rPPG Pipeline — PulseRoute")
    print("="*50)
    print(f"Mode:       {PIPELINE_MODE}")
    print(f"Patient ID: {PATIENT_ID or '(not set)'}")
    print(f"Patient age: {PATIENT_AGE or '(unknown)'}")
    print(f"Headless:   {SHOW_DISPLAY is False}")

    publish_status("starting", 0, "Initialising pipeline")

    # ── Step 1 ────────────────────────────────────
    print("\nSTEP 1 — Camera initialisation")
    cap, actual_fps = initialize_camera(camera_index=0)
    if cap is None:
        publish("measurement_failed", {
            "status":  "failed",
            "message": "Camera failed to open",
        })
        sys.exit(1)

    publish_step(1, {"fps": round(actual_fps, 1)})
    publish_status("running", 5, "Camera ready")

    # ── Session logger ─────────────────────────────
    logger = SessionLogger()
    logger.set_patient_id(PATIENT_ID)
    logger.set_mode(PIPELINE_MODE)
    logger.set_patient_age(PATIENT_AGE)
    logger.set_fps(actual_fps)

    _CAP = cap
    _LOGGER = logger

    signal.signal(signal.SIGINT, _cleanup_and_exit)
    signal.signal(signal.SIGTERM, _cleanup_and_exit)

    palm_mode = (PIPELINE_MODE == "palm")

    # ══════════════════════════════════════════════
    # PALM MODE — direct
    # ══════════════════════════════════════════════
    if palm_mode:

        (hr_results, rr_bpm, snr_score, snr_db,
         route_palm_again, quality_level, snr_report,
         palm_state, palm_profile) = run_palm_pipeline(
            cap, actual_fps,
            face_state=None, face_profile=None,
            palm_mode=True, logger=logger,
            show_display=SHOW_DISPLAY
        )
        logger.set_routing(None, 'direct_palm_mode')

        if route_palm_again:
            measurement_failed(
                cap,
                reason=(
                    f"Palm signal too weak (SNR {snr_score:.3f}). "
                    "Improve lighting and try again."
                ),
                suggestions=[
                    "Move closer to a lamp or face a window",
                    "Ensure palm is flat, centered, fully visible",
                    "Avoid strong backlight",
                    "Clean the camera lens",
                ],
                logger=logger
            )

        print(f"\n{'='*50}\nPALM MEASUREMENT COMPLETE\n{'='*50}")
        print(f"Modality:  PALM (direct)")
        print_hr_line(hr_results)
        print(f"Respiratory: {rr_bpm if rr_bpm else 'N/A'} BrPM")
        print_rmssd_lines(hr_results, snr_report)
        print(f"SNR Score: {snr_score:.4f} ({quality_level.upper()})")

        publish_final(
            hr_results=hr_results, snr_score=snr_score,
            quality_level=quality_level, route_palm=False,
            ita=palm_state.ita, fitzpatrick=palm_profile.fitzpatrick,
            rr_bpm=rr_bpm, snr_report=snr_report,
        )
        publish_status("complete", 100, "Measurement complete")
        logger.set_final("palm", hr_results, rr_bpm)
        logger.write()

    # ══════════════════════════════════════════════
    # AUTO MODE — face first
    # ══════════════════════════════════════════════
    else:

        # ── Step 2 — Face ROI + Calibration ───────
        print(f"\n{'='*50}\nSTEP 2 — Face ROI + Calibration\n{'='*50}")
        face_state = FaceROIState()
        profile    = run_face_roi_extraction(
            cap, actual_fps, face_state,
            duration_sec=10,
            show_display=SHOW_DISPLAY
        )

        if face_state.combined_mask is None:
            measurement_failed(
                cap,
                reason="Face not detected during setup.",
                suggestions=[
                    "Ensure your face is centered and fully visible",
                    "Move closer to the camera",
                    "Improve lighting — avoid strong backlight",
                    "Remove glasses or objects blocking your face",
                ],
                logger=logger
            )

        publish_step(2, {
            "ita":           round(face_state.ita, 1),
            "fitzpatrick":   profile.fitzpatrick,
            "profile_valid": profile.is_valid,
            "hr_estimate":   profile.hr_estimate_bpm,
            "modality":      "face",
        })
        publish_status("running", 15, "Face detected — calibrating")
        logger.set_profile(profile, face_state)

        # ── Step 3 — Face RGB signal ───────────────
        print(f"\n{'='*50}\nSTEP 3 — RGB signal (face)\n{'='*50}")

        publish_recording_progress(
            progress=0,
            message="Recording face signal",
            active=True,
        )

        face_frame_cb = make_frame_callback(
            mask_color=(0, 220, 0)   # green for face
        )
        face_progress_cb = make_recording_progress_callback('face')

        r, g, b, fps_measured, quality_ok, issues, _ = \
            run_face_signal_extraction(
                cap, actual_fps, face_state,
                duration_sec=35, profile=profile,
                on_frame=face_frame_cb,
                on_progress=face_progress_cb,
                show_display=SHOW_DISPLAY
            )

        try:
            publish_recording_progress(
                progress=100,
                message="",
                active=False,
            )
        except Exception:
            pass

        logger.set_fps(fps_measured)

        if not quality_ok:
            print("Signal quality issues:")
            for issue in issues:
                print(f"  - {issue}")

        publish_step(3, {
            "quality_ok": quality_ok,
            "fps":        round(fps_measured, 1),
            "modality":   "face",
        })
        publish_status("running", 28, "Face signal captured")

        # ── Steps 4–11 (face) ─────────────────────
        (hr_results, rr_bpm, snr_score, snr_db,
         route_palm, quality_level, snr_report) = \
            run_steps_4_to_11(
                r, g, b, fps_measured,
                quality_ok, profile, modality="face"
            )

        logger.set_face_results(
            hr_results=hr_results, snr_score=snr_score,
            quality_level=quality_level, rr_bpm=rr_bpm,
        )
        routing_reason = ''
        if route_palm:
            routing_reason = (
                'std_floor'
                if snr_report.get('std_floor_triggered')
                else 'snr_score'
            )
        logger.set_routing(route_palm, routing_reason)

        print(f"\n{'='*50}\nFACE MEASUREMENT COMPLETE\n{'='*50}")
        print_hr_line(hr_results)
        print(f"Respiratory: {rr_bpm if rr_bpm else 'N/A'} BrPM")
        print_rmssd_lines(hr_results, snr_report)
        print(f"SNR: {snr_score:.4f} ({quality_level.upper()})")
        print(f"Routing: "
              f"{'⚠ PALM' if route_palm else '✓ FACE ACCEPTED'}")

        # ── Face accepted ──────────────────────────
        if not route_palm:
            publish_final(
                hr_results=hr_results, snr_score=snr_score,
                quality_level=quality_level, route_palm=False,
                ita=face_state.ita, fitzpatrick=profile.fitzpatrick,
                rr_bpm=rr_bpm, snr_report=snr_report,
            )
            publish_status("complete", 100, "Measurement complete")
            logger.set_final("face", hr_results, rr_bpm)
            logger.write()

        # ── Route to palm ──────────────────────────
        else:
            publish_routing(
                route_palm=True, reason=routing_reason,
                snr_score=snr_score,
                std_floor_triggered=snr_report.get(
                    'std_floor_triggered', False),
            )
            (hr_results, rr_bpm, snr_score, snr_db,
             route_palm_again, quality_level, snr_report,
             palm_state, palm_profile) = run_palm_pipeline(
                cap, actual_fps,
                face_state=face_state, face_profile=profile,
                palm_mode=False, logger=logger,
                show_display=SHOW_DISPLAY
            )

            if route_palm_again:
                measurement_failed(
                    cap,
                    reason=(
                        "Both face and palm signals too weak. "
                        f"Palm SNR: {snr_score:.3f} (need ≥ 0.45)."
                    ),
                    suggestions=[
                        "Improve lighting",
                        "Move closer to a lamp or face a window",
                        "Ensure palm is flat, centered, visible",
                        "Try again in a brighter room",
                    ],
                    logger=logger
                )

            print(f"\n{'='*50}\nPALM MEASUREMENT COMPLETE\n{'='*50}")
            print(f"Modality: PALM (auto-routed)")
            print_hr_line(hr_results)
            print(f"Respiratory: {rr_bpm if rr_bpm else 'N/A'} BrPM")
            print_rmssd_lines(hr_results, snr_report)
            print(f"SNR: {snr_score:.4f} ({quality_level.upper()})")

            publish_final(
                hr_results=hr_results, snr_score=snr_score,
                quality_level=quality_level, route_palm=True,
                ita=palm_state.ita,
                fitzpatrick=palm_profile.fitzpatrick,
                rr_bpm=rr_bpm, snr_report=snr_report,
            )
            publish_status("complete", 100, "Measurement complete")
            logger.set_final("palm", hr_results, rr_bpm)
            logger.write()

    print(f"\nReady for Step 12 (display)")
    cap.release()