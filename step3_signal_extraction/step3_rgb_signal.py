import numpy as np
import cv2
import time
import os
import sys
from collections import deque

# ─────────────────────────────────────────
# Step 3 — RGB Signal Extraction
#
# Purpose: Convert each video frame into three
# numbers (mean R, G, B) from the ROI pixels.
# Accumulate over N seconds to build the raw
# pulse waveform for POS algorithm in Step 5.
#
# AI contribution: ITA-adaptive quality filter
# adjusts pixel quality thresholds per subject
# based on skin tone — improves SNR for FST V-VI
# without degrading FST I-II performance.
#
# NEW — Motion rejection (MotionDetector):
#   Frames where the green channel deviates more
#   than the patient's personal threshold from
#   the rolling mean are silently skipped.
#
# NEW — on_frame callback:
#   Optional callback(frame_bgr, combined_mask)
#   called on every accepted clean frame.
#   Used by run_web.py to publish annotated JPEG
#   frames to the patient's live camera feed via
#   WebSocket. Throttling is handled by the caller.
# ─────────────────────────────────────────

# Default signal buffer duration in seconds
SIGNAL_DURATION_SEC = 30

# Minimum pixels required for valid extraction
MIN_ROI_PIXELS = 50

# Rolling window for motion baseline (frames)
MOTION_WINDOW_SIZE = 10

# Max recording time multiplier
MAX_DURATION_MULTIPLIER = 2.5


# ─────────────────────────────────────────
# ITA-adaptive quality thresholds
# ─────────────────────────────────────────

def compute_adaptive_thresholds(ita_angle):
    """
    Compute pixel quality thresholds adapted to
    subject skin tone via ITA angle.
    """
    ita_clamped    = max(-60.0, min(90.0, float(ita_angle)))
    ita_normalized = (ita_clamped + 60.0) / 150.0

    overexposure_threshold  = int(200 + ita_normalized * 40)
    underexposure_threshold = int(60 - ita_normalized * 30)

    return overexposure_threshold, underexposure_threshold


# ─────────────────────────────────────────
# Motion Detector
# ─────────────────────────────────────────

class MotionDetector:
    """
    Per-frame motion artifact detector.

    Uses the patient's personal motion threshold
    from SubjectProfile (5× their own baseline
    green channel noise floor).
    """

    def __init__(self, profile=None):
        if profile is not None and \
           profile.motion_threshold is not None:
            self.threshold = profile.motion_threshold
            self._source   = "personal"
        else:
            self.threshold = 6.0
            self._source   = "default"

        self.window   = deque(maxlen=MOTION_WINDOW_SIZE)
        self.rejected = 0
        self.accepted = 0

    def is_clean(self, mean_g):
        if len(self.window) < 3:
            self.window.append(mean_g)
            self.accepted += 1
            return True

        rolling_mean = float(np.mean(self.window))
        delta        = abs(mean_g - rolling_mean)

        if delta > self.threshold:
            self.rejected += 1
            return False

        self.window.append(mean_g)
        self.accepted += 1
        return True

    def get_rejection_rate(self):
        total = self.rejected + self.accepted
        if total == 0:
            return 0.0
        return round(self.rejected / total, 3)

    def print_summary(self):
        rate = self.get_rejection_rate()
        print(f"  Motion detector:  "
              f"threshold={self.threshold:.4f} "
              f"({self._source})  "
              f"rejected={self.rejected}  "
              f"rate={rate:.1%}")


# ─────────────────────────────────────────
# Core extraction function
# ─────────────────────────────────────────

def extract_rgb_signal(frame_rgb, mask, ita_angle=55.0):
    """Extract mean R, G, B from ROI pixels."""
    roi_pixels = frame_rgb[mask == 1]

    if len(roi_pixels) < MIN_ROI_PIXELS:
        return 0.0, 0.0, 0.0, 0, 0.0, (240, 30)

    over_thresh, under_thresh = \
        compute_adaptive_thresholds(ita_angle)

    not_overexposed  = np.all(
        roi_pixels < over_thresh, axis=1
    )
    not_underexposed = np.any(
        roi_pixels > under_thresh, axis=1
    )

    quality_filter = not_overexposed & not_underexposed
    quality_ratio  = float(quality_filter.sum()) / \
                     len(roi_pixels)

    min_threshold = max(
        MIN_ROI_PIXELS,
        int(len(roi_pixels) * 0.3)
    )

    if quality_filter.sum() >= min_threshold:
        pixels_to_use = roi_pixels[quality_filter]
    else:
        pixels_to_use = roi_pixels
        quality_ratio = 0.0

    means  = np.mean(pixels_to_use, axis=0)
    mean_r = float(means[0])
    mean_g = float(means[1])
    mean_b = float(means[2])

    return (
        mean_r, mean_g, mean_b,
        len(pixels_to_use),
        round(quality_ratio, 3),
        (over_thresh, under_thresh)
    )


# ─────────────────────────────────────────
# Signal buffer
# ─────────────────────────────────────────

class RGBSignalBuffer:
    def __init__(self, actual_fps,
                 duration_sec=SIGNAL_DURATION_SEC):
        self.actual_fps   = actual_fps
        self.duration_sec = duration_sec
        self.buffer_size  = int(actual_fps * duration_sec)

        self.r_signal       = []
        self.g_signal       = []
        self.b_signal       = []
        self.timestamps     = []
        self.pixel_counts   = []
        self.quality_ratios = []
        self.ita_values     = []
        self.thresholds     = []

        print(f"Signal buffer: {self.buffer_size} frames "
              f"({duration_sec}s @ {actual_fps:.1f}fps)")

    def add_frame(self, mean_r, mean_g, mean_b,
                  pixel_count, quality_ratio,
                  ita_angle=55.0, thresholds=(240, 30)):
        if len(self.r_signal) >= self.buffer_size:
            return

        self.r_signal.append(mean_r)
        self.g_signal.append(mean_g)
        self.b_signal.append(mean_b)
        self.timestamps.append(time.time())
        self.pixel_counts.append(pixel_count)
        self.quality_ratios.append(quality_ratio)
        self.ita_values.append(ita_angle)
        self.thresholds.append(thresholds)

    def is_full(self):
        return len(self.r_signal) >= self.buffer_size

    def get_signals(self):
        return (
            np.array(self.r_signal),
            np.array(self.g_signal),
            np.array(self.b_signal),
            np.array(self.timestamps)
        )

    def get_actual_fps(self):
        if len(self.timestamps) < 2:
            return self.actual_fps
        total_time = self.timestamps[-1] - \
                     self.timestamps[0]
        if total_time == 0:
            return self.actual_fps
        return (len(self.timestamps) - 1) / total_time

    def get_progress(self):
        n        = len(self.r_signal)
        progress = n / self.buffer_size
        elapsed  = n / self.actual_fps
        return round(progress, 3), round(elapsed, 1)

    def get_mean_quality(self):
        if len(self.quality_ratios) == 0:
            return 0.0
        return round(float(np.mean(self.quality_ratios)), 3)

    def get_mean_ita(self):
        if len(self.ita_values) == 0:
            return 0.0
        return round(float(np.mean(self.ita_values)), 1)

    def get_threshold_summary(self):
        if len(self.thresholds) == 0:
            return {}
        over_vals  = [t[0] for t in self.thresholds]
        under_vals = [t[1] for t in self.thresholds]
        return {
            "overexposure_mean":  round(np.mean(over_vals), 1),
            "overexposure_std":   round(np.std(over_vals), 2),
            "underexposure_mean": round(np.mean(under_vals), 1),
            "underexposure_std":  round(np.std(under_vals), 2),
            "mean_ita":           self.get_mean_ita()
        }

    def reset(self):
        self.r_signal       = []
        self.g_signal       = []
        self.b_signal       = []
        self.timestamps     = []
        self.pixel_counts   = []
        self.quality_ratios = []
        self.ita_values     = []
        self.thresholds     = []


# ─────────────────────────────────────────
# Signal quality assessment
# ─────────────────────────────────────────

def assess_signal_quality(buffer):
    issues = []
    r, g, b, t = buffer.get_signals()

    if len(g) == 0:
        return False, ["No signal collected"], 0.0, {}

    min_frames = int(buffer.actual_fps * 10)
    if len(g) < min_frames:
        issues.append(
            f"Insufficient data: {len(g)} frames "
            f"(need {min_frames})"
        )

    mean_quality = buffer.get_mean_quality()
    if mean_quality < 0.5:
        issues.append(
            f"Low pixel quality: {mean_quality:.2f}"
        )

    g_std = float(np.std(g))
    if g_std < 0.05:
        issues.append(
            f"Flat green signal: std={g_std:.4f} "
            f"(no pulse detected)"
        )

    g_mean = float(np.mean(g))
    if g_mean > 250 or g_mean < 5:
        issues.append(
            f"Signal clipping: mean_g={g_mean:.1f}"
        )

    snr_estimate      = (g_std / g_mean * 100) \
                        if g_mean > 0 else 0.0
    threshold_summary = buffer.get_threshold_summary()
    quality_ok        = len(issues) == 0

    return (
        quality_ok, issues,
        round(snr_estimate, 3), threshold_summary
    )


# ─────────────────────────────────────────
# Main extraction runner
# ─────────────────────────────────────────

def run_signal_extraction(cap, actual_fps,
                          get_mask_fn,
                          get_ita_fn=None,
                          modality="face",
                          duration_sec=30,
                          profile=None,
                          on_frame=None):   # ← NEW
    """
    Run RGB signal extraction for duration_sec seconds.

    Parameters
    ----------
    cap          : VideoCapture from Step 1
    actual_fps   : measured fps from Step 1
    get_mask_fn  : callable → (combined, fore, cheek) masks
    get_ita_fn   : callable → ITA float
    modality     : 'face' or 'palm'
    duration_sec : target clean recording seconds
    profile      : SubjectProfile (optional)
    on_frame     : optional callback(frame_bgr, combined_mask)
                   called on every accepted clean frame.
                   Used by run_web.py to publish annotated
                   JPEG frames to the patient's live camera
                   feed via WebSocket. Throttling handled
                   by the caller (make_frame_callback).

    Returns
    -------
    r, g, b        : NumPy signal arrays (clean frames only)
    fps_measured   : actual fps during recording
    quality_ok     : bool
    issues         : list of issue strings
    thresh_summary : adaptive threshold stats
    """
    buffer   = RGBSignalBuffer(actual_fps,
                               duration_sec=duration_sec)
    detector = MotionDetector(profile=profile)

    max_wall_time = duration_sec * MAX_DURATION_MULTIPLIER
    wall_deadline = time.time() + max_wall_time

    current_ita = 55.0
    motion_on   = profile is not None

    print(f"\nStarting {modality} signal extraction...")
    print(f"Target: {duration_sec}s of clean signal")
    if motion_on:
        print(f"Motion rejection: ON  "
              f"(threshold={detector.threshold:.4f})")
    else:
        print(f"Motion rejection: OFF  (no profile)")
    print("Press Q to stop early")

    consecutive_failures = 0

    while not buffer.is_full():
        if time.time() > wall_deadline:
            print(f"\nMax recording time reached "
                  f"({max_wall_time:.0f}s). Stopping.")
            break

        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures >= 30:
                print(f"\n⚠ Camera disconnected — "
                      f"30 consecutive read failures.")
                break
            continue
        consecutive_failures = 0

        frame_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        if get_ita_fn is not None:
            current_ita = get_ita_fn()

        combined_mask, forehead_mask, cheek_mask = \
            get_mask_fn(frame_rgb)

        if combined_mask is None or \
           np.sum(combined_mask) < MIN_ROI_PIXELS:
            continue

        mean_r, mean_g, mean_b, \
        pixel_count, quality_ratio, \
        thresholds_used = extract_rgb_signal(
            frame_rgb, combined_mask, current_ita
        )

        if mean_g == 0.0:
            continue

        # ── Motion rejection ──────────────────────
        is_clean_frame = True
        if motion_on:
            is_clean_frame = detector.is_clean(mean_g)

        if not is_clean_frame:
            _draw_rejected_hud(
                frame, buffer, duration_sec,
                detector, current_ita,
                forehead_mask, cheek_mask
            )
            cv2.imshow(
                f"Signal Extraction — {modality.upper()}",
                frame
            )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ── Accept frame ──────────────────────────
        buffer.add_frame(
            mean_r, mean_g, mean_b,
            pixel_count, quality_ratio,
            current_ita, thresholds_used
        )

        # ── Live frame callback ───────────────────
        # Called on every accepted clean frame.
        # run_web.py's make_frame_callback() throttles
        # to every 3rd frame (~10fps) before publishing.
        if on_frame is not None:
            try:
                on_frame(frame, combined_mask)
            except Exception:
                pass   # never crash pipeline for a frame

        # ── Display ───────────────────────────────
        progress, elapsed = buffer.get_progress()
        remaining         = duration_sec - elapsed
        display           = frame.copy()

        if forehead_mask is not None and \
           np.sum(forehead_mask) > 0:
            colored              = np.zeros_like(display)
            colored[forehead_mask == 1] = (0, 255, 0)
            cv2.addWeighted(
                colored, 0.4, display, 0.6, 0, display
            )

        if cheek_mask is not None and \
           np.sum(cheek_mask) > 0:
            colored            = np.zeros_like(display)
            colored[cheek_mask == 1] = (255, 0, 0)
            cv2.addWeighted(
                colored, 0.4, display, 0.6, 0, display
            )

        y = [30]
        def put(text, color=(0, 255, 0), scale=0.55):
            cv2.putText(display, text, (10, y[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 2)
            y[0] += 26

        put(f"Modality: {modality.upper()}")
        put(f"Clean signal: {elapsed:.1f}s / "
            f"{duration_sec}s  ({remaining:.1f}s left)",
            (0, 255, 255))
        put(f"Frames: {len(buffer.r_signal)} / "
            f"{buffer.buffer_size}")
        put(f"G signal: {mean_g:.2f}  "
            f"Pixels: {pixel_count}")
        put(
            f"Quality: {quality_ratio:.2f}  "
            f"ITA: {current_ita:.1f}",
            (0, 255, 0) if quality_ratio > 0.7
            else (0, 165, 255)
        )

        if motion_on:
            rate = detector.get_rejection_rate()
            put(
                f"Rejected: {detector.rejected} frames "
                f"({rate:.0%} artifacts)",
                (0, 200, 255) if rate < 0.2
                else (0, 100, 255)
            )

        put(
            f"Thresholds  over:{thresholds_used[0]}"
            f"  under:{thresholds_used[1]}",
            (200, 200, 0)
        )

        bar_w   = int(progress * (frame.shape[1] - 20))
        bar_col = (0, 255, 0) if quality_ratio > 0.7 \
                  else (0, 165, 255)
        cv2.rectangle(
            display,
            (10, frame.shape[0] - 30),
            (10 + bar_w, frame.shape[0] - 10),
            bar_col, -1
        )
        cv2.rectangle(
            display,
            (10, frame.shape[0] - 30),
            (frame.shape[1] - 10, frame.shape[0] - 10),
            (100, 100, 100), 2
        )

        cv2.imshow(
            f"Signal Extraction — {modality.upper()}",
            display
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    quality_ok, issues, snr, thresh_summary = \
        assess_signal_quality(buffer)

    r, g, b, t   = buffer.get_signals()
    fps_measured = buffer.get_actual_fps()

    print(f"\nExtraction complete:")
    print(f"  Clean frames:   {len(g)}")
    print(f"  Actual fps:     {fps_measured:.1f}")
    print(f"  Mean quality:   {buffer.get_mean_quality():.3f}")
    print(f"  Mean ITA:       {buffer.get_mean_ita():.1f}")
    print(f"  SNR estimate:   {snr:.3f}%")
    print(f"  Quality OK:     {quality_ok}")
    if motion_on:
        detector.print_summary()
    if thresh_summary:
        print(f"  Thresholds:     "
              f"over={thresh_summary['overexposure_mean']:.0f}"
              f"  under="
              f"{thresh_summary['underexposure_mean']:.0f}")
    for issue in issues:
        print(f"  Issue: {issue}")

    return r, g, b, fps_measured, \
           quality_ok, issues, thresh_summary


def _draw_rejected_hud(frame, buffer, duration_sec,
                        detector, current_ita,
                        forehead_mask, cheek_mask):
    progress, elapsed = buffer.get_progress()

    red_overlay        = np.zeros_like(frame)
    red_overlay[:,:,2] = 60
    cv2.addWeighted(red_overlay, 0.3, frame, 0.7, 0, frame)

    cv2.putText(
        frame,
        "MOTION DETECTED — waiting...",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 0, 255), 2
    )
    cv2.putText(
        frame,
        f"Clean signal so far: {elapsed:.1f}s / "
        f"{duration_sec}s",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (0, 200, 255), 2
    )
    cv2.putText(
        frame,
        f"Artifacts rejected: {detector.rejected}",
        (10, 86), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (100, 100, 255), 1
    )