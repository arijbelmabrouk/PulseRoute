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
# Accumulate over 30 seconds to build the raw
# pulse waveform for POS algorithm in Step 5.
#
# AI contribution: ITA-adaptive quality filter
# adjusts pixel quality thresholds per subject
# based on skin tone — improves SNR for FST V-VI
# without degrading FST I-II performance.
#
# No neural network — pure NumPy mathematics
# informed by skin tone from Step 2.
# ─────────────────────────────────────────

# Signal buffer duration in seconds
SIGNAL_DURATION_SEC = 30

# Minimum pixels required for valid extraction
MIN_ROI_PIXELS = 50


# ─────────────────────────────────────────
# ITA-adaptive quality thresholds
# ─────────────────────────────────────────

def compute_adaptive_thresholds(ita_angle):
    """
    Compute pixel quality thresholds adapted to
    subject skin tone via ITA angle.

    Scientific basis:
        Melanin absorbs light before it reaches
        capillaries — darker skin has lower baseline
        pixel values for the same lighting conditions.
        Fixed thresholds designed for light skin
        misclassify dark skin pixels as underexposed
        and overexposed at different absolute values.

    Threshold adaptation logic:

        Overexposure threshold:
            Light skin (high ITA) → 240 (standard)
            Dark skin  (low ITA)  → 200 (tighter)
            Why: specular reflection on dark skin
            corrupts signal at lower absolute values
            than on light skin. A value of 220 on
            dark skin may represent saturation while
            on light skin it is normal.

        Underexposure threshold:
            Light skin (high ITA) → 30 (standard)
            Dark skin  (low ITA)  → 50 (higher)
            Why: dark skin pixels have lower baseline
            values — a pixel at value 35 on dark skin
            may be well-lit while on light skin it
            indicates shadow. Raising the threshold
            filters more aggressively for dark skin
            to remove shadow-contaminated pixels.

    ITA scale:
        >  55 → FST I-II   (Very Light)
        28–55 → FST III    (Light-Medium)
        10–28 → FST IV     (Medium)
       -30–10 → FST V      (Medium-Dark)
        < -30 → FST VI     (Dark)

    Input:
        ita_angle — float, ITA value from Step 2

    Output:
        overexposure_threshold  — int (180-240)
        underexposure_threshold — int (30-60)
    """
    # Normalize ITA to 0-1 range
    # Clamp to expected range [-60, 90]
    ita_clamped    = max(-60.0, min(90.0, float(ita_angle)))
    ita_normalized = (ita_clamped + 60.0) / 150.0  # 0=dark, 1=light

    # Overexposure threshold — tighter for dark skin
    # Light skin: 240 (standard)
    # Dark skin:  200 (tighter — catches specular earlier)
    overexposure_threshold = int(200 + ita_normalized * 40)

    # Underexposure threshold — higher for dark skin
    # Light skin: 30 (standard)
    # Dark skin:  60 (higher — filters shadow pixels)
    underexposure_threshold = int(60 - ita_normalized * 30)

    return overexposure_threshold, underexposure_threshold


# ─────────────────────────────────────────
# Core extraction function
# ─────────────────────────────────────────

def extract_rgb_signal(frame_rgb, mask, ita_angle=55.0):
    """
    Extract mean R, G, B from ROI pixels with
    ITA-adaptive quality filtering.

    Two-stage process:
        1. Adaptive quality filter — removes pixels
           that corrupt the rPPG signal, with
           thresholds adjusted for subject skin tone
        2. Spatial averaging — collapses remaining
           pixels to three scalar mean values

    Quality filter removes:
        Overexposed pixels  — saturated, cannot
            reflect blood flow color changes.
            Threshold adapted to skin tone.
        Underexposed pixels — too dark, dominated
            by noise not blood flow.
            Threshold adapted to skin tone.

    Fallback to unfiltered mean if fewer than 30%
    of pixels pass the filter — prevents signal
    dropout in poor lighting conditions.

    Input:
        frame_rgb  — RGB frame (H, W, 3) uint8
        mask       — binary ROI mask (H, W)
                     1 = inside ROI, 0 = outside
        ita_angle  — ITA skin tone value from Step 2
                     default 55.0 = light skin
                     (safe fallback if not available)

    Output:
        mean_r            — mean red   channel (float)
        mean_g            — mean green channel (float)
        mean_b            — mean blue  channel (float)
        pixel_count       — number of pixels used
        quality_ratio     — fraction of pixels kept
        thresholds_used   — (over, under) tuple
                            for logging and validation
    """
    # Extract ROI pixels via boolean indexing
    # roi_pixels shape: (N, 3)
    roi_pixels = frame_rgb[mask == 1]

    if len(roi_pixels) < MIN_ROI_PIXELS:
        return 0.0, 0.0, 0.0, 0, 0.0, (240, 30)

    # ── Adaptive thresholds from skin tone ────────
    over_thresh, under_thresh = \
        compute_adaptive_thresholds(ita_angle)

    # ── Quality filter ────────────────────────────
    # Overexposed: any channel at or above threshold
    not_overexposed  = np.all(
        roi_pixels < over_thresh, axis=1
    )
    # Underexposed: all channels below threshold
    not_underexposed = np.any(
        roi_pixels > under_thresh, axis=1
    )

    quality_filter = not_overexposed & not_underexposed
    quality_ratio  = float(quality_filter.sum()) / \
                     len(roi_pixels)

    # Apply filter if enough pixels remain
    min_threshold = max(
        MIN_ROI_PIXELS,
        int(len(roi_pixels) * 0.3)
    )

    if quality_filter.sum() >= min_threshold:
        pixels_to_use = roi_pixels[quality_filter]
    else:
        # Fallback — use all pixels
        pixels_to_use = roi_pixels
        quality_ratio = 0.0

    # ── Spatial averaging ─────────────────────────
    means  = np.mean(pixels_to_use, axis=0)
    mean_r = float(means[0])
    mean_g = float(means[1])
    mean_b = float(means[2])

    return (
        mean_r,
        mean_g,
        mean_b,
        len(pixels_to_use),
        round(quality_ratio, 3),
        (over_thresh, under_thresh)
    )


# ─────────────────────────────────────────
# Signal buffer
# ─────────────────────────────────────────

class RGBSignalBuffer:
    """
    Accumulates per-frame RGB mean values over time.

    Stores three signal arrays plus metadata:
        timestamps     — for accurate FFT frequency bins
        pixel_counts   — pixels used per frame
        quality_ratios — filter quality per frame
        ita_values     — skin tone per frame
                         (tracks lighting-induced drift)
        thresholds     — adaptive thresholds used
                         (for validation logging)

    At 30fps × 30s = 900 values per channel.
    Adapts to any fps from Step 1 measurement.
    """

    def __init__(self, actual_fps,
                 duration_sec=SIGNAL_DURATION_SEC):
        """
        Input:
            actual_fps   — measured fps from Step 1
            duration_sec — recording window in seconds
        """
        self.actual_fps   = actual_fps
        self.duration_sec = duration_sec
        self.buffer_size  = int(actual_fps * duration_sec)

        self.r_signal      = []
        self.g_signal      = []
        self.b_signal      = []
        self.timestamps    = []
        self.pixel_counts  = []
        self.quality_ratios = []
        self.ita_values    = []
        self.thresholds    = []

        print(f"Signal buffer: {self.buffer_size} frames "
              f"({duration_sec}s @ {actual_fps:.1f}fps)")

    def add_frame(self, mean_r, mean_g, mean_b,
                  pixel_count, quality_ratio,
                  ita_angle=55.0, thresholds=(240, 30)):
        """
        Add one frame to the buffer.

        Input:
            mean_r, mean_g, mean_b — channel means
            pixel_count            — pixels used
            quality_ratio          — filter ratio
            ita_angle              — skin tone this frame
            thresholds             — (over, under) used
        """
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

    def is_ready(self):
        """True when 10+ seconds collected — FFT minimum."""
        return len(self.r_signal) >= \
               int(self.actual_fps * 10)

    def is_full(self):
        """True when full 30 seconds collected."""
        return len(self.r_signal) >= self.buffer_size

    def get_signals(self):
        """
        Return collected signals as NumPy arrays.
        Feeds directly into Step 4 (normalization)
        and Step 5 (POS algorithm).

        Output:
            r, g, b — channel arrays (N,)
            t       — timestamp array (N,)
        """
        return (
            np.array(self.r_signal),
            np.array(self.g_signal),
            np.array(self.b_signal),
            np.array(self.timestamps)
        )

    def get_actual_fps(self):
        """
        Calculate real fps from timestamps.
        More accurate than Step 1 initial measurement.
        Used for FFT frequency bin calculation in Step 7.
        """
        if len(self.timestamps) < 2:
            return self.actual_fps
        total_time = self.timestamps[-1] - \
                     self.timestamps[0]
        if total_time == 0:
            return self.actual_fps
        return (len(self.timestamps) - 1) / total_time

    def get_progress(self):
        """Recording progress as fraction and seconds."""
        n        = len(self.r_signal)
        progress = n / self.buffer_size
        elapsed  = n / self.actual_fps
        return round(progress, 3), round(elapsed, 1)

    def get_mean_quality(self):
        """Mean quality ratio across all frames."""
        if len(self.quality_ratios) == 0:
            return 0.0
        return round(float(np.mean(self.quality_ratios)), 3)

    def get_mean_ita(self):
        """
        Mean ITA across recording session.
        Used for per-subject skin tone logging
        and Fitzpatrick group assignment in results.
        """
        if len(self.ita_values) == 0:
            return 0.0
        return round(float(np.mean(self.ita_values)), 1)

    def get_threshold_summary(self):
        """
        Summary of adaptive thresholds used.
        For validation — shows how much thresholds
        varied across the recording session.
        Useful for your research methodology section.
        """
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
        """Clear all buffers — start fresh recording."""
        self.r_signal       = []
        self.g_signal       = []
        self.b_signal       = []
        self.timestamps     = []
        self.pixel_counts   = []
        self.quality_ratios = []
        self.ita_values     = []
        self.thresholds     = []
        print("Signal buffer reset")


# ─────────────────────────────────────────
# Signal quality assessment
# ─────────────────────────────────────────

def assess_signal_quality(buffer):
    """
    Assess collected RGB signal quality before
    passing to POS algorithm.

    Four checks:
        1. Enough frames (minimum 10 seconds)
        2. Mean pixel quality ratio > 0.5
        3. Green channel has detectable variance
           (flat = no pulse detected)
        4. Signal not clipping at 0 or 255

    Also returns SNR estimate and adaptive
    threshold summary for validation logging.

    Input:
        buffer — RGBSignalBuffer instance

    Output:
        quality_ok         — bool
        issues             — list of problem strings
        snr_estimate       — rough SNR from green variance
        threshold_summary  — adaptive threshold stats
    """
    issues = []
    r, g, b, t = buffer.get_signals()

    if len(g) == 0:
        return False, ["No signal collected"], 0.0, {}

    # Check 1 — enough data
    min_frames = int(buffer.actual_fps * 10)
    if len(g) < min_frames:
        issues.append(
            f"Insufficient data: {len(g)} frames "
            f"(need {min_frames})"
        )

    # Check 2 — pixel quality
    mean_quality = buffer.get_mean_quality()
    if mean_quality < 0.5:
        issues.append(
            f"Low pixel quality: {mean_quality:.2f}"
        )

    # Check 3 — green channel variance
    g_std = float(np.std(g))
    if g_std < 0.05:
        issues.append(
            f"Flat green signal: std={g_std:.4f} "
            f"(no pulse detected)"
        )

    # Check 4 — clipping
    g_mean = float(np.mean(g))
    if g_mean > 250 or g_mean < 5:
        issues.append(
            f"Signal clipping: mean_g={g_mean:.1f}"
        )

    # SNR estimate from green variance
    snr_estimate = (g_std / g_mean * 100) \
                   if g_mean > 0 else 0.0

    threshold_summary = buffer.get_threshold_summary()
    quality_ok        = len(issues) == 0

    return (
        quality_ok,
        issues,
        round(snr_estimate, 3),
        threshold_summary
    )


# ─────────────────────────────────────────
# Main extraction runner
# ─────────────────────────────────────────

def run_signal_extraction(cap, actual_fps,
                          get_mask_fn,
                          get_ita_fn=None,
                          modality="face"):
    """
    Run RGB signal extraction for 30 seconds.

    Receives camera from Step 1 and mask function
    from Step 2. Works identically for face or palm.

    ITA angle updated each frame via get_ita_fn
    so adaptive thresholds track any drift in
    lighting conditions during the session.

    Input:
        cap        — VideoCapture from Step 1
        actual_fps — measured fps from Step 1
        get_mask_fn — callable returning
                      (combined_mask, fore_mask,
                       cheek_mask) per frame
        get_ita_fn  — callable returning current
                      ITA float (optional —
                      defaults to 55.0 if None)
        modality   — 'face' or 'palm' for display

    Output:
        r, g, b       — NumPy signal arrays
        fps_measured  — actual fps during recording
        quality_ok    — bool
        issues        — list of issue strings
        thresh_summary — adaptive threshold stats
    """
    buffer  = RGBSignalBuffer(actual_fps)
    current_ita = 55.0  # default light skin fallback

    print(f"\nStarting {modality} signal extraction...")
    print(f"Recording {SIGNAL_DURATION_SEC} seconds")
    print("Press Q to stop early")

    while not buffer.is_full():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        # Get current ITA from Step 2 if available
        if get_ita_fn is not None:
            current_ita = get_ita_fn()

        # Get current ROI mask from Step 2
        combined_mask, forehead_mask, cheek_mask = \
            get_mask_fn(frame_rgb)

        if combined_mask is None or \
           np.sum(combined_mask) < MIN_ROI_PIXELS:
            continue

        # Extract RGB with adaptive quality filter
        mean_r, mean_g, mean_b, \
        pixel_count, quality_ratio, \
        thresholds_used = extract_rgb_signal(
            frame_rgb, combined_mask, current_ita
        )

        if mean_g == 0.0:
            continue

        buffer.add_frame(
            mean_r, mean_g, mean_b,
            pixel_count, quality_ratio,
            current_ita, thresholds_used
        )

        # ── Display ───────────────────────────────
        progress, elapsed = buffer.get_progress()
        remaining         = SIGNAL_DURATION_SEC - elapsed
        display           = frame.copy()

        # Draw ROI overlays
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

        # HUD
        y = [30]
        def put(text, color=(0, 255, 0), scale=0.55):
            cv2.putText(display, text, (10, y[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 2)
            y[0] += 26

        put(f"Modality: {modality.upper()}")
        put(f"Time: {elapsed:.1f}s / "
            f"{SIGNAL_DURATION_SEC}s  "
            f"({remaining:.1f}s left)",
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
        put(
            f"Thresholds  over:{thresholds_used[0]}"
            f"  under:{thresholds_used[1]}",
            (200, 200, 0)
        )

        # Progress bar
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
            (frame.shape[1] - 10,
             frame.shape[0] - 10),
            (100, 100, 100), 2
        )

        cv2.imshow(
            f"Signal Extraction — {modality.upper()}",
            display
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Assess signal quality
    quality_ok, issues, snr, thresh_summary = \
        assess_signal_quality(buffer)

    r, g, b, t   = buffer.get_signals()
    fps_measured = buffer.get_actual_fps()

    print(f"\nExtraction complete:")
    print(f"  Frames:       {len(g)}")
    print(f"  Actual fps:   {fps_measured:.1f}")
    print(f"  Mean quality: {buffer.get_mean_quality():.3f}")
    print(f"  Mean ITA:     {buffer.get_mean_ita():.1f}")
    print(f"  SNR estimate: {snr:.3f}%")
    print(f"  Quality OK:   {quality_ok}")
    if thresh_summary:
        print(f"  Thresholds:   "
              f"over={thresh_summary['overexposure_mean']:.0f}"
              f"  under="
              f"{thresh_summary['underexposure_mean']:.0f}")
    for issue in issues:
        print(f"  Issue: {issue}")

    return r, g, b, fps_measured, \
           quality_ok, issues, thresh_summary


# ─────────────────────────────────────────
# Entry point — standalone test
# ─────────────────────────────────────────

"""if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))
    from step1_video_capture.step1_video_captureV3 \
        import initialize_camera

    cap, actual_fps = initialize_camera(camera_index=0)
    if cap is None:
        print("Camera failed")
        sys.exit(1)

    # Dummy mask — center rectangle for standalone test
    # Replace with real Step 2 mask in full pipeline
    def dummy_mask_fn(frame_rgb):
        h, w = frame_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h//4: 3*h//4, w//4: 3*w//4] = 1
        return mask, mask, None

    # Simulate FST VI (dark skin) for testing
    # adaptive thresholds — replace with real ITA
    def dummy_ita_fn():
        return -40.0  # FST VI

    r, g, b, fps, ok, issues, thresh = \
        run_signal_extraction(
            cap, actual_fps,
            dummy_mask_fn,
            dummy_ita_fn,
            modality="test"
        )

    print(f"\nSignal shapes: {r.shape}")
    print(f"Green mean:    {np.mean(g):.2f}")
    print(f"Green std:     {np.std(g):.4f}")
    print(f"Threshold summary: {thresh}")

    cap.release()"""