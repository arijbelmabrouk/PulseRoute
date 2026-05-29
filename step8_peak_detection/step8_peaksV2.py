import numpy as np
from scipy.signal import find_peaks

# ─────────────────────────────────────────
# Step 8 — Peak Detection
#
# Purpose: Find dominant heart rate peak in
# FFT power spectrum with harmonic rejection
# and sub-bin interpolation for precision.
# Also detects individual beat peaks in time
# domain for HRV calculation in Step 9.
#
# Pure functions only.
# No camera, no imports from other steps.
# Called by main.py with FFT output from Step 7
# and filtered pulse from Step 6.
#
# Input:  freqs    — frequency axis Hz (Step 7)
#         power    — power spectrum (Step 7)
#         filtered — filtered pulse (Step 6)
#         fps      — actual fps (Step 1)
#         hr_bpm   — FFT heart rate (Step 7/8)
# Output: hr_bpm   — heart rate in BPM
#         peak_hz  — peak frequency in Hz
#         rr_intervals — beat intervals for HRV
# ─────────────────────────────────────────

HR_MIN_BPM         = 55
HR_MAX_BPM         = 180
HR_MIN_HZ          = HR_MIN_BPM / 60   # 0.800 Hz
HR_MAX_HZ          = HR_MAX_BPM / 60   # 3.000 Hz
HARMONIC_TOLERANCE = 0.05


def interpolate_peak(freqs, power, peak_idx):
    """
    Quadratic interpolation around FFT peak bin
    for sub-bin frequency precision.

    Input:
        freqs    — frequency axis in Hz
        power    — power spectrum
        peak_idx — index of peak bin

    Output:
        refined_freq — interpolated frequency Hz
        refined_bpm  — interpolated frequency BPM
    """
    if peak_idx <= 0 or \
       peak_idx >= len(power) - 1:
        return float(freqs[peak_idx]), \
               float(freqs[peak_idx] * 60)

    y1 = float(power[peak_idx - 1])
    y2 = float(power[peak_idx])
    y3 = float(power[peak_idx + 1])

    denominator = 2.0 * (2.0 * y2 - y1 - y3)
    if abs(denominator) < 1e-10:
        return float(freqs[peak_idx]), \
               float(freqs[peak_idx] * 60)

    delta        = (y3 - y1) / denominator
    bin_spacing  = float(freqs[1] - freqs[0])
    refined_freq = float(freqs[peak_idx]) + \
                   delta * bin_spacing
    refined_bpm  = refined_freq * 60.0

    return round(refined_freq, 5), \
           round(refined_bpm, 2)


def reject_harmonics(candidate_freqs,
                      candidate_powers):
    """
    Remove harmonic AND sub-harmonic peaks.

    Checks both directions:
        Harmonic:     peak at 2x or 3x a lower peak
        Sub-harmonic: peak at 0.5x a higher peak

    Keeps only fundamental frequencies.

    Input:
        candidate_freqs  — peak frequencies in Hz
        candidate_powers — peak powers

    Output:
        fundamental_freqs  — non-harmonic frequencies
        fundamental_powers — corresponding powers
    """
    if len(candidate_freqs) == 0:
        return candidate_freqs, candidate_powers

    sort_idx = np.argsort(candidate_freqs)
    freqs_s  = candidate_freqs[sort_idx]
    powers_s = candidate_powers[sort_idx]

    is_harmonic = np.zeros(len(freqs_s), dtype=bool)

    for i in range(len(freqs_s)):
        if is_harmonic[i]:
            continue
        # Check if higher peaks are harmonics of i
        for j in range(i + 1, len(freqs_s)):
            ratio = freqs_s[j] / freqs_s[i]
            if abs(ratio - 2.0) < HARMONIC_TOLERANCE \
            or abs(ratio - 3.0) < HARMONIC_TOLERANCE:
                is_harmonic[j] = True
        # Check if i is sub-harmonic of a lower peak
        for j in range(i):
            ratio = freqs_s[i] / freqs_s[j]
            if abs(ratio - 0.5) < HARMONIC_TOLERANCE:
                is_harmonic[i] = True

    fundamental_freqs  = freqs_s[~is_harmonic]
    fundamental_powers = powers_s[~is_harmonic]

    return fundamental_freqs, fundamental_powers


def find_dominant_peak(freqs, power):
    """
    Find dominant heart rate peak in power spectrum.

    Steps:
        1. Isolate valid HR band (48-180 BPM)
        2. Find all local peaks in HR band
        3. Reject harmonics and sub-harmonics
        4. Select peak with highest power
        5. Refine with quadratic interpolation

    Input:
        freqs — frequency axis in Hz
        power — power spectrum

    Output:
        hr_bpm      — heart rate in BPM
        peak_hz     — peak frequency in Hz
        peak_power  — power at peak
        n_peaks     — peaks found before rejection
        n_harmonics — harmonics rejected
    """
    hr_mask  = (freqs >= HR_MIN_HZ) & \
               (freqs <= HR_MAX_HZ)
    hr_freqs = freqs[hr_mask]
    hr_power = power[hr_mask]
    hr_idxs  = np.where(hr_mask)[0]

    if len(hr_power) == 0:
        return None, None, None, 0, 0

    peaks_local, _ = find_peaks(
        hr_power,
        height=0.1 * np.max(hr_power),
        distance=3
    )

    if len(peaks_local) == 0:
        max_local_idx   = np.argmax(hr_power)
        peak_global_idx = hr_idxs[max_local_idx]
        refined_hz, refined_bpm = interpolate_peak(
            freqs, power, peak_global_idx
        )
        return (
            refined_bpm,
            refined_hz,
            float(hr_power[max_local_idx]),
            1, 0
        )

    n_peaks     = len(peaks_local)
    peak_freqs  = hr_freqs[peaks_local]
    peak_powers = hr_power[peaks_local]

    fund_freqs, fund_powers = reject_harmonics(
        peak_freqs, peak_powers
    )
    n_harmonics = n_peaks - len(fund_freqs)

    if len(fund_freqs) == 0:
        fund_freqs  = peak_freqs
        fund_powers = peak_powers

    best_idx        = np.argmax(fund_powers)
    best_freq_hz    = float(fund_freqs[best_idx])
    best_global_idx = np.argmin(
        np.abs(freqs - best_freq_hz)
    )

    refined_hz, refined_bpm = interpolate_peak(
        freqs, power, best_global_idx
    )

    return (
        refined_bpm,
        refined_hz,
        float(fund_powers[best_idx]),
        n_peaks,
        n_harmonics
    )


def filter_rr_intervals(rr_intervals):
    if len(rr_intervals) < 3:
        return rr_intervals

    # Pass 1 — remove extreme outliers
    # using physiological hard limits
    # 55 BPM = 1090ms max, 180 BPM = 333ms min
    physiological = rr_intervals[
        (rr_intervals >= 333) &
        (rr_intervals <= 1090)
    ]

    if len(physiological) < 3:
        physiological = rr_intervals

    # Pass 2 — remove statistical outliers
    # using median of physiologically valid intervals
    median_rr = np.median(physiological)
    valid     = np.abs(
        physiological - median_rr
    ) < 0.15 * median_rr  # tightened to 15%

    result = physiological[valid]

    # Safety — if too many removed return pass 1
    if len(result) < 5:
        return physiological

    return result

def detect_beats_from_period(filtered, fps,
                              hr_bpm_fft):
    """
    FFT-guided beat detection using weighted
    centroid instead of maximum for broad
    rPPG waveform peaks.
    """
    period_samples = int((60.0 / hr_bpm_fft) * fps)
    period_samples = max(period_samples, 3)
    min_window     = int(period_samples * 0.60)
    max_window     = int(period_samples * 1.40)
    n              = len(filtered)

    beat_indices = []

    # First peak — search in first two periods
    first_end   = min(period_samples * 2, n)
    first_peak  = _find_centroid_peak(
        filtered, 0, first_end
    )
    beat_indices.append(first_peak)

    # Adaptive forward search from each peak
    while True:
        last_peak    = beat_indices[-1]
        search_start = last_peak + min_window
        search_end   = min(
            last_peak + max_window, n
        )

        if search_start >= n:
            break

        if search_end <= search_start:
            break

        local_peak = _find_centroid_peak(
            filtered, search_start, search_end
        )
        beat_indices.append(local_peak)

    return np.array(beat_indices)


def _find_centroid_peak(signal, start, end):
    """
    Find weighted centroid of signal above
    threshold in window [start, end).

    More stable than argmax for broad smooth
    rPPG waveform peaks — reduces positional
    uncertainty from 50-100ms to 10-20ms.

    Input:
        signal — full filtered signal
        start  — window start index
        end    — window end index

    Output:
        centroid_idx — weighted center index
    """
    window = signal[start:end]

    if len(window) == 0:
        return start

    # Threshold — only use top 30% of window
    w_min      = np.min(window)
    w_max      = np.max(window)
    w_range    = w_max - w_min

    if w_range < 1e-10:
        return start + int(np.argmax(window))

    threshold = w_min + 0.70 * w_range

    # Indices above threshold
    above_mask = window >= threshold
    if not np.any(above_mask):
        return start + int(np.argmax(window))

    indices    = np.where(above_mask)[0]
    weights    = window[above_mask] - threshold
    weight_sum = np.sum(weights)

    if weight_sum < 1e-10:
        return start + int(np.argmax(window))

    centroid = np.sum(indices * weights) / weight_sum
    return start + int(round(centroid))

def detect_beat_peaks(filtered, fps,
                       hr_bpm_fft=None):
    """
    Detect individual heartbeat peaks in time domain.

    Two-stage approach:
        Stage 1 — Adaptive FFT-guided detection
                  uses known period from FFT
                  searches in adaptive window
                  around each expected beat
                  robust for smooth rPPG waveforms

        Stage 2 — Prominence-based fallback
                  used only if FFT HR unavailable
                  or FFT completely failed

    Applies RR consistency filter after detection
    to remove physiologically impossible intervals.

    Input:
        filtered    — filtered pulse (Step 6)
        fps         — actual fps
        hr_bpm_fft  — HR from FFT (Step 7) optional

    Output:
        peak_indices   — sample indices of beats
        peak_times_sec — times of beats in seconds
        rr_intervals   — filtered RR intervals ms
    """
    # ── Stage 1 — Adaptive FFT-guided ─────────────
    if hr_bpm_fft is not None and \
       HR_MIN_BPM <= hr_bpm_fft <= HR_MAX_BPM:

        beat_indices = detect_beats_from_period(
            filtered, fps, hr_bpm_fft
        )

    else:
        # ── Stage 2 — Prominence fallback ─────────
        min_distance = int(fps / (HR_MAX_BPM / 60))
        min_distance = max(min_distance, 3)

        signal_range   = float(
            np.max(filtered) - np.min(filtered)
        )
        min_height     = float(np.min(filtered)) + \
                         0.15 * signal_range
        min_prominence = 0.05 * signal_range

        beat_indices, _ = find_peaks(
            filtered,
            distance=min_distance,
            height=min_height,
            prominence=min_prominence
        )

    if len(beat_indices) < 2:
        return beat_indices, np.array([]), np.array([])

    # Convert to time in seconds
    peak_times_sec = beat_indices / fps

    # RR intervals in milliseconds
    rr_raw = np.diff(peak_times_sec) * 1000.0

    # Filter physiologically impossible intervals
    rr_filtered = filter_rr_intervals(rr_raw)

    return beat_indices, peak_times_sec, rr_filtered


def assess_peak_quality(hr_bpm, peak_hz,
                         peak_power, rr_intervals,
                         n_peaks, n_harmonics):
    """
    Assess peak detection quality.

    Three checks:
        1. HR in valid physiological range
        2. Sufficient beats for HRV (at least 5)
        3. RR intervals physiologically plausible
           with 20% buffer above minimum HR

    Input:
        hr_bpm       — detected heart rate BPM
        peak_hz      — peak frequency Hz
        peak_power   — power at peak
        rr_intervals — filtered RR intervals ms
        n_peaks      — peaks before rejection
        n_harmonics  — harmonics rejected

    Output:
        quality_ok — bool
        issues     — list of problem descriptions
        report     — dict of statistics
    """
    issues = []
    report = {}

    report['hr_bpm']      = round(hr_bpm, 2) \
                            if hr_bpm else None
    report['peak_hz']     = round(peak_hz, 4) \
                            if peak_hz else None
    report['n_peaks']     = n_peaks
    report['n_harmonics'] = n_harmonics

    # Check 1 — HR in valid range
    if hr_bpm is None:
        issues.append("No peak detected")
    elif not (HR_MIN_BPM <= hr_bpm <= HR_MAX_BPM):
        issues.append(
            f"HR outside valid range: "
            f"{hr_bpm:.1f} BPM"
        )

    # Check 2 — enough beats for HRV
    n_beats = len(rr_intervals) + 1 \
              if len(rr_intervals) > 0 else 0
    report['n_beats'] = n_beats

    if n_beats < 5:
        issues.append(
            f"Too few beats for HRV: {n_beats} "
            f"(need at least 5)"
        )

    # Check 3 — RR intervals plausible
    # 20% buffer above minimum HR threshold
    rr_max_threshold = (60.0 / HR_MIN_BPM) \
                       * 1000 * 1.20

    if len(rr_intervals) > 0:
        rr_min  = float(np.min(rr_intervals))
        rr_max  = float(np.max(rr_intervals))
        rr_mean = float(np.mean(rr_intervals))
        rr_std  = float(np.std(rr_intervals))

        report['rr_min_ms']  = round(rr_min, 1)
        report['rr_max_ms']  = round(rr_max, 1)
        report['rr_mean_ms'] = round(rr_mean, 1)
        report['rr_std_ms']  = round(rr_std, 1)

        if rr_min < 333:
            issues.append(
                f"RR too short: {rr_min:.0f}ms "
                f"(> 180 BPM)"
            )
        if rr_max > rr_max_threshold:
            issues.append(
                f"RR too long: {rr_max:.0f}ms "
                f"(< {HR_MIN_BPM} BPM)"
            )

    quality_ok = len(issues) == 0
    return quality_ok, issues, report


def print_peak_report(hr_bpm, peak_hz, peak_power,
                       rr_intervals, peak_times,
                       n_peaks, n_harmonics):
    """
    Print peak detection report to terminal.
    """
    quality_ok, issues, report = assess_peak_quality(
        hr_bpm, peak_hz, peak_power,
        rr_intervals, n_peaks, n_harmonics
    )

    print(f"\n{'='*45}")
    print(f"STEP 8 — PEAK DETECTION REPORT")
    print(f"{'='*45}")

    print(f"\nFrequency domain peaks:")
    print(f"  Peaks found:       {n_peaks}")
    print(f"  Harmonics removed: {n_harmonics}")
    print(f"  Dominant peak:     "
          f"{hr_bpm:.2f} BPM  "
          f"({peak_hz:.4f} Hz)")
    print(f"  Peak power:        {peak_power:.4f}")

    print(f"\nTime domain beats:")
    print(f"  Beats detected:    "
          f"{report.get('n_beats', 0)}")

    if len(rr_intervals) > 0:
        print(f"  RR mean:           "
              f"{report.get('rr_mean_ms', 0):.1f} ms")
        print(f"  RR std:            "
              f"{report.get('rr_std_ms', 0):.1f} ms")
        print(f"  RR min:            "
              f"{report.get('rr_min_ms', 0):.1f} ms")
        print(f"  RR max:            "
              f"{report.get('rr_max_ms', 0):.1f} ms")

    print(f"\nQuality OK: {quality_ok}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues — ready for HR/HRV")

    print(f"{'='*45}")
    return quality_ok, report