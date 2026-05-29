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
# HRV NOTE: At 30fps temporal resolution is
# 33ms per sample. RR interval precision is
# therefore ±33ms minimum. RMSSD values from
# this system are indicative only — not
# clinically validated absolute values.
# Clinical HRV requires 60fps+ camera.
#
# Pure functions only.
# No camera, no imports from other steps.
# Called by main.py with FFT output from Step 7.
# ─────────────────────────────────────────

HR_MIN_BPM = 55
HR_MAX_BPM = 180
HR_MIN_HZ  = HR_MIN_BPM / 60  # 0.917 Hz
HR_MAX_HZ  = HR_MAX_BPM / 60  # 3.000 Hz

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
        Harmonic:     peak at 2x or 3x lower peak
        Sub-harmonic: peak at 0.5x higher peak

    Input:
        candidate_freqs  — peak frequencies Hz
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
        for j in range(i + 1, len(freqs_s)):
            ratio = freqs_s[j] / freqs_s[i]
            if abs(ratio - 2.0) < HARMONIC_TOLERANCE \
            or abs(ratio - 3.0) < HARMONIC_TOLERANCE:
                is_harmonic[j] = True
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
        1. Isolate valid HR band (55-180 BPM)
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
    """
    Two-pass RR interval filtering.

    Pass 1 — physiological hard limits:
        Remove intervals outside 55-180 BPM range
        333ms (180 BPM) to 1090ms (55 BPM)

    Pass 2 — statistical filtering:
        Keep intervals within 20% of median
        of physiologically valid intervals

    Works for all heart rates 55-180 BPM.
    Relative thresholds — universal for all users.

    Note: At 30fps, minimum RR precision is ±33ms.
    RMSSD values are indicative only.

    Input:
        rr_intervals — raw RR intervals in ms

    Output:
        filtered_rr — cleaned RR intervals
    """
    if len(rr_intervals) < 3:
        return rr_intervals

    # Pass 1 — physiological hard limits
    physiological = rr_intervals[
        (rr_intervals >= 333) &
        (rr_intervals <= 1090)
    ]

    if len(physiological) < 3:
        physiological = rr_intervals

    # Pass 2 — statistical filtering
    median_rr = np.median(physiological)
    valid     = np.abs(
        physiological - median_rr
    ) < 0.20 * median_rr

    result = physiological[valid]

    # Safety — if too many removed return pass 1
    if len(result) < 5:
        return physiological

    return result


def detect_beat_peaks(filtered, fps,
                       hr_bpm_fft=None):
    """
    Detect individual heartbeat peaks in time domain.

    Primary: FFT-guided adaptive window detection.
        Uses known period from FFT to search for
        next beat within 60%-140% of expected period.
        Robust for smooth rPPG waveforms.

    Fallback: Prominence-based detection.
        Used when FFT result unavailable.

    Applies two-pass RR consistency filter.

    Input:
        filtered    — filtered pulse (Step 6)
        fps         — actual fps
        hr_bpm_fft  — HR from FFT (Step 7)

    Output:
        peak_indices   — sample indices of beats
        peak_times_sec — beat timestamps seconds
        rr_intervals   — filtered RR intervals ms
    """
    if hr_bpm_fft is not None and \
       HR_MIN_BPM <= hr_bpm_fft <= HR_MAX_BPM:

        # ── FFT-guided adaptive window ─────────────
        period_samples = int(
            (60.0 / hr_bpm_fft) * fps
        )
        period_samples = max(period_samples, 3)

        min_window = int(period_samples * 0.60)
        max_window = int(period_samples * 1.40)
        n          = len(filtered)

        beat_indices = []

        # First peak in first two periods
        first_end  = min(period_samples * 2, n)
        first_peak = int(
            np.argmax(filtered[:first_end])
        )
        beat_indices.append(first_peak)

        # Adaptive forward search
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

            window     = filtered[
                search_start:search_end
            ]
            local_peak = int(np.argmax(window)) \
                         + search_start
            beat_indices.append(local_peak)

        beat_indices = np.array(beat_indices)

    else:
        # ── Prominence-based fallback ──────────────
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
        return beat_indices, \
               np.array([]), np.array([])

    peak_times_sec = beat_indices / fps
    rr_raw         = np.diff(peak_times_sec) * 1000.0
    rr_filtered    = filter_rr_intervals(rr_raw)

    return beat_indices, peak_times_sec, rr_filtered


def assess_peak_quality(hr_bpm, peak_hz,
                         peak_power, rr_intervals,
                         n_peaks, n_harmonics):
    """
    Assess peak detection quality.

    Three checks:
        1. HR in valid range (55-180 BPM)
        2. Sufficient beats for HRV (at least 5)
        3. RR intervals physiologically plausible

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
    if len(rr_intervals) > 0:
        rr_min  = float(np.min(rr_intervals))
        rr_max  = float(np.max(rr_intervals))
        rr_mean = float(np.mean(rr_intervals))
        rr_std  = float(np.std(rr_intervals))

        report['rr_min_ms']  = round(rr_min, 1)
        report['rr_max_ms']  = round(rr_max, 1)
        report['rr_mean_ms'] = round(rr_mean, 1)
        report['rr_std_ms']  = round(rr_std, 1)

        rr_max_threshold = (60.0 / HR_MIN_BPM) \
                           * 1000 * 1.20
        if rr_min < 333:
            issues.append(
                f"RR too short: {rr_min:.0f}ms"
            )
        if rr_max > rr_max_threshold:
            issues.append(
                f"RR too long: {rr_max:.0f}ms"
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