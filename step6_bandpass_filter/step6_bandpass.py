import numpy as np
from scipy.signal import butter, filtfilt, freqz

# ─────────────────────────────────────────
# Step 6 — Bandpass Filter
#
# Tool: Butterworth bandpass filter (order 4)
# scipy.signal.butter + filtfilt
#
# Purpose: Remove all frequencies outside the
# valid heart rate range (40-180 BPM / 0.67-3.0 Hz)
# from the raw POS pulse signal.
#
# Pure functions only.
# No camera, no imports from other steps.
# Called by main.py with pulse array from Step 5.
#
# Input:  pulse    — shape (N,) from Step 5
#         fps      — actual fps from Step 1
# Output: filtered — shape (N,) clean pulse
# ─────────────────────────────────────────

# Valid physiological heart rate range
HR_MIN_BPM = 55
HR_MAX_BPM = 180
HR_MIN_HZ  = HR_MIN_BPM / 60  # 0.667 Hz
HR_MAX_HZ  = HR_MAX_BPM / 60  # 3.000 Hz

# Butterworth filter order
# Order 4 — standard in rPPG literature
# Sharp enough cutoff, no ringing artifacts
FILTER_ORDER = 4


def design_bandpass_filter(fps, low_hz=HR_MIN_HZ,
                            high_hz=HR_MAX_HZ,
                            order=FILTER_ORDER):
    """
    Design Butterworth bandpass filter coefficients.

    Uses scipy.signal.butter to compute filter
    coefficients for a bandpass filter between
    low_hz and high_hz at the given sample rate.

    Nyquist frequency = fps / 2
    All frequencies normalized to Nyquist before
    passing to butter() — this is required by scipy.

    Input:
        fps     — sample rate in Hz (frames per second)
        low_hz  — lower cutoff frequency in Hz
        high_hz — upper cutoff frequency in Hz
        order   — filter order (default 4)

    Output:
        b, a — filter coefficients for filtfilt()
    """
    nyquist   = fps / 2.0
    low_norm  = low_hz  / nyquist
    high_norm = high_hz / nyquist

    # Clamp to valid range — avoid numerical issues
    # if fps is very low or cutoffs are at boundary
    low_norm  = np.clip(low_norm,  0.001, 0.999)
    high_norm = np.clip(high_norm, 0.001, 0.999)

    if low_norm >= high_norm:
        raise ValueError(
            f"Invalid filter range: "
            f"low={low_norm:.3f} >= high={high_norm:.3f}. "
            f"Check fps={fps} and cutoff frequencies."
        )

    b, a = butter(
        order,
        [low_norm, high_norm],
        btype='bandpass'
    )
    return b, a


def apply_bandpass_filter(pulse, fps,
                           low_hz=HR_MIN_HZ,
                           high_hz=HR_MAX_HZ,
                           order=FILTER_ORDER):
    """
    Apply Butterworth bandpass filter to pulse signal.

    Uses filtfilt (zero-phase filtering) — applies
    filter forward then backward to eliminate phase
    distortion. This preserves the timing of the
    pulse peaks which is critical for accurate
    heart rate calculation in Step 9.

    Why zero-phase matters:
        A standard one-pass filter shifts the signal
        in time — peaks arrive slightly late. For heart
        rate this causes timing errors in peak detection.
        filtfilt applies the filter twice in opposite
        directions, cancelling the phase shift exactly.

    Input:
        pulse   — raw pulse waveform (N,) from Step 5
        fps     — actual fps from Step 1
        low_hz  — lower cutoff (default 0.667 Hz = 40 BPM)
        high_hz — upper cutoff (default 3.000 Hz = 180 BPM)
        order   — filter order (default 4)

    Output:
        filtered — clean pulse waveform (N,)
                   same length as input
                   only HR frequencies remain
    """
    # Need minimum samples for filter to work
    # Rule of thumb: at least 3x the filter order
    min_samples = 3 * order * 2
    if len(pulse) < min_samples:
        raise ValueError(
            f"Signal too short for filtering: "
            f"{len(pulse)} samples, "
            f"need at least {min_samples}"
        )

    b, a     = design_bandpass_filter(
        fps, low_hz, high_hz, order
    )
    filtered = filtfilt(b, a, pulse)

    return filtered


def assess_filter_quality(pulse_raw, pulse_filtered,
                           fps):
    """
    Assess bandpass filter output quality.

    Three checks:
        1. Signal not flat after filtering
        2. HR power ratio improved vs raw signal
           filtered signal should have much higher
           fraction of power in HR range than raw
        3. Dominant frequency in valid HR range

    Input:
        pulse_raw      — raw pulse from Step 5
        pulse_filtered — filtered pulse
        fps            — actual fps

    Output:
        quality_ok — bool
        issues     — list of problem descriptions
        report     — dict of statistics
    """
    issues = []
    report = {}

    # Check 1 — signal not flat
    filtered_std = float(np.std(pulse_filtered))
    report['filtered_std'] = round(filtered_std, 6)

    if filtered_std < 1e-6:
        issues.append("Flat signal after filtering")
        return False, issues, report

    # Check 2 — HR power ratio improved
    n     = len(pulse_filtered)
    freqs = np.fft.rfftfreq(n, d=1.0/fps)

    valid_mask = (freqs >= HR_MIN_HZ) & \
                 (freqs <= HR_MAX_HZ)

    # Raw signal HR power ratio
    raw_power      = np.abs(np.fft.rfft(pulse_raw)) ** 2
    raw_total      = raw_power.sum()
    raw_hr_ratio   = float(
        raw_power[valid_mask].sum() / raw_total
    ) if raw_total > 0 else 0.0

    # Filtered signal HR power ratio
    filt_power     = np.abs(
        np.fft.rfft(pulse_filtered)
    ) ** 2
    filt_total     = filt_power.sum()
    filt_hr_ratio  = float(
        filt_power[valid_mask].sum() / filt_total
    ) if filt_total > 0 else 0.0

    report['raw_hr_power_ratio']      = round(raw_hr_ratio, 3)
    report['filtered_hr_power_ratio'] = round(filt_hr_ratio, 3)
    report['power_improvement']       = round(
        filt_hr_ratio - raw_hr_ratio, 3
    )

    if filt_hr_ratio < 0.3:
        issues.append(
            f"Low HR power after filtering: "
            f"{filt_hr_ratio:.1%} — possible motion artifact"
        )

    # Check 3 — dominant frequency in valid range
    if valid_mask.sum() > 0:
        valid_power    = filt_power[valid_mask]
        dominant_freq  = float(
            freqs[valid_mask][np.argmax(valid_power)]
        )
        dominant_bpm   = dominant_freq * 60
        report['dominant_freq_hz']  = round(dominant_freq, 3)
        report['dominant_bpm']      = round(dominant_bpm, 1)

        if not (HR_MIN_BPM <= dominant_bpm <= HR_MAX_BPM):
            issues.append(
                f"Dominant frequency outside HR range: "
                f"{dominant_bpm:.1f} BPM"
            )
    else:
        issues.append("No valid HR frequencies found")

    quality_ok = len(issues) == 0
    return quality_ok, issues, report


def print_filter_report(pulse_raw, pulse_filtered, fps):
    """
    Print bandpass filter report to terminal.

    Input:
        pulse_raw      — raw pulse from Step 5
        pulse_filtered — filtered pulse
        fps            — actual fps
    """
    quality_ok, issues, report = \
        assess_filter_quality(
            pulse_raw, pulse_filtered, fps
        )

    print(f"\n{'='*45}")
    print(f"STEP 6 — BANDPASS FILTER REPORT")
    print(f"{'='*45}")

    print(f"\nFilter settings:")
    print(f"  Type:      Butterworth order {FILTER_ORDER}")
    print(f"  Low cut:   {HR_MIN_HZ:.3f} Hz "
          f"({HR_MIN_BPM} BPM)")
    print(f"  High cut:  {HR_MAX_HZ:.3f} Hz "
          f"({HR_MAX_BPM} BPM)")
    print(f"  Method:    filtfilt (zero-phase)")

    print(f"\nSignal stats:")
    print(f"  Raw std:       {np.std(pulse_raw):.6f}")
    print(f"  Filtered std:  {report['filtered_std']:.6f}")

    print(f"\nHR power ratio (fraction in 40-180 BPM):")
    print(f"  Before filter: "
          f"{report.get('raw_hr_power_ratio', 0):.1%}")
    print(f"  After filter:  "
          f"{report.get('filtered_hr_power_ratio', 0):.1%}")
    print(f"  Improvement:   "
          f"+{report.get('power_improvement', 0):.1%}")

    if 'dominant_bpm' in report:
        print(f"\nDominant frequency after filter:")
        print(f"  {report['dominant_bpm']:.1f} BPM  "
              f"({report['dominant_freq_hz']:.3f} Hz)")

    print(f"\nQuality OK: {quality_ok}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues — ready for FFT")

    print(f"{'='*45}")
    return quality_ok