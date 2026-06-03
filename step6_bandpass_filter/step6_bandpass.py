import numpy as np
from scipy.signal import butter, filtfilt, freqz

# ─────────────────────────────────────────
# Step 6 — Bandpass Filter
#
# Tool: Butterworth bandpass filter (order 4)
# scipy.signal.butter + filtfilt
#
# Purpose: Remove all frequencies outside the
# valid heart rate range from the POS pulse signal.
#
# NEW — Artifact clipping (clip_signal_artifacts):
#   Even after Step 3 motion rejection, mild
#   artifacts survive as spikes in the signal.
#   A Butterworth filter rings — one spike causes
#   oscillations for dozens of samples afterward.
#   Clipping samples beyond 4× personal std before
#   filtering prevents one bad moment from
#   contaminating several seconds of signal.
#   Uses SubjectProfile's baseline_g_std when
#   available — otherwise computes from signal itself.
#
# NEW — Profile bandpass hint:
#   If SubjectProfile provides an HR estimate from
#   calibration, the bandpass window is narrowed
#   around that estimate (±30 BPM) instead of
#   using the full 40-180 BPM range.
#   This improves SNR by excluding HR frequencies
#   far from the patient's actual rate.
#   Falls back to full range if no hint available.
#
# Pure functions only.
# No camera, no imports from other steps.
# Called by run.py with pulse array from Step 5.
# ─────────────────────────────────────────

HR_MIN_BPM = 40
HR_MAX_BPM = 180
HR_MIN_HZ  = HR_MIN_BPM / 60   # 0.667 Hz
HR_MAX_HZ  = HR_MAX_BPM / 60   # 3.000 Hz

FILTER_ORDER = 4

# Clip threshold multiplier — samples beyond
# this many std from the mean are clipped
# before filtering to prevent filter ringing.
# 4.0 is conservative — only clips genuine spikes,
# not real physiological variation.
CLIP_STD_MULTIPLIER_DEFAULT = 4.0

# ─────────────────────────────────────────
# Artifact clipping
# ─────────────────────────────────────────

def clip_signal_artifacts(pulse, profile=None):
    """
    Clip motion artifact spikes before bandpass filter.

    Why this is needed:
        A Butterworth filter applied via filtfilt
        rings when the input contains sharp spikes.
        One cough that survived frame rejection
        appears as a spike in the pulse signal.
        filtfilt propagates this spike as oscillations
        ~20-40 samples in both directions — corrupting
        nearly 1-2 seconds of signal around each event.
        Clipping the spike to ±4std before filtering
        prevents this contamination entirely.

    Clipping strategy:
        Any sample more than CLIP_STD_MULTIPLIER
        standard deviations from the signal mean
        is clipped to that boundary.
        This preserves real physiological variation
        (which is typically within ±3std) while
        removing artifact spikes (which are >±4std).

    The clip boundaries use the signal's own
    statistics (mean and std). No fixed thresholds.

    Input:
        pulse   — raw pulse waveform (N,) from Step 5
        profile — SubjectProfile (optional)
                  not currently used for clipping
                  (signal-level clipping is self-
                  calibrating from the signal itself)

    Output:
        clipped       — artifact-clipped pulse (N,)
        n_clipped     — number of samples clipped
        clip_low      — lower clip boundary used
        clip_high     — upper clip boundary used
    """
    signal_mean = float(np.mean(pulse))
    signal_std  = float(np.std(pulse))

    if signal_std < 1e-10:
        # Flat signal — nothing to clip
        return pulse.copy(), 0, 0.0, 0.0

    if profile is not None:
        multiplier = profile.get_clip_multiplier()
    else:
        multiplier = CLIP_STD_MULTIPLIER_DEFAULT

    clip_low  = signal_mean - multiplier * signal_std
    clip_high = signal_mean + multiplier * signal_std

    clipped   = np.clip(pulse, clip_low, clip_high)
    n_clipped = int(np.sum(
        (pulse < clip_low) | (pulse > clip_high)
    ))

    return clipped, n_clipped, clip_low, clip_high


# ─────────────────────────────────────────
# Profile-aware cutoff selection
# ─────────────────────────────────────────

def get_cutoffs_from_profile(profile, low_hz_override=None):
    """
    Get bandpass cutoffs using SubjectProfile hint.

    If profile has a reliable HR estimate from the
    calibration phase, narrow the bandpass window
    around that estimate (±30 BPM) rather than
    using the full 40-180 BPM range.

    Narrowing the bandpass around the patient's
    actual HR improves SNR by excluding power from
    HR frequencies the patient clearly doesn't have.
    A patient with resting HR of 70 BPM gains nothing
    from passing 150-180 BPM frequencies through —
    they only add noise to downstream steps.

    The ±30 BPM window is wide enough to accommodate:
        - Natural HR variation during recording
        - HR response to breathing (RSA)
        - Mild exercise or anxiety effect

    Falls back to full range if:
        - No profile provided
        - Profile HR estimate was unreliable
        - Caller provides explicit low_hz_override
          (respiratory-adaptive cutoff from run.py
           takes priority over profile hint)

    Input:
        profile         — SubjectProfile (optional)
        low_hz_override — explicit lower cutoff in Hz
                          (from respiratory detection)
                          overrides profile hint if set

    Output:
        low_hz  — lower cutoff in Hz
        high_hz — upper cutoff in Hz
        source  — string describing which source was used
    """
    # Explicit override from respiratory detection
    # always takes priority over profile hint
    if low_hz_override is not None:
        return low_hz_override, HR_MAX_HZ, "respiratory"

    # Profile HR hint — narrow the window
    if profile is not None:
        hint = profile.get_bandpass_hint()
        if hint is not None:
            low_hz, high_hz = hint
            return low_hz, high_hz, "profile_hint"

    # Full range fallback
    return HR_MIN_HZ, HR_MAX_HZ, "default"


# ─────────────────────────────────────────
# Filter design
# ─────────────────────────────────────────

def design_bandpass_filter(fps, low_hz=HR_MIN_HZ,
                            high_hz=HR_MAX_HZ,
                            order=FILTER_ORDER):
    """
    Design Butterworth bandpass filter coefficients.

    Uses scipy.signal.butter. All frequencies
    normalized to Nyquist before passing to butter().

    Input:
        fps     — sample rate (fps from Step 1)
        low_hz  — lower cutoff in Hz
        high_hz — upper cutoff in Hz
        order   — filter order (default 4)

    Output:
        b, a — filter coefficients for filtfilt()
    """
    nyquist   = fps / 2.0
    low_norm  = np.clip(low_hz  / nyquist, 0.001, 0.999)
    high_norm = np.clip(high_hz / nyquist, 0.001, 0.999)

    if low_norm >= high_norm:
        raise ValueError(
            f"Invalid filter range: "
            f"low={low_norm:.3f} >= high={high_norm:.3f}. "
            f"Check fps={fps} and cutoff frequencies."
        )

    b, a = butter(order, [low_norm, high_norm],
                  btype='bandpass')
    return b, a


# ─────────────────────────────────────────
# Main filter entry point
# ─────────────────────────────────────────

def apply_bandpass_filter(pulse, fps,
                           low_hz=None,
                           high_hz=HR_MAX_HZ,
                           order=FILTER_ORDER,
                           profile=None):
    """
    Apply Butterworth bandpass filter to pulse signal.

    NEW: Two additions before the filter itself:
        1. Artifact clipping — clips samples beyond
           ±4std to prevent filter ringing from spikes
        2. Profile hint — narrows bandpass window
           around patient's calibration HR estimate
           if available and no explicit low_hz given

    Uses filtfilt (zero-phase) — applies filter
    forward then backward to eliminate phase shift.
    Phase preservation is critical for accurate
    beat timing in Step 8.

    Input:
        pulse   — raw pulse waveform (N,) from Step 5
        fps     — actual fps from Step 1
        low_hz  — lower cutoff in Hz (optional)
                  None → determined from profile or default
                  Set → used directly (e.g. from respiratory)
        high_hz — upper cutoff in Hz (default 3.0 Hz)
        order   — filter order (default 4)
        profile — SubjectProfile (optional)
                  used for bandpass hint only

    Output:
        filtered   — clean pulse waveform (N,)
        clip_report — dict with clipping statistics
    """
    min_samples = 3 * order * 2
    if len(pulse) < min_samples:
        raise ValueError(
            f"Signal too short: {len(pulse)} samples, "
            f"need at least {min_samples}"
        )

    # ── Step 1: Artifact clipping ──────────────────
    clipped, n_clipped, clip_low, clip_high = \
        clip_signal_artifacts(pulse, profile)

    clip_report = {
        'n_clipped':  n_clipped,
        'clip_low':   round(float(clip_low),  6),
        'clip_high':  round(float(clip_high), 6),
        'clip_rate':  round(n_clipped / len(pulse), 4)
    }

    # ── Step 2: Determine cutoffs ──────────────────
    effective_low, effective_high, cutoff_source = \
        get_cutoffs_from_profile(profile, low_hz)

    clip_report['cutoff_source'] = cutoff_source
    clip_report['low_hz']        = round(effective_low, 4)
    clip_report['high_hz']       = round(effective_high, 4)

    # ── Step 3: Apply filter ───────────────────────
    b, a     = design_bandpass_filter(
        fps, effective_low, effective_high, order
    )
    filtered = filtfilt(b, a, clipped)

    return filtered, clip_report


# ─────────────────────────────────────────
# Quality assessment
# ─────────────────────────────────────────

def assess_filter_quality(pulse_raw, pulse_filtered,
                           fps, low_hz=HR_MIN_HZ):
    """
    Assess bandpass filter output quality.

    Three checks:
        1. Signal not flat after filtering
        2. HR power ratio improved vs raw signal
        3. Dominant frequency in valid HR range
    """
    issues = []
    report = {}

    filtered_std = float(np.std(pulse_filtered))
    report['filtered_std'] = round(filtered_std, 6)

    if filtered_std < 1e-6:
        issues.append("Flat signal after filtering")
        return False, issues, report

    n     = len(pulse_filtered)
    freqs = np.fft.rfftfreq(n, d=1.0/fps)

    valid_mask = (freqs >= low_hz) & \
                 (freqs <= HR_MAX_HZ)

    raw_power    = np.abs(np.fft.rfft(pulse_raw)) ** 2
    raw_total    = raw_power.sum()
    raw_hr_ratio = float(
        raw_power[valid_mask].sum() / raw_total
    ) if raw_total > 0 else 0.0

    filt_power    = np.abs(
        np.fft.rfft(pulse_filtered)
    ) ** 2
    filt_total    = filt_power.sum()
    filt_hr_ratio = float(
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
            f"{filt_hr_ratio:.1%}"
        )

    if valid_mask.sum() > 0:
        valid_power   = filt_power[valid_mask]
        dominant_freq = float(
            freqs[valid_mask][np.argmax(valid_power)]
        )
        dominant_bpm  = dominant_freq * 60
        report['dominant_freq_hz'] = round(dominant_freq, 3)
        report['dominant_bpm']     = round(dominant_bpm, 1)

        if not (HR_MIN_BPM <= dominant_bpm <= HR_MAX_BPM):
            issues.append(
                f"Dominant frequency outside HR range: "
                f"{dominant_bpm:.1f} BPM"
            )
    else:
        issues.append("No valid HR frequencies found")

    quality_ok = len(issues) == 0
    return quality_ok, issues, report


# ─────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────

def print_filter_report(pulse_raw, pulse_filtered,
                         fps, clip_report=None):
    """
    Print bandpass filter report to terminal.
    """
    low_hz = clip_report.get('low_hz', HR_MIN_HZ) \
             if clip_report else HR_MIN_HZ

    quality_ok, issues, report = \
        assess_filter_quality(
            pulse_raw, pulse_filtered, fps, low_hz
        )

    print(f"\n{'='*45}")
    print(f"STEP 6 — BANDPASS FILTER REPORT")
    print(f"{'='*45}")

    if clip_report:
        source = clip_report.get('cutoff_source', 'default')
        print(f"\nCutoff source: {source}")
        print(f"  Low:  {clip_report.get('low_hz', HR_MIN_HZ):.3f} Hz  "
              f"({clip_report.get('low_hz', HR_MIN_HZ)*60:.1f} BPM)")
        print(f"  High: {clip_report.get('high_hz', HR_MAX_HZ):.3f} Hz  "
              f"({clip_report.get('high_hz', HR_MAX_HZ)*60:.1f} BPM)")

        n_clip = clip_report.get('n_clipped', 0)
        rate   = clip_report.get('clip_rate', 0)
        print(f"\nArtifact clipping:")
        print(f"  Samples clipped: {n_clip}  "
              f"({rate:.1%} of signal)")
        if n_clip > 0:
            print(f"  Clip range: "
                  f"[{clip_report['clip_low']:.4f}, "
                  f"{clip_report['clip_high']:.4f}]")
    else:
        print(f"\nFilter settings:")
        print(f"  Low cut:  {HR_MIN_HZ:.3f} Hz "
              f"({HR_MIN_BPM} BPM)")
        print(f"  High cut: {HR_MAX_HZ:.3f} Hz "
              f"({HR_MAX_BPM} BPM)")

    print(f"  Type:   Butterworth order {FILTER_ORDER}")
    print(f"  Method: filtfilt (zero-phase)")

    print(f"\nSignal stats:")
    print(f"  Raw std:      {np.std(pulse_raw):.6f}")
    print(f"  Filtered std: {report['filtered_std']:.6f}")

    print(f"\nHR power ratio ({HR_MIN_BPM}-{HR_MAX_BPM} BPM):")
    print(f"  Before: {report.get('raw_hr_power_ratio', 0):.1%}")
    print(f"  After:  {report.get('filtered_hr_power_ratio', 0):.1%}")
    print(f"  Gain:   +{report.get('power_improvement', 0):.1%}")

    if 'dominant_bpm' in report:
        print(f"\nDominant frequency:")
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