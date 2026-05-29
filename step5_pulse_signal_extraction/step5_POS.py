import numpy as np

# ─────────────────────────────────────────
# Step 5 — Pulse Signal Extraction
#
# Algorithm: POS (Plane-Orthogonal-to-Skin)
# Wang et al. 2017
#
# Purpose: Combine three normalized RGB channels
# into one single pulse waveform by projecting
# onto the plane orthogonal to the skin color
# vector — cancels noise, reinforces pulse.
#
# Pure functions only.
# No camera, no imports from other steps.
# Called by main.py with normalized r, g, b
# arrays from Step 4.
#
# Input:  r_norm, g_norm, b_norm — shape (N,)
# Output: pulse                  — shape (N,)
# ─────────────────────────────────────────


def compute_pos(r_norm, g_norm, b_norm):
    """
    Apply POS algorithm to extract pulse waveform.

    POS (Plane-Orthogonal-to-Skin) — Wang et al. 2017.

    Mathematical derivation:
        When blood pulses through skin capillaries,
        the RGB reflectance changes in a specific
        direction in color space — the skin color
        vector. POS removes this direction and keeps
        only the orthogonal component — the pulse.

    Two projection signals:
        S1 = R - G
            difference between red and green
            captures one axis of skin color plane

        S2 = R + G - 2B
            captures the other axis of skin color plane
            blue has different absorption than red+green

        alpha = std(S1) / std(S2)
            adaptive weight — balances S1 and S2
            based on their variance
            this adaptation is what makes POS
            more robust than fixed-weight methods

        pulse = S1 + alpha * S2
            weighted combination that projects
            out the skin color direction

    Why this works for dark skin:
        CHROM uses fixed skin color coefficients
        designed for light skin. POS uses adaptive
        alpha computed from the actual signal variance
        — no fixed skin color assumption. More robust
        across all Fitzpatrick types.

    Input:
        r_norm — normalized red   channel (N,)
        g_norm — normalized green channel (N,)
        b_norm — normalized blue  channel (N,)
                 All centered around 0.0 from Step 4.

    Output:
        pulse  — raw pulse waveform (N,)
                 oscillates at heartbeat frequency
                 still contains noise — Step 6
                 bandpass filter removes it
        s1     — first projection signal (for debug)
        s2     — second projection signal (for debug)
        alpha  — adaptive weight used (for logging)
    """
    # ── Two projection signals ────────────────────
    s1 = r_norm - g_norm
    s2 = r_norm + g_norm - 2.0 * b_norm

    # ── Adaptive weight ───────────────────────────
    std_s1 = float(np.std(s1))
    std_s2 = float(np.std(s2))

    if std_s2 == 0:
        # Fallback — flat S2 means no signal in that
        # projection, use S1 alone
        alpha = 1.0
    else:
        alpha = std_s1 / std_s2

    # ── Pulse waveform ────────────────────────────
    pulse = s1 + alpha * s2

    return pulse, s1, s2, round(alpha, 4)


def assess_pos_quality(pulse, fps):
    """
    Assess quality of POS output pulse signal.

    Three checks:
        1. Signal not flat — variance above threshold
        2. Signal not clipping — no extreme values
        3. Dominant frequency in valid HR range
           40-180 BPM = 0.67-3.0 Hz
           (rough check before proper FFT in Step 7)

    Input:
        pulse — raw pulse waveform (N,)
        fps   — actual fps from Step 1

    Output:
        quality_ok — bool
        issues     — list of problem descriptions
        report     — dict of signal statistics
    """
    issues = []
    report = {}

    # Check 1 — signal variance
    pulse_std = float(np.std(pulse))
    pulse_mean = float(np.mean(np.abs(pulse)))
    report['pulse_std']  = round(pulse_std, 6)
    report['pulse_mean'] = round(pulse_mean, 6)

    if pulse_std < 1e-6:
        issues.append("Flat pulse signal — no heartbeat detected")

    # Check 2 — no extreme clipping
    pulse_max = float(np.max(np.abs(pulse)))
    report['pulse_max'] = round(pulse_max, 6)

    if pulse_max > 10.0:
        issues.append(
            f"Pulse clipping detected: max={pulse_max:.3f}"
        )

    # Check 3 — rough frequency check via FFT
    # Full FFT in Step 7 — this is just a sanity check
    n      = len(pulse)
    freqs  = np.fft.rfftfreq(n, d=1.0/fps)
    power  = np.abs(np.fft.rfft(pulse)) ** 2

    # Valid heart rate range: 40-180 BPM
    valid_mask = (freqs >= 0.67) & (freqs <= 3.0)

    if valid_mask.sum() > 0:
        valid_power   = power[valid_mask]
        total_power   = power.sum()
        valid_ratio   = float(valid_power.sum() / total_power) \
                        if total_power > 0 else 0.0
        dominant_freq = float(freqs[valid_mask][np.argmax(valid_power)])
        dominant_bpm  = round(dominant_freq * 60, 1)

        report['valid_hr_power_ratio'] = round(valid_ratio, 3)
        report['dominant_freq_hz']     = round(dominant_freq, 3)
        report['dominant_bpm_rough']   = dominant_bpm

        if valid_ratio < 0.1:
            issues.append(
                f"Low power in HR range: "
                f"{valid_ratio:.1%} of total power"
            )
    else:
        issues.append("Cannot assess frequency content")

    quality_ok = len(issues) == 0
    return quality_ok, issues, report


def print_pos_report(pulse, fps, alpha, s1, s2):
    """
    Print POS extraction report to terminal.

    Input:
        pulse — raw pulse waveform (N,)
        fps   — actual fps
        alpha — adaptive weight used
        s1    — first projection signal
        s2    — second projection signal
    """
    quality_ok, issues, report = \
        assess_pos_quality(pulse, fps)

    print(f"\n{'='*45}")
    print(f"STEP 5 — POS PULSE EXTRACTION")
    print(f"{'='*45}")

    print(f"\nPOS parameters:")
    print(f"  alpha (adaptive weight): {alpha:.4f}")
    print(f"  S1 std: {np.std(s1):.6f}")
    print(f"  S2 std: {np.std(s2):.6f}")

    print(f"\nPulse signal:")
    print(f"  Length:    {len(pulse)} samples")
    print(f"  Std:       {report['pulse_std']:.6f}")
    print(f"  Max abs:   {report['pulse_max']:.6f}")

    if 'dominant_bpm_rough' in report:
        print(f"\nRough frequency estimate (pre-filter):")
        print(f"  Dominant:  {report['dominant_bpm_rough']:.1f} BPM")
        print(f"  HR power:  {report['valid_hr_power_ratio']:.1%} of total")

    print(f"\nQuality OK: {quality_ok}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues — ready for bandpass filter")

    print(f"{'='*45}")
    return quality_ok