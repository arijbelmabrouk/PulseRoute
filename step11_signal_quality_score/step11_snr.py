import numpy as np

# ─────────────────────────────────────────
# Step 11 — Signal Quality Score (SNR)
#
# Purpose: Compute final comprehensive signal
# quality score from multiple metrics.
# This is the definitive routing gate —
# low score triggers palm fallback.
#
# Four metrics combined into one score:
#   1. Spectral SNR     — FFT peak dominance
#   2. SNR in dB        — band power ratio
#   3. Waveform regularity — RR interval CV
#   4. Signal amplitude — pulse strength
#
# Pure functions only.
# No camera, no imports from other steps.
# Called by main.py with outputs from
# Steps 6, 7, 8.
#
# Input:  filtered, freqs, power,
#         hr_bpm, rr_intervals, fps
# Output: snr_score, snr_db,
#         route_palm, quality_level, report
# ─────────────────────────────────────────

# HR band constants — must match Steps 6-8
HR_MIN_BPM = 55
HR_MAX_BPM = 180
HR_MIN_HZ  = HR_MIN_BPM / 60  # 0.917 Hz
HR_MAX_HZ  = HR_MAX_BPM / 60  # 3.000 Hz

# Routing thresholds
# To be calibrated from pilot dataset
# Initial values based on development testing
SNR_HIGH_THRESHOLD   = 0.60  # >= HIGH quality
SNR_MEDIUM_THRESHOLD = 0.40  # >= MEDIUM quality
                              # <  LOW → route palm


# ─────────────────────────────────────────
# Metric 1 — Spectral SNR
# ─────────────────────────────────────────

def compute_spectral_snr(freqs, power, hr_bpm):
    """
    Spectral SNR — how clearly the HR peak
    dominates the HR band.

    Compares power at dominant HR frequency
    vs mean power of all other HR band bins.

    High spectral SNR means one clean frequency
    dominates — reliable heart rate.
    Low spectral SNR means multiple competing
    frequencies — noisy or ambiguous signal.

    Input:
        freqs  — frequency axis Hz
        power  — power spectrum
        hr_bpm — dominant heart rate BPM

    Output:
        spectral_snr — ratio (not dB)
        score        — 0.0 to 1.0
    """
    hr_mask  = (freqs >= HR_MIN_HZ) & \
               (freqs <= HR_MAX_HZ)
    hr_freqs = freqs[hr_mask]
    hr_power = power[hr_mask]

    if len(hr_power) == 0:
        return 0.0, 0.0

    # Peak bin
    peak_idx   = int(np.argmax(hr_power))
    peak_power = float(hr_power[peak_idx])

    # Mean of all other bins in HR band
    other_mask  = np.ones(len(hr_power), dtype=bool)
    other_mask[peak_idx] = False

    # Also exclude ±1 bin around peak
    if peak_idx > 0:
        other_mask[peak_idx - 1] = False
    if peak_idx < len(hr_power) - 1:
        other_mask[peak_idx + 1] = False

    if other_mask.sum() == 0:
        return peak_power, 1.0

    noise_mean = float(np.mean(hr_power[other_mask]))

    if noise_mean == 0:
        return float('inf'), 1.0

    spectral_snr = peak_power / noise_mean

    # Score: SNR > 50x → 1.0, SNR < 3x → 0.0
    score = float(
        np.clip((spectral_snr - 2) / 13, 0.0, 1.0)
    )

    return round(spectral_snr, 2), round(score, 4)


# ─────────────────────────────────────────
# Metric 2 — SNR in dB
# ─────────────────────────────────────────

def compute_snr_db(freqs, power):
    """
    Signal-to-Noise Ratio in decibels.

    Compares total power in HR band vs
    total power outside HR band.

    SNR_dB = 10 * log10(HR_power / noise_power)

    Higher dB = cleaner signal:
        > 10 dB → good
        5-10 dB → acceptable
        < 5  dB → poor

    Input:
        freqs — frequency axis Hz
        power — power spectrum

    Output:
        snr_db — SNR in decibels
        score  — 0.0 to 1.0
    """
    hr_mask   = (freqs >= HR_MIN_HZ) & \
                (freqs <= HR_MAX_HZ)
    noise_mask = ~hr_mask

    hr_power    = float(power[hr_mask].sum()) \
                  if hr_mask.sum() > 0 else 0.0
    noise_power = float(power[noise_mask].sum()) \
                  if noise_mask.sum() > 0 else 1e-10

    if noise_power == 0 or hr_power == 0:
        return 0.0, 0.0

    snr_db = 10.0 * np.log10(
        hr_power / noise_power
    )

    # Score: > 15 dB → 1.0, < 0 dB → 0.0
    score = float(
        np.clip(snr_db / 15.0, 0.0, 1.0)
    )

    return round(float(snr_db), 2), round(score, 4)


# ─────────────────────────────────────────
# Metric 3 — Waveform regularity
# ─────────────────────────────────────────

def compute_regularity(rr_intervals):
    """
    Waveform regularity via RR interval
    coefficient of variation (CV).

    CV = std(RR) / mean(RR)

    Low CV = regular heartbeat = clean signal.
    High CV = irregular intervals = noise peaks
    mixed with real beats.

    Note: Some real HRV exists even in healthy
    subjects. CV < 0.15 is considered regular
    for rPPG purposes (not clinical HRV).

    Input:
        rr_intervals — filtered RR intervals ms

    Output:
        cv    — coefficient of variation
        score — 0.0 to 1.0
    """
    if len(rr_intervals) < 3:
        return 1.0, 0.3  # insufficient data

    mean_rr = float(np.mean(rr_intervals))
    std_rr  = float(np.std(rr_intervals))

    if mean_rr == 0:
        return 1.0, 0.0

    cv = std_rr / mean_rr

    # Score: CV < 0.05 → 1.0, CV > 0.30 → 0.0
    score = float(
        np.clip(1.0 - (cv - 0.05) / 0.25,
                0.0, 1.0)
    )

    return round(cv, 4), round(score, 4)


# ─────────────────────────────────────────
# Metric 4 — Signal amplitude
# ─────────────────────────────────────────

def compute_amplitude_score(filtered):
    """
    Signal amplitude score.

    Measures standard deviation of filtered
    pulse signal. Too small means the heartbeat
    oscillation is too weak to measure reliably.

    Typical values from development testing:
        Good signal:  std > 0.010
        Weak signal:  std 0.003-0.010
        Too weak:     std < 0.003

    Input:
        filtered — filtered pulse waveform

    Output:
        amplitude — std of filtered signal
        score     — 0.0 to 1.0
    """
    amplitude = float(np.std(filtered))

    # Score: std > 0.020 → 1.0, std < 0.003 → 0.0
    score = float(
        np.clip((amplitude - 0.003) / 0.017,
                0.0, 1.0)
    )

    return round(amplitude, 6), round(score, 4)


# ─────────────────────────────────────────
# Combined SNR score
# ─────────────────────────────────────────

def compute_snr_score(filtered, freqs, power,
                       hr_bpm, rr_intervals, fps):
    """
    Compute final comprehensive SNR score.

    Combines four metrics with weights:
        Spectral SNR    0.35  — most important
        SNR in dB       0.25  — band power ratio
        Regularity      0.25  — RR consistency
        Amplitude       0.15  — signal strength

    Weights reflect clinical importance:
        Spectral SNR highest — directly measures
        how cleanly one HR frequency dominates.
        Regularity and amplitude are supporting
        evidence of signal quality.

    Input:
        filtered     — filtered pulse (Step 6)
        freqs        — frequency axis (Step 7)
        power        — power spectrum (Step 7)
        hr_bpm       — heart rate BPM (Step 8)
        rr_intervals — RR intervals ms (Step 8)
        fps          — actual fps

    Output:
        snr_score    — combined 0.0 to 1.0
        snr_db       — SNR in decibels
        route_palm   — bool routing decision
        quality_level — 'high', 'medium', 'low'
        report       — dict of all metrics
    """
    # Compute all four metrics
    spectral_snr, spectral_score = \
        compute_spectral_snr(freqs, power, hr_bpm)

    snr_db, db_score = \
        compute_snr_db(freqs, power)

    rr_cv, regularity_score = \
        compute_regularity(rr_intervals)

    amplitude, amplitude_score = \
        compute_amplitude_score(filtered)

    # Weighted combination
    snr_score = (
        spectral_score    * 0.35 +
        db_score          * 0.25 +
        regularity_score  * 0.25 +
        amplitude_score   * 0.15
    )
    snr_score = round(snr_score, 4)

    # Routing decision
    if snr_score >= SNR_HIGH_THRESHOLD:
        quality_level = 'high'
        route_palm    = False
    elif snr_score >= SNR_MEDIUM_THRESHOLD:
        quality_level = 'medium'
        route_palm    = False
    else:
        quality_level = 'low'
        route_palm    = True

    report = {
        'snr_score':        snr_score,
        'snr_db':           snr_db,
        'spectral_snr':     spectral_snr,
        'spectral_score':   spectral_score,
        'db_score':         db_score,
        'rr_cv':            rr_cv,
        'regularity_score': regularity_score,
        'amplitude':        amplitude,
        'amplitude_score':  amplitude_score,
        'quality_level':    quality_level,
        'route_palm':       route_palm,
    }

    return (
        snr_score,
        snr_db,
        route_palm,
        quality_level,
        report
    )


# ─────────────────────────────────────────
# Report
# ─────────────────────────────────────────

def print_snr_report(snr_score, snr_db,
                      route_palm, quality_level,
                      report):
    """
    Print SNR quality report to terminal.

    Input:
        snr_score     — combined score 0.0-1.0
        snr_db        — SNR in dB
        route_palm    — routing decision bool
        quality_level — 'high', 'medium', 'low'
        report        — full metrics dict
    """
    print(f"\n{'='*45}")
    print(f"STEP 11 — SIGNAL QUALITY SCORE")
    print(f"{'='*45}")

    print(f"\nIndividual metrics:")
    print(f"  Spectral SNR:    "
          f"{report['spectral_snr']:.1f}x  "
          f"(score: {report['spectral_score']:.3f})")
    print(f"  SNR dB:          "
          f"{report['snr_db']:.1f} dB  "
          f"(score: {report['db_score']:.3f})")
    print(f"  RR regularity:   "
          f"CV={report['rr_cv']:.3f}  "
          f"(score: {report['regularity_score']:.3f})")
    print(f"  Amplitude:       "
          f"std={report['amplitude']:.6f}  "
          f"(score: {report['amplitude_score']:.3f})")

    print(f"\nWeighted combination:")
    print(f"  Spectral  0.35 × {report['spectral_score']:.3f} = "
          f"{0.35 * report['spectral_score']:.4f}")
    print(f"  dB        0.25 × {report['db_score']:.3f} = "
          f"{0.25 * report['db_score']:.4f}")
    print(f"  Regularity 0.25 × {report['regularity_score']:.3f} = "
          f"{0.25 * report['regularity_score']:.4f}")
    print(f"  Amplitude  0.15 × {report['amplitude_score']:.3f} = "
          f"{0.15 * report['amplitude_score']:.4f}")

    print(f"\n  SNR Score: {snr_score:.4f}")
    print(f"  Quality:   {quality_level.upper()}")

    print(f"\nThresholds:")
    print(f"  HIGH   >= {SNR_HIGH_THRESHOLD}  "
          f"→ accept face result")
    print(f"  MEDIUM >= {SNR_MEDIUM_THRESHOLD}  "
          f"→ acceptable, use face")
    print(f"  LOW    <  {SNR_MEDIUM_THRESHOLD}  "
          f"→ route to palm")

    print(f"\nRouting decision: "
          f"{'⚠ ROUTE TO PALM' if route_palm else '✓ FACE ACCEPTED'}")

    print(f"{'='*45}")