import numpy as np

# ─────────────────────────────────────────
# Step 11 — Signal Quality Score (SNR)
#
# FIX 1: SNR dB score recalibrated (ceiling 10 dB)
# FIX 2: Amplitude score — now DYNAMIC via profile
#         Uses profile.get_amplitude_target() as
#         the personal ceiling instead of 0.0055.
#         Dark-skinned / low-light patients are
#         scored against their own realistic best,
#         not a lab benchmark they can never reach.
# FIX 3: HR confidence feeds routing (weight 0.15)
# NEW  : ITA-based routing threshold adjustment
#         Darker skin (lower ITA) → lower thresholds
#         so face is not unfairly routed to palm
#         purely because of skin-tone physics.
# ─────────────────────────────────────────

HR_MIN_BPM = 55
HR_MAX_BPM = 180
HR_MIN_HZ  = HR_MIN_BPM / 60
HR_MAX_HZ  = HR_MAX_BPM / 60

# Default routing thresholds (used when ITA >= 28)
SNR_HIGH_THRESHOLD   = 0.60
SNR_MEDIUM_THRESHOLD = 0.40


def get_routing_thresholds(profile=None):
    """
    Get routing thresholds adjusted for skin tone.

    Why ITA affects routing:
        The amplitude score is the main component
        that differs by skin tone — darker skin
        produces lower filtered signal std because
        melanin absorbs more light before it reaches
        capillaries. The amplitude score is now
        personal (via profile), but the routing
        thresholds should still reflect that darker
        skin tends to produce lower overall SNR scores
        even when the HR measurement is valid.

        Reducing the routing threshold slightly for
        darker skin prevents the system from routing
        a valid face measurement to palm simply
        because physics makes the signal weaker.
        This directly serves your inclusivity thesis.

    Adjustment scale (based on ITA):
        ITA > 28  (FST I-III) → standard thresholds
        ITA 10-28 (FST IV)    → 5% reduction
        ITA -30-10(FST V)     → 10% reduction
        ITA < -30 (FST VI)    → 15% reduction

    Input:
        profile — SubjectProfile (optional)
                  None → standard thresholds

    Output:
        high_threshold   — float (route palm if below)
        medium_threshold — float
        ita_adjustment   — float (reduction applied)
        source           — string for logging
    """
    if profile is None:
        return (SNR_HIGH_THRESHOLD,
                SNR_MEDIUM_THRESHOLD,
                0.0, "default")

    ita = profile.ita

    if ita > 28:
        adjustment = 0.00    # FST I-III: no change
    elif ita > 10:
        adjustment = 0.05    # FST IV: -5%
    elif ita > -30:
        adjustment = 0.10    # FST V:  -10%
    else:
        adjustment = 0.15    # FST VI: -15%

    high_thresh   = round(SNR_HIGH_THRESHOLD   - adjustment, 3)
    medium_thresh = round(SNR_MEDIUM_THRESHOLD - adjustment, 3)

    source = f"ITA={ita:.1f} ({profile.fitzpatrick})"
    return high_thresh, medium_thresh, adjustment, source


def compute_spectral_snr(freqs, power, hr_bpm):
    """
    Spectral SNR — how clearly HR peak dominates.
    Peak power vs mean of all other HR band bins.
    """
    hr_mask  = (freqs >= HR_MIN_HZ) & \
               (freqs <= HR_MAX_HZ)
    hr_power = power[hr_mask]

    if len(hr_power) == 0:
        return 0.0, 0.0

    peak_idx   = int(np.argmax(hr_power))
    peak_power = float(hr_power[peak_idx])

    other_mask = np.ones(len(hr_power), dtype=bool)
    other_mask[peak_idx] = False
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
    score = float(
        np.clip((spectral_snr - 2) / 13, 0.0, 1.0)
    )

    return round(spectral_snr, 2), round(score, 4)


def compute_snr_db(freqs, power):
    """
    SNR in dB — HR band power vs noise power.
    Ceiling: 10 dB (realistic excellent webcam result).
    """
    hr_mask    = (freqs >= HR_MIN_HZ) & \
                 (freqs <= HR_MAX_HZ)
    noise_mask = ~hr_mask

    hr_power    = float(power[hr_mask].sum()) \
                  if hr_mask.sum() > 0 else 0.0
    noise_power = float(power[noise_mask].sum()) \
                  if noise_mask.sum() > 0 else 1e-10

    if noise_power == 0 or hr_power == 0:
        return 0.0, 0.0

    snr_db = 10.0 * np.log10(hr_power / noise_power)
    score  = float(np.clip(snr_db / 10.0, 0.0, 1.0))

    return round(float(snr_db), 2), round(score, 4)


def compute_regularity(rr_intervals):
    """
    RR interval coefficient of variation.
    Low CV = regular heartbeat = clean signal.
    """
    if len(rr_intervals) < 3:
        return 1.0, 0.3

    mean_rr = float(np.mean(rr_intervals))
    std_rr  = float(np.std(rr_intervals))

    if mean_rr == 0:
        return 1.0, 0.0

    cv    = std_rr / mean_rr
    score = float(
        np.clip(1.0 - (cv - 0.05) / 0.25, 0.0, 1.0)
    )

    return round(cv, 4), round(score, 4)


def compute_amplitude_score(filtered, profile=None):
    """
    Signal amplitude score.

    NEW: Uses profile.get_amplitude_target() as the
    personal ceiling when profile is available.

    Why this matters:
        A dark-skinned patient under home lighting
        may only reach filtered std of 0.001.
        Scoring them against a fixed ceiling of 0.006
        (calibrated on lab subjects) gives them 0.000
        — not because their heart isn't beating, but
        because physics gives them a weaker signal.

        Using their personal calibration amplitude
        as the ceiling means they're scored against
        their own realistic best. If they reach 80%
        of their calibration amplitude, they score
        well — because that's what good looks like
        for them specifically.

    Fallback (no profile):
        Uses the recalibrated fixed formula:
        floor=0.0005, ceiling=0.006
        Same as the previous fix.

    Input:
        filtered — filtered pulse waveform (Step 6)
        profile  — SubjectProfile (optional)

    Output:
        amplitude — float (std of filtered signal)
        score     — float 0.0-1.0
    """
    amplitude = float(np.std(filtered))

    if profile is not None and \
       profile.get_amplitude_target() is not None:
        # Personal ceiling from calibration
        # floor = 10% of their target (near-zero signal)
        # ceiling = their personal best (from calibration)
        personal_target = profile.get_amplitude_target()
        floor   = personal_target * 0.10
        ceiling = personal_target

        if ceiling <= floor:
            score = 0.0
        else:
            score = float(
                np.clip(
                    (amplitude - floor) / (ceiling - floor),
                    0.0, 1.0
                )
            )
        score_source = "personal"
    else:
        # Fallback — recalibrated fixed formula
        # floor=0.0005 (dark skin minimum)
        # ceiling=0.006 (realistic webcam maximum)
        score = float(
            np.clip(
                (amplitude - 0.0005) / 0.0055,
                0.0, 1.0
            )
        )
        score_source = "fixed"

    return round(amplitude, 6), round(score, 4), \
           score_source


def compute_snr_score(filtered, freqs, power,
                       hr_bpm, rr_intervals, fps,
                       hr_confidence=None,
                       profile=None):
    """
    Compute final SNR score and routing decision.

    NEW: profile parameter wires in two changes:
        1. Amplitude score uses personal ceiling
           (profile.get_amplitude_target())
        2. Routing thresholds adjusted for skin tone
           (get_routing_thresholds(profile))

    Weights (unchanged):
        Spectral SNR    0.30
        SNR in dB       0.20
        Regularity      0.20
        Amplitude       0.15
        HR confidence   0.15

    Input:
        filtered       — filtered pulse (Step 6)
        freqs          — frequency axis (Step 7)
        power          — power spectrum (Step 7)
        hr_bpm         — heart rate BPM (Step 8)
        rr_intervals   — RR intervals ms (Step 8)
        fps            — actual fps
        hr_confidence  — confidence from Step 9
        profile        — SubjectProfile (optional)
    """
    spectral_snr, spectral_score = \
        compute_spectral_snr(freqs, power, hr_bpm)

    snr_db, db_score = \
        compute_snr_db(freqs, power)

    rr_cv, regularity_score = \
        compute_regularity(rr_intervals)

    amplitude, amplitude_score, amp_source = \
        compute_amplitude_score(filtered, profile)

    if hr_confidence is not None:
        confidence_score = float(
            np.clip(hr_confidence, 0.0, 1.0)
        )
    else:
        confidence_score = 0.5

    snr_score = (
        spectral_score   * 0.30 +
        db_score         * 0.20 +
        regularity_score * 0.20 +
        amplitude_score  * 0.15 +
        confidence_score * 0.15
    )
    snr_score = round(snr_score, 4)

    # ITA-adjusted routing thresholds
    high_thresh, medium_thresh, \
    ita_adjustment, thresh_source = \
        get_routing_thresholds(profile)

    if snr_score >= high_thresh:
        quality_level = 'high'
        route_palm    = False
    elif snr_score >= medium_thresh:
        quality_level = 'medium'
        route_palm    = False
    else:
        quality_level = 'low'
        route_palm    = True

    report = {
        'snr_score':          snr_score,
        'snr_db':             snr_db,
        'spectral_snr':       spectral_snr,
        'spectral_score':     spectral_score,
        'db_score':           db_score,
        'rr_cv':              rr_cv,
        'regularity_score':   regularity_score,
        'amplitude':          amplitude,
        'amplitude_score':    amplitude_score,
        'amplitude_source':   amp_source,
        'confidence_score':   confidence_score,
        'quality_level':      quality_level,
        'route_palm':         route_palm,
        'high_threshold':     high_thresh,
        'medium_threshold':   medium_thresh,
        'ita_adjustment':     ita_adjustment,
        'threshold_source':   thresh_source,
    }

    return (
        snr_score,
        snr_db,
        route_palm,
        quality_level,
        report
    )


def print_snr_report(snr_score, snr_db,
                      route_palm, quality_level,
                      report):
    print(f"\n{'='*45}")
    print(f"STEP 11 — SIGNAL QUALITY SCORE")
    print(f"{'='*45}")

    print(f"\nIndividual metrics:")
    print(f"  Spectral SNR:   "
          f"{report['spectral_snr']:.1f}x  "
          f"(score: {report['spectral_score']:.3f})")
    print(f"  SNR dB:         "
          f"{report['snr_db']:.1f} dB  "
          f"(score: {report['db_score']:.3f})")
    print(f"  RR regularity:  "
          f"CV={report['rr_cv']:.3f}  "
          f"(score: {report['regularity_score']:.3f})")
    print(f"  Amplitude:      "
          f"std={report['amplitude']:.6f}  "
          f"(score: {report['amplitude_score']:.3f})  "
          f"[{report['amplitude_source']}]")
    print(f"  HR confidence:  "
          f"(score: {report['confidence_score']:.3f})"
          f"  ← Step 9")

    print(f"\nWeighted combination:")
    print(f"  Spectral   0.30 × "
          f"{report['spectral_score']:.3f} = "
          f"{0.30 * report['spectral_score']:.4f}")
    print(f"  dB         0.20 × "
          f"{report['db_score']:.3f} = "
          f"{0.20 * report['db_score']:.4f}")
    print(f"  Regularity 0.20 × "
          f"{report['regularity_score']:.3f} = "
          f"{0.20 * report['regularity_score']:.4f}")
    print(f"  Amplitude  0.15 × "
          f"{report['amplitude_score']:.3f} = "
          f"{0.15 * report['amplitude_score']:.4f}")
    print(f"  Confidence 0.15 × "
          f"{report['confidence_score']:.3f} = "
          f"{0.15 * report['confidence_score']:.4f}")

    print(f"\n  SNR Score: {snr_score:.4f}")
    print(f"  Quality:   {quality_level.upper()}")

    print(f"\nRouting thresholds "
          f"[{report['threshold_source']}]:")
    adj = report['ita_adjustment']
    if adj > 0:
        print(f"  ITA adjustment: -{adj:.2f} "
              f"(skin-tone fairness)")
    print(f"  HIGH   >= {report['high_threshold']:.3f}  "
          f"→ accept face result")
    print(f"  MEDIUM >= {report['medium_threshold']:.3f}  "
          f"→ acceptable, use face")
    print(f"  LOW    <  {report['medium_threshold']:.3f}  "
          f"→ route to palm")

    routing_str = '⚠ ROUTE TO PALM' \
                  if route_palm else '✓ FACE ACCEPTED'
    print(f"\nRouting decision: {routing_str}")
    print(f"{'='*45}")