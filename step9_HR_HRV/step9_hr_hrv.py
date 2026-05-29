import numpy as np

# ─────────────────────────────────────────
# Step 9 — Heart Rate and HRV Calculation
#
# Purpose: Compute final validated heart rate
# and HRV metrics from Step 8 peak detection.
#
# Heart rate: weighted combination of FFT
# frequency estimate and RR interval mean.
# FFT gets higher weight — more robust.
#
# HRV metrics:
#   RMSSD — primary, valid for short recordings
#   SDNN  — secondary, requires longer recordings
#
# Pure functions only.
# No camera, no imports from other steps.
# Called by main.py with Step 8 output.
# ─────────────────────────────────────────

# Clinical HRV thresholds
RMSSD_GOOD     = 50.0   # ms — good autonomic function
RMSSD_NORMAL   = 20.0   # ms — normal lower bound
SDNN_GOOD      = 50.0   # ms — healthy
SDNN_BORDERLINE = 20.0  # ms — borderline

# HR source weights
# FFT more robust than RR mean for short recordings
HR_FFT_WEIGHT = 0.7
HR_RR_WEIGHT  = 0.3

# Maximum acceptable disagreement between
# FFT and RR mean before flagging low confidence
HR_AGREEMENT_THRESHOLD = 10.0  # BPM

# Minimum beats for reliable HRV
MIN_BEATS_HRV = 10


# ─────────────────────────────────────────
# Heart Rate computation
# ─────────────────────────────────────────

def compute_heart_rate(hr_bpm_fft, rr_intervals):
    """
    Compute final validated heart rate.

    Combines two independent estimates:
        1. FFT dominant frequency (weight 0.7)
           more robust — uses entire 30s signal
        2. RR interval mean (weight 0.3)
           more variable — depends on beat detection

    If estimates disagree by more than 10 BPM:
        trust FFT only — RR detection unreliable
        flag as lower confidence

    Input:
        hr_bpm_fft   — FFT heart rate in BPM
        rr_intervals — RR intervals in ms (N,)

    Output:
        final_hr   — validated heart rate BPM
        hr_rr_mean — RR-derived heart rate BPM
        hr_source  — which source was used
        agreement  — difference between estimates BPM
    """
    # Compute RR-derived heart rate
    if len(rr_intervals) >= 2:
        rr_mean_ms = float(np.mean(rr_intervals))
        hr_rr_mean = 60000.0 / rr_mean_ms
    else:
        rr_mean_ms = None
        hr_rr_mean = None

    # If no RR intervals — use FFT only
    if hr_rr_mean is None:
        return (
            round(hr_bpm_fft, 1),
            None,
            'fft_only',
            None
        )

    # Agreement between estimates
    agreement = abs(hr_bpm_fft - hr_rr_mean)

    # If disagreement too large — trust FFT only
    if agreement > HR_AGREEMENT_THRESHOLD:
        return (
            round(hr_bpm_fft, 1),
            round(hr_rr_mean, 1),
            'fft_only',
            round(agreement, 1)
        )

    # Weighted combination
    final_hr = (
        HR_FFT_WEIGHT * hr_bpm_fft +
        HR_RR_WEIGHT  * hr_rr_mean
    )

    return (
        round(final_hr, 1),
        round(hr_rr_mean, 1),
        'combined',
        round(agreement, 1)
    )


# ─────────────────────────────────────────
# HRV computation
# ─────────────────────────────────────────

def compute_rmssd(rr_intervals):
    """
    Compute RMSSD — Root Mean Square of
    Successive Differences.

    Primary HRV metric for short recordings.
    Valid for recordings as short as 10 seconds.
    Recommended by 2017 Task Force guidelines
    for ultra-short HRV analysis.

    More robust to outliers than SDNN because
    it uses differences between consecutive
    intervals — a single outlier only affects
    two differences not the whole distribution.

    Input:
        rr_intervals — RR intervals in ms (N,)
                       need at least 2 intervals

    Output:
        rmssd — in ms, or None if insufficient data
    """
    if len(rr_intervals) < 2:
        return None

    # Successive differences
    successive_diffs = np.diff(rr_intervals)

    # Root mean square
    rmssd = float(
        np.sqrt(np.mean(successive_diffs ** 2))
    )

    return round(rmssd, 1)


def compute_sdnn(rr_intervals):
    """
    Compute SDNN — Standard Deviation of
    Normal-to-Normal intervals.

    Overall HRV over recording period.
    Clinically valid for 5-minute recordings.
    For 30-second recordings — report as
    approximate indicator only.

    Input:
        rr_intervals — RR intervals in ms (N,)
                       need at least 3 intervals

    Output:
        sdnn — in ms, or None if insufficient data
    """
    if len(rr_intervals) < 3:
        return None

    sdnn = float(np.std(rr_intervals, ddof=1))
    return round(sdnn, 1)


def interpret_hrv(rmssd, sdnn):
    """
    Interpret HRV metrics clinically.

    Note: These interpretations are approximate
    for 30-second recordings. Standard clinical
    HRV requires 5-minute recordings. Short-term
    RMSSD is validated but SDNN is indicative only.

    Input:
        rmssd — RMSSD in ms
        sdnn  — SDNN in ms

    Output:
        rmssd_interpretation — string
        sdnn_interpretation  — string
        overall              — string
    """
    if rmssd is None:
        rmssd_interp = "Insufficient data"
    elif rmssd >= RMSSD_GOOD:
        rmssd_interp = "Good autonomic function"
    elif rmssd >= RMSSD_NORMAL:
        rmssd_interp = "Normal"
    else:
        rmssd_interp = "Reduced HRV — possible stress or fatigue"

    if sdnn is None:
        sdnn_interp = "Insufficient data"
    elif sdnn >= SDNN_GOOD:
        sdnn_interp = "Healthy"
    elif sdnn >= SDNN_BORDERLINE:
        sdnn_interp = "Borderline"
    else:
        sdnn_interp = "Poor HRV"

    # Overall based on RMSSD (more reliable short-term)
    if rmssd is None:
        overall = "Cannot assess — insufficient beats"
    elif rmssd >= RMSSD_GOOD:
        overall = "Good"
    elif rmssd >= RMSSD_NORMAL:
        overall = "Normal"
    else:
        overall = "Reduced"

    return rmssd_interp, sdnn_interp, overall


# ─────────────────────────────────────────
# Confidence score
# ─────────────────────────────────────────

def compute_confidence(hr_bpm_fft, hr_source,
                        agreement, snr_ratio,
                        n_beats, quality_ok,
                        rmssd):
    """
    Compute overall measurement confidence score.

    Four factors contribute:
        1. FFT SNR ratio      weight 0.35
           signal clarity in frequency domain
        2. HR source agreement weight 0.25
           FFT and RR mean consistency
        3. Beat count         weight 0.25
           enough beats for reliable HRV
        4. Signal quality     weight 0.15
           Step 3 quality flag

    Output:
        confidence — 0.0 to 1.0
        factors    — dict of individual scores
    """
    factors = {}

    # Factor 1 — SNR score
    # SNR > 100x → 1.0, SNR < 10x → 0.0
    snr_score = float(
        np.clip((snr_ratio - 10) / 90, 0.0, 1.0)
    )
    factors['snr_score'] = round(snr_score, 3)

    # Factor 2 — Agreement score
    if agreement is None or \
       hr_source == 'fft_only':
        # No RR data or large disagreement
        agreement_score = 0.5
    else:
        # Perfect agreement (0 BPM diff) → 1.0
        # 10 BPM diff → 0.0
        agreement_score = float(
            np.clip(1.0 - agreement / 10.0,
                    0.0, 1.0)
        )
    factors['agreement_score'] = round(
        agreement_score, 3
    )

    # Factor 3 — Beat count score
    # 30+ beats → 1.0, < 10 beats → 0.0
    beat_score = float(
        np.clip((n_beats - 10) / 20, 0.0, 1.0)
    )
    factors['beat_score'] = round(beat_score, 3)

    # Factor 4 — Signal quality score
    quality_score = 1.0 if quality_ok else 0.3
    factors['quality_score'] = quality_score

    # Weighted combination
    confidence = (
        snr_score       * 0.35 +
        agreement_score * 0.25 +
        beat_score      * 0.25 +
        quality_score   * 0.15
    )

    return round(confidence, 3), factors


def interpret_confidence(confidence):
    """
    Interpret confidence score for routing decision.

    High   (>= 0.75) — reliable measurement
    Medium (0.50-0.74) — acceptable, flag for review
    Low    (< 0.50)  — unreliable, route to palm

    Input:
        confidence — 0.0 to 1.0

    Output:
        level       — 'high', 'medium', 'low'
        route_palm  — bool, True if should switch
        description — string explanation
    """
    if confidence >= 0.75:
        return (
            'high',
            False,
            "Measurement reliable"
        )
    elif confidence >= 0.50:
        return (
            'medium',
            False,
            "Acceptable — minor quality concerns"
        )
    else:
        return (
            'low',
            True,
            "Low confidence — palm fallback recommended"
        )


# ─────────────────────────────────────────
# Main computation
# ─────────────────────────────────────────

def compute_hr_hrv(hr_bpm_fft, rr_intervals,
                   peak_times, snr_ratio,
                   quality_ok, fps):
    """
    Compute all heart rate and HRV metrics.

    Master function called by main.py.
    Combines all sub-computations into one call.

    Input:
        hr_bpm_fft   — FFT heart rate BPM (Step 8)
        rr_intervals — RR intervals ms (Step 8)
        peak_times   — beat timestamps sec (Step 8)
        snr_ratio    — FFT SNR ratio (Step 7)
        quality_ok   — Step 3 quality flag
        fps          — actual fps

    Output:
        results — dict containing all metrics
    """
    # Heart rate
    final_hr, hr_rr_mean, hr_source, agreement = \
        compute_heart_rate(hr_bpm_fft, rr_intervals)

    # HRV
    rmssd = compute_rmssd(rr_intervals)
    sdnn  = compute_sdnn(rr_intervals)

    # Interpretations
    rmssd_interp, sdnn_interp, hrv_overall = \
        interpret_hrv(rmssd, sdnn)

    # Beat count
    n_beats = len(rr_intervals) + 1 \
              if len(rr_intervals) > 0 else 0

    # Confidence
    confidence, factors = compute_confidence(
        hr_bpm_fft, hr_source, agreement,
        snr_ratio, n_beats, quality_ok, rmssd
    )

    level, route_palm, conf_description = \
        interpret_confidence(confidence)

    results = {
        # Heart rate
        'final_hr':     final_hr,
        'hr_bpm_fft':   round(hr_bpm_fft, 1),
        'hr_rr_mean':   hr_rr_mean,
        'hr_source':    hr_source,
        'hr_agreement': agreement,

        # HRV
        'rmssd':        rmssd,
        'sdnn':         sdnn,
        'n_beats':      n_beats,
        'n_rr':         len(rr_intervals),

        # Interpretations
        'rmssd_interp':  rmssd_interp,
        'sdnn_interp':   sdnn_interp,
        'hrv_overall':   hrv_overall,

        # Confidence
        'confidence':       confidence,
        'confidence_level': level,
        'confidence_desc':  conf_description,
        'route_palm':       route_palm,
        'factors':          factors,
    }

    return results


# ─────────────────────────────────────────
# Report
# ─────────────────────────────────────────

def print_hr_hrv_report(results):
    """
    Print heart rate and HRV report to terminal.

    Input:
        results — dict from compute_hr_hrv()
    """
    print(f"\n{'='*45}")
    print(f"STEP 9 — HEART RATE & HRV REPORT")
    print(f"{'='*45}")

    print(f"\nHeart Rate:")
    print(f"  Final HR:      {results['final_hr']} BPM")
    print(f"  FFT estimate:  {results['hr_bpm_fft']} BPM")
    if results['hr_rr_mean']:
        print(f"  RR estimate:   "
              f"{results['hr_rr_mean']} BPM")
    print(f"  Source:        {results['hr_source']}")
    if results['hr_agreement']:
        print(f"  Agreement:     "
              f"±{results['hr_agreement']} BPM")

    print(f"\nHRV Metrics "
          f"(30s recording — RMSSD primary):")
    print(f"  Beats used:    {results['n_beats']}")
    print(f"  RR intervals:  {results['n_rr']}")

    if results['rmssd'] is not None:
        print(f"  RMSSD:         "
              f"{results['rmssd']} ms  "
              f"← {results['rmssd_interp']}")
    else:
        print(f"  RMSSD:         Insufficient data")

    if results['sdnn'] is not None:
        print(f"  SDNN:          "
              f"{results['sdnn']} ms  "
              f"← {results['sdnn_interp']} "
              f"(indicative only for 30s)")
    else:
        print(f"  SDNN:          Insufficient data")

    print(f"\nHRV Overall:   {results['hrv_overall']}")

    print(f"\nConfidence:")
    print(f"  Score:         {results['confidence']}")
    print(f"  Level:         "
          f"{results['confidence_level'].upper()}")
    print(f"  Description:   "
          f"{results['confidence_desc']}")

    print(f"\nConfidence factors:")
    for factor, value in results['factors'].items():
        print(f"  {factor:20s}: {value:.3f}")

    if results['route_palm']:
        print(f"\n⚠ ROUTING: Low confidence — "
              f"palm fallback recommended")
    else:
        print(f"\n✓ ROUTING: Face measurement accepted")

    print(f"{'='*45}")