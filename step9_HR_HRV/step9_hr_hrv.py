import numpy as np

# ─────────────────────────────────────────
# Step 9 — Heart Rate and HRV Calculation
#
# Purpose: Compute final validated heart rate
# and HRV metrics from Step 8 peak detection.
#
# CHANGE from original:
#   Step 9 no longer makes routing decisions.
#   route_palm is always False here.
#   Routing is exclusively Step 11's job.
#   Step 9 only computes HR, HRV, confidence.
#   Confidence score is passed to Step 11
#   as an additional input for its decision.
#
# NEW — Age-adjusted HRV interpretation:
#   HRV (RMSSD) declines naturally with age.
#   A 60-year-old with RMSSD=20ms is normal;
#   the same value in a 25-year-old indicates
#   reduced autonomic function.
#   Labeling both as "Normal" is misleading.
#
#   Age groups (derived from published norms):
#     < 30   — Young adult
#     30–44  — Adult
#     45–59  — Middle-aged
#     60–74  — Older adult
#     >= 75  — Senior
#
#   Age is passed via SubjectProfile.
#   If age is not available, falls back to
#   the original population-average thresholds.
#
# NEW — FPS-dependent RMSSD reporting:
#   RMSSD requires precise beat timing.
#   At 30fps, one sample = 33ms — too coarse
#   for reliable RR interval timing.
#   The system self-reports its own limitation:
#
#     fps < 60  → RMSSD suppressed entirely
#                 "HRV requires 60fps+ camera"
#     60–90 fps → RMSSD reported, low confidence
#     90+ fps   → RMSSD reported normally
#
#   This means a 120fps camera gets full HRV
#   automatically — no code change needed.
# ─────────────────────────────────────────

# ── FPS thresholds for RMSSD reliability ──────
HRV_FPS_MINIMUM    = 60   # below this: not reported
HRV_FPS_FULL       = 90   # above this: full confidence

# ── Population-average thresholds (no age) ────
RMSSD_GOOD   = 50.0
RMSSD_NORMAL = 20.0
SDNN_GOOD    = 50.0
SDNN_BORDERLINE = 20.0

# ── Age-adjusted RMSSD norms ──────────────────
# Reference: Shaffer & Ginsberg (2017), Nunan et al. (2010)
# Values are (good_threshold, normal_threshold)
# "good"   — above average for age group
# "normal" — within expected range for age group
# "reduced"— below expected range for age group
#
# All values in ms for 30-second recordings (RMSSD).
AGE_RMSSD_NORMS = {
    (18,  29): (45.0, 25.0),
    (30,  44): (38.0, 20.0),
    (45,  59): (30.0, 15.0),
    (60,  74): (22.0, 10.0),
    (75, 120): (15.0,  7.0),
}

# HR source weights
HR_FFT_WEIGHT = 0.7
HR_RR_WEIGHT  = 0.3

# Maximum acceptable disagreement
HR_AGREEMENT_THRESHOLD = 10.0  # BPM

# Minimum beats for reliable HRV
MIN_BEATS_HRV = 10


# ─────────────────────────────────────────
# FPS-based HRV availability
# ─────────────────────────────────────────

def get_hrv_fps_status(fps):
    """
    Determine HRV availability based on camera fps.

    At 30fps, one sample = 33ms. Beat timing errors
    of ±33ms per sample produce RR interval errors
    of ±66ms, making RMSSD values unreliable.

    At 120fps, one sample = 8ms — sufficient for
    clinically useful RMSSD from rPPG.

    Input:
        fps — measured camera fps

    Output:
        available     — bool: should RMSSD be computed
        confidence    — "suppressed" / "low" / "normal"
        message       — string for reporting
    """
    if fps < HRV_FPS_MINIMUM:
        return (
            False,
            "suppressed",
            f"HRV requires 60fps+ camera "
            f"(current: {fps:.0f}fps — "
            f"beat timing too coarse at 30fps)"
        )
    elif fps < HRV_FPS_FULL:
        return (
            True,
            "low",
            f"HRV available at reduced confidence "
            f"(current: {fps:.0f}fps — "
            f"full confidence requires 90fps+)"
        )
    else:
        return (
            True,
            "normal",
            f"HRV at full confidence "
            f"(current: {fps:.0f}fps)"
        )


# ─────────────────────────────────────────
# Age-adjusted threshold lookup
# ─────────────────────────────────────────

def get_rmssd_thresholds(age=None):
    """
    Get RMSSD interpretation thresholds for this age.

    If age is None or outside table, returns the
    population-average thresholds (original behavior).

    Input:
        age — integer years (from SubjectProfile)
               None → population average

    Output:
        good_thresh   — RMSSD above this = "Good"
        normal_thresh — RMSSD above this = "Normal"
        age_group     — string label for reporting
    """
    if age is None:
        return RMSSD_GOOD, RMSSD_NORMAL, "population average"

    age = int(age)
    for (age_min, age_max), (good, normal) in \
        AGE_RMSSD_NORMS.items():
        if age_min <= age <= age_max:
            label = f"age {age_min}–{age_max}"
            return good, normal, label

    return RMSSD_GOOD, RMSSD_NORMAL, "population average"


def interpret_hrv(rmssd, sdnn, age=None):
    """
    Interpret RMSSD and SDNN with optional age adjustment.
    Only called when RMSSD is available (fps sufficient).

    Input:
        rmssd — RMSSD in ms (or None)
        sdnn  — SDNN in ms (or None, indicative only)
        age   — integer years (optional)

    Output:
        rmssd_interp — string interpretation
        sdnn_interp  — string interpretation
        overall      — string overall assessment
        age_group    — string for reporting
    """
    good_thresh, normal_thresh, age_group = \
        get_rmssd_thresholds(age)

    if rmssd is None:
        rmssd_interp = "Insufficient data"
    elif rmssd >= good_thresh:
        rmssd_interp = "Good autonomic function"
    elif rmssd >= normal_thresh:
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

    if rmssd is None:
        overall = "Cannot assess — insufficient beats"
    elif rmssd >= good_thresh:
        overall = "Good"
    elif rmssd >= normal_thresh:
        overall = "Normal"
    else:
        overall = "Reduced"

    return rmssd_interp, sdnn_interp, overall, age_group


# ─────────────────────────────────────────
# Core HR computation
# ─────────────────────────────────────────

def compute_heart_rate(hr_bpm_fft, rr_intervals):
    """Compute final validated heart rate."""
    if len(rr_intervals) >= 2:
        rr_mean_ms = float(np.mean(rr_intervals))
        hr_rr_mean = 60000.0 / rr_mean_ms
    else:
        rr_mean_ms = None
        hr_rr_mean = None

    if hr_rr_mean is None:
        return (
            round(hr_bpm_fft, 1),
            None, 'fft_only', None
        )

    agreement = abs(hr_bpm_fft - hr_rr_mean)

    if agreement > HR_AGREEMENT_THRESHOLD:
        return (
            round(hr_bpm_fft, 1),
            round(hr_rr_mean, 1),
            'fft_only',
            round(agreement, 1)
        )

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


def compute_rmssd(rr_intervals):
    if len(rr_intervals) < 2:
        return None
    successive_diffs = np.diff(rr_intervals)
    rmssd = float(
        np.sqrt(np.mean(successive_diffs ** 2))
    )
    return round(rmssd, 1)


def compute_sdnn(rr_intervals):
    if len(rr_intervals) < 3:
        return None
    sdnn = float(np.std(rr_intervals, ddof=1))
    return round(sdnn, 1)


def compute_confidence(hr_bpm_fft, hr_source,
                        agreement, snr_ratio,
                        n_beats, quality_ok):
    """
    Compute measurement confidence score (0.0–1.0).
    Passed to Step 11 for routing decision.

    Four factors:
        1. FFT SNR ratio       weight 0.35
        2. HR source agreement weight 0.25
        3. Beat count          weight 0.25
        4. Signal quality      weight 0.15
    """
    factors = {}

    snr_score = float(
        np.clip((snr_ratio - 10) / 40, 0.0, 1.0)
    )
    factors['snr_score'] = round(snr_score, 3)

    if agreement is None or hr_source == 'fft_only':
        agreement_score = 0.5
    else:
        agreement_score = float(
            np.clip(1.0 - agreement / 10.0, 0.0, 1.0)
        )
    factors['agreement_score'] = round(agreement_score, 3)

    beat_score = float(
        np.clip((n_beats - 10) / 20, 0.0, 1.0)
    )
    factors['beat_score'] = round(beat_score, 3)

    quality_score = 1.0 if quality_ok else 0.3
    factors['quality_score'] = quality_score

    confidence = (
        snr_score       * 0.35 +
        agreement_score * 0.25 +
        beat_score      * 0.25 +
        quality_score   * 0.15
    )

    return round(confidence, 3), factors


# ─────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────

def compute_hr_hrv(hr_bpm_fft, rr_intervals,
                   peak_times, snr_ratio,
                   quality_ok, fps,
                   profile=None):
    """
    Compute all heart rate and HRV metrics.

    RMSSD is only computed when fps is sufficient:
        fps < 60  → RMSSD = None, hrv suppressed
        60–90 fps → RMSSD computed, low confidence
        90+ fps   → RMSSD computed, full confidence

    This means a 120fps camera automatically gets
    full HRV — no configuration needed.

    Input:
        hr_bpm_fft   — dominant HR from FFT (Step 7/8)
        rr_intervals — filtered RR intervals in ms
        peak_times   — beat times in seconds
        snr_ratio    — FFT SNR ratio
        quality_ok   — bool from Step 3
        fps          — measured camera fps
        profile      — SubjectProfile (optional)
                       used for age-adjusted HRV

    Output:
        dict with all HR, HRV, and confidence fields
    """
    final_hr, hr_rr_mean, hr_source, agreement = \
        compute_heart_rate(hr_bpm_fft, rr_intervals)

    n_beats = len(rr_intervals) + 1 \
              if len(rr_intervals) > 0 else 0

    # ── FPS check — determines RMSSD availability ─
    hrv_available, hrv_confidence, hrv_fps_message = \
        get_hrv_fps_status(fps)

    age = None
    if profile is not None:
        age = getattr(profile, 'age', None)

    if hrv_available:
        # Compute RMSSD and SDNN normally
        rmssd = compute_rmssd(rr_intervals)
        sdnn  = compute_sdnn(rr_intervals)

        rmssd_interp, sdnn_interp, \
        hrv_overall, age_group = \
            interpret_hrv(rmssd, sdnn, age)

        # If fps is in the low-confidence range,
        # append a note to the interpretation
        if hrv_confidence == "low" and rmssd is not None:
            rmssd_interp += " (low confidence — 60fps)"
            hrv_overall  += " (low confidence)"

    else:
        # fps too low — suppress RMSSD entirely
        rmssd        = None
        sdnn         = None
        rmssd_interp = "Not available — camera fps too low"
        sdnn_interp  = "Not available — camera fps too low"
        hrv_overall  = "Requires 60fps+ camera"
        age_group    = "N/A"

    confidence, factors = compute_confidence(
        hr_bpm_fft, hr_source, agreement,
        snr_ratio, n_beats, quality_ok
    )

    results = {
        'final_hr':          final_hr,
        'hr_bpm_fft':        round(hr_bpm_fft, 1),
        'hr_rr_mean':        hr_rr_mean,
        'hr_source':         hr_source,
        'hr_agreement':      agreement,
        'rmssd':             rmssd,
        'sdnn':              sdnn,
        'n_beats':           n_beats,
        'n_rr':              len(rr_intervals),
        'rmssd_interp':      rmssd_interp,
        'sdnn_interp':       sdnn_interp,
        'hrv_overall':       hrv_overall,
        'age_group':         age_group,
        'hrv_available':     hrv_available,
        'hrv_confidence':    hrv_confidence,
        'hrv_fps_message':   hrv_fps_message,
        'confidence':        confidence,
        'confidence_level':  'see_step11',
        'confidence_desc':   'Routing decided by Step 11',
        'route_palm':        False,
        'factors':           factors,
    }

    return results


def print_hr_hrv_report(results):
    print(f"\n{'='*45}")
    print(f"STEP 9 — HEART RATE & HRV REPORT")
    print(f"{'='*45}")

    print(f"\nHeart Rate:")
    print(f"  Final HR:      {results['final_hr']} BPM")
    print(f"  FFT estimate:  {results['hr_bpm_fft']} BPM")
    if results['hr_rr_mean']:
        print(f"  RR estimate:   {results['hr_rr_mean']} BPM")
    print(f"  Source:        {results['hr_source']}")
    if results['hr_agreement']:
        print(f"  Agreement:     ±{results['hr_agreement']} BPM")

    print(f"\nHRV ({results['hrv_fps_message']}):")
    print(f"  Age group:     {results['age_group']}")
    print(f"  Beats used:    {results['n_beats']}")
    print(f"  RR intervals:  {results['n_rr']}")

    if results['hrv_available']:
        if results['rmssd'] is not None:
            print(f"  RMSSD:         {results['rmssd']} ms"
                  f"  ← {results['rmssd_interp']}")
        else:
            print(f"  RMSSD:         Insufficient beats")

        if results['sdnn'] is not None:
            print(f"  SDNN:          {results['sdnn']} ms"
                  f"  ← {results['sdnn_interp']}"
                  f" (indicative only for short recordings)")
        else:
            print(f"  SDNN:          Insufficient beats")

        print(f"\nHRV Overall:   {results['hrv_overall']}")
        if results['age_group'] != "N/A":
            print(f"  (norms: {results['age_group']})")
    else:
        # fps too low — show clear message, no numbers
        print(f"  RMSSD:         ✗ Not reported")
        print(f"  SDNN:          ✗ Not reported")
        print(f"\nHRV Overall:   {results['hrv_overall']}")
        print(f"  → {results['hrv_fps_message']}")

    print(f"\nConfidence score: {results['confidence']}")
    print(f"  (passed to Step 11 for routing decision)")
    print(f"\nConfidence factors:")
    for factor, value in results['factors'].items():
        print(f"  {factor:20s}: {value:.3f}")

    print(f"\nNote: Routing decision made by Step 11")
    print(f"{'='*45}")