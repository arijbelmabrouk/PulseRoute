import numpy as np
from scipy.signal import butter, filtfilt

# ─────────────────────────────────────────
# Step 10 — Respiratory Rate
#
# Purpose: Extract breathing rate from the
# normalized green channel signal.
#
# Method:
#   Stage 1 — Low-pass filter g_norm to
#             isolate respiratory band
#             (0.10-0.50 Hz = 6-30 BrPM)
#   Stage 2 — FFT on filtered signal to
#             find dominant breathing frequency
#   Stage 3 — Adaptive notch filter design
#             for Step 6 pipeline improvement
#
# Why green channel:
#   Breathing creates low-frequency oscillation
#   in skin color via two mechanisms:
#     1. Chest/shoulder motion → face motion
#     2. Respiratory Sinus Arrhythmia (RSA)
#        heart rate modulated by breathing
#
# Pure functions only.
# No camera, no imports from other steps.
# Called by main.py with g_norm from Step 4.
#
# Input:  g_norm — normalized green (Step 4)
#         fps    — actual fps (Step 1)
# Output: rr_bpm — breaths per minute
#         rr_hz  — breathing frequency Hz
# ─────────────────────────────────────────

# Respiratory rate physiological range
RR_MIN_BPM = 6     # very slow / deep breathing
RR_MAX_BPM = 30    # exercise / fast breathing
RR_MIN_HZ  = RR_MIN_BPM / 60   # 0.100 Hz
RR_MAX_HZ  = RR_MAX_BPM / 60   # 0.500 Hz

# Normal resting respiratory range
RR_NORMAL_MIN_BPM = 12
RR_NORMAL_MAX_BPM = 20


# ─────────────────────────────────────────
# Stage 1 — Respiratory signal extraction
# ─────────────────────────────────────────

def extract_respiratory_signal(g_norm, fps):
    """
    Extract low-frequency respiratory signal
    from normalized green channel.

    Applies bandpass filter isolating the
    respiratory frequency band (0.10-0.50 Hz).
    This removes:
        - DC component (already removed in Step 4)
        - Heartbeat (0.917+ Hz)
        - High frequency noise

    Leaving only slow oscillations caused by
    breathing motion and RSA modulation.

    Input:
        g_norm — normalized green channel (N,)
                 centered around 0 from Step 4
        fps    — actual fps from Step 1

    Output:
        resp_signal — respiratory signal (N,)
        valid       — bool, True if extraction ok
    """
    nyquist  = fps / 2.0
    low_norm = RR_MIN_HZ / nyquist
    high_norm = RR_MAX_HZ / nyquist

    # Clamp to valid range
    low_norm  = np.clip(low_norm,  0.001, 0.499)
    high_norm = np.clip(high_norm, 0.002, 0.499)

    if low_norm >= high_norm:
        return np.zeros_like(g_norm), False

    # Check if we have enough samples
    # Need at least 2 full breathing cycles
    # At 6 BrPM = 0.1 Hz, period = 10 seconds
    # So need at least 20 seconds = 20 * fps samples
    min_samples = int(20 * fps)
    if len(g_norm) < min_samples:
        return np.zeros_like(g_norm), False

    try:
        b, a        = butter(4, [low_norm, high_norm],
                             btype='bandpass')
        resp_signal = filtfilt(b, a, g_norm)
        return resp_signal, True
    except Exception:
        return np.zeros_like(g_norm), False


# ─────────────────────────────────────────
# Stage 2 — Respiratory rate via FFT
# ─────────────────────────────────────────
def find_respiratory_peak(freqs, power, rr_mask):
    """
    Find dominant respiratory peak with
    sub-harmonic rejection.

    If detected peak has a harmonic at 2x
    with at least 20% of its power —
    the detected peak is a sub-harmonic.
    Use the 2x frequency instead.

    Input:
        freqs   — full frequency axis Hz
        power   — full power spectrum
        rr_mask — boolean mask for RR band

    Output:
        peak_global_idx — index in freqs array
        peak_hz         — frequency in Hz
    """
    rr_power = power[rr_mask]
    rr_freqs = freqs[rr_mask]

    if len(rr_power) == 0:
        return None, None

    # Initial dominant peak
    peak_local_idx  = int(np.argmax(rr_power))
    peak_global_idx = np.where(rr_mask)[0][
        peak_local_idx
    ]
    peak_hz    = float(freqs[peak_global_idx])
    peak_power = float(power[peak_global_idx])

    # Check if 2x harmonic exists with
    # at least 20% of detected peak power
    harmonic_hz   = peak_hz * 2.0
    harmonic_mask = np.abs(freqs - harmonic_hz) \
                    < 0.05

    if harmonic_mask.sum() > 0:
        harmonic_power = float(
            np.max(power[harmonic_mask])
        )

        if harmonic_power > 0.20 * peak_power \
           and RR_MIN_HZ <= harmonic_hz <= RR_MAX_HZ:
            # Sub-harmonic detected —
            # use 2x frequency instead
            peak_global_idx = int(
                np.argmin(np.abs(freqs - harmonic_hz))
            )
            peak_hz = float(freqs[peak_global_idx])

    return peak_global_idx, peak_hz

def compute_respiratory_rate(g_norm, fps):
    """
    Compute respiratory rate from green channel.

    Master function called by main.py.

    Steps:
        1. Extract respiratory signal via bandpass
        2. Compute FFT of respiratory signal
        3. Find dominant peak in RR band
        4. Refine with quadratic interpolation
        5. Assess quality and confidence

    Input:
        g_norm — normalized green channel (N,)
        fps    — actual fps

    Output:
        rr_bpm        — respiratory rate BrPM
        rr_hz         — breathing frequency Hz
        rr_confidence — 0.0 to 1.0
        rr_snr        — SNR ratio of RR peak
        valid         — bool
    """
    # Stage 1 — extract respiratory signal
    resp_signal, valid = extract_respiratory_signal(
        g_norm, fps
    )

    if not valid:
        return None, None, 0.0, 0.0, False

    # Stage 2 — FFT
    n     = len(resp_signal)
    freqs = np.fft.rfftfreq(n, d=1.0/fps)
    power = np.abs(np.fft.rfft(resp_signal)) ** 2

    # Isolate respiratory band
    rr_mask  = (freqs >= RR_MIN_HZ) & \
               (freqs <= RR_MAX_HZ)
    rr_freqs = freqs[rr_mask]
    rr_power = power[rr_mask]

    if len(rr_power) == 0 or \
       np.max(rr_power) == 0:
        return None, None, 0.0, 0.0, False

    # Find dominant peak sub-harmonic aware
    peak_global_idx, _ = find_respiratory_peak(
        freqs, power, rr_mask
    )
    if peak_global_idx is None:
        return None, None, 0.0, 0.0, False

    # Quadratic interpolation for sub-bin precision
    rr_hz, rr_bpm = _interpolate_rr_peak(
        freqs, power, peak_global_idx
    )

    # SNR — peak power vs noise floor
    noise_mask  = np.abs(freqs - rr_hz) > 0.05
    if noise_mask.sum() > 0:
        noise_floor = float(np.mean(power[noise_mask]))
        peak_power  = float(power[peak_global_idx])
        rr_snr      = peak_power / noise_floor \
                      if noise_floor > 0 else 0.0
    else:
        rr_snr = 0.0

    # Confidence — based on SNR and valid range
    rr_confidence = _compute_rr_confidence(
        rr_bpm, rr_snr, rr_power, rr_freqs
    )

    return (
        round(rr_bpm, 1),
        round(rr_hz, 4),
        round(rr_confidence, 3),
        round(rr_snr, 2),
        True
    )


def _interpolate_rr_peak(freqs, power, peak_idx):
    """
    Quadratic interpolation for sub-bin RR precision.

    Input:
        freqs    — frequency axis Hz
        power    — power spectrum
        peak_idx — index of peak bin

    Output:
        refined_hz  — interpolated frequency Hz
        refined_bpm — interpolated frequency BrPM
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
    refined_hz   = float(freqs[peak_idx]) + \
                   delta * bin_spacing
    refined_bpm  = refined_hz * 60.0

    return round(refined_hz, 5), round(refined_bpm, 2)


def _compute_rr_confidence(rr_bpm, rr_snr,
                            rr_power, rr_freqs):
    """
    Compute confidence in respiratory rate estimate.

    Three factors:
        1. SNR of dominant peak (weight 0.5)
        2. Peak in normal range 12-20 BrPM (weight 0.3)
        3. Peak dominance in RR band (weight 0.2)

    Input:
        rr_bpm   — detected respiratory rate
        rr_snr   — SNR ratio of peak
        rr_power — power in RR band
        rr_freqs — frequencies in RR band Hz

    Output:
        confidence — 0.0 to 1.0
    """
    # Factor 1 — SNR score
    # SNR > 10x → 1.0, SNR < 2x → 0.0
    snr_score = float(
        np.clip((rr_snr - 2) / 8, 0.0, 1.0)
    )

    # Factor 2 — normal range score
    if RR_NORMAL_MIN_BPM <= rr_bpm <= RR_NORMAL_MAX_BPM:
        range_score = 1.0
    elif RR_MIN_BPM <= rr_bpm <= RR_MAX_BPM:
        range_score = 0.5
    else:
        range_score = 0.0

    # Factor 3 — peak dominance
    # How much of total RR band power is at peak
    total_rr_power = float(np.sum(rr_power))
    peak_power     = float(np.max(rr_power))
    dominance      = peak_power / total_rr_power \
                     if total_rr_power > 0 else 0.0
    dominance_score = float(
        np.clip(dominance / 0.5, 0.0, 1.0)
    )

    confidence = (
        snr_score       * 0.5 +
        range_score     * 0.3 +
        dominance_score * 0.2
    )

    return confidence


# ─────────────────────────────────────────
# Stage 3 — Adaptive notch filter
# ─────────────────────────────────────────

def design_notch_filter(rr_hz, fps,
                         notch_width_hz=0.05):
    """
    Design notch filter to remove breathing artifact
    from pulse signal before Step 6 bandpass filter.

    Uses the respiratory frequency detected in
    Stage 2 to create a narrow stopband centered
    exactly at the breathing frequency.

    This is the correct scientific solution that
    replaces the fixed 55 BPM lower cutoff.
    After Step 10 detects the breathing frequency,
    main.py can apply this notch to the raw pulse
    before Step 6 to remove the breathing artifact
    regardless of where it falls in the spectrum.

    Input:
        rr_hz          — breathing frequency Hz
        fps            — actual fps
        notch_width_hz — width of notch in Hz
                         default 0.05 Hz = 3 BPM

    Output:
        b, a — filter coefficients for filtfilt
        valid — bool, True if design succeeded
    """
    nyquist   = fps / 2.0
    low_stop  = max(rr_hz - notch_width_hz, 0.01)
    high_stop = min(rr_hz + notch_width_hz,
                    nyquist - 0.01)

    low_norm  = low_stop  / nyquist
    high_norm = high_stop / nyquist

    if low_norm <= 0 or high_norm >= 1 or \
       low_norm >= high_norm:
        return None, None, False

    try:
        # Bandstop = notch
        b, a = butter(
            2,
            [low_norm, high_norm],
            btype='bandstop'
        )
        return b, a, True
    except Exception:
        return None, None, False


def apply_notch_filter(pulse, rr_hz, fps):
    """
    Apply adaptive notch filter to pulse signal.

    Removes breathing artifact at exact respiratory
    frequency before bandpass filtering in Step 6.

    Called by main.py between Step 5 (POS) and
    Step 6 (bandpass) when respiratory rate is known.

    Input:
        pulse  — raw POS pulse signal (Step 5)
        rr_hz  — breathing frequency from Step 10
        fps    — actual fps

    Output:
        notched — pulse with breathing artifact removed
        applied — bool, True if filter was applied
    """
    b, a, valid = design_notch_filter(rr_hz, fps)

    if not valid:
        return pulse, False

    try:
        notched = filtfilt(b, a, pulse)
        return notched, True
    except Exception:
        return pulse, False


# ─────────────────────────────────────────
# Interpretation
# ─────────────────────────────────────────

def interpret_respiratory_rate(rr_bpm):
    """
    Interpret respiratory rate clinically.

    Normal adult resting: 12-20 BrPM
    Bradypnea (slow):     < 12 BrPM
    Tachypnea (fast):     > 20 BrPM
    Exercise range:       20-30 BrPM

    Input:
        rr_bpm — respiratory rate BrPM

    Output:
        interpretation — string
        normal         — bool
    """
    if rr_bpm is None:
        return "Cannot assess", False
    elif rr_bpm < 6:
        return "Below measurable range", False
    elif rr_bpm < 12:
        return "Bradypnea — slow breathing", False
    elif rr_bpm <= 20:
        return "Normal resting range", True
    elif rr_bpm <= 30:
        return "Elevated — exercise or stress", False
    else:
        return "Above measurable range", False


# ─────────────────────────────────────────
# Report
# ─────────────────────────────────────────

def print_respiratory_report(rr_bpm, rr_hz,
                               rr_confidence,
                               rr_snr, valid):
    """
    Print respiratory rate report to terminal.

    Input:
        rr_bpm        — respiratory rate BrPM
        rr_hz         — breathing frequency Hz
        rr_confidence — confidence 0.0-1.0
        rr_snr        — SNR ratio of peak
        valid         — bool
    """
    print(f"\n{'='*45}")
    print(f"STEP 10 — RESPIRATORY RATE REPORT")
    print(f"{'='*45}")

    if not valid or rr_bpm is None:
        print("Respiratory rate: Could not be determined")
        print("Reason: Insufficient signal length or")
        print("        signal quality too low")
        print(f"{'='*45}")
        return

    interpretation, normal = \
        interpret_respiratory_rate(rr_bpm)

    print(f"\nRespiratory rate:")
    print(f"  Rate:          {rr_bpm:.1f} BrPM")
    print(f"  Frequency:     {rr_hz:.4f} Hz")
    print(f"  Interpretation:{interpretation}")
    print(f"  Normal range:  "
          f"{'Yes' if normal else 'No'} "
          f"(12-20 BrPM)")

    print(f"\nSignal quality:")
    print(f"  SNR ratio:     {rr_snr:.2f}x")
    print(f"  Confidence:    {rr_confidence:.3f}")
    print(f"  Level:         "
          f"{'HIGH' if rr_confidence > 0.7 else 'MEDIUM' if rr_confidence > 0.4 else 'LOW'}")

    print(f"\nAdaptive notch filter:")
    print(f"  Target freq:   {rr_hz:.4f} Hz")
    print(f"  Will remove breathing artifact")
    print(f"  at {rr_bpm:.1f} BrPM from pulse signal")

    print(f"{'='*45}")