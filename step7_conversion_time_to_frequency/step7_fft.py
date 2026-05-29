import numpy as np

# ─────────────────────────────────────────
# Step 7 — FFT (Fast Fourier Transform)
#
# Purpose: Convert filtered pulse signal from
# time domain to frequency domain to identify
# the dominant heartbeat frequency.
#
# Tool: NumPy rfft (real FFT — input is real-valued
# so only positive frequencies needed)
#
# Pure functions only.
# No camera, no imports from other steps.
# Called by main.py with filtered pulse from Step 6.
#
# Input:  filtered — shape (N,) from Step 6
#         fps      — actual fps from Step 1
# Output: freqs    — frequency axis in Hz
#         power    — power spectrum
#         bpm_axis — frequency axis in BPM
# ─────────────────────────────────────────

# Valid heart rate range
HR_MIN_BPM = 55
HR_MAX_BPM = 180
HR_MIN_HZ  = HR_MIN_BPM / 60  # 0.667 Hz
HR_MAX_HZ  = HR_MAX_BPM / 60  # 3.000 Hz


def compute_fft(filtered, fps):
    """
    Compute power spectrum of filtered pulse signal.

    Converts time-domain pulse signal to frequency
    domain using NumPy real FFT. Returns the full
    frequency axis, power spectrum, and BPM axis
    for use in peak detection (Step 8).

    Why rfft not fft:
        The pulse signal is real-valued (not complex).
        rfft exploits this and returns only the
        positive frequency half of the spectrum —
        the negative half is a mirror image and
        contains no additional information.
        Result: N/2 + 1 frequency bins instead of N.

    Frequency resolution:
        resolution = fps / N
        At 30fps with 906 samples = 0.033 Hz = 2.0 BPM
        This means we can distinguish heart rates
        that differ by at least 2 BPM — clinically
        acceptable (medical devices require ±3 BPM).

    Input:
        filtered — filtered pulse waveform (N,)
                   from Step 6 bandpass filter
        fps      — actual fps from Step 1

    Output:
        freqs    — frequency axis in Hz (N/2+1,)
        power    — power at each frequency (N/2+1,)
        bpm_axis — frequency axis in BPM (N/2+1,)
        freq_resolution_hz  — Hz per bin
        freq_resolution_bpm — BPM per bin
    """
    n = len(filtered)

    # Frequency axis in Hz
    # d = time between samples = 1/fps
    freqs = np.fft.rfftfreq(n, d=1.0/fps)

    # Power spectrum
    # rfft returns complex values
    # power = magnitude squared
    fft_vals = np.fft.rfft(filtered)
    power    = np.abs(fft_vals) ** 2

    # Frequency axis in BPM
    bpm_axis = freqs * 60.0

    # Frequency resolution
    freq_resolution_hz  = fps / n
    freq_resolution_bpm = freq_resolution_hz * 60.0

    return (
        freqs,
        power,
        bpm_axis,
        round(freq_resolution_hz, 4),
        round(freq_resolution_bpm, 2)
    )


def get_hr_band(freqs, power):
    """
    Extract only the valid heart rate band
    from the full power spectrum.

    Isolates frequencies between 40 and 180 BPM
    for use in peak detection and SNR calculation.

    Input:
        freqs — full frequency axis in Hz
        power — full power spectrum

    Output:
        hr_freqs — frequencies in HR band (Hz)
        hr_power — power in HR band
        hr_bpm   — frequencies in HR band (BPM)
        hr_mask  — boolean mask for indexing
    """
    hr_mask  = (freqs >= HR_MIN_HZ) & \
               (freqs <= HR_MAX_HZ)
    hr_freqs = freqs[hr_mask]
    hr_power = power[hr_mask]
    hr_bpm   = hr_freqs * 60.0

    return hr_freqs, hr_power, hr_bpm, hr_mask


def assess_fft_quality(freqs, power, fps):
    """
    Assess FFT output quality.

    Three checks:
        1. HR band has sufficient power
        2. Dominant peak is clearly above noise floor
           peak-to-noise ratio > 3.0
        3. Frequency resolution is acceptable
           should be < 5 BPM

    Input:
        freqs — frequency axis in Hz
        power — power spectrum
        fps   — actual fps

    Output:
        quality_ok — bool
        issues     — list of problem descriptions
        report     — dict of statistics
    """
    issues = []
    report = {}

    # Frequency resolution
    n                   = len(power) * 2 - 2
    freq_res_bpm        = (fps / n) * 60.0
    report['freq_resolution_bpm'] = round(freq_res_bpm, 2)

    if freq_res_bpm > 5.0:
        issues.append(
            f"Low frequency resolution: "
            f"{freq_res_bpm:.1f} BPM per bin — "
            f"need longer recording"
        )

    # HR band power
    hr_freqs, hr_power, hr_bpm, hr_mask = \
        get_hr_band(freqs, power)

    if len(hr_power) == 0:
        issues.append("No frequencies in HR band")
        return False, issues, report

    total_power  = float(power.sum())
    hr_power_sum = float(hr_power.sum())
    hr_ratio     = hr_power_sum / total_power \
                   if total_power > 0 else 0.0

    report['hr_power_ratio'] = round(hr_ratio, 3)
    report['total_power']    = round(total_power, 4)

    if hr_ratio < 0.3:
        issues.append(
            f"Low HR band power: {hr_ratio:.1%}"
        )

    # Peak to noise ratio
    peak_power   = float(np.max(hr_power))
    peak_idx     = np.argmax(hr_power)
    peak_freq_hz = float(hr_freqs[peak_idx])
    peak_bpm     = peak_freq_hz * 60.0

    # Noise = mean power outside peak ± 0.2 Hz window
    noise_mask = np.abs(freqs - peak_freq_hz) > 0.2
    if noise_mask.sum() > 0:
        noise_floor = float(np.mean(power[noise_mask]))
        snr_ratio   = peak_power / noise_floor \
                      if noise_floor > 0 else 0.0
    else:
        noise_floor = 0.0
        snr_ratio   = 0.0

    report['peak_bpm']    = round(peak_bpm, 1)
    report['peak_freq_hz'] = round(peak_freq_hz, 4)
    report['peak_power']  = round(peak_power, 4)
    report['noise_floor'] = round(noise_floor, 4)
    report['snr_ratio']   = round(snr_ratio, 2)

    if snr_ratio < 3.0:
        issues.append(
            f"Weak peak: SNR ratio = {snr_ratio:.1f} "
            f"(need > 3.0)"
        )

    quality_ok = len(issues) == 0
    return quality_ok, issues, report


def print_fft_report(freqs, power, bpm_axis,
                     freq_res_hz, freq_res_bpm, fps):
    """
    Print FFT analysis report to terminal.

    Input:
        freqs        — frequency axis in Hz
        power        — power spectrum
        bpm_axis     — frequency axis in BPM
        freq_res_hz  — Hz per bin
        freq_res_bpm — BPM per bin
        fps          — actual fps
    """
    quality_ok, issues, report = \
        assess_fft_quality(freqs, power, fps)

    print(f"\n{'='*45}")
    print(f"STEP 7 — FFT REPORT")
    print(f"{'='*45}")

    print(f"\nFFT parameters:")
    print(f"  Samples:         {(len(freqs)-1)*2}")
    print(f"  Frequency bins:  {len(freqs)}")
    print(f"  Resolution:      {freq_res_hz:.4f} Hz  "
          f"({freq_res_bpm:.2f} BPM per bin)")
    print(f"  HR band:         "
          f"{HR_MIN_BPM}-{HR_MAX_BPM} BPM  "
          f"({HR_MIN_HZ:.3f}-{HR_MAX_HZ:.3f} Hz)")

    print(f"\nPower spectrum:")
    print(f"  Total power:     "
          f"{report.get('total_power', 0):.4f}")
    print(f"  HR band ratio:   "
          f"{report.get('hr_power_ratio', 0):.1%}")

    print(f"\nDominant peak:")
    print(f"  Frequency:  {report.get('peak_freq_hz', 0):.4f} Hz")
    print(f"  Heart rate: {report.get('peak_bpm', 0):.1f} BPM")
    print(f"  Peak power: {report.get('peak_power', 0):.4f}")
    print(f"  Noise floor:{report.get('noise_floor', 0):.4f}")
    print(f"  SNR ratio:  {report.get('snr_ratio', 0):.2f}x")

    print(f"\nQuality OK: {quality_ok}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues — ready for peak detection")

    print(f"{'='*45}")
    return quality_ok, report