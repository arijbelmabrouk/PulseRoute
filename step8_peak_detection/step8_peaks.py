import numpy as np
from scipy.signal import find_peaks

# ─────────────────────────────────────────
# Step 8 — Peak Detection
#
# FIX 1: Overlapping windows (80% advance)
# FIX 2: Deduplication after overlap
# FIX 3: RR filter tolerance — DYNAMIC
#         via SubjectProfile.get_rr_tolerance()
# FIX 4: Hard physiological limits 400-1500ms
# FIX 5: Min prominence 2% of signal range
# FIX 6: Gap filling — scans deduplicated beat
#         list for gaps > 1.5x median RR and
#         inserts a missed beat in the middle
#         third of each gap. Fixes the specific
#         failure mode where deduplication removes
#         one of two close detections and leaves
#         a double-length gap that inflates RMSSD.
# NEW  : Quality-weighted peak detection
#         Each candidate beat is scored by
#         local signal quality (SNR of the
#         window around it). Beats in noisy
#         windows are downweighted, not deleted.
#         The dominant FFT peak still gates
#         the period — quality weighting only
#         decides which of two nearby candidates
#         to prefer when deduplication fires.
# ─────────────────────────────────────────

HR_MIN_BPM = 48
HR_MAX_BPM = 180
HR_MIN_HZ  = HR_MIN_BPM / 60
HR_MAX_HZ  = HR_MAX_BPM / 60

HARMONIC_TOLERANCE = 0.05

RR_MIN_MS = 400
RR_MAX_MS = 1500


def interpolate_peak(freqs, power, peak_idx):
    """
    Quadratic interpolation around FFT peak bin
    for sub-bin frequency precision.
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
    Keeps only fundamental frequencies.
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

    return freqs_s[~is_harmonic], powers_s[~is_harmonic]


def find_dominant_peak(freqs, power):
    """
    Find dominant heart rate peak in power spectrum.

    FIX: Harmonic-support scoring.

        Previous behavior: after rejecting harmonics,
        pick the surviving peak with highest raw power.
        Problem: a sub-harmonic at 52 BPM can have
        higher raw power than the true fundamental at
        68 BPM, causing the wrong peak to be selected.

        New behavior: score each candidate by how much
        harmonic energy exists ABOVE it at 2× and 3×
        its frequency. A true fundamental at 68 BPM
        will have detectable power at 136 and 204 BPM.
        A sub-harmonic at 52 BPM will have little or
        no support below the true fundamental.

        Final score = raw_power × (1 + harmonic_bonus)
        harmonic_bonus = sum of power at 2× and 3×
                         normalized to peak's own power.

    Steps:
        1. Isolate valid HR band
        2. Find all local peaks above 10% of max
        3. Score each by harmonic support
        4. Select peak with highest harmonic-weighted score
        5. Refine with quadratic interpolation
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

    # ── Harmonic support scoring ───────────────────
    scored_powers = np.zeros(len(peak_freqs))

    for i, (freq, pwr) in enumerate(
        zip(peak_freqs, peak_powers)
    ):
        harmonic_bonus = 0.0

        for multiple in [2.0, 3.0]:
            harmonic_freq = freq * multiple
            if harmonic_freq > HR_MAX_HZ * 1.5:
                continue

            closest_idx = np.argmin(
                np.abs(freqs - harmonic_freq)
            )
            if abs(freqs[closest_idx] - harmonic_freq) \
               < HARMONIC_TOLERANCE * harmonic_freq:
                harmonic_power = float(power[closest_idx])
                harmonic_bonus += min(
                    harmonic_power / (pwr + 1e-10),
                    1.0
                )

        scored_powers[i] = pwr * (1.0 + harmonic_bonus)

    best_idx     = np.argmax(scored_powers)
    best_freq_hz = float(peak_freqs[best_idx])

    n_harmonics = 0
    for i, freq in enumerate(peak_freqs):
        if i == best_idx:
            continue
        ratio = freq / best_freq_hz
        if abs(ratio - 2.0) < HARMONIC_TOLERANCE or \
           abs(ratio - 3.0) < HARMONIC_TOLERANCE or \
           abs(ratio - 0.5) < HARMONIC_TOLERANCE:
            n_harmonics += 1

    best_global_idx = np.argmin(
        np.abs(freqs - best_freq_hz)
    )

    refined_hz, refined_bpm = interpolate_peak(
        freqs, power, best_global_idx
    )

    return (
        refined_bpm,
        refined_hz,
        float(peak_powers[best_idx]),
        n_peaks,
        n_harmonics
    )


def fill_missing_beats(beat_indices, filtered, fps,
                        median_rr_ms):
    """
    Find beats missed by the overlapping window search.

    After deduplication, scan RR intervals for gaps
    larger than 1.5x the median. These are almost
    certainly missed beats. For each gap, search the
    filtered signal in that window and insert the
    best candidate peak.

    This fixes the specific failure mode where
    deduplication removes one of two close detections
    and leaves a double-length gap that inflates RMSSD.

    The search targets the middle 50% of each gap
    (25%-75%) to avoid the edges where the known
    beats already sit. The local maximum within that
    window is the inserted beat.

    Safety limits:
        - Max 2 fills per gap (prevents infinite loop
          on genuinely flat or noisy regions)
        - Inserted beat must be within physiological
          RR limits (RR_MIN_MS to RR_MAX_MS) relative
          to its neighbors after insertion

    Input:
        beat_indices  — deduplicated beat indices (N,)
        filtered      — filtered pulse waveform
        fps           — actual fps
        median_rr_ms  — median RR interval in ms
                        used as the gap threshold

    Output:
        filled_indices — beat indices with gaps filled
                         sorted ascending
    """
    if len(beat_indices) < 2 or median_rr_ms <= 0:
        return beat_indices

    filled = list(beat_indices)
    i      = 0
    fills_this_gap = 0

    while i < len(filled) - 1:
        gap_samples = filled[i+1] - filled[i]
        gap_ms      = gap_samples / fps * 1000.0

        # Gap larger than 1.5x median → missed beat
        if gap_ms > 1.5 * median_rr_ms and \
           fills_this_gap < 2:

            # Search middle 50% of gap
            search_start = filled[i] + \
                           int(gap_samples * 0.25)
            search_end   = filled[i] + \
                           int(gap_samples * 0.75)
            search_end   = min(search_end,
                               len(filtered) - 1)

            if search_end > search_start + 3:
                window    = filtered[
                    search_start:search_end
                ]
                local_max = int(np.argmax(window)) + \
                            search_start

                # Verify inserted beat creates valid
                # RR intervals on both sides
                rr_left  = (local_max - filled[i]) \
                           / fps * 1000.0
                rr_right = (filled[i+1] - local_max) \
                           / fps * 1000.0

                if RR_MIN_MS <= rr_left  <= RR_MAX_MS \
                and RR_MIN_MS <= rr_right <= RR_MAX_MS:
                    filled.insert(i + 1, local_max)
                    fills_this_gap += 1
                    # Don't increment i — recheck this
                    # gap in case two beats are missing
                    continue
                else:
                    # Inserted beat would create invalid
                    # intervals — skip this gap
                    i += 1
                    fills_this_gap = 0
                    continue
            else:
                i += 1
                fills_this_gap = 0
                continue
        else:
            i += 1
            fills_this_gap = 0

    return np.array(sorted(filled))


def filter_rr_intervals(rr_intervals,
                         tolerance=0.40):
    """
    Remove physiologically impossible RR intervals
    then filter statistical outliers.

    Dynamic tolerance driven by SubjectProfile.get_rr_tolerance():
        Clean signal (quality > 0.7) → 0.30 (tight)
        Average signal               → 0.40 (default)
        Noisy signal (quality < 0.3) → 0.50 (loose)

    Stage 1 — Hard physiological limits
        Removes any interval outside 400-1500ms.

    Stage 2 — Statistical outlier filter (dynamic)
        Removes intervals more than `tolerance` from median.
    """
    if len(rr_intervals) < 2:
        return rr_intervals

    phys_valid = (rr_intervals >= RR_MIN_MS) & \
                 (rr_intervals <= RR_MAX_MS)
    rr_phys = rr_intervals[phys_valid]

    if len(rr_phys) < 2:
        return rr_phys

    median_rr = np.median(rr_phys)
    valid     = np.abs(rr_phys - median_rr) \
                < tolerance * median_rr

    return rr_phys[valid]


# ─────────────────────────────────────────
# Quality-weighted peak utilities
# ─────────────────────────────────────────

def compute_local_snr(signal, center_idx,
                       half_window):
    """
    Compute local SNR around a candidate beat index.

    Method: peak-to-RMS ratio of surrounding window.
    RMS is sensitive to spikes — artifact windows
    have high RMS even with near-zero mean, so
    peak/RMS drops sharply for artifact windows.

    Input:
        signal      — filtered pulse waveform (N,)
        center_idx  — sample index of candidate beat
        half_window — half-width of analysis window

    Output:
        local_snr — float (0.0 if window invalid)
    """
    n     = len(signal)
    start = max(0, center_idx - half_window)
    end   = min(n, center_idx + half_window)

    if end - start < 3:
        return 0.0

    window = signal[start:end]
    rms    = float(np.sqrt(np.mean(window ** 2)))
    peak   = float(np.max(np.abs(window)))

    if rms < 1e-10:
        return 0.0

    return round(peak / rms, 3)


def score_beat_candidates(beat_indices, signal, fps,
                            hr_bpm_fft=None):
    """
    Score each candidate beat by local signal quality.

    Two factors:
        1. local_snr   — peak/RMS of surrounding window
                         (weight 0.6)
        2. prominence  — peak height above neighboring
                         valleys (weight 0.4)

    Used exclusively in deduplication — when two
    candidates are too close, the higher-scored one
    wins. Does not add or remove beats.

    SNR is relative to local noise — fair across
    skin tones unlike absolute amplitude thresholds.

    Input:
        beat_indices — candidate beat sample indices
        signal       — filtered pulse waveform (N,)
        fps          — actual fps
        hr_bpm_fft   — FFT HR estimate (for window sizing)

    Output:
        scores — float array (N_beats,)
    """
    if len(beat_indices) == 0:
        return np.array([])

    if hr_bpm_fft is not None and \
       HR_MIN_BPM <= hr_bpm_fft <= HR_MAX_BPM:
        period_samples = int((60.0 / hr_bpm_fft) * fps)
        half_window    = max(period_samples // 4, 5)
    else:
        half_window = max(int(fps * 0.3), 5)

    signal_range = float(np.max(signal) - np.min(signal))
    if signal_range < 1e-10:
        signal_range = 1.0

    scores = np.zeros(len(beat_indices), dtype=float)

    for i, idx in enumerate(beat_indices):
        snr = compute_local_snr(signal, idx, half_window)

        left_start   = max(0, idx - half_window)
        right_end    = min(len(signal), idx + half_window)
        left_valley  = float(np.min(signal[left_start:idx])) \
                       if idx > left_start \
                       else float(signal[idx])
        right_valley = float(np.min(signal[idx:right_end])) \
                       if right_end > idx \
                       else float(signal[idx])
        valley       = max(left_valley, right_valley)
        prominence   = (float(signal[idx]) - valley) \
                       / signal_range

        scores[i] = snr * 0.6 + prominence * 0.4

    return scores


def deduplicate_peaks_quality(beat_indices, min_distance,
                               scores):
    """
    Quality-aware deduplication of overlapping peaks.

    When two candidates are closer than min_distance,
    keep the one with the higher quality score instead
    of defaulting to the earlier one.

    Input:
        beat_indices  — candidate beat sample indices
        min_distance  — minimum allowed gap in samples
        scores        — quality score per candidate

    Output:
        deduped indices (subset of beat_indices)
    """
    if len(beat_indices) == 0:
        return beat_indices

    sorted_order  = np.argsort(beat_indices)
    sorted_idx    = beat_indices[sorted_order]
    sorted_scores = scores[sorted_order] \
                    if len(scores) == len(beat_indices) \
                    else np.zeros(len(beat_indices))

    kept = [0]

    for i in range(1, len(sorted_idx)):
        last_kept = kept[-1]
        gap       = sorted_idx[i] - sorted_idx[last_kept]

        if gap >= min_distance:
            kept.append(i)
        else:
            if sorted_scores[i] > sorted_scores[last_kept]:
                kept[-1] = i

    return sorted_idx[kept]


def deduplicate_peaks(beat_indices, min_distance):
    """
    Original deduplication (no scores).
    Fallback when score computation fails.
    """
    if len(beat_indices) == 0:
        return beat_indices

    sorted_idx = np.sort(beat_indices)
    deduped    = [sorted_idx[0]]

    for idx in sorted_idx[1:]:
        if idx - deduped[-1] >= min_distance:
            deduped.append(idx)

    return np.array(deduped)


# ─────────────────────────────────────────
# Main detection entry point
# ─────────────────────────────────────────

def detect_beat_peaks(filtered, fps,
                       hr_bpm_fft=None,
                       profile=None,
                       signal_quality=None):
    """
    Detect individual heartbeat peaks in time domain.

    Pipeline:
        1. FFT-guided overlapping window search (80% advance)
        2. Quality-weighted deduplication
        3. Gap filling — insert missed beats in double gaps
        4. RR interval filtering (dynamic tolerance)

    Input:
        filtered       — filtered pulse (Step 6)
        fps            — actual fps
        hr_bpm_fft     — HR from FFT (Step 7)
        profile        — SubjectProfile (optional)
        signal_quality — float 0.0-1.0 from Step 11

    Output:
        peak_indices   — sample indices of beat peaks
        peak_times_sec — times of beats in seconds
        rr_intervals   — filtered RR intervals in ms
        rr_tolerance   — tolerance actually used
    """
    # ── Determine RR tolerance ─────────────────────
    if profile is not None:
        rr_tolerance = profile.get_rr_tolerance(
            signal_quality
        )
    else:
        rr_tolerance = 0.40

    # ── Stage 1 — FFT-guided period detection ──────
    if hr_bpm_fft is not None and \
       HR_MIN_BPM <= hr_bpm_fft <= HR_MAX_BPM:

        period_samples = int((60.0 / hr_bpm_fft) * fps)
        period_samples = max(period_samples, 3)
        n              = len(filtered)
        beat_indices   = []

        advance = max(int(period_samples * 0.80), 1)

        start = 0
        while start + period_samples <= n:
            window     = filtered[
                start:start + period_samples
            ]
            local_peak = int(np.argmax(window)) + start
            beat_indices.append(local_peak)
            start += advance

        remaining = n - start
        if remaining > period_samples * 0.5:
            window     = filtered[start:n]
            local_peak = int(np.argmax(window)) + start
            beat_indices.append(local_peak)

        beat_indices = np.array(beat_indices)
        min_sep      = max(int(period_samples * 0.60), 1)

        # ── Quality-weighted deduplication ─────────
        try:
            scores = score_beat_candidates(
                beat_indices, filtered, fps, hr_bpm_fft
            )
            beat_indices = deduplicate_peaks_quality(
                beat_indices, min_sep, scores
            )
        except Exception:
            beat_indices = deduplicate_peaks(
                beat_indices, min_sep
            )

    else:
        # ── Stage 2 — Prominence-based fallback ────
        min_distance = int(fps / (HR_MAX_BPM / 60))
        min_distance = max(min_distance, 3)

        signal_range   = float(
            np.max(filtered) - np.min(filtered)
        )
        min_height     = float(np.min(filtered)) + \
                         0.10 * signal_range
        min_prominence = 0.02 * signal_range

        beat_indices, _ = find_peaks(
            filtered,
            distance=min_distance,
            height=min_height,
            prominence=min_prominence
        )

    if len(beat_indices) < 2:
        return beat_indices, np.array([]), \
               np.array([]), rr_tolerance

    # ── FIX 6: Gap filling ─────────────────────────
    # Scan for double-length gaps created by
    # deduplication removing one of two close beats.
    # Insert a missed beat in the middle of each gap.
    try:
        peak_times_check = beat_indices / fps
        rr_check         = np.diff(peak_times_check) \
                           * 1000.0
        if len(rr_check) > 0:
            median_rr_check = float(np.median(rr_check))
            n_before        = len(beat_indices)
            beat_indices    = fill_missing_beats(
                beat_indices, filtered, fps,
                median_rr_check
            )
            n_after = len(beat_indices)
            if n_after > n_before:
                pass  # beats were inserted silently
    except Exception:
        pass  # gap filling failed — proceed without it

    if len(beat_indices) < 2:
        return beat_indices, np.array([]), \
               np.array([]), rr_tolerance

    # ── RR intervals ───────────────────────────────
    peak_times_sec = beat_indices / fps
    rr_raw         = np.diff(peak_times_sec) * 1000.0
    rr_filtered    = filter_rr_intervals(
        rr_raw, tolerance=rr_tolerance
    )

    return beat_indices, peak_times_sec, \
           rr_filtered, rr_tolerance


def assess_peak_quality(hr_bpm, peak_hz,
                         peak_power, rr_intervals,
                         n_peaks, n_harmonics):
    """
    Assess peak detection quality.
    """
    issues = []
    report = {}

    report['hr_bpm']      = round(hr_bpm, 2) \
                            if hr_bpm else None
    report['peak_hz']     = round(peak_hz, 4) \
                            if peak_hz else None
    report['n_peaks']     = n_peaks
    report['n_harmonics'] = n_harmonics

    if hr_bpm is None:
        issues.append("No peak detected")
    elif not (HR_MIN_BPM <= hr_bpm <= HR_MAX_BPM):
        issues.append(
            f"HR outside valid range: "
            f"{hr_bpm:.1f} BPM"
        )

    n_beats = len(rr_intervals) + 1 \
              if len(rr_intervals) > 0 else 0
    report['n_beats'] = n_beats

    if n_beats < 10:
        issues.append(
            f"Too few beats for HRV: {n_beats}"
        )

    if len(rr_intervals) > 0:
        rr_min  = float(np.min(rr_intervals))
        rr_max  = float(np.max(rr_intervals))
        rr_mean = float(np.mean(rr_intervals))
        rr_std  = float(np.std(rr_intervals))

        report['rr_min_ms']  = round(rr_min, 1)
        report['rr_max_ms']  = round(rr_max, 1)
        report['rr_mean_ms'] = round(rr_mean, 1)
        report['rr_std_ms']  = round(rr_std, 1)

        if rr_min < 333:
            issues.append(
                f"RR too short: {rr_min:.0f}ms"
            )
        if rr_max > 1500:
            issues.append(
                f"RR too long: {rr_max:.0f}ms"
            )

    quality_ok = len(issues) == 0
    return quality_ok, issues, report


def print_peak_report(hr_bpm, peak_hz, peak_power,
                       rr_intervals, peak_times,
                       n_peaks, n_harmonics,
                       rr_tolerance=0.40):
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
    print(f"  RR tolerance:      "
          f"{rr_tolerance:.2f}  "
          f"({'dynamic' if rr_tolerance != 0.40 else 'default'})")
    print(f"  Deduplication:     quality-weighted")
    print(f"  Gap filling:       active (FIX 6)")

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