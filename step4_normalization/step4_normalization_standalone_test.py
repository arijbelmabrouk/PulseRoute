import numpy as np
from scipy.signal import detrend

# ─────────────────────────────────────────
# Step 4 — Normalization
#
# Purpose: Remove DC component (absolute brightness)
# and slow baseline drift from raw RGB signals.
# Keeps only AC component (heartbeat oscillation).
#
# Two-stage approach:
#   Stage 1 — Division by mean
#              removes fixed lighting baseline
#   Stage 2 — Linear detrending
#              removes slow lighting drift
#
# No AI — pure NumPy/SciPy mathematics.
# Works identically for all skin tones.
# ─────────────────────────────────────────


def normalize_mean(signal):
    """
    Stage 1: Normalize signal by dividing by its mean.

    Removes the DC component — the absolute brightness
    baseline caused by lighting intensity.

    Physical meaning:
        raw signal    = skin_reflectance × lighting
        divided by mean cancels lighting component
        result        = relative skin_reflectance only

    Input signal centered around absolute value (e.g. 75).
    Output signal centered around 1.0.

    Example:
        input:  [75.2, 75.8, 75.1, 76.0, 75.5]
        mean:   75.52
        output: [0.996, 1.004, 0.995, 1.006, 1.000]

    Input:
        signal — 1D NumPy array of raw channel values

    Output:
        normalized — 1D NumPy array centered around 1.0
        mean_value — the mean that was divided out
                     stored for reference and logging
    """
    mean_value = float(np.mean(signal))

    if mean_value == 0:
        # Avoid division by zero — return zeros
        return np.zeros_like(signal), 0.0

    normalized = signal / mean_value

    return normalized, round(mean_value, 4)


def detrend_signal(signal):
    """
    Stage 2: Remove linear trend from signal.

    After mean normalization the signal is centered
    around 1.0 but may still have slow linear drift
    caused by gradual lighting changes during the
    30-second recording (e.g. cloud passing a window,
    someone walking past a light source).

    scipy.signal.detrend fits a least-squares linear
    trend to the signal and subtracts it.

    Result is centered around 0.0 — the signal now
    represents only fast oscillations (heartbeat)
    with no slow baseline or linear drift.

    Input:
        signal — 1D NumPy array (after mean normalization)

    Output:
        detrended — 1D NumPy array centered around 0.0
    """
    return detrend(signal, type='linear')


def normalize_signal(signal):
    """
    Full two-stage normalization pipeline for one channel.

    Stage 1 — divide by mean (remove DC baseline)
    Stage 2 — linear detrend (remove slow drift)

    Input:
        signal — 1D NumPy array of raw channel values
                 shape (N,) where N = number of frames

    Output:
        normalized   — fully normalized 1D array
                       centered around 0.0
                       ready for POS algorithm
        mean_value   — original mean (for logging)
        stage1       — after mean normalization only
                       (for comparison/debugging)
    """
    # Stage 1 — mean normalization
    stage1, mean_value = normalize_mean(signal)

    # Stage 2 — linear detrending
    normalized = detrend_signal(stage1)

    return normalized, mean_value, stage1


def normalize_rgb(r, g, b):
    """
    Normalize all three RGB channels.

    Applies the full two-stage normalization pipeline
    to each channel independently.

    Each channel has its own mean and trend —
    normalizing independently preserves the relative
    differences between channels that POS uses
    to separate pulse from noise.

    Input:
        r — raw red   channel array (N,)
        g — raw green channel array (N,)
        b — raw blue  channel array (N,)

    Output:
        r_norm   — normalized red   channel (N,)
        g_norm   — normalized green channel (N,)
        b_norm   — normalized blue  channel (N,)
        means    — dict of original means per channel
        stages1  — dict of stage1 results per channel
                   (after mean norm, before detrend)
                   useful for debugging and comparison
    """
    r_norm, r_mean, r_s1 = normalize_signal(r)
    g_norm, g_mean, g_s1 = normalize_signal(g)
    b_norm, b_mean, b_s1 = normalize_signal(b)

    means = {
        'r': r_mean,
        'g': g_mean,
        'b': b_mean
    }

    stages1 = {
        'r': r_s1,
        'g': g_s1,
        'b': b_s1
    }

    return r_norm, g_norm, b_norm, means, stages1


def assess_normalization_quality(r_norm, g_norm, b_norm,
                                  r_raw, g_raw, b_raw):
    """
    Assess normalization quality by comparing
    raw vs normalized signal statistics.

    Checks three things:

        1. Mean removal — normalized mean should be ~0
           If not zero, normalization failed

        2. SNR improvement — normalized std should be
           similar to or better than raw std/mean ratio
           significant drop suggests over-smoothing

        3. Channel balance — all three channels should
           have similar variance after normalization
           large imbalance indicates quality issue

    Input:
        r_norm, g_norm, b_norm — normalized arrays
        r_raw, g_raw, b_raw   — original raw arrays

    Output:
        quality_ok — bool
        report     — dict of quality statistics
        issues     — list of problem descriptions
    """
    issues = []
    report = {}

    # Check 1 — mean removal
    for name, arr in [('r', r_norm),
                       ('g', g_norm),
                       ('b', b_norm)]:
        mean_after = float(np.abs(np.mean(arr)))
        report[f'{name}_mean_after'] = round(mean_after, 6)
        if mean_after > 0.01:
            issues.append(
                f"Mean not removed for {name.upper()}: "
                f"{mean_after:.6f} (expected ~0)"
            )

    # Check 2 — variance preserved
    # normalized std should be similar to raw SNR
    for name, raw, norm in [
        ('r', r_raw, r_norm),
        ('g', g_raw, g_norm),
        ('b', b_raw, b_norm)
    ]:
        raw_snr  = float(np.std(raw) / np.mean(raw)) \
                   if np.mean(raw) > 0 else 0
        norm_std = float(np.std(norm))
        report[f'{name}_raw_snr']  = round(raw_snr, 6)
        report[f'{name}_norm_std'] = round(norm_std, 6)

        if norm_std < 1e-6:
            issues.append(
                f"Flat signal after normalization "
                f"for {name.upper()}: std={norm_std:.8f}"
            )

    # Check 3 — channel balance
    stds = [
        float(np.std(r_norm)),
        float(np.std(g_norm)),
        float(np.std(b_norm))
    ]
    max_std = max(stds)
    min_std = min(stds)

    if min_std > 0:
        balance_ratio = max_std / min_std
        report['channel_balance_ratio'] = \
            round(balance_ratio, 3)

        if balance_ratio > 10:
            issues.append(
                f"Channel imbalance after normalization: "
                f"ratio={balance_ratio:.2f} "
                f"(one channel dominates)"
            )

    quality_ok = len(issues) == 0
    return quality_ok, report, issues


def print_normalization_report(r_norm, g_norm, b_norm,
                                r_raw, g_raw, b_raw,
                                means):
    """
    Print detailed normalization report to terminal.

    Shows before/after statistics for each channel
    and overall quality assessment.

    Input:
        r_norm, g_norm, b_norm — normalized arrays
        r_raw, g_raw, b_raw   — raw arrays
        means                 — dict of original means
    """
    quality_ok, report, issues = \
        assess_normalization_quality(
            r_norm, g_norm, b_norm,
            r_raw, g_raw, b_raw
        )

    print("\n" + "="*45)
    print("STEP 4 — NORMALIZATION REPORT")
    print("="*45)

    print("\nOriginal means (DC component removed):")
    print(f"  R mean: {means['r']:.4f}")
    print(f"  G mean: {means['g']:.4f}")
    print(f"  B mean: {means['b']:.4f}")

    print("\nBefore normalization (raw):")
    for name, arr in [('R', r_raw),
                       ('G', g_raw),
                       ('B', b_raw)]:
        print(f"  {name}: mean={np.mean(arr):.4f}  "
              f"std={np.std(arr):.4f}  "
              f"SNR={np.std(arr)/np.mean(arr)*100:.3f}%")

    print("\nAfter normalization:")
    for name, arr in [('R', r_norm),
                       ('G', g_norm),
                       ('B', b_norm)]:
        print(f"  {name}: mean={np.mean(arr):.6f}  "
              f"std={np.std(arr):.6f}")

    print(f"\nChannel balance ratio: "
          f"{report.get('channel_balance_ratio', 'N/A')}")
    print(f"Quality OK: {quality_ok}")

    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues — normalized signal ready for POS")

    print("="*45)


# ─────────────────────────────────────────
# Entry point — standalone test
# ─────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    # Add project root to path
    sys.path.insert(0, os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))

    print("Step 4 — Normalization standalone test")
    print("Generating synthetic test signal...")

    # Generate synthetic rPPG-like signal
    # 906 frames at 30fps = 30 seconds
    # 70 BPM = 1.167 Hz heartbeat
    fps        = 30.2
    n_frames   = 906
    heart_rate = 70  # BPM
    freq       = heart_rate / 60  # Hz

    t = np.linspace(0, n_frames / fps, n_frames)

    # Simulate raw signal:
    # baseline + slow drift + heartbeat + noise
    baseline    = 75.0
    slow_drift  = 2.0 * t / t[-1]      # linear drift
    heartbeat   = 0.5 * np.sin(
        2 * np.pi * freq * t
    )
    noise       = 0.1 * np.random.randn(n_frames)

    g_raw = baseline + slow_drift + heartbeat + noise
    r_raw = g_raw * 1.1 + 0.05 * np.random.randn(n_frames)
    b_raw = g_raw * 0.9 + 0.05 * np.random.randn(n_frames)

    r_raw = r_raw.astype(np.float64)
    g_raw = g_raw.astype(np.float64)
    b_raw = b_raw.astype(np.float64)

    print(f"Synthetic signal: {n_frames} frames @ {fps}fps")
    print(f"Simulated heart rate: {heart_rate} BPM")
    print(f"Simulated drift: +{slow_drift[-1]:.2f} "
          f"over 30 seconds")

    # Run normalization
    r_norm, g_norm, b_norm, means, stages1 = \
        normalize_rgb(r_raw, g_raw, b_raw)

    # Print report
    print_normalization_report(
        r_norm, g_norm, b_norm,
        r_raw, g_raw, b_raw,
        means
    )

    # Show what happened to the heartbeat
    # The normalized signal should still contain
    # the 70 BPM oscillation — just centered at 0
    print("\nVerification — heartbeat preserved:")
    print(f"  Raw G std:       {np.std(g_raw):.4f}")
    print(f"  Norm G std:      {np.std(g_norm):.6f}")
    print(f"  Stage1 G std:    {np.std(stages1['g']):.6f}")
    print(f"  Expected ~0.5 amplitude heartbeat "
          f"after normalization: "
          f"{np.std(g_norm):.6f} "
          f"({'OK' if np.std(g_norm) > 1e-4 else 'FLAT'} )")