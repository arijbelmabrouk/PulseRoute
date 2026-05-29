import numpy as np
from scipy.signal import detrend

# ─────────────────────────────────────────
# Step 4 — Normalization
#
# Pure functions only.
# No camera, no imports from other steps.
# Called by main.py with raw r, g, b arrays
# from Step 3. Returns normalized arrays
# ready for Step 5 POS algorithm.
# ─────────────────────────────────────────


def normalize_mean(signal):
    """
    Stage 1: Divide signal by its mean.
    Removes DC baseline — absolute brightness.
    Output centered around 1.0.

    Input:
        signal — 1D NumPy array

    Output:
        normalized — array centered around 1.0
        mean_value — original mean
    """
    mean_value = float(np.mean(signal))
    if mean_value == 0:
        return np.zeros_like(signal), 0.0
    return signal / mean_value, round(mean_value, 4)


def detrend_signal(signal):
    """
    Stage 2: Remove linear trend.
    Removes slow lighting drift during recording.
    Output centered around 0.0.

    Input:
        signal — 1D NumPy array (after mean normalization)

    Output:
        detrended — array centered around 0.0
    """
    return detrend(signal, type='linear')


def normalize_signal(signal):
    """
    Full two-stage normalization for one channel.

    Stage 1 — divide by mean (remove DC baseline)
    Stage 2 — linear detrend (remove slow drift)

    Input:
        signal — 1D NumPy array

    Output:
        normalized — fully normalized array (~0 mean)
        mean_value — original mean
        stage1     — after mean norm, before detrend
    """
    stage1, mean_value = normalize_mean(signal)
    normalized         = detrend_signal(stage1)
    return normalized, mean_value, stage1


def normalize_rgb(r, g, b):
    """
    Normalize all three RGB channels independently.

    Each channel normalized separately to preserve
    inter-channel differences used by POS algorithm.

    Input:
        r, g, b — raw channel arrays (N,) from Step 3

    Output:
        r_norm, g_norm, b_norm — normalized arrays
        means   — dict of original means per channel
        stages1 — dict of stage1 results per channel
    """
    r_norm, r_mean, r_s1 = normalize_signal(r)
    g_norm, g_mean, g_s1 = normalize_signal(g)
    b_norm, b_mean, b_s1 = normalize_signal(b)

    return (
        r_norm, g_norm, b_norm,
        {'r': r_mean, 'g': g_mean, 'b': b_mean},
        {'r': r_s1,   'g': g_s1,   'b': b_s1}
    )


def assess_normalization_quality(r_norm, g_norm, b_norm,
                                  r_raw,  g_raw,  b_raw):
    """
    Three quality checks on normalized signal:
        1. Mean removed — should be ~0
        2. Variance preserved — signal not flat
        3. Channel balance — no channel dominates

    Input:
        r_norm, g_norm, b_norm — normalized arrays
        r_raw,  g_raw,  b_raw  — original raw arrays

    Output:
        quality_ok — bool
        report     — dict of statistics
        issues     — list of problem descriptions
    """
    issues = []
    report = {}

    for name, arr in [('r', r_norm),
                       ('g', g_norm),
                       ('b', b_norm)]:
        mean_after = float(np.abs(np.mean(arr)))
        report[f'{name}_mean_after'] = round(mean_after, 6)
        if mean_after > 0.01:
            issues.append(
                f"Mean not removed for {name.upper()}"
            )

    for name, norm in [('r', r_norm),
                        ('g', g_norm),
                        ('b', b_norm)]:
        norm_std = float(np.std(norm))
        report[f'{name}_norm_std'] = round(norm_std, 6)
        if norm_std < 1e-6:
            issues.append(
                f"Flat signal for {name.upper()}"
            )

    stds    = [np.std(r_norm), np.std(g_norm), np.std(b_norm)]
    max_std = max(stds)
    min_std = min(stds)
    if min_std > 0:
        balance = max_std / min_std
        report['channel_balance'] = round(balance, 3)
        if balance > 10:
            issues.append(
                f"Channel imbalance: {balance:.2f}"
            )

    return len(issues) == 0, report, issues


def print_normalization_report(r_norm, g_norm, b_norm,
                                r_raw,  g_raw,  b_raw,
                                means,  modality):
    """
    Print normalization report to terminal.

    Input:
        r_norm, g_norm, b_norm — normalized arrays
        r_raw,  g_raw,  b_raw  — raw arrays
        means                  — dict of original means
        modality               — 'face' or 'palm'
    """
    quality_ok, report, issues = \
        assess_normalization_quality(
            r_norm, g_norm, b_norm,
            r_raw,  g_raw,  b_raw
        )

    print(f"\n{'='*45}")
    print(f"STEP 4 — NORMALIZATION ({modality.upper()})")
    print(f"{'='*45}")
    print(f"DC removed — "
          f"R:{means['r']:.2f}  "
          f"G:{means['g']:.2f}  "
          f"B:{means['b']:.2f}")

    print("\nBefore normalization:")
    for name, arr in [('R', r_raw),
                       ('G', g_raw),
                       ('B', b_raw)]:
        snr = float(np.std(arr) / np.mean(arr) * 100) \
              if np.mean(arr) > 0 else 0
        print(f"  {name}: mean={np.mean(arr):.4f}  "
              f"std={np.std(arr):.6f}  "
              f"SNR={snr:.3f}%")

    print("\nAfter normalization:")
    for name, arr in [('R', r_norm),
                       ('G', g_norm),
                       ('B', b_norm)]:
        print(f"  {name}: mean={np.mean(arr):.8f}  "
              f"std={np.std(arr):.6f}")

    print(f"\nChannel balance: "
          f"{report.get('channel_balance', 'N/A')}")
    print(f"Quality OK: {quality_ok}")

    if issues:
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues — ready for POS algorithm")

    print(f"{'='*45}")
    return quality_ok