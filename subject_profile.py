import numpy as np

# ─────────────────────────────────────────
# Subject Profile — Per-Subject Calibration
#
# Purpose: Replace every fixed threshold in
# the pipeline with a value derived from
# THIS specific patient's own signal during
# the first 10 seconds of setup.
#
# Why this matters:
#   Fixed thresholds calibrated on lab subjects
#   under controlled lighting fail on:
#     - Dark skin (lower pixel baseline)
#     - Weak webcams (lower signal amplitude)
#     - Home lighting (variable intensity)
#     - Sick patients (movement, coughing)
#
#   A threshold anchored to the patient's OWN
#   baseline is fair across all skin tones,
#   lighting conditions, and camera qualities.
#
# Built during Step 2 extended setup phase.
# Passed through the entire pipeline.
# ─────────────────────────────────────────


class SubjectProfile:
    """
    Per-subject calibration profile.

    Built once per session during the 10-second
    setup phase. Passed to every downstream step
    that previously used a fixed threshold.

    Attributes (set during calibration):
        ita              — ITA skin tone angle
        fitzpatrick      — FST group I-VI string
        baseline_g_mean  — mean green value at rest
        baseline_g_std   — green channel noise floor
        baseline_r_mean  — mean red at rest
        baseline_b_mean  — mean blue at rest
        motion_threshold — 3× personal noise floor
                           for frame rejection
        amplitude_target — 80% of personal best
                           signal amplitude seen
                           during calibration
        hr_estimate_bpm  — rough HR from first 10s
                           (used to narrow bandpass)
        hr_estimate_hz   — same in Hz
        n_calibration_frames — frames used to build
                               this profile
        is_valid         — True when enough data
                           was collected (≥100 frames)

    Dynamic threshold methods:
        get_motion_threshold()  — for Step 3
        get_amplitude_target()  — for Step 11
        get_bandpass_hint()     — for Step 6
        get_rr_tolerance()      — for Step 8
    """

    def __init__(self):
        # Skin tone
        self.ita         = 55.0
        self.fitzpatrick = "FST III"

        # Green channel baseline (from calibration)
        self.baseline_g_mean = None
        self.baseline_g_std  = None
        self.baseline_r_mean = None
        self.baseline_b_mean = None

        # Dynamic thresholds (computed after calibration)
        self.motion_threshold  = None  # Step 3
        self.amplitude_target  = None  # Step 11
        self.hr_estimate_bpm   = None  # Step 6 hint
        self.hr_estimate_hz    = None
        self.calib_to_filtered_scale = None  # measured scale factor

        # Calibration metadata
        self.n_calibration_frames = 0
        self.is_valid             = False

    # ─────────────────────────────────────
    # Build from calibration data
    # ─────────────────────────────────────

    def build_from_calibration(self, g_samples,
                                r_samples,
                                b_samples,
                                ita_value,
                                fps):
        """
        Compute all profile attributes from
        the calibration pixel samples.

        Called at end of Step 2 extended setup
        after 5 seconds of clean pixel collection.

        Input:
            g_samples — list of per-frame mean G
            r_samples — list of per-frame mean R
            b_samples — list of per-frame mean B
            ita_value — ITA from Step 2
            fps       — camera fps from Step 1
        """
        g = np.array(g_samples, dtype=float)
        r = np.array(r_samples, dtype=float)
        b = np.array(b_samples, dtype=float)

        self.n_calibration_frames = len(g)
        self.ita = float(ita_value)
        self.fitzpatrick = self._ita_to_fitzpatrick(
            self.ita
        )

        if len(g) < 30:
            # Not enough calibration data
            # Fall back to population defaults
            self._set_population_defaults()
            return

        # ── Channel baselines ──────────────────
        self.baseline_g_mean = float(np.mean(g))
        self.baseline_g_std  = float(np.std(g))
        self.baseline_r_mean = float(np.mean(r))
        self.baseline_b_mean = float(np.mean(b))

        # ── Motion threshold ───────────────────
        # 3× the patient's own noise floor.
        # A frame whose green channel deviates by
        # more than this from the rolling mean is
        # a motion artifact (cough, movement, etc).
        #
        # This is personal — a fidgety patient has
        # a higher natural variation, so their
        # threshold is proportionally higher too.
        # Fixed thresholds punish them unfairly.
        #
        # Minimum of 0.5 so threshold is never zero
        # on very stable signals.
        self.motion_threshold = max(
            5.0 * self.baseline_g_std,
            2.0
        )

        # ── Amplitude target ───────────────────
        # 80% of the patient's own best green std
        # seen during calibration.
        # Step 11 amplitude score uses this as
        # the ceiling instead of a fixed 0.006.
        #
        # A dark-skinned patient under home lighting
        # may only reach std=0.001 — this is their
        # honest best, and scoring them against a
        # lab benchmark of 0.006 is unfair.
        # AFTER
        # Raw green std is ~0.3-2.0
        # Filtered pulse std is ~0.0005-0.003
        # The pipeline reduces amplitude by roughly 500x
        # through normalization + POS + bandpass.
        # Scale the calibration std to the filtered domain.
        CALIB_TO_FILTERED_SCALE = self.calibrate_scale_factor(
            g_samples, fps
        )
        self.calib_to_filtered_scale = CALIB_TO_FILTERED_SCALE
        self.amplitude_target = max(
            0.80 * self.baseline_g_std * CALIB_TO_FILTERED_SCALE,
            0.0002
        )

        # ── Rough HR estimate from calibration ─
        # 5 seconds is barely enough for a reliable
        # HR estimate, so we only use it to narrow
        # the bandpass hint — not as a final result.
        if len(g) >= int(fps * 5):
            hr_est = self._estimate_hr_from_signal(
                g, fps
            )
            self.hr_estimate_bpm = hr_est
            self.hr_estimate_hz  = hr_est / 60.0 \
                if hr_est else None

        self.is_valid = True

    def _estimate_hr_from_signal(self, g_signal, fps):
        """
        Rough HR estimate from short calibration signal.
        Used only as bandpass hint — not a result.

        Returns BPM float or None if unreliable.
        """
        try:
            from scipy.signal import butter, filtfilt
            from scipy.fft import rfft, rfftfreq

            # Quick bandpass 0.667–3.0 Hz
            nyq    = fps / 2.0
            low    = 0.667 / nyq
            high   = min(3.0 / nyq, 0.99)
            b, a   = butter(2, [low, high],
                            btype='band')
            filtered = filtfilt(b, a, g_signal)

            # FFT
            N      = len(filtered)
            freqs  = rfftfreq(N, d=1.0 / fps)
            power  = np.abs(rfft(filtered)) ** 2

            hr_mask = (freqs >= 0.667) & \
                      (freqs <= 3.0)
            if hr_mask.sum() == 0:
                return None

            peak_idx = np.argmax(power[hr_mask])
            peak_hz  = freqs[hr_mask][peak_idx]
            peak_bpm = peak_hz * 60.0

            # Only trust if in plausible range
            if 45 <= peak_bpm <= 150:
                return round(float(peak_bpm), 1)
            return None

        except Exception:
            return None

    def _ita_to_fitzpatrick(self, ita):
        """Map ITA angle to Fitzpatrick type string."""
        if ita > 55:
            return "FST I-II"
        elif ita > 28:
            return "FST III"
        elif ita > 10:
            return "FST IV"
        elif ita > -30:
            return "FST V"
        else:
            return "FST VI"

    def _set_population_defaults(self):
        """
        Fallback when calibration data insufficient.
        Uses conservative population-average values.
        """
        self.baseline_g_mean  = 120.0
        self.baseline_g_std   = 2.0
        self.baseline_r_mean  = 100.0
        self.baseline_b_mean  = 80.0
        self.motion_threshold = 6.0   # 3 × 2.0
        self.amplitude_target = 0.001
        self.is_valid         = False  # flag as fallback

    # ─────────────────────────────────────
    # Dynamic threshold getters
    # (used by downstream steps)
    # ─────────────────────────────────────

    def get_motion_threshold(self):
        """
        Per-frame motion rejection threshold for Step 3.

        A frame whose green channel deviates from the
        rolling mean by more than this value is a
        motion artifact and should be skipped.

        Returns patient's personal threshold (5× their
        own noise floor) or a safe default if calibration
        was incomplete.
        """
        if self.motion_threshold is not None:
            return self.motion_threshold
        return 6.0  # safe fallback

    def get_amplitude_target(self):
        """
        Personal amplitude ceiling for Step 11 scoring.

        Step 11 uses this instead of a fixed 0.006 ceiling
        so dark-skinned / low-light patients are scored
        against their own realistic best, not a lab benchmark.
        """
        if self.amplitude_target is not None:
            return self.amplitude_target
        return 0.001  # conservative fallback

    def get_bandpass_hint(self):
        """
        Rough HR estimate for Step 6 bandpass tuning.

        Returns (low_hz, high_hz) window around the
        estimated HR ± 30 BPM.
        Returns None if calibration HR estimate
        was not reliable enough.
        """
        if self.hr_estimate_hz is None:
            return None

        margin_hz = 30.0 / 60.0  # ±30 BPM
        low_hz    = max(
            self.hr_estimate_hz - margin_hz,
            0.667  # absolute minimum 40 BPM
        )
        high_hz   = min(
            self.hr_estimate_hz + margin_hz,
            3.0    # absolute maximum 180 BPM
        )
        return low_hz, high_hz

    def get_rr_tolerance(self, signal_quality=None):
        """
        Dynamic RR filter tolerance for Step 8.

        Tightens when signal is clean, relaxes when noisy.
        Prevents the filter from discarding real beats
        during natural HR variation on noisy signals.

        Input:
            signal_quality — float 0.0–1.0 from Step 11
                             None → returns 0.40 default

        Output:
            tolerance — fraction of median RR
                        (0.30 clean → 0.50 noisy)
        """
        if signal_quality is None:
            return 0.40

        # Clean signal (quality > 0.7): tighter filter
        # Noisy signal (quality < 0.3): looser filter
        tolerance = 0.30 + (1.0 - signal_quality) * 0.20
        return round(float(np.clip(tolerance,
                                   0.30, 0.50)), 3)
    

    def get_std_floor(self):
        """
        Personal HRV reliability floor for Step 11.
        Derived as 20% of the patient's personal
        amplitude_target — already correctly scaled
        to the filtered signal domain.
        Falls back to 0.002 if calibration failed.
        """
        if self.amplitude_target is not None:
            return max(
                self.amplitude_target * 0.10,
                0.0002
            )
        return 0.002

    def get_clip_multiplier(self):
        """
        Personal artifact clipping multiplier for Step 6.

        Fixed 4.0 is too tight for patients with naturally
        high signal variance (strong pulse) and too loose
        for patients with weak signal.

        Derived from calibration:
            - Low baseline_g_std (weak signal) → tighter
            clip (3.0) — artifacts stand out more
            - High baseline_g_std (strong signal) → looser
            clip (5.0) — more natural variance to preserve

        Range: 3.0 – 5.0
        Falls back to 4.0 if calibration failed.
        """
        if self.baseline_g_std is None:
            return 4.0
        # Scale: std=0.3 → 3.0, std=1.0 → 4.0, std=2.0+ → 5.0
        multiplier = 3.0 + (self.baseline_g_std / 2.0)
        return round(float(np.clip(multiplier, 3.0, 5.0)), 2)
    

    def calibrate_scale_factor(self, g_samples, fps):
        """
        Measure the actual ratio between raw green std
        and filtered pulse std for THIS patient on THIS
        device under THIS lighting.
        
        Replaces the hardcoded 0.004 with a measured value.
        """
        try:
            from scipy.signal import butter, filtfilt
            
            g = np.array(g_samples, dtype=float)
            g_norm = g - np.mean(g)
            
            raw_std = float(np.std(g_norm))
            if raw_std < 1e-10:
                return 0.004  # flat signal, use default
            
            # Quick bandpass identical to Step 6
            nyq  = fps / 2.0
            low  = np.clip(0.667 / nyq, 0.001, 0.999)
            high = np.clip(3.0   / nyq, 0.001, 0.999)
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, g_norm)
            
            filtered_std = float(np.std(filtered))
            if filtered_std < 1e-10:
                return 0.004
            
            scale = filtered_std / raw_std
            # Clamp to plausible range — sanity check
            return float(np.clip(scale, 0.001, 0.015))
        
        except Exception:
            return 0.004  # fallback if anything fails
    
    # ─────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────

    def print_profile(self):
        """Print calibration profile to terminal."""
        print(f"\n{'─'*45}")
        print(f"SUBJECT PROFILE (calibration results)")
        print(f"{'─'*45}")
        print(f"  ITA:               {self.ita:.1f}  "
              f"({self.fitzpatrick})")
        print(f"  Calibration frames:{self.n_calibration_frames}")
        print(f"  Valid:             {self.is_valid}")
        if self.baseline_g_mean is not None:
            print(f"  Green baseline:    "
                  f"mean={self.baseline_g_mean:.2f}  "
                  f"std={self.baseline_g_std:.4f}")
            print(f"  Motion threshold:  "
                  f"{self.motion_threshold:.4f}  "
                  f"(3× personal noise floor)")
            print(f"  Amplitude target:  "
                  f"{self.amplitude_target:.6f}  "
                  f"(80% of personal best)")
        if self.hr_estimate_bpm is not None:
            print(f"  HR estimate:       "
                  f"{self.hr_estimate_bpm:.1f} BPM  "
                  f"(calibration hint only)")
        print(f"{'─'*45}")
