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

# ── Calibration quality bounds ─────────────────
# baseline_g_std bounds:
#   Too low  (< 0.05): face not detected, static image,
#                      or covered — no real signal
#   Too high (> 3.0):  occluded face, hand in front,
#                      strong motion during calibration,
#                      or very poor lighting
#
# These bounds come from empirical pilot data.
# Normal face calibration: 0.1 – 1.5
# Normal palm calibration: 0.5 – 3.0 (stronger signal)
CALIB_STD_MIN  = 0.05
CALIB_STD_MAX  = 3.0

# Minimum pixel count for a usable mask
CALIB_MIN_MASK_PIXELS = 500

# Minimum calibration frames for a valid profile
CALIB_MIN_FRAMES = 30


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
        motion_threshold — 5× personal noise floor
                           for frame rejection
        amplitude_target — 80% of personal best
                           signal amplitude seen
                           during calibration
        hr_estimate_bpm  — rough HR from first 10s
                           (used to narrow bandpass)
        hr_estimate_hz   — same in Hz
        n_calibration_frames — frames used to build
                               this profile
        is_valid         — True when enough good data
                           was collected
        validation_reason — why validation failed
                            (None if valid)

    Dynamic threshold methods:
        get_motion_threshold()  — for Step 3
        get_amplitude_target()  — for Step 11
        get_bandpass_hint()     — for Step 6
        get_rr_tolerance()      — for Step 8
        get_std_floor()         — for Step 11
        get_clip_multiplier()   — for Step 6
        validate()              — quality gate
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
        self.motion_threshold        = None
        self.amplitude_target        = None
        self.hr_estimate_bpm         = None
        self.hr_estimate_hz          = None
        self.calib_to_filtered_scale = None

        # Calibration metadata
        self.n_calibration_frames = 0
        self.is_valid             = False
        self.validation_reason    = None

    # ─────────────────────────────────────
    # Calibration quality gate
    # ─────────────────────────────────────

    def validate(self, mask_pixels=None):
        """
        Check whether this calibration profile is
        trustworthy enough to proceed with recording.

        Called after build_from_calibration in Step 2.
        If this returns False, Step 2 should prompt
        the patient to reposition and retry — not
        silently proceed with garbage thresholds.

        Checks:
            1. Enough calibration frames collected
            2. Green std is within plausible range
               (not too low = no face,
                not too high = occluded/moving)
            3. Mask has enough pixels (face visible)
            4. Green mean is in a plausible range
               (not pure black, not overexposed)

        Input:
            mask_pixels — int pixel count of ROI mask
                          (optional, from state)

        Output:
            valid  — bool
            reason — string explaining failure
                     (None if valid)
        """
        # Check 1 — enough frames
        if self.n_calibration_frames < CALIB_MIN_FRAMES:
            reason = (
                f"Too few calibration frames: "
                f"{self.n_calibration_frames} "
                f"(need {CALIB_MIN_FRAMES}). "
                f"Face may not have been visible."
            )
            self.is_valid          = False
            self.validation_reason = reason
            return False, reason

        # Check 2 — green std in plausible range
        if self.baseline_g_std is None:
            reason = "Calibration failed — no signal data."
            self.is_valid          = False
            self.validation_reason = reason
            return False, reason

        if self.baseline_g_std < CALIB_STD_MIN:
            reason = (
                f"Signal too flat: std={self.baseline_g_std:.4f} "
                f"(min {CALIB_STD_MIN}). "
                f"Face may be covered or camera blocked."
            )
            self.is_valid          = False
            self.validation_reason = reason
            return False, reason

        if self.baseline_g_std > CALIB_STD_MAX:
            reason = (
                f"Signal too noisy: std={self.baseline_g_std:.4f} "
                f"(max {CALIB_STD_MAX}). "
                f"Patient moved during calibration, "
                f"face was partially occluded, "
                f"or lighting changed suddenly."
            )
            self.is_valid          = False
            self.validation_reason = reason
            return False, reason

        # Check 3 — mask has enough pixels
        if mask_pixels is not None and \
           mask_pixels < CALIB_MIN_MASK_PIXELS:
            reason = (
                f"ROI too small: {mask_pixels} pixels "
                f"(need {CALIB_MIN_MASK_PIXELS}). "
                f"Move closer to the camera."
            )
            self.is_valid          = False
            self.validation_reason = reason
            return False, reason

        # Check 4 — green mean in plausible range
        if self.baseline_g_mean is not None:
            if self.baseline_g_mean < 10:
                reason = (
                    f"Face too dark: mean_g={self.baseline_g_mean:.1f}. "
                    f"Improve lighting — face is underlit."
                )
                self.is_valid          = False
                self.validation_reason = reason
                return False, reason

            if self.baseline_g_mean > 245:
                reason = (
                    f"Face overexposed: mean_g={self.baseline_g_mean:.1f}. "
                    f"Reduce direct light on face or "
                    f"move away from bright source."
                )
                self.is_valid          = False
                self.validation_reason = reason
                return False, reason

        self.is_valid          = True
        self.validation_reason = None
        return True, None

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

        if len(g) < CALIB_MIN_FRAMES:
            self._set_population_defaults()
            return

        # ── Channel baselines ──────────────────
        self.baseline_g_mean = float(np.mean(g))
        self.baseline_g_std  = float(np.std(g))
        self.baseline_r_mean = float(np.mean(r))
        self.baseline_b_mean = float(np.mean(b))

        # ── Motion threshold ───────────────────
        self.motion_threshold = max(
            5.0 * self.baseline_g_std,
            2.0
        )

        # ── Amplitude target ───────────────────
        CALIB_TO_FILTERED_SCALE = self.calibrate_scale_factor(
            g_samples, fps
        )
        self.calib_to_filtered_scale = CALIB_TO_FILTERED_SCALE
        self.amplitude_target = max(
            0.80 * self.baseline_g_std * CALIB_TO_FILTERED_SCALE,
            0.0002
        )

        # ── Rough HR estimate from calibration ─
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
        Used only as bandpass hint — not a final result.
        Returns BPM float or None if unreliable.
        """
        try:
            from scipy.signal import butter, filtfilt
            from scipy.fft import rfft, rfftfreq

            nyq    = fps / 2.0
            low    = 0.667 / nyq
            high   = min(3.0 / nyq, 0.99)
            b, a   = butter(2, [low, high], btype='band')
            filtered = filtfilt(b, a, g_signal)

            N      = len(filtered)
            freqs  = rfftfreq(N, d=1.0 / fps)
            power  = np.abs(rfft(filtered)) ** 2

            hr_mask = (freqs >= 0.667) & (freqs <= 3.0)
            if hr_mask.sum() == 0:
                return None

            peak_idx = np.argmax(power[hr_mask])
            peak_hz  = freqs[hr_mask][peak_idx]
            peak_bpm = peak_hz * 60.0

            if 45 <= peak_bpm <= 150:
                return round(float(peak_bpm), 1)
            return None

        except Exception:
            return None

    def _ita_to_fitzpatrick(self, ita):
        if ita > 55:
            return "FST I"
        elif ita > 41:
            return "FST II"
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
        is_valid stays False — flag as fallback.
        """
        self.baseline_g_mean  = 120.0
        self.baseline_g_std   = 2.0
        self.baseline_r_mean  = 100.0
        self.baseline_b_mean  = 80.0
        self.motion_threshold = 6.0
        self.amplitude_target = 0.001
        self.is_valid         = False

    # ─────────────────────────────────────
    # Dynamic threshold getters
    # ─────────────────────────────────────

    def get_motion_threshold(self):
        if self.motion_threshold is not None:
            return self.motion_threshold
        return 6.0

    def get_amplitude_target(self):
        if self.amplitude_target is not None:
            return self.amplitude_target
        return 0.001

    def get_bandpass_hint(self):
        if self.hr_estimate_hz is None:
            return None
        margin_hz = 30.0 / 60.0
        low_hz    = max(self.hr_estimate_hz - margin_hz, 0.667)
        high_hz   = min(self.hr_estimate_hz + margin_hz, 3.0)
        return low_hz, high_hz

    def get_rr_tolerance(self, signal_quality=None):
        if signal_quality is None:
            return 0.40
        tolerance = 0.30 + (1.0 - signal_quality) * 0.20
        return round(float(np.clip(tolerance, 0.30, 0.50)), 3)

    def get_std_floor(self):
        if self.amplitude_target is not None:
            return max(self.amplitude_target * 0.10, 0.0002)
        return 0.002

    def get_clip_multiplier(self):
        if self.baseline_g_std is None:
            return 4.0
        multiplier = 3.0 + (self.baseline_g_std / 2.0)
        return round(float(np.clip(multiplier, 3.0, 5.0)), 2)

    def calibrate_scale_factor(self, g_samples, fps):
        try:
            from scipy.signal import butter, filtfilt

            g        = np.array(g_samples, dtype=float)
            g_norm   = g - np.mean(g)
            raw_std  = float(np.std(g_norm))

            if raw_std < 1e-10:
                return 0.004

            nyq      = fps / 2.0
            low      = np.clip(0.667 / nyq, 0.001, 0.999)
            high     = np.clip(3.0   / nyq, 0.001, 0.999)
            b, a     = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, g_norm)

            filtered_std = float(np.std(filtered))
            if filtered_std < 1e-10:
                return 0.004

            scale = filtered_std / raw_std
            return float(np.clip(scale, 0.001, 0.015))

        except Exception:
            return 0.004

    # ─────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────

    def print_profile(self):
        print(f"\n{'─'*45}")
        print(f"SUBJECT PROFILE (calibration results)")
        print(f"{'─'*45}")
        print(f"  ITA:               {self.ita:.1f}  "
              f"({self.fitzpatrick})")
        print(f"  Calibration frames:{self.n_calibration_frames}")
        print(f"  Valid:             {self.is_valid}")
        if self.validation_reason:
            print(f"  ⚠ Reason:         {self.validation_reason}")
        if self.baseline_g_mean is not None:
            print(f"  Green baseline:    "
                  f"mean={self.baseline_g_mean:.2f}  "
                  f"std={self.baseline_g_std:.4f}")
            print(f"  Motion threshold:  "
                  f"{self.motion_threshold:.4f}")
            print(f"  Amplitude target:  "
                  f"{self.amplitude_target:.6f}")
        if self.hr_estimate_bpm is not None:
            print(f"  HR estimate:       "
                  f"{self.hr_estimate_bpm:.1f} BPM  "
                  f"(calibration hint only)")
        print(f"{'─'*45}")