# PulseRoute — rPPG Vital Signs System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A contactless vital signs monitor that measures **heart rate**, **heart rate variability (HRV)**, and **respiratory rate** using only a standard webcam — no wearable, no contact, no hardware beyond a camera.

The system uses **remote photoplethysmography (rPPG)**, a technique that detects the microscopic color changes in skin caused by blood pulsing through capillaries with each heartbeat. These changes are invisible to the naked eye but measurable from video frames.

---

## What makes this different

Most existing rPPG systems only target the face, which fails silently for users with dark skin, heavy beards, or face coverings — introducing bias by design.

This system solves that with a **dual-modality routing architecture**:

- **Face first** — better user experience, no hand positioning needed
- **Automatic palm fallback** — triggered when face signal quality is insufficient
- **Signal quality gate** — the system decides which modality to use based on measured SNR, not assumptions about the user's appearance
- **Per-subject adaptive calibration** — every threshold is derived from the patient's own signal, not population averages

This means the system works for **dark skin tones**, **bearded faces**, **niqab wearers**, and any other case where the face reflects insufficient light. The palm signal is 31% stronger in such cases (validated in peer-reviewed literature), and the routing is automatic and transparent.

---

## Per-subject adaptive calibration

Every fixed threshold in the pipeline has been replaced by a value derived from **the patient's own signal** during a 10-second calibration phase at startup.

A `SubjectProfile` object is built during Step 2 and passed through every downstream step. Nothing in the pipeline uses population-average constants for clinical decisions.

| Threshold | Old approach | New approach |
|---|---|---|
| Motion rejection | Fixed delta of 6.0 | 5× patient's own noise floor |
| Amplitude scoring ceiling | Fixed 0.006 | 80% of patient's calibration amplitude, scaled to filtered domain |
| HRV reliability floor | Fixed 0.002 | 20% of patient's personal amplitude target |
| Artifact clipping multiplier | Fixed 4.0 std | 3.0–5.0 std scaled to patient's signal variance |
| Bandpass window | Always 40–180 BPM | ±30 BPM around patient's estimated HR |
| RR filter tolerance | Fixed 40% | 30–50% based on measured signal quality |
| Routing thresholds | Fixed 0.60 / 0.40 | ITA-adjusted per skin tone group |
| HRV interpretation | Population average | Age-adjusted norms when age is available |
| Scale factor (raw→filtered) | Fixed 0.004 empirical | Measured per session via `calibrate_scale_factor()` |

### How the scale factor works

The raw green channel std during calibration (~0.3–2.0) is in a completely different domain from the filtered pulse std after normalization + POS + bandpass (~0.0005–0.003). A fixed bridge factor of 0.004 was derived from one webcam under one lighting condition and would be wrong on other devices.

`calibrate_scale_factor()` runs a mini bandpass filter on the calibration signal and measures the actual ratio between raw std and filtered std **for this patient, on this device, under this lighting**. This measured ratio then drives `amplitude_target` and `get_std_floor()`, making both thresholds self-calibrating every session.

---

## Measured vitals

| Vital | Method | Notes |
|---|---|---|
| Heart Rate | POS algorithm + FFT peak detection | Primary output |
| HRV (RMSSD) | Beat-to-beat RR interval analysis | Autonomic nervous system indicator |
| HRV (SDNN) | Standard deviation of RR intervals | Indicative for short recordings |
| Respiratory Rate | Green channel FFT analysis | Feeds adaptive bandpass filter |
| Signal Quality Score | Multi-metric SNR (5 components) | Drives face/palm routing decision |
| Skin Tone (ITA) | BiSeNet mask photometry | Used for adaptive thresholds |

---

## Pipeline architecture

```
Camera
  │
  ▼
Step 1 ── Camera initialization & FPS measurement
  │
  ▼
Step 2 ── Face ROI extraction (BiSeNet) + Subject Calibration  ← 10s total
  │        Phase 1 (0–5s):  Forehead + cheek mask via semantic segmentation
  │        Phase 2 (5–10s): Pixel sampling → SubjectProfile
  │           - baseline_g_std → motion threshold, amplitude target
  │           - calibrate_scale_factor() → measured raw→filtered ratio
  │           - _estimate_hr_from_signal() → bandpass hint
  │           - ITA → Fitzpatrick type → routing threshold adjustment
  │        SubjectProfile passed to ALL downstream steps
  ▼
Step 3 ── RGB signal extraction + motion rejection
  │        MotionDetector uses profile.get_motion_threshold() (personal)
  │        Rejected frames silently skipped; recording extends automatically
  │        Up to 2.5× target duration if patient is moving
  ▼
Step 4 ── Normalization
  │        DC removal + linear detrending
  │
  ▼
Step 10 ── Respiratory rate detection
  │         FFT on green channel → breathing Hz
  │         Feeds adaptive notch + lower cutoff to Step 6
  │
  ▼
Step 5 ── POS pulse extraction (Wang et al. 2017)
  │        RGB → single pulse waveform
  ▼
Step 6 ── Adaptive bandpass filter
  │        Artifact clipping: ±profile.get_clip_multiplier() std (personal, 3.0–5.0)
  │        Bandpass narrowed around patient's estimated HR if available
  │        Respiratory cutoff takes priority when detected
  ▼
Step 7 ── FFT frequency analysis
  │        Time domain → power spectrum
  ▼
Step 8 ── Peak detection  [two-pass with Step 11 feedback]
  │        Frequency: harmonic-support scoring selects true fundamental
  │        FFT cross-check override catches sub-harmonics
  │        Time: overlapping windows (80% advance) + quality-weighted dedup
  │        Gap filling inserts missed beats in double-length intervals
  │        Dynamic RR tolerance: profile.get_rr_tolerance(signal_quality)
  ▼
Step 9 ── Heart rate & HRV (pass 1 → feeds Step 11 confidence)
  │        Age-adjusted HRV interpretation via SubjectProfile
  ▼
Step 11 ── Signal quality score + routing decision
  │         5-component SNR score
  │         Personal amplitude ceiling: profile.get_amplitude_target()
  │         ITA-adjusted routing thresholds: profile.get_routing_thresholds()
  │         Personal HRV floor: profile.get_std_floor() = amplitude_target × 0.10
  │         Two independent routing triggers (see below)
  ▼
Step 8/9 ── Second pass (quality score now known)
  │          RR re-filtered with real quality-driven tolerance
  │          HR/HRV recomputed from refined intervals
  │
  ├─── FACE ACCEPTED → final summary
  │
  └─── ROUTE TO PALM ──────────────────────────────────────────────┐
                                                                    │
Step 2b ── Palm ROI extraction (MediaPipe) + Palm Calibration  ← 10s
  │         Phase 1 (0–5s): MediaPipe hand landmarks → palm mask
  │         Phase 2 (5–10s): Pixel sampling → palm SubjectProfile
  │         Palm profile is INDEPENDENT from face profile —
  │         palm baseline_g_std is higher (less melanin),
  │         so motion threshold and amplitude target are re-anchored
  │         to the palm's own signal characteristics
  ▼
Step 3b ── Palm RGB signal extraction (35s)
  │         Identical motion rejection using palm profile thresholds
  ▼
Steps 4–11 (palm) — identical signal processing
  │         All thresholds driven by palm SubjectProfile
  │         Modality label switches to "palm" in all reports
  ▼
Step 12 ── Display (in development)
```

---

## Routing decision — when and why

The system routes to palm in two independent ways. Either condition alone triggers the switch.

### Way 1 — Composite SNR score below threshold

The weighted 5-component score falls below the ITA-adjusted routing threshold.

| Component | Weight | What it measures |
|---|---|---|
| Spectral SNR | 0.30 | HR peak dominance over noise bins |
| SNR in dB | 0.20 | HR band power vs total noise power |
| RR regularity | 0.20 | Beat-to-beat consistency (CV of RR intervals) |
| Amplitude | 0.15 | Signal strength vs personal calibration target |
| HR confidence | 0.15 | Step 9 measurement reliability |

Routing thresholds are ITA-adjusted because darker skin produces lower signal amplitude by physics:

| Skin tone | HIGH threshold | MEDIUM threshold |
|---|---|---|
| FST I–III (ITA > 28) | ≥ 0.60 | ≥ 0.40 |
| FST IV (ITA 10–28) | ≥ 0.55 | ≥ 0.35 |
| FST V (ITA −30–10) | ≥ 0.50 | ≥ 0.30 |
| FST VI (ITA < −30) | ≥ 0.45 | ≥ 0.25 |

Score ≥ HIGH → face accepted, result reliable.  
Score ≥ MEDIUM → face accepted, result usable.  
Score < MEDIUM → route to palm.

### Way 2 — Signal amplitude below personal HRV reliability floor

Even when the composite score passes, the system checks whether the filtered signal std is above the patient's personal floor (`profile.get_std_floor()`).

The floor is computed as:
```
std_floor = amplitude_target × 0.10
amplitude_target = 0.80 × baseline_g_std × calibrate_scale_factor()
```

This means the floor is anchored to **this patient's own calibration signal on this device** — not a fixed constant. A patient with a naturally strong signal has a proportionally higher floor. A patient with weak signal (dark skin, dim room) has a lower floor that reflects their realistic capability.

If filtered std falls below this personal floor, individual beat peaks are too close to the noise floor for precise timing. argmax finds slightly wrong samples, RR interval errors of ±50–100ms cascade into RMSSD values of 200–300ms — inflated 3–5× the true value. This is a physics limit, not an algorithm limit.

When triggered, the terminal output shows:
```
Std floor check: std=0.000892  floor=0.000494  [personal]  ⚠ TRIGGERED
Routing decision: ⚠ ROUTE TO PALM (signal too weak for HRV)
```

---

## Motion robustness

Designed for real teleconsultation patients — not lab subjects:

- **Coughing** → motion artifact frames silently rejected, recording extends automatically
- **Talking** → same rejection mechanism
- **Swallowing** → same
- **Slow drift** → artifact clipping in Step 6 (personal multiplier) prevents filter contamination
- **Head turn** → missed beats recovered by gap-filling in Step 8
- **Poor lighting** → personal amplitude scoring adapts to actual signal strength
- **Variable FPS** → measured FPS used throughout, not assumed

The patient does not need to stay perfectly still. The system adapts.

---

## SubjectProfile — full API

Built once per session during Step 2. Passed to every downstream step.

```python
profile = SubjectProfile()
profile.build_from_calibration(
    g_samples,   # per-frame mean green during calibration
    r_samples,   # per-frame mean red
    b_samples,   # per-frame mean blue
    ita_value,   # ITA from BiSeNet mask
    fps          # measured camera FPS
)
```

**Attributes set after calibration:**

| Attribute | Description |
|---|---|
| `ita` | ITA skin tone angle |
| `fitzpatrick` | FST group string (FST I-II through FST VI) |
| `baseline_g_mean` | Mean green pixel value at rest |
| `baseline_g_std` | Green channel noise floor |
| `motion_threshold` | 5× personal noise floor |
| `amplitude_target` | 80% of personal best, scaled to filtered domain |
| `calib_to_filtered_scale` | Measured per session by `calibrate_scale_factor()` |
| `hr_estimate_bpm` | Rough HR from calibration (bandpass hint only) |
| `is_valid` | True when ≥30 calibration frames collected |

**Dynamic threshold getters:**

| Method | Used by | Returns |
|---|---|---|
| `get_motion_threshold()` | Step 3 | Personal frame rejection threshold |
| `get_amplitude_target()` | Step 11 | Personal amplitude scoring ceiling |
| `get_std_floor()` | Step 11 | Personal HRV reliability floor |
| `get_clip_multiplier()` | Step 6 | Personal artifact clipping std multiplier (3.0–5.0) |
| `get_bandpass_hint()` | Step 6 | (low_hz, high_hz) around estimated HR ±30 BPM |
| `get_rr_tolerance(quality)` | Step 8 | Dynamic RR filter tolerance (0.30–0.50) |

---

## Step-by-step detail

### Step 1 — Camera Initialization
Opens the webcam, discards warm-up frames, measures real FPS over 5 seconds. The measured FPS (not the claimed FPS) is used as the timing reference for all downstream calculations.

### Step 2 — Face ROI Extraction + Calibration (10s)
**Phase 1 (0–5s):** BiSeNet semantic segmentation finds forehead and cheek pixels, excluding hair, beard, eyes, and glasses. Computes ITA as a numerical skin tone value.

**Phase 2 (5–10s):** Pixel values are sampled through the locked mask. `calibrate_scale_factor()` runs a mini bandpass on the calibration signal to measure the actual raw→filtered amplitude ratio for this session. `SubjectProfile` is built with all personal thresholds derived from this data.

### Step 3 — RGB Signal Extraction + Motion Rejection
`MotionDetector` compares each frame's green channel mean to a rolling 10-frame mean. Frames deviating by more than `profile.get_motion_threshold()` are silently skipped. Recording extends up to 2.5× target duration if needed.

### Step 4 — Normalization
Divides each channel by its temporal mean, then applies linear detrending. Required input format for POS.

### Step 10 — Respiratory Rate
Detects breathing frequency from the green channel's low-frequency oscillation. Output feeds Step 6 as adaptive notch frequency and lower bandpass cutoff.

### Step 5 — POS Pulse Extraction
Plane-Orthogonal-to-Skin algorithm (Wang et al. 2017). Combines three normalized RGB channels into one pulse waveform.

### Step 6 — Bandpass Filter (Adaptive)
1. **Artifact clipping** — samples beyond ±`profile.get_clip_multiplier()` std are clipped to prevent filter ringing. Multiplier is personal: 3.0 for weak signals (artifacts stand out more), up to 5.0 for strong signals (more natural variance to preserve).
2. **Profile hint** — bandpass narrowed to ±30 BPM around calibration HR estimate if available; respiratory cutoff takes priority.

### Step 7 — FFT
NumPy rfft converts filtered pulse to frequency domain. Resolution ≈ 1.7 BPM per bin for 35-second recording at 30 FPS.

### Step 8 — Peak Detection
**Frequency domain:** Harmonic-support scoring evaluates each candidate by energy at 2× and 3× its frequency. FFT cross-check override: if Step 8 picks a sub-harmonic more than 15 BPM below the Step 7 peak with SNR > 5×, the FFT result overrides.

**Time domain:** Overlapping windows (80% advance) + quality-weighted deduplication + gap filling for intervals >1.5× median RR.

**Two-pass:** First pass with `signal_quality=None`. After Step 11, second pass re-filters with real quality-driven tolerance from `profile.get_rr_tolerance(snr_score)`.

### Step 9 — Heart Rate & HRV
Final HR: weighted combination of FFT (70%) and RR mean (30%), falls back to FFT-only if disagreement >10 BPM. RMSSD (primary) and SDNN (indicative). Age-adjusted HRV interpretation when `profile.age` is set.

### Step 11 — Signal Quality Score + Routing
Two independent routing triggers — see Routing section above.

### Step 2b / Step 3b — Palm Pipeline
Runs only when `route_palm=True`. MediaPipe Hands detects landmarks → three ROI regions (thenar, central, hypothenar). A fresh `SubjectProfile` is built from palm calibration — **independent from the face profile** because palm baseline_g_std is typically higher (less melanin), so all thresholds need to be re-anchored to the palm signal. Steps 4–11 run identically on the palm signal.

---

## Installation

**Requirements:** Python 3.10, webcam

```bash
git clone https://github.com/arijbelmabrouk/PulseRoute.git
cd PulseRoute

python -m venv rppg_env
rppg_env\Scripts\activate        # Windows
# source rppg_env/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

**BiSeNet model weights** (not in repo — too large):

Place `bisenet_resnet18.pth` at:
```
step2_face_ROI_extraction/face_parsing_mask/models/bisenet_resnet18.pth
```

**mediapipe version** — must be 0.10.9 for palm detection compatibility:
```bash
pip install mediapipe==0.10.9
```

---

## Usage

```bash
python run.py
```

- Camera warms up, face ROI established (5s), calibration runs (5s)
- 35-second clean signal recording begins (extends automatically if you move)
- Results printed to terminal at each step
- If face signal quality is insufficient, palm fallback activates automatically

---

## Web dashboard (Step 12)

Two-page real-time dashboard over WebSocket.

| Page | URL | Audience |
|---|---|---|
| Patient view | `http://localhost:5173/patient` | Patient screen during session |
| Doctor dashboard | `http://localhost:5173/doctor` | Clinician monitor |

**Run (three terminals):**

```powershell
# Terminal 1 — FastAPI server
uvicorn step12_display.server:app --port 8000 --reload

# Terminal 2 — React dev server
cd step12_display/frontend
npm run dev

# Terminal 3 — Pipeline
python run_web.py
```

Doctor page features: live metrics per step, normal range indicators with clinical bands, re-measurement trigger button, routing decision with reason, signal quality timeline.

---

## File structure

```
rPPG_project/
├── run.py                               # Full pipeline entry point (Steps 1–11 + palm branch)
├── run_web.py                           # Pipeline with WebSocket publishing for Step 12
├── subject_profile.py                   # Per-subject adaptive calibration profile
│
├── step1_video_capture/
├── step2_face_ROI_extraction/           # BiSeNet semantic face parsing
├── step2_palm_ROI_extraction/           # MediaPipe hand landmark detection
├── step3_signal_extraction/
│   ├── step3_face_signal_bisenet.py     # Face modality + calibration phase
│   ├── step3_palm_signal.py             # Palm modality + palm calibration phase
│   └── step3_rgb_signal.py             # Core extraction + MotionDetector
├── step4_normalization/
├── step5_pulse_signal_extraction/       # POS algorithm (Wang et al. 2017)
├── step6_bandpass_filter/               # Butterworth bandpass + adaptive clipping
├── step7_conversion_time_to_frequency/
├── step8_peak_detection/                # HR peak + beat detection + gap filling
├── step9_HR_HRV/                        # Final HR + RMSSD/SDNN + age norms
├── step10_respiratory_rate/
├── step11_signal_quality_score/         # SNR scoring + ITA-adjusted routing
└── step12_display/                      # FastAPI + React web dashboard
    ├── server.py
    ├── publisher.py
    └── frontend/
```

---

## Current status

| Component | Status |
|---|---|
| Step 1 — Camera | ✅ Complete |
| Step 2 — Face ROI + Calibration | ✅ Complete |
| Step 2b — Palm ROI + Calibration | ✅ Complete |
| Step 3 — RGB extraction + motion rejection | ✅ Complete |
| Step 3b — Palm RGB extraction | ✅ Complete |
| Step 4 — Normalization | ✅ Complete |
| Step 5 — POS | ✅ Complete |
| Step 6 — Bandpass + adaptive clipping | ✅ Complete |
| Step 7 — FFT | ✅ Complete |
| Step 8 — Peak detection (all fixes + two-pass) | ✅ Complete |
| Step 9 — HR & HRV + age norms | ✅ Complete |
| Step 10 — Respiratory rate | ✅ Complete |
| Step 11 — SNR + ITA routing + personal floors | ✅ Complete |
| SubjectProfile — fully dynamic, no hardcoded thresholds | ✅ Complete |
| Palm routing activation in run.py | ✅ Complete |
| Step 12 — Web dashboard (face pipeline) | ✅ Complete |
| Step 12 — Palm routing in run_web.py | 🔄 In development |

---

## Scientific basis

- **POS algorithm:** Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
- **rPPG feasibility:** Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. *Optics Express*, 16(26), 21434–21445.
- **HRV standards:** Task Force of the European Society of Cardiology (1996). Heart rate variability: standards of measurement. *Circulation*, 93(5), 1043–1065.
- **Age-adjusted HRV norms:** Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health*, 5, 258.
- **Skin tone classification:** ITA (Individual Typology Angle) per Chardon et al. (1991), mapped to Fitzpatrick scale for threshold adaptation.
- **Palm signal advantage:** Supported by literature documenting higher superficial capillary density and lower melanin variation in the palm versus facial skin across Fitzpatrick types.

---

## License

MIT

# PulseRoute — rPPG Vital Signs System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A contactless vital signs monitor that measures **heart rate**, **heart rate variability (HRV)**, and **respiratory rate** using only a standard webcam — no wearable, no contact, no hardware beyond a camera.

The system uses **remote photoplethysmography (rPPG)**, a technique that detects the microscopic color changes in skin caused by blood pulsing through capillaries with each heartbeat. These changes are invisible to the naked eye but measurable from video frames.

---

## What makes this different

Most existing rPPG systems only target the face, which fails silently for users with dark skin, heavy beards, or face coverings — introducing bias by design.

This system solves that with a **dual-modality routing architecture**:

- **Face first** — better user experience, no hand positioning needed
- **Automatic palm fallback** — triggered when face signal quality is insufficient
- **Signal quality gate** — the system decides which modality to use based on measured SNR, not assumptions about the user's appearance
- **Per-subject adaptive calibration** — every threshold is derived from the patient's own signal, not population averages
- **Hardware-aware HRV** — RMSSD is suppressed at 30fps and reported automatically at 60fps+, so the system never outputs numbers it cannot reliably produce

This means the system works for **dark skin tones**, **bearded faces**, **niqab wearers**, and any other case where the face reflects insufficient light. The palm signal is 31% stronger in such cases (validated in peer-reviewed literature), and the routing is automatic and transparent.

---

## Per-subject adaptive calibration

Every fixed threshold in the pipeline has been replaced by a value derived from **the patient's own signal** during a 10-second calibration phase at startup.

A `SubjectProfile` object is built during Step 2 and passed through every downstream step. Nothing in the pipeline uses population-average constants for clinical decisions.

| Threshold | Old approach | New approach |
|---|---|---|
| Motion rejection | Fixed delta of 6.0 | 5× patient's own noise floor |
| Amplitude scoring ceiling | Fixed 0.006 | 80% of patient's calibration amplitude, scaled to filtered domain |
| HRV reliability floor | Fixed 0.002 | 10% of patient's personal amplitude target |
| Artifact clipping multiplier | Fixed 4.0 std | 3.0–5.0 std scaled to patient's signal variance |
| Bandpass window | Always 40–180 BPM | ±30 BPM around patient's estimated HR |
| RR filter tolerance | Fixed 40% | 30–50% based on measured signal quality |
| Routing thresholds | Fixed 0.60 / 0.40 | ITA-adjusted per skin tone group |
| HRV interpretation | Population average | Age-adjusted norms when age is available |
| Scale factor (raw→filtered) | Fixed 0.004 empirical | Measured per session via full normalize→POS→bandpass mini-pipeline |

### How the scale factor works

The raw green channel std during calibration (~0.3–2.0) is in a completely different domain from the filtered pulse std after normalization + POS + bandpass (~0.0005–0.003). The old approach approximated this with a fixed 0.004 constant derived from one webcam under one lighting condition.

`_measure_scale_factor_via_pipeline()` runs the actual normalize → POS → bandpass pipeline on stored calibration frames and measures `filtered_pulse_std / raw_g_std` directly. This captures the real attenuation for this patient, this device, and this lighting — not an approximation.

### Calibration quality gate

Before recording starts, `SubjectProfile.validate()` checks whether calibration data is trustworthy:

- Frame count ≥ 30 (face was visible long enough)
- `baseline_g_std` between 0.05 and 3.0 (not flat, not noisy)
- Mask pixels ≥ 500 (face not too far from camera)
- `baseline_g_mean` between 10 and 245 (not underlit, not overexposed)

If validation fails, the system shows a specific actionable message and retries calibration up to 3 times before falling back to population defaults.

---

## Hardware-aware HRV reporting

RMSSD requires precise beat timing. At 30fps, one sample = 33ms — too coarse for reliable RR interval estimation. The system self-reports its own hardware limitation:

| Camera FPS | RMSSD status |
|---|---|
| < 60 fps | ✗ Not reported — "HRV requires 60fps+ camera" |
| 60–90 fps | Reported with low confidence flag |
| 90+ fps | Reported normally |

A 120fps camera automatically gets full HRV — no configuration needed. A standard 30fps laptop webcam correctly suppresses RMSSD rather than reporting inflated values.

---

## HR reliability flag

When the FFT and RR-based HR estimates disagree by more than 10 BPM, the final HR is flagged in the summary:

```
Heart Rate:    67.2 BPM  ⚠ LOW CONFIDENCE (FFT/RR disagreement ±14.3 BPM)
```

This disagreement also applies a 25% penalty to the HR confidence component of the Step 11 SNR score, pushing borderline cases toward palm routing when HR estimation is unreliable.

---

## Measurement failure path

When both face and palm signals are insufficient, the system exits cleanly with a specific actionable message rather than showing unreliable numbers:

```
MEASUREMENT FAILED

Both face and palm signals are too weak for a reliable measurement.
Palm SNR score: 0.312 (need >= 0.45).

To fix this:
  • Improve lighting — move closer to a lamp or face a window
  • Ensure your palm is flat, centered, and fully visible
  • Avoid strong backlight behind you
  • Try again in a brighter room
```

The same clean exit applies when the face mask or palm mask was never established during setup.

---

## Measured vitals

| Vital | Method | Notes |
|---|---|---|
| Heart Rate | POS algorithm + FFT peak detection | Primary output |
| HRV (RMSSD) | Beat-to-beat RR interval analysis | Only reported at 60fps+ |
| HRV (SDNN) | Standard deviation of RR intervals | Only reported at 60fps+ |
| Respiratory Rate | Green channel FFT analysis | Feeds adaptive bandpass filter |
| Signal Quality Score | Multi-metric SNR (5 components) | Drives face/palm routing decision |
| Skin Tone (ITA) | BiSeNet mask photometry | Used for adaptive thresholds |

---

## Pipeline architecture

```
Camera
  |
  v
Step 1 -- Camera initialization & FPS measurement
  |
  v
Step 2 -- Face ROI extraction (BiSeNet) + Subject Calibration  <- 10s total
  |        Phase 1 (0-5s):  Forehead + cheek mask via semantic segmentation
  |        Phase 2 (5-10s): Pixel sampling -> SubjectProfile
  |           - _measure_scale_factor_via_pipeline() -> real raw->filtered ratio
  |           - baseline_g_std -> motion threshold, amplitude target
  |           - _estimate_hr_from_signal() -> bandpass hint
  |           - ITA -> Fitzpatrick type -> routing threshold adjustment
  |        validate() -> calibration quality gate (retries up to 3x)
  |        SubjectProfile passed to ALL downstream steps
  v
Step 3 -- RGB signal extraction + motion rejection
  |        MotionDetector uses profile.get_motion_threshold() (personal)
  |        Rejected frames silently skipped; recording extends automatically
  v
Step 4 -- Normalization
  v
Step 10 -- Respiratory rate detection
  v
Step 5 -- POS pulse extraction (Wang et al. 2017)
  v
Step 6 -- Adaptive bandpass filter
  |        Artifact clipping: personal multiplier (3.0-5.0 std)
  |        Bandpass narrowed around patient's estimated HR if available
  v
Step 7 -- FFT frequency analysis
  v
Step 8 -- Peak detection  [two-pass with Step 11 feedback]
  |        Harmonic-support scoring + FFT cross-check override
  |        Overlapping windows + quality-weighted dedup + gap filling
  |        Dynamic RR tolerance: profile.get_rr_tolerance(signal_quality)
  v
Step 9 -- Heart rate & HRV (pass 1 -> feeds Step 11 confidence)
  |        FPS check: RMSSD suppressed if fps < 60
  |        hr_reliable flag set when FFT/RR agree within 10 BPM
  v
Step 11 -- Signal quality score + routing decision
  |         HR reliability penalty: -25% to confidence when hr_reliable=False
  |         Personal amplitude ceiling, ITA-adjusted thresholds, personal HRV floor
  v
Step 8/9 -- Second pass (quality score now known)
  |
  |--- FACE ACCEPTED -> final summary
  |
  |--- ROUTE TO PALM
  |      Step 2b -- Palm ROI (MediaPipe) + Palm Calibration <- 10s
  |      Step 3b -- Palm RGB extraction (live mask update every frame)
  |      Steps 4-11 (palm) -- identical processing
  |
  |--- PALM ACCEPTED -> final summary
  |
  +--- BOTH FAILED -> measurement_failed() -> clean exit with suggestions
```

---

## Routing decision — when and why

Two independent checks. Either condition alone triggers palm routing.

### Check 1 — Composite SNR score below ITA-adjusted threshold

| Component | Weight | What it measures |
|---|---|---|
| Spectral SNR | 0.30 | HR peak dominance over noise bins |
| SNR in dB | 0.20 | HR band power vs total noise power |
| RR regularity | 0.20 | Beat-to-beat consistency |
| Amplitude | 0.15 | Signal strength vs personal calibration target |
| HR confidence | 0.15 | Step 9 reliability (−25% when FFT/RR disagree) |

| Skin tone | HIGH threshold | MEDIUM threshold |
|---|---|---|
| FST I–III (ITA > 28) | ≥ 0.60 | ≥ 0.40 |
| FST IV (ITA 10–28) | ≥ 0.55 | ≥ 0.35 |
| FST V (ITA −30–10) | ≥ 0.50 | ≥ 0.30 |
| FST VI (ITA < −30) | ≥ 0.45 | ≥ 0.25 |

### Check 2 — Signal amplitude below personal HRV reliability floor

```
std_floor = amplitude_target × 0.10
amplitude_target = 0.80 × baseline_g_std × measured_scale_factor
```

The floor is personal — anchored to this patient's own calibration on this device and lighting.

---

## SubjectProfile — full API

| Attribute | Description |
|---|---|
| `ita` | ITA skin tone angle |
| `fitzpatrick` | FST group string |
| `baseline_g_std` | Green channel noise floor |
| `motion_threshold` | 5× personal noise floor |
| `amplitude_target` | 80% of personal best, scaled via measured pipeline ratio |
| `calib_to_filtered_scale` | Measured per session by `_measure_scale_factor_via_pipeline()` |
| `hr_estimate_bpm` | Rough HR from calibration (bandpass hint only) |
| `is_valid` | True when calibration passed quality gate |
| `validation_reason` | Human-readable reason if calibration failed |

| Method | Used by | Returns |
|---|---|---|
| `validate(mask_pixels)` | Step 2 | (bool, reason_string) |
| `get_motion_threshold()` | Step 3 | Personal frame rejection threshold |
| `get_amplitude_target()` | Step 11 | Personal amplitude scoring ceiling |
| `get_std_floor()` | Step 11 | Personal HRV reliability floor |
| `get_clip_multiplier()` | Step 6 | Personal artifact clipping multiplier (3.0–5.0) |
| `get_bandpass_hint()` | Step 6 | (low_hz, high_hz) around estimated HR ±30 BPM |
| `get_rr_tolerance(quality)` | Step 8 | Dynamic RR filter tolerance (0.30–0.50) |

---

## Installation

**Requirements:** Python 3.10, webcam

```bash
git clone https://github.com/arijbelmabrouk/PulseRoute.git
cd PulseRoute

python -m venv rppg_env
rppg_env\Scripts\activate        # Windows
# source rppg_env/bin/activate   # macOS/Linux

pip install -r requirements.txt
pip install mediapipe==0.10.9
```

**BiSeNet model weights** — place at:
```
step2_face_ROI_extraction/face_parsing_mask/models/bisenet_resnet18.pth
```

---

## Usage

```bash
python run.py
```

- Camera warms up, face ROI established (5s), calibration runs (5s) with quality validation
- 35-second clean signal recording begins (extends automatically if you move)
- If calibration fails, system prompts to reposition and retries up to 3 times
- If face signal is insufficient, palm fallback activates automatically
- If both fail, system exits cleanly with actionable suggestions

---

## Web dashboard (Step 12)

```powershell
# Terminal 1
uvicorn step12_display.server:app --port 8000 --reload

# Terminal 2
cd step12_display/frontend && npm run dev

# Terminal 3
python run_web.py
```

| Page | URL | Audience |
|---|---|---|
| Patient view | `http://localhost:5173/patient` | Patient screen |
| Doctor dashboard | `http://localhost:5173/doctor` | Clinician monitor |

---

## File structure

```
rPPG_project/
├── run.py                               # Full pipeline (Steps 1-11 + palm + failure paths)
├── run_web.py                           # Pipeline with WebSocket publishing
├── subject_profile.py                   # Per-subject adaptive calibration profile
├── step1_video_capture/
├── step2_face_ROI_extraction/           # BiSeNet semantic face parsing
├── step2_palm_ROI_extraction/           # MediaPipe hand landmark detection
├── step3_signal_extraction/
│   ├── step3_face_signal_bisenet.py     # Face modality + calibration + quality gate
│   ├── step3_palm_signal.py             # Palm modality + live mask update
│   └── step3_rgb_signal.py             # Core extraction + MotionDetector
├── step4_normalization/
├── step5_pulse_signal_extraction/       # POS algorithm (Wang et al. 2017)
├── step6_bandpass_filter/               # Butterworth bandpass + adaptive clipping
├── step7_conversion_time_to_frequency/
├── step8_peak_detection/                # HR peak + beat detection + gap filling
├── step9_HR_HRV/                        # HR + RMSSD/SDNN + fps gate + age norms
├── step10_respiratory_rate/
├── step11_signal_quality_score/         # SNR + ITA routing + HR reliability penalty
└── step12_display/                      # FastAPI + React web dashboard
```

---

## Current status

| Component | Status |
|---|---|
| Step 1 — Camera | ✅ Complete |
| Step 2 — Face ROI + Calibration + Quality gate | ✅ Complete |
| Step 2b — Palm ROI + Calibration | ✅ Complete |
| Step 3 — RGB extraction + motion rejection | ✅ Complete |
| Step 3b — Palm RGB extraction (live mask) | ✅ Complete |
| Step 4 — Normalization | ✅ Complete |
| Step 5 — POS | ✅ Complete |
| Step 6 — Bandpass + adaptive clipping | ✅ Complete |
| Step 7 — FFT | ✅ Complete |
| Step 8 — Peak detection (all fixes + two-pass) | ✅ Complete |
| Step 9 — HR & HRV + fps gate + hr_reliable flag | ✅ Complete |
| Step 10 — Respiratory rate | ✅ Complete |
| Step 11 — SNR + ITA routing + HR reliability penalty | ✅ Complete |
| SubjectProfile — fully dynamic + pipeline scale factor | ✅ Complete |
| Palm routing in run.py + measurement failure path | ✅ Complete |
| Step 12 — Web dashboard (face pipeline) | ✅ Complete |
| Step 12 — Palm routing in run_web.py | 🔄 In development |

---

## Scientific basis

- **POS algorithm:** Wang et al. (2017). *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
- **rPPG feasibility:** Verkruysse et al. (2008). *Optics Express*, 16(26), 21434–21445.
- **HRV standards:** Task Force of the ESC (1996). *Circulation*, 93(5), 1043–1065.
- **Age-adjusted HRV norms:** Shaffer & Ginsberg (2017). *Frontiers in Public Health*, 5, 258.
- **Skin tone classification:** ITA per Chardon et al. (1991), mapped to Fitzpatrick scale.
- **Palm signal advantage:** Higher superficial capillary density and lower melanin variation in palm vs face across Fitzpatrick types.

---

## License

MIT