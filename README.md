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

This means the system works for **dark skin tones**, **bearded faces**, **niqab wearers**, and any other case where the face reflects insufficient light. The palm signal is 31% stronger in such cases (validated in peer-reviewed literature), and the routing is automatic and transparent.

---

## Per-subject adaptive calibration

Every fixed threshold in the pipeline has been replaced by a value derived from **the patient's own signal** during a 10-second calibration phase at startup.

A `SubjectProfile` object is built during Step 2 and passed through every downstream step:

| Threshold | Old approach | New approach |
|---|---|---|
| Motion rejection | Fixed delta of 6.0 | 5× patient's own noise floor |
| Amplitude scoring ceiling | Fixed 0.006 | 80% of patient's own calibration amplitude |
| Bandpass window | Always 40–180 BPM | ±30 BPM around patient's estimated HR |
| RR filter tolerance | Fixed 40% | 30–50% based on measured signal quality |
| Routing thresholds | Fixed 0.60 / 0.40 | ITA-adjusted per skin tone group |
| HRV interpretation | Population average | Age-adjusted norms when age is available |

This eliminates the performance gap between lab conditions and real-world teleconsultation use.

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
  │        SubjectProfile passed to all downstream steps
  ▼
Step 3 ── RGB signal extraction + motion rejection
  │        Per-frame motion detector with personal threshold
  │        Rejected frames silently skipped; recording extends automatically
  ▼
Step 4 ── Normalization
  │        DC removal + linear detrending
  │
  ├──────────────────────────────────────────────┐
  │                                              │
  ▼                                             (continues)
Step 10 ── Respiratory rate detection
  │         FFT on green channel → breathing Hz
  │         Feeds adaptive notch + lower cutoff to Step 6
  │
  ▼
Step 5 ── POS pulse extraction (Wang et al. 2017)
  │        RGB → single pulse waveform
  ▼
Step 6 ── Adaptive bandpass filter
  │        Artifact clipping (±4std) before filter to prevent ringing
  │        Bandpass narrowed around patient's estimated HR if available
  │        Respiratory cutoff takes priority when detected
  ▼
Step 7 ── FFT frequency analysis
  │        Time domain → power spectrum
  ▼
Step 8 ── Peak detection  [two-pass with Step 11 feedback]
  │        Frequency: harmonic-support scoring selects true fundamental
  │        Time: overlapping windows + quality-weighted deduplication
  │        Gap filling inserts missed beats in double-length intervals
  │        Dynamic RR tolerance tightens/loosens with signal quality
  ▼
Step 9 ── Heart rate & HRV (pass 1 → feeds Step 11 confidence)
  │        Age-adjusted HRV interpretation via SubjectProfile
  ▼
Step 11 ── Signal quality score + routing decision
  │         5-component SNR score (spectral, dB, regularity,
  │         amplitude, confidence)
  │         Personal amplitude ceiling from SubjectProfile
  │         ITA-adjusted routing thresholds
  │         score < threshold → route to palm
  ▼
Step 8/9 ── Second pass (quality score now known)
  │          RR re-filtered with real quality-driven tolerance
  │          HR/HRV recomputed from refined intervals
  ▼
Step 12 ── Display (in development)
```

---

## Motion robustness

Designed for real teleconsultation patients — not lab subjects:

- **Coughing** → motion artifact frames silently rejected, recording extends
- **Talking** → same rejection mechanism
- **Swallowing** → same
- **Slow drift** → artifact clipping in Step 6 prevents filter contamination
- **Head turn** → missed beats recovered by gap-filling in Step 8
- **Poor lighting** → personal amplitude scoring adapts to actual signal strength

The patient does not need to stay perfectly still. The system adapts.

---

## Step-by-step detail

### Step 1 — Camera Initialization
Opens the webcam, discards warm-up frames, measures real FPS over 5 seconds. The measured FPS (not the claimed FPS) is used as the timing reference for all downstream calculations.

### Step 2 — Face ROI Extraction + Calibration (10s)
**Phase 1 (0–5s):** BiSeNet semantic segmentation finds forehead and cheek pixels, excluding hair, beard, eyes, and glasses. Computes ITA (Individual Typology Angle) as a numerical skin tone value.

**Phase 2 (5–10s):** Pixel values are sampled through the locked mask to build a `SubjectProfile`. This profile holds the patient's personal green channel baseline, noise floor, motion threshold, amplitude target, and rough HR estimate. Every downstream threshold is derived from this profile — not population averages.

### Step 3 — RGB Signal Extraction + Motion Rejection
Records clean signal until the buffer is full. A `MotionDetector` compares each frame's green channel mean to a rolling 10-frame mean. Frames deviating by more than the patient's personal threshold are silently skipped. The buffer waits for clean frames; a red "MOTION DETECTED" indicator shows during rejections. Recording extends up to 2.5× the target duration if needed.

### Step 4 — Normalization
Divides each channel by its temporal mean (removes absolute brightness dependence), then applies linear detrending (removes slow lighting drift). Required input format for POS.

### Step 10 — Respiratory Rate
Detects breathing frequency from the green channel's low-frequency oscillation. Output feeds Step 6 as an adaptive notch frequency and lower bandpass cutoff.

### Step 5 — POS Pulse Extraction
Plane-Orthogonal-to-Skin algorithm (Wang et al. 2017). Combines three normalized RGB channels into one pulse waveform by projecting onto the plane orthogonal to the skin color vector.

### Step 6 — Bandpass Filter (Adaptive)
Two preprocessing steps before the Butterworth filter:
1. **Artifact clipping** — samples beyond ±4 standard deviations are clipped to prevent filter ringing from spikes
2. **Profile hint** — if the patient's calibration HR estimate is available, the bandpass is narrowed to ±30 BPM around it; respiratory cutoff takes priority when detected

### Step 7 — FFT
NumPy rfft converts the filtered pulse to frequency domain. Frequency resolution ≈ 1.7 BPM per bin for a 35-second recording at 30 FPS.

### Step 8 — Peak Detection
**Frequency domain:** Harmonic-support scoring evaluates each candidate peak by how much energy exists at 2× and 3× its frequency. True fundamentals score higher than sub-harmonics even when raw power is similar.

**Time domain:** FFT-guided overlapping windows (80% advance) find individual beat peaks. Quality-weighted deduplication prefers the beat with higher local SNR when two candidates are too close. Gap filling inserts missed beats in intervals >1.5× the median with physiological validity checks. Dynamic RR tolerance driven by `SubjectProfile.get_rr_tolerance(signal_quality)`.

**Two-pass structure:** First pass uses `signal_quality=None`. After Step 11 computes the quality score, a second pass re-filters RR intervals with the real tolerance.

### Step 9 — Heart Rate & HRV
Final HR: weighted combination of FFT estimate (70%) and RR mean (30%). Falls back to FFT-only if estimates disagree by >10 BPM. Computes RMSSD (primary) and SDNN (indicative for short recordings). HRV interpretation uses age-adjusted norms when `SubjectProfile.age` is set.

### Step 11 — Signal Quality Score + Routing

Five-component weighted score:

| Component | Weight | What it measures |
|---|---|---|
| Spectral SNR | 0.30 | HR peak dominance over noise bins |
| SNR in dB | 0.20 | HR band power vs total noise power |
| RR regularity | 0.20 | Beat-to-beat consistency (CV of RR intervals) |
| Amplitude | 0.15 | Signal strength vs personal calibration target |
| HR confidence | 0.15 | Step 9 measurement reliability |

---

#### When and why the system routes to palm

The system routes to palm in two independent ways. Either condition alone is enough to trigger the switch.

**Way 1 — Composite SNR score below threshold**

The weighted score falls below the ITA-adjusted routing threshold. This means overall signal quality is too low — the HR frequency doesn't dominate the spectrum clearly enough, beat timing is irregular, or the signal is too weak relative to the patient's own calibration baseline.

Routing thresholds are adjusted for skin tone because darker skin produces lower signal amplitude by physics (melanin absorbs more light before it reaches capillaries). Without adjustment, darker-skinned patients would be routed to palm even when their measurement is valid — just quieter.

| Skin tone | HIGH threshold | MEDIUM threshold |
|---|---|---|
| FST I–III (ITA > 28) | ≥ 0.60 | ≥ 0.40 |
| FST IV (ITA 10–28) | ≥ 0.55 | ≥ 0.35 |
| FST V (ITA −30–10) | ≥ 0.50 | ≥ 0.30 |
| FST VI (ITA < −30) | ≥ 0.45 | ≥ 0.25 |

Score ≥ HIGH → face accepted, result reliable.
Score ≥ MEDIUM → face accepted, result usable.
Score < MEDIUM → route to palm.

**Way 2 — Signal amplitude below HRV reliability floor**

Even when the composite score passes, the system checks whether the filtered signal standard deviation is above `0.002`. If not, it routes to palm regardless of the score.

This exists because the composite score can look clean while HRV is still unreliable. The spectral SNR can be high (the HR frequency is visible in the FFT), regularity can be good (beats are found at roughly the right rate), but if the filtered signal std is below 0.002, the individual beat peaks are too close to the noise floor for the argmax peak finder to locate the correct sample precisely. Timing errors of ±50–100ms per beat cascade into RMSSD values of 200–300ms — inflated by 3–5× the true value.

This is a physics limit, not an algorithm limit. No amount of peak detection improvement fixes it at this signal strength. The palm, which has higher superficial capillary density and minimal melanin variation, consistently produces 31% stronger signal and crosses this floor reliably.

The value `0.002` was set empirically from pilot recordings:

| Filtered std | Observed RMSSD | Beat timing |
|---|---|---|
| > 0.005 | 40–80ms (realistic) | Precise |
| 0.002–0.005 | 100–180ms (borderline) | Imprecise |
| < 0.002 | 200–300ms (inflated) | Unreliable |

`0.002` is deliberately conservative — lower than the literature threshold of ~0.003 — to avoid routing valid signals to palm unnecessarily. It should be validated against a larger dataset and adjusted if the boundary shifts.

When this check triggers, the terminal output shows:
```
Std floor check: std=0.000892  floor=0.002  ⚠ TRIGGERED
Routing decision: ⚠ ROUTE TO PALM (signal too weak for HRV)
```

### Step 12 — Display (in development)
Will render a live dashboard showing all vitals with confidence indicators, quality level, and palm prompt when routing is triggered.

---

## Palm modality

The palm pipeline mirrors steps 3–11 with a different ROI. The palm has higher superficial capillary density and minimal melanin variation regardless of skin tone. Signal is 31% stronger than face in challenging cases. Palm ROI extraction uses MediaPipe hand landmarks. All signal processing steps are identical to the face pipeline. Palm routing is the fallback when face quality is insufficient.

---

## Installation

**Requirements:** Python 3.10, webcam

```bash
git clone https://github.com/arijbelmabrouk/rPPG.git
cd rPPG

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

---

## Usage

```bash
python run.py
```

- Camera warms up, face ROI is established (5s), calibration runs (5s)
- 35-second clean signal recording begins (extends automatically if you move)
- Results printed to terminal at each pipeline step
- If face signal quality is insufficient, palm fallback is recommended

---

## Project structure

```
rPPG_project/
├── run.py                               # Full pipeline entry point (Steps 1–11)
├── subject_profile.py                   # Per-subject adaptive calibration profile
│
├── step1_video_capture/                 # Camera init & FPS measurement
├── step2_face_ROI_extraction/           # BiSeNet semantic face parsing
│   └── face_parsing_mask/
├── step2_palm_ROI_extraction/           # MediaPipe hand landmark detection
├── step3_signal_extraction/             # RGB extraction + motion rejection
│   ├── step3_face_signal_bisenet.py     # Face modality + calibration phase
│   └── step3_rgb_signal.py             # Core extraction + MotionDetector
├── step4_normalization/                 # DC removal + detrending
├── step5_pulse_signal_extraction/       # POS algorithm (Wang et al. 2017)
├── step6_bandpass_filter/               # Butterworth bandpass + artifact clipping
├── step7_conversion_time_to_frequency/  # FFT power spectrum
├── step8_peak_detection/                # HR peak + beat detection + gap filling
├── step9_HR_HRV/                        # Final HR + RMSSD/SDNN + age norms
├── step10_respiratory_rate/             # Breathing rate + adaptive notch
├── step11_signal_quality_score/         # SNR scoring + ITA-adjusted routing
└── step12_display/                      # (in development)
```

---

## Current status

| Component | Status |
|---|---|
| Step 1 — Camera |  Complete |
| Step 2 — Face ROI + Calibration |  Complete |
| Step 2 — Palm ROI |  Complete |
| Step 3 — RGB extraction + motion rejection |  Complete |
| Step 4 — Normalization |  Complete |
| Step 5 — POS |  Complete |
| Step 6 — Bandpass + artifact clipping |  Complete |
| Step 7 — FFT |  Complete |
| Step 8 — Peak detection (all fixes) |  Complete |
| Step 9 — HR & HRV + age norms |  Complete |
| Step 10 — Respiratory rate |  Complete |
| Step 11 — SNR + ITA routing |  Complete |
| SubjectProfile — adaptive calibration |  Complete |
| Step 12 — Display | In development |
| Palm pipeline routing activation | In development |

---

## Scientific basis

- **POS algorithm:** Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
- **rPPG feasibility:** Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. *Optics Express*, 16(26), 21434–21445.
- **HRV standards:** Task Force of the European Society of Cardiology (1996). Heart rate variability: standards of measurement. *Circulation*, 93(5), 1043–1065.
- **Age-adjusted HRV norms:** Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health*, 5, 258. Nunan, D. et al. (2010). A quantitative systematic review of normal values for short-term heart rate variability. *PACE*, 33(11), 1407–1417.
- **Skin tone classification:** ITA (Individual Typology Angle) per Chardon et al. (1991), mapped to Fitzpatrick scale for threshold adaptation.

---

## License

MIT