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

PulseRoute solves that with a **dual-modality routing architecture**:

- **Face first** — better user experience, no hand positioning needed
- **Automatic palm fallback** — triggered when face signal quality is insufficient
- **Signal quality gate** — the system decides which modality to use based on measured SNR, not assumptions about appearance
- **Per-subject adaptive calibration** — every threshold is derived from the patient's own signal, not population averages

This means the system works for dark skin tones, bearded faces, and any other case where the face reflects insufficient light. The palm signal is 31% stronger in such cases (validated in peer-reviewed literature), and the routing is automatic and transparent.

---

## Per-subject adaptive calibration

Every fixed threshold in the pipeline has been replaced by a value derived from the patient's own signal during a 10-second calibration phase at startup.

A `SubjectProfile` object is built during Step 2 and passed through every downstream step. Nothing in the pipeline uses population-average constants for clinical decisions.

| Threshold | Old approach | New approach |
|---|---|---|
| Motion rejection | Fixed delta of 6.0 | 5x patient's own noise floor |
| Amplitude scoring ceiling | Fixed 0.006 | 80% of patient's calibration amplitude, scaled to filtered domain |
| HRV reliability floor | Fixed 0.002 | 10% of patient's personal amplitude target |
| Artifact clipping multiplier | Fixed 4.0 std | 3.0–5.0 std scaled to patient's signal variance |
| Bandpass window | Always 40–180 BPM | +/-30 BPM around patient's estimated HR |
| RR filter tolerance | Fixed 40% | 30–50% based on measured signal quality |
| Routing thresholds | Fixed 0.60 / 0.40 | ITA-adjusted per skin tone group |
| HRV interpretation | Population average | Age-adjusted norms when age is available |
| Scale factor (raw to filtered) | Fixed 0.004 empirical | Measured per session via `calibrate_scale_factor()` |

### How the scale factor works

The raw green channel std during calibration (~0.3–2.0) is in a completely different domain from the filtered pulse std after normalization, POS, and bandpass (~0.0005–0.003). A fixed bridge factor derived from one webcam under one lighting condition is wrong on other devices.

`calibrate_scale_factor()` runs a mini bandpass filter on the calibration signal and measures the actual ratio between raw std and filtered std for this patient, on this device, under this lighting. This measured ratio drives `amplitude_target` and `get_std_floor()`, making both thresholds self-calibrating every session.

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
  |
  v
Step 1  -- Camera initialization & FPS measurement
  |
  v
Step 2  -- Face ROI extraction (BiSeNet) + Subject Calibration   [10s total]
  |         Phase 1 (0-5s):  Forehead + cheek mask via semantic segmentation
  |         Phase 2 (5-10s): Pixel sampling -> SubjectProfile
  |            - baseline_g_std -> motion threshold, amplitude target
  |            - calibrate_scale_factor() -> measured raw->filtered ratio
  |            - _estimate_hr_from_signal() -> bandpass hint
  |            - ITA -> Fitzpatrick type -> routing threshold adjustment
  |         SubjectProfile passed to ALL downstream steps
  v
Step 3  -- RGB signal extraction + motion rejection
  |         MotionDetector uses profile.get_motion_threshold() (personal)
  |         Rejected frames silently skipped; recording extends automatically
  |         Up to 2.5x target duration if patient is moving
  v
Step 4  -- Normalization
  |         DC removal + linear detrending
  v
Step 10 -- Respiratory rate detection
  |         FFT on green channel -> breathing Hz
  |         Feeds adaptive notch + lower cutoff to Step 6
  v
Step 5  -- POS pulse extraction (Wang et al. 2017)
  |         RGB -> single pulse waveform
  v
Step 6  -- Adaptive bandpass filter
  |         Artifact clipping: +/- profile.get_clip_multiplier() std (personal, 3.0-5.0)
  |         Bandpass narrowed around patient's estimated HR if available
  |         Respiratory cutoff takes priority when detected
  v
Step 7  -- FFT frequency analysis
  |         Time domain -> power spectrum
  v
Step 8  -- Peak detection  [two-pass with Step 11 feedback]
  |         Frequency: harmonic-support scoring selects true fundamental
  |         FFT cross-check override catches sub-harmonics
  |         Time: overlapping windows (80% advance) + quality-weighted dedup
  |         Gap filling inserts missed beats in double-length intervals
  |         Dynamic RR tolerance: profile.get_rr_tolerance(signal_quality)
  v
Step 9  -- Heart rate & HRV (pass 1 -> feeds Step 11 confidence)
  |         Age-adjusted HRV interpretation via SubjectProfile
  v
Step 11 -- Signal quality score + routing decision
  |         5-component SNR score
  |         Personal amplitude ceiling: profile.get_amplitude_target()
  |         ITA-adjusted routing thresholds: profile.get_routing_thresholds()
  |         Personal HRV floor: profile.get_std_floor() = amplitude_target x 0.10
  |         Two independent routing triggers (see Routing section)
  v
Step 8/9 -- Second pass (quality score now known)
  |          RR re-filtered with real quality-driven tolerance
  |          HR/HRV recomputed from refined intervals
  |
  |--- FACE ACCEPTED -> final summary
  |
  +-- ROUTE TO PALM ---------------------------------------------------+
                                                                        |
Step 2b -- Palm ROI extraction (MediaPipe) + Palm Calibration   [10s] |
  |         Phase 1 (0-5s): MediaPipe hand landmarks -> palm mask      |
  |         Phase 2 (5-10s): Pixel sampling -> palm SubjectProfile     |
  |         Palm profile is INDEPENDENT from face profile              |
  v
Step 3b -- Palm RGB signal extraction (35s)
  v
Steps 4-11 (palm) -- identical signal processing
  |         All thresholds driven by palm SubjectProfile
  v
Step 12 -- Web dashboard
```

---

## Routing decision — when and why

The system routes to palm in two independent ways. Either condition alone triggers the switch.

### Composite SNR score below threshold

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
| FST I–III (ITA > 28) | >= 0.60 | >= 0.40 |
| FST IV (ITA 10–28) | >= 0.55 | >= 0.35 |
| FST V (ITA -30 to 10) | >= 0.50 | >= 0.30 |
| FST VI (ITA < -30) | >= 0.45 | >= 0.25 |

Score >= HIGH: face accepted, result reliable.
Score >= MEDIUM: face accepted, result usable.
Score < MEDIUM: route to palm.

### Signal amplitude below personal HRV reliability floor

Even when the composite score passes, the system checks whether the filtered signal std is above the patient's personal floor (`profile.get_std_floor()`).

The floor is computed as:
```
std_floor        = amplitude_target x 0.10
amplitude_target = 0.80 x baseline_g_std x calibrate_scale_factor()
```

This floor is anchored to this patient's own calibration signal on this device — not a fixed constant. When triggered, individual beat peaks are too close to the noise floor for precise timing, and RMSSD becomes unreliable (inflated 3–5x). Routing to palm is the correct clinical response.

---

## Motion robustness

Designed for real teleconsultation patients, not lab subjects:

- Coughing, talking, swallowing — motion artifact frames silently rejected, recording extends automatically
- Slow drift — artifact clipping in Step 6 prevents filter contamination
- Head turns — missed beats recovered by gap-filling in Step 8
- Poor lighting — personal amplitude scoring adapts to actual signal strength
- Variable FPS — measured FPS used throughout, not assumed

---

## Web dashboard (Step 12)

The web interface is the primary user-facing product. No terminal interaction is required during a teleconsultation session.

### Session flow

```
Patient opens /patient page
  |
  v
Patient enters their name or ID and selects measurement mode (Auto / Palm)
  |
  v
Server stores patient selection and notifies doctor page
  |
  v
Doctor reviews patient info and clicks Start
  |
  v
Server spawns run_web.py as a subprocess with mode and patient ID via env vars
  |
  v
Pipeline runs and publishes events + annotated camera frames via WebSocket
  |
  v
Patient page: live camera feed with ROI overlay + step progress
Doctor page: live metrics updating per step + SNR trace + subject profile
  |
  v
Results published to both pages simultaneously
Doctor page: routing decision, range indicators, history tab
```

### Pages

| Page | URL | Audience | Key features |
|---|---|---|---|
| Patient view | `http://localhost:5173/patient` | Patient screen | Name/ID entry, mode selection, live camera feed with ROI overlay, step progress, results |
| Doctor dashboard | `http://localhost:5173/doctor` | Clinician | Start button, live metrics per step, normal range indicators, SNR chart, routing decision with reason, session history |

### Doctor page features

- Start button triggers pipeline via `POST /api/start` — no terminal needed
- Live metrics update as each pipeline step completes
- Normal range bands on all vitals (HR, RR, RMSSD, SNR, confidence)
- SNR quality trace over the session
- Subject profile panel: ITA, Fitzpatrick type, calibration validity, signal std, bandpass cutoff
- Session history tab: reads `logs/sessions.csv` via `GET /api/history`, filterable by patient ID
- Re-measure button kills the current pipeline and resets state
- Routing decision card explains why face or palm was used

### Patient page features

- Name/ID entry and mode selection before measurement starts
- Live annotated camera feed during Step 3 (ROI overlay shows active measurement regions)
- Step-by-step progress checklist
- Results card on completion with HR and respiratory rate

### Launch

```powershell
start.bat
```

This starts the FastAPI server and React frontend in minimised windows and opens both pages in the browser. The pipeline starts when the doctor clicks Start — no terminal interaction required.

For development (manual):

```powershell
# Terminal 1 — FastAPI server
uvicorn step12_display.server:app --port 8000 --reload

# Terminal 2 — React dev server
cd step12_display/frontend
npm run dev
```

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
| `fitzpatrick` | FST group string |
| `baseline_g_mean` | Mean green pixel value at rest |
| `baseline_g_std` | Green channel noise floor |
| `motion_threshold` | 5x personal noise floor |
| `amplitude_target` | 80% of personal best, scaled to filtered domain |
| `calib_to_filtered_scale` | Measured per session by `calibrate_scale_factor()` |
| `hr_estimate_bpm` | Rough HR from calibration (bandpass hint only) |
| `is_valid` | True when >= 30 calibration frames collected |

**Dynamic threshold getters:**

| Method | Used by | Returns |
|---|---|---|
| `get_motion_threshold()` | Step 3 | Personal frame rejection threshold |
| `get_amplitude_target()` | Step 11 | Personal amplitude scoring ceiling |
| `get_std_floor()` | Step 11 | Personal HRV reliability floor |
| `get_clip_multiplier()` | Step 6 | Personal artifact clipping std multiplier (3.0–5.0) |
| `get_bandpass_hint()` | Step 6 | (low_hz, high_hz) around estimated HR +/- 30 BPM |
| `get_rr_tolerance(quality)` | Step 8 | Dynamic RR filter tolerance (0.30–0.50) |

---

## Installation

**Requirements:** Python 3.10, Node.js 18+, webcam

```bash
git clone https://github.com/arijbelmabrouk/PulseRoute.git
cd PulseRoute

python -m venv rppg_env
rppg_env\Scripts\activate        # Windows
# source rppg_env/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

**BiSeNet model weights** (not in repo — too large for Git):

Place `bisenet_resnet18.pth` at:
```
step2_face_ROI_extraction/face_parsing_mask/models/bisenet_resnet18.pth
```

**MediaPipe version** — must be 0.10.9 for palm detection compatibility:
```bash
pip install mediapipe==0.10.9
```

**Frontend dependencies:**
```bash
cd step12_display/frontend
npm install
```

---

## Usage

### Terminal pipeline (development / testing)

```bash
python run.py
```

Camera warms up, face ROI established (5s), calibration runs (5s), 35-second clean signal recording begins. If face signal quality is insufficient, palm fallback activates automatically.

### Web dashboard (teleconsultation mode)

```powershell
start.bat
```

Patient and doctor pages open in the browser. Pipeline starts from the doctor interface.

---

## File structure

```
rPPG_project/
|
+-- run.py                               # Terminal pipeline (Steps 1-11 + palm branch)
+-- run_web.py                           # Pipeline with WebSocket publishing for Step 12
+-- subject_profile.py                   # Per-subject adaptive calibration profile
+-- session_logger.py                    # CSV session logging (one row per measurement)
+-- start.bat                            # One-click launcher for the web dashboard
|
+-- step1_video_capture/
+-- step2_face_ROI_extraction/           # BiSeNet semantic face parsing
+-- step2_palm_ROI_extraction/           # MediaPipe hand landmark detection
+-- step3_signal_extraction/
|   +-- step3_face_signal_bisenet.py     # Face modality + calibration phase
|   +-- step3_palm_signal.py             # Palm modality + palm calibration phase
|   +-- step3_rgb_signal.py             # Core extraction + MotionDetector + on_frame callback
+-- step4_normalization/
+-- step5_pulse_signal_extraction/       # POS algorithm (Wang et al. 2017)
+-- step6_bandpass_filter/               # Butterworth bandpass + adaptive clipping
+-- step7_conversion_time_to_frequency/
+-- step8_peak_detection/                # HR peak + beat detection + gap filling
+-- step9_HR_HRV/                        # Final HR + RMSSD/SDNN + age norms
+-- step10_respiratory_rate/
+-- step11_signal_quality_score/         # SNR scoring + ITA-adjusted routing
+-- step12_display/                      # Web dashboard
    +-- server.py                        # FastAPI: /api/start /api/history /api/frame
    +-- publisher.py                     # Pipeline -> server event publishing
    +-- frontend/
        +-- src/
            +-- pages/
            |   +-- DoctorPage.jsx       # Start panel, metrics, SNR chart, history tab
            |   +-- PatientPage.jsx      # ID entry, mode selection, live camera feed
            +-- hooks/
                +-- useWebSocket.js      # Shared WebSocket hook with frame handling
```

---

## Current status

| Component | Status |
|---|---|
| Step 1 — Camera | Complete |
| Step 2 — Face ROI + Calibration | Complete |
| Step 2b — Palm ROI + Calibration | Complete |
| Step 3 — RGB extraction + motion rejection | Complete |
| Step 3b — Palm RGB extraction | Complete |
| Step 4 — Normalization | Complete |
| Step 5 — POS | Complete |
| Step 6 — Bandpass + adaptive clipping | Complete |
| Step 7 — FFT | Complete |
| Step 8 — Peak detection (all fixes + two-pass) | Complete |
| Step 9 — HR & HRV + age norms | Complete |
| Step 10 — Respiratory rate | Complete |
| Step 11 — SNR + ITA routing + personal floors | Complete |
| SubjectProfile — fully dynamic, no hardcoded thresholds | Complete |
| Palm routing in run.py | Complete |
| Palm routing in run_web.py | Complete |
| Step 12 — Web dashboard architecture | Complete |
| Step 12 — Live camera feed via WebSocket | Complete |
| Step 12 — Session history tab | Complete |
| Step 12 — Doctor Start button (pipeline on demand) | Complete |
| Step 12 — Patient mode selection (Auto / Palm) | In development |
| Step 12 — Patient-ready handshake | In development |
| Logging — sessions.csv with patient ID | Complete |
| OpenCV headless mode for web deployment | In development |

---

## Known limitations

**Single-machine architecture.** The current implementation assumes the patient and the pipeline server are on the same physical machine. `run_web.py` opens the local webcam via `cv2.VideoCapture(0)`. A real teleconsultation deployment would require the patient's browser to stream their camera to the server, which is a different architecture (WebRTC or similar). This is the primary gap between the current research prototype and a deployable product.

**Camera access requires HTTPS in production.** Browsers only allow camera access on `localhost` or HTTPS origins. Any LAN or remote deployment requires a valid certificate.

**RMSSD reliability.** At 30fps, RMSSD is reported but flagged as low confidence for patients whose face signal filtered std falls below their personal floor. This is by design — the system routes to palm precisely for these cases. Palm signal results for RMSSD have not yet been formally validated against ECG ground truth.

**Sample size.** Pilot testing has been conducted on 2 subjects. Clinical validation requires a larger dataset spanning all Fitzpatrick types with ECG ground truth for HR and HRV.

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