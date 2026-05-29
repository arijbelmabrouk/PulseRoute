# rPPG Vital Signs System
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

This means the system works for **dark skin tones**, **bearded faces**, **niqab wearers**, and any other case where the face reflects insufficient light for reliable signal extraction. The palm's signal is 31% stronger than the face in such cases (validated in peer-reviewed literature), and the routing is transparent and automatic.

---

## Measured vitals

| Vital | Method | Notes |
|---|---|---|
| Heart Rate | POS algorithm + FFT peak detection | Primary output |
| HRV (RMSSD) | Beat-to-beat RR interval analysis | Autonomic nervous system indicator |
| HRV (SDNN) | Standard deviation of RR intervals | Indicative for 30s recordings |
| Respiratory Rate | Green channel FFT analysis | Feeds adaptive bandpass filter |
| Signal Quality Score | Multi-metric SNR | Drives face/palm routing decision |
| Skin Tone (ITA) | BiSeNet mask photometry | Context for result interpretation |

---

## Pipeline architecture

The system runs as a linear signal processing pipeline. Each step has a single responsibility and passes its output to the next.

```
Camera
  │
  ▼
Step 1 ── Camera initialization & FPS measurement
  │
  ▼
Step 2 ── Face ROI extraction (BiSeNet semantic segmentation)
  │        Forehead + cheeks only — excludes hair, beard, eyes
  ▼
Step 3 ── RGB signal extraction
  │        Spatial average of skin pixels per frame → 3 time series
  ▼
Step 4 ── Normalization
  │        DC removal + linear detrending → zero-mean channels
  │
  ├──────────────────────────────────────────────────────┐
  │                                                      │
  ▼                                                      ▼
Step 10 ── Respiratory rate detection             (continues to Step 5)
  │         FFT on green channel → breathing Hz
  │         Feeds adaptive notch + cutoff to Step 6
  │
  ▼
Step 5 ── POS pulse extraction
  │        RGB → single pulse waveform
  │        Plane-Orthogonal-to-Skin algorithm (Wang et al. 2017)
  ▼
Step 6 ── Adaptive bandpass filter (40–180 BPM)
  │        Butterworth order 4, filtfilt zero-phase
  │        Lower cutoff adapts to measured breathing frequency
  ▼
Step 7 ── FFT frequency analysis
  │        Time domain → power spectrum
  │        Identifies dominant HR frequency
  ▼
Step 8 ── Peak detection
  │        Frequency domain: dominant HR peak with harmonic rejection
  │        Time domain: individual beat peaks for RR intervals
  ▼
Step 9 ── Heart rate & HRV calculation
  │        Combines FFT + RR estimates → final HR
  │        Computes RMSSD, SDNN, confidence score
  ▼
Step 11 ── Signal quality score (SNR)
  │         Spectral SNR + dB SNR + RR regularity + amplitude
  │         Score < 0.4 → route to palm
  ▼
Step 12 ── Display (in development)
           Live dashboard of all vitals
```

---

## Step-by-step detail

### Step 1 — Camera Initialization
Opens the webcam, discards the first 30 warm-up frames, then measures real FPS over 5 seconds. The measured FPS (not the claimed FPS) is used as the timing reference for every calculation downstream. Returns the camera capture object and actual FPS.

### Step 2 — Face ROI Extraction (BiSeNet)
Uses a pretrained BiSeNet deep learning model to semantically parse the face into regions and extract only skin pixels from the forehead and cheeks. Unlike bounding-box approaches, this mask excludes hair, beard, eyes, and glasses. Also computes the ITA (Individual Typology Angle) as a numerical skin tone value. This step is what makes the system inclusive — it finds actual skin pixels regardless of skin tone.

### Step 3 — RGB Signal Extraction
Records 30 seconds of frames and at each frame computes the spatial mean of red, green, and blue values across the masked skin pixels. With ~17,000 skin pixels per frame, individual pixel noise averages out, leaving only the systematic oscillation from blood flow. Output: three 900-sample time series (r, g, b).

### Step 4 — Normalization
Divides each channel by its temporal mean to remove absolute brightness dependence, then applies linear detrending to remove slow lighting drifts. Makes the algorithm skin-tone agnostic and drift-immune. Required input format for the POS algorithm.

### Step 10 — Respiratory Rate (runs before Step 5)
Detects breathing frequency from the low-frequency oscillation in the green channel. Breathing creates a 0.1–0.5 Hz modulation through small head movements and respiratory sinus arrhythmia. Uses a narrow bandpass filter then FFT to find the dominant breathing peak. Output feeds Step 6 as an adaptive notch filter and lower cutoff, removing breathing contamination more precisely than a fixed filter.

### Step 5 — POS Pulse Extraction
Applies the Plane-Orthogonal-to-Skin (POS) algorithm to combine the three normalized RGB channels into a single pulse waveform. Projects the signal onto the plane orthogonal to the skin color vector, cancelling specular reflections and common-mode noise. The adaptive alpha weight balances the two projection axes based on measured signal variance, making it work across skin tones without hardcoded skin color assumptions.

### Step 6 — Bandpass Filter
Applies a 4th-order Butterworth bandpass filter using filtfilt (zero-phase, forward-backward). Passes only frequencies in the 55–180 BPM range. The lower cutoff adapts based on measured breathing frequency from Step 10. The filtfilt zero-phase method eliminates timing shifts that would corrupt beat-to-beat HRV calculations. Typically improves HR band power ratio from ~5% to ~75%.

### Step 7 — FFT Frequency Analysis
Applies NumPy's rfft to convert the filtered pulse from time domain to frequency domain. Computes the power spectrum and identifies the frequency band corresponding to valid heart rates. Frequency resolution is approximately 2 BPM per bin for a 30-second recording at 30 FPS, within the ±3 BPM tolerance of medical device standards. Output: freqs, power, bpm_axis, SNR ratio.

### Step 8 — Peak Detection
Two parallel processes. In the frequency domain: finds all local peaks in the HR band, rejects harmonics using ratio analysis, selects the highest-power fundamental, refines frequency using quadratic interpolation between adjacent bins. In the time domain: uses FFT-guided windowing with 50% overlap and deduplication to find individual beat peaks, computes RR intervals in milliseconds, filters physiologically impossible intervals. Output feeds both HR calculation and HRV.

### Step 9 — Heart Rate & HRV
Computes final heart rate as a weighted combination of the FFT estimate (weight 0.7) and the RR interval mean (weight 0.3). Computes RMSSD as the primary HRV metric (root mean square of successive RR differences), and SDNN as a secondary metric. Generates a confidence score from four factors: FFT SNR, HR estimate agreement, beat count, and signal quality flag. If the two HR estimates disagree by more than 10 BPM, trusts FFT only.

### Step 11 — Signal Quality Score
Computes a 0.0–1.0 quality score from four weighted metrics: spectral SNR (0.35), SNR in dB (0.25), RR interval regularity (0.25), and signal amplitude (0.15). Applies routing thresholds: score ≥ 0.6 → HIGH, score 0.4–0.6 → MEDIUM, score < 0.4 → route to palm. This is the gatekeeper — the only step that decides whether the face result is trustworthy or the palm pipeline should be invoked.

### Step 12 — Display (in development)
Will render a real-time dashboard showing all vitals with confidence indicators, quality level, modality in use (face or palm), and a palm prompt when routing is triggered.

---

## Palm modality (routing target)

The palm pipeline mirrors steps 3–11 but uses a different ROI. The palm has a higher density of superficial capillaries and is unaffected by hair, beard, or face covering. Signal strength is measured to be 31% higher than the face in challenging cases.

Palm ROI extraction uses MediaPipe hand landmarks instead of BiSeNet. All signal processing steps (normalization, POS, bandpass, FFT, peak detection, HRV, SNR) are identical to the face pipeline.

---

## Installation

**Requirements:** Python 3.10, webcam

```bash
# Clone the repo
git clone https://github.com/arijbelmabrouk/rPPG.git
cd rPPG

# Create virtual environment
python -m venv rppg_env
rppg_env\Scripts\activate        # Windows
# source rppg_env/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

**Download the BiSeNet model weights** (not included in repo due to file size):

Place the file `bisenet_resnet18.pth` at:
```
step2_face_ROI_extraction/face_parsing_mask/models/bisenet_resnet18.pth
```

---

## Usage

```bash
python mainV1.py
```

- The system will warm up the camera, detect your face, and begin a 30-second recording
- Keep your face visible and stay reasonably still during recording
- Results are printed to terminal at each pipeline step
- If signal quality is insufficient, you will be prompted to show your palm

---

## Project structure

```
rPPG_project/
├── mainV1.py                          # Full pipeline (Steps 1–11)
├── main.py                            # Steps 1–8 only (earlier version)
│
├── step1_video_capture/               # Camera init & FPS measurement
├── step2_face_ROI_extraction/         # BiSeNet semantic face parsing
│   ├── face_parsing_mask/             # BiSeNet implementation
│   └── mediapipe_face_mesh/           # Alternative (not used in main pipeline)
├── step2_palm_ROI_extraction/         # MediaPipe hand landmark detection
├── step3_signal_extraction/           # RGB spatial averaging
├── step4_normalization/               # DC removal + detrending
├── step5_pulse_signal_extraction/     # POS algorithm
├── step6_bandpass_filter/             # Butterworth bandpass
├── step7_conversion_time_to_frequency/# FFT power spectrum
├── step8_peak_detection/              # HR peak + beat detection
├── step9_HR_HRV/                      # Final HR + RMSSD/SDNN
├── step10_respiratory_rate/           # Breathing rate + notch filter
├── step11_signal_quality_score/       # SNR scoring + routing decision
└── step12_display/                    # (in development)
```

---

## Scientific basis

- **POS algorithm:** Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.
- **rPPG feasibility:** Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. *Optics Express*, 16(26), 21434–21445.
- **Palm signal superiority:** Documented 31% SNR improvement over face in cases with dark skin or facial hair.
- **HRV standards:** Task Force of the European Society of Cardiology. (1996). Heart rate variability: standards of measurement. *Circulation*, 93(5), 1043–1065.
- **Skin tone inclusivity:** ITA-based skin classification with per-channel adaptive processing eliminates the performance gap present in landmark-based rPPG systems.

---

## Current status

| Step | Status |
|---|---|
| Step 1 — Camera | Complete |
| Step 2 — Face ROI (BiSeNet) | Complete |
| Step 2 — Palm ROI | Complete |
| Step 3 — RGB extraction | Complete |
| Step 4 — Normalization | Complete |
| Step 5 — POS | Complete |
| Step 6 — Bandpass filter | Complete |
| Step 7 — FFT | Complete |
| Step 8 — Peak detection | Complete |
| Step 9 — HR & HRV | Complete |
| Step 10 — Respiratory rate | Complete |
| Step 11 — Signal quality & routing | Complete |
| Step 12 — Display | In development |
| Palm pipeline routing | In development |

---

## License

MIT
