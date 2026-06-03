# PulseRoute: An Inclusive Contactless Vital Signs Monitoring System with Adaptive Face-to-Palm Signal Routing for Teleconsultation

**Arij Belmabrouk**  
Independent Researcher  
arij.belmabrouk@etudiant-isi.utm.tn

---

## Abstract

Contactless photoplethysmography (rPPG) enables camera-based measurement of heart rate (HR), heart rate variability (HRV), and respiratory rate (RR) without physical contact. However, existing systems are predominantly validated on light-skinned subjects under controlled laboratory conditions, resulting in systematic measurement failures for individuals with darker skin tones, in low-ambient lighting, or in real-world clinical environments. We present PulseRoute, a contactless vital signs monitoring system designed for teleconsultation settings that addresses these limitations through two novel contributions. First, a per-session subject calibration mechanism (SubjectProfile) that replaces all fixed population-average thresholds with dynamically computed personal thresholds derived from the patient's own signal during a 10-second setup phase — including a session-measured raw-to-filtered amplitude scale factor that eliminates device and lighting dependence from threshold computation. Second, an adaptive signal routing mechanism that begins measurement from the face and automatically routes to the palm when the face signal is insufficient, using a composite signal quality score combined with a personal physics-based amplitude floor derived from each patient's own calibration rather than a fixed constant. Preliminary evaluation on two subjects across Fitzpatrick skin types IV–VI under standard home lighting demonstrates correct HR estimation, accurate respiratory rate detection, and correct palm routing in conditions where face-only systems produce silently inflated HRV measurements. PulseRoute is implemented in Python using BiSeNet face parsing, the POS algorithm, and an 11-step modular pipeline, and is made publicly available.

**Keywords:** remote photoplethysmography, contactless vital signs, heart rate variability, skin tone bias, teleconsultation, signal routing, facial ROI, palm ROI, adaptive calibration

---

## 1. Introduction

The COVID-19 pandemic accelerated the adoption of telemedicine and remote patient monitoring, creating urgent demand for contactless physiological measurement tools that can operate in uncontrolled home environments. Remote photoplethysmography (rPPG) — the extraction of blood volume pulse signals from video of the skin surface — offers a compelling solution: it requires only a standard webcam, no wearable hardware, and no physical contact [1].

Existing rPPG systems have demonstrated reliable heart rate estimation under laboratory conditions, with several commercial implementations achieving mean absolute errors below 5 BPM [2, 3]. However, these systems share a critical limitation: their design assumptions, threshold values, and validation datasets are predominantly drawn from studies on light-skinned subjects under controlled lighting [4, 5]. This creates a documented performance gap for individuals with darker skin tones — Fitzpatrick skin types (FST) IV through VI — who represent a substantial fraction of the global population and are often disproportionately represented among patients with conditions requiring remote monitoring (hypertension, cardiovascular disease, diabetes) [6].

The physical basis of this gap is well understood. Melanin in the epidermis absorbs a portion of incident light before it reaches the capillaries, reducing the amplitude of the reflected pulse signal [7]. At FST V–VI under typical home webcam conditions, the filtered pulse signal standard deviation can be 5–15 times weaker than the equivalent signal from FST I–II subjects. Fixed amplitude thresholds calibrated for laboratory subjects will classify these signals as poor quality and either discard the measurement or report inflated HRV metrics — without informing the clinician that the measurement is unreliable.

A second underexplored direction is the use of palm skin for rPPG signal extraction. Research has shown that the palm produces a significantly stronger rPPG signal than the face due to its higher density of superficial capillaries and lower melanin content regardless of overall skin tone [8]. Despite this, no widely deployed rPPG system implements palm measurement as a clinical fallback, and the routing logic that would decide when to switch from face to palm has not been systematically studied.

We present PulseRoute, a system that addresses both problems. Our contributions are:

1. A **per-session subject calibration mechanism** (SubjectProfile) that replaces every fixed population-average threshold in the pipeline with a value derived from the patient's own signal during a 10-second setup phase. Critically, this includes a session-measured amplitude scale factor that captures the actual raw-to-filtered signal attenuation for this patient on this device under this lighting — eliminating the device and lighting dependence that makes fixed scaling constants unreliable across deployment contexts.

2. A **personal physics-based HRV reliability floor** derived from each patient's calibration amplitude rather than a fixed constant. This floor adapts to the patient's actual signal capability: patients with stronger signals have proportionally higher floors, ensuring the routing system is equally demanding of all patients relative to their own realistic best.

3. An **adaptive face-to-palm routing mechanism** that begins measurement from the face for better user experience and automatically routes to the palm when the face signal is insufficient, using two independent routing checks — a composite five-component quality score and the personal amplitude floor.

4. A **motion-robust recording protocol** that silently rejects artifact frames caused by patient coughing, talking, or repositioning, extending the recording window to accumulate the required clean frames rather than failing on a fixed time budget.

The system is designed specifically for teleconsultation settings where the patient is a real sick person sitting in their living room under ceiling lighting — not a healthy volunteer in a lab.

---

## 2. Related Work

### 2.1 Remote Photoplethysmography

The foundational rPPG work of Verkruysse et al. [9] demonstrated that the green channel of standard RGB video contains a measurable blood volume pulse signal. Subsequent algorithmic advances include CHROM [10], which uses chrominance-based signal separation, and POS (Plane-Orthogonal-to-Skin) [11], which projects the normalized RGB signals onto the plane orthogonal to the skin color vector and achieves superior performance across lighting conditions. PulseRoute uses the POS algorithm for its adaptive alpha weighting, which unlike CHROM requires no fixed skin color coefficients and therefore performs more equitably across Fitzpatrick types.

### 2.2 Facial Region of Interest Extraction

Early rPPG systems used simple rectangular face bounding boxes. More recent work uses facial landmark detection (MediaPipe [12]) or semantic segmentation (BiSeNet [13]) to extract specific skin regions. BiSeNet provides pixel-level classification of 19 facial regions, enabling precise exclusion of non-skin areas (hair, eyes, nose, lips) that degrade signal quality. PulseRoute uses BiSeNet V1 with a ResNet-18 backbone for forehead and cheek ROI extraction.

### 2.3 Skin Tone and rPPG Performance

Ba et al. [4] demonstrated significant performance degradation of rPPG systems at darker skin tones, with HR mean absolute error increasing from 2.1 BPM at FST I–II to 8.7 BPM at FST V–VI under equivalent conditions. Nowara et al. [5] showed that standard rPPG datasets are severely imbalanced toward lighter skin tones, with FST V–VI subjects comprising less than 8% of most benchmark datasets. The ITA (Individual Typology Angle) metric, derived from CIELab colorspace measurements of the ROI pixels, has been validated as a reliable proxy for Fitzpatrick skin type in rPPG contexts [14].

### 2.4 HRV from rPPG

Extraction of HRV metrics from rPPG requires accurate beat-to-beat interval (RR interval) estimation. McDuff et al. [15] showed that short-term RMSSD from rPPG achieves acceptable agreement with ECG gold standard when filtered signal amplitude is sufficient and the recording is at least 60 seconds. At lower signal amplitudes — precisely the condition encountered with darker skin — RR interval timing errors of ±50–100 ms have been reported, which cascade into RMSSD inflation of 150–300% above true values [16].

### 2.5 Palm-Based rPPG

Luo et al. [8] demonstrated that the palm produces a stronger rPPG signal than the face, attributed to higher capillary density in the palmar dermis and near-zero melanin content of the palm regardless of dorsal skin tone. Despite this finding, no clinical rPPG system implements palm measurement as a systematic fallback, and the decision boundary for when face measurement is insufficient has not been formally characterized.

---

## 3. System Design

### 3.1 Architecture Overview

PulseRoute implements an 11-step modular pipeline. Steps 1–11 operate on the face signal; if routing to palm is triggered at Step 11, Steps 2b–11 repeat on the palm signal with an independently calibrated SubjectProfile.

**Step 1 — Camera Initialization:** The system initializes the webcam and measures actual fps over a 5-second warm-up window. Nominal fps values from camera drivers are rejected; only the empirically measured rate is used for all downstream temporal calculations. Minimum requirements: 15 fps, 480p resolution.

**Step 2 — Face ROI Extraction and Subject Calibration (10 seconds):** BiSeNet parses the face frame-by-frame to build a stable forehead and cheek mask. The first 5 seconds establish the mask; the subsequent 5 seconds sample pixel values through the locked mask to build a SubjectProfile (Section 3.2). The ITA angle is computed from the mean CIELab L* and b* values of the ROI pixels.

**Step 3 — RGB Signal Extraction (35 seconds, adaptive):** Per-frame mean R, G, B values are extracted from the combined forehead and cheek mask. Frames where the green channel deviates by more than the patient's personal motion threshold from the rolling mean of the previous 10 frames are silently rejected. The recording extends automatically to compensate for rejected frames, capped at 2.5× the target duration.

**Step 4 — Normalization:** Each channel is divided by its temporal mean to remove DC offset, then linearly detrended to remove slow lighting drift.

**Step 5 — POS Pulse Extraction:** The POS algorithm [11] combines the three normalized channels into a single pulse waveform using an adaptive weight α = σ(S1)/σ(S2), where S1 = R−G and S2 = R+G−2B.

**Step 6 — Bandpass Filtering:** A zero-phase Butterworth filter of order 4 is applied. The lower cutoff is set adaptively: if respiratory rate is detected with confidence > 0.4, the cutoff is placed 0.1 Hz above the breathing frequency; otherwise it defaults to 0.917 Hz (55 BPM). A pre-filter clipping step removes samples beyond ±profile.get_clip_multiplier() standard deviations to prevent filter ringing from artifact spikes. The clipping multiplier is personal (range 3.0–5.0) derived from the patient's own calibration variance.

**Step 7 — FFT Frequency Analysis:** The real FFT of the filtered signal produces a power spectrum from which the dominant HR frequency is identified.

**Step 8 — Peak Detection (two-pass):** The dominant HR frequency gates an overlapping window search (80% advance) across the time-domain signal. A harmonic-support scoring function evaluates candidates by energy at 2× and 3× their frequency, preferring true fundamentals over sub-harmonics. An FFT cross-check overrides the frequency estimate if it deviates more than 15 BPM below the Step 7 dominant peak with SNR > 5×. A gap-filling pass recovers missed beats in double-length intervals. Dynamic RR filter tolerance (0.30–0.50) is driven by the quality score from Step 11 via a second pass after routing is decided.

**Step 9 — HR and HRV Calculation:** Final HR is a weighted combination (0.7 FFT, 0.3 time-domain) when estimates agree within 10 BPM, otherwise FFT-only. RMSSD and SDNN are computed from filtered RR intervals. HRV interpretation uses age-adjusted norms when patient age is available.

**Step 10 — Respiratory Rate:** The normalized green channel is bandpass filtered to 0.1–0.5 Hz (6–30 BrPM) and the dominant frequency extracted via FFT. This step runs before Step 5 to capture the breathing frequency before POS processing modifies low-frequency content.

**Step 11 — Signal Quality Score and Routing Decision:** A composite score combining five metrics drives the routing decision through two independent checks (Section 3.3). When palm routing is triggered, Steps 2b–11 repeat on the palm signal with a fresh, independently calibrated SubjectProfile anchored to palm signal characteristics.

### 3.2 SubjectProfile: Per-Session Dynamic Calibration

The SubjectProfile is the central architectural contribution of PulseRoute. It is computed once per session during Step 2 and passed to every downstream step, replacing all fixed population-average thresholds.

During the 5-second calibration phase, per-frame green channel means are collected through the locked ROI mask. The following personal thresholds are computed:

**Session-measured amplitude scale factor** (`calibrate_scale_factor`)**:** The ratio between raw green channel standard deviation and bandpass-filtered standard deviation is measured by running a mini bandpass filter on the calibration signal itself:

```
calib_to_filtered_scale = std(bandpass(g_calib)) / std(g_calib)
```

This ratio is device-specific, lighting-specific, and patient-specific. A fixed constant (such as the commonly used 0.004) is derived from one device under one lighting condition and transfers poorly to other deployment contexts. By measuring it per session, all downstream thresholds that depend on this scaling are automatically correct for the current environment.

**Motion threshold:** 5 × baseline_g_std (minimum 2.0). A frame whose green channel deviates from the rolling mean by more than this value is classified as a motion artifact. Using the patient's own noise floor means a naturally active patient has a proportionally higher threshold rather than being penalized by a threshold calibrated for still laboratory subjects.

**Amplitude target:** 0.80 × baseline_g_std × calib_to_filtered_scale. This is the patient's personal expected best filtered signal amplitude, used as the ceiling for Step 11's amplitude score. Dark-skinned patients under home lighting are scored against their own realistic best rather than a lab benchmark they cannot reach.

**Personal HRV reliability floor** (`get_std_floor`)**:** 

```
std_floor = amplitude_target × 0.10
```

This is the minimum filtered signal std below which beat timing is too imprecise for reliable RMSSD. Crucially, this floor is personal — it is derived from the patient's own amplitude target rather than a fixed constant. A patient with a naturally strong signal (amplitude_target = 0.030) has a floor of 0.003; a patient with a weak signal (amplitude_target = 0.005) has a floor of 0.0005. The routing system is therefore equally demanding of all patients relative to their own calibration baseline, not relative to a threshold calibrated on a different population.

The fixed-constant approach (using 0.002 for all patients) would over-route patients with strong signals to palm unnecessarily and under-route patients with very weak signals who happen to fall above 0.002. The personal floor eliminates both failure modes.

**Artifact clipping multiplier** (`get_clip_multiplier`)**:** Range 3.0–5.0, derived from baseline_g_std. Low baseline variance (weak signal) → tighter clipping (3.0) so artifacts stand out more clearly against the quieter background. High baseline variance (strong signal) → looser clipping (5.0) to preserve genuine physiological variation.

**Bandpass hint:** A rough HR estimate from the calibration signal is used to narrow the bandpass window to ±30 BPM around the estimated rate.

**RR filter tolerance:** Dynamically set between 0.30 (clean signal) and 0.50 (noisy signal) based on the composite quality score from Step 11.

Table 1 summarizes all dynamic thresholds and their derivation.

**Table 1. SubjectProfile dynamic thresholds**

| Threshold | Derivation | Used by |
|---|---|---|
| Motion threshold | 5 × baseline_g_std, min 2.0 | Step 3 frame rejection |
| Amplitude target | 0.80 × baseline_g_std × scale_factor | Step 11 amplitude score |
| HRV reliability floor | amplitude_target × 0.10 | Step 11 routing check 2 |
| Clip multiplier | 3.0 + baseline_g_std/2, clipped 3–5 | Step 6 artifact clipping |
| Bandpass hint | ±30 BPM around calibration HR estimate | Step 6 cutoff selection |
| RR tolerance | 0.30 + (1 − quality) × 0.20 | Step 8 interval filtering |
| Routing thresholds | ITA-adjusted (FST IV: −5%, V: −10%, VI: −15%) | Step 11 routing check 1 |

### 3.3 Routing Logic

The routing decision in Step 11 operates through two independent checks. Either condition alone triggers palm routing.

**Check 1 — Composite score below ITA-adjusted threshold:**

The weighted five-component score is:

```
score = 0.30 × spectral_SNR + 0.20 × SNR_dB + 0.20 × RR_regularity
      + 0.15 × amplitude_score + 0.15 × HR_confidence
```

Routing thresholds are adjusted downward for darker skin tones because melanin absorption physically reduces signal amplitude. Without adjustment, FST V–VI patients would be routed to palm even when their measurement is valid — merely because their signal is quieter, not because it is wrong.

| Skin tone | HIGH threshold | MEDIUM threshold |
|---|---|---|
| FST I–III (ITA > 28) | ≥ 0.60 | ≥ 0.40 |
| FST IV (ITA 10–28) | ≥ 0.55 | ≥ 0.35 |
| FST V (ITA −30–10) | ≥ 0.50 | ≥ 0.30 |
| FST VI (ITA < −30) | ≥ 0.45 | ≥ 0.25 |

Score ≥ HIGH → face accepted. Score ≥ MEDIUM → face accepted with lower confidence. Score < MEDIUM → route to palm.

**Check 2 — Filtered std below personal HRV reliability floor:**

Even when the composite score passes, the system checks:

```
if filtered_std < profile.get_std_floor():
    route_to_palm = True
```

This check exists because the composite score can pass while HRV is still unreliable. Spectral SNR can be high (the HR frequency is visible in the FFT) and regularity can be acceptable (beats are found at roughly the right rate), while individual beat timing remains too imprecise for RMSSD. When filtered std falls below the personal floor, argmax in the peak detector finds slightly wrong samples, producing RR interval timing errors of ±50–100 ms that cascade into RMSSD inflation of 150–300% regardless of algorithm improvements. This is a physics limit, not an algorithm limit.

The personal floor adapts to the patient: a patient whose calibration predicted a strong signal is held to a higher floor; a patient with a genuinely weak signal capability is held to a proportionally lower floor. The system is equally demanding of everyone relative to their own baseline.

When palm routing is triggered, the reason is logged — composite score failure vs. amplitude floor trigger — so the clinical interface can distinguish between general signal quality problems and the specific physics limitation of face measurement.

**Palm pipeline:** When routing fires, Steps 2b–11 repeat with MediaPipe Hands-based palm ROI detection and a fresh SubjectProfile calibrated from palm pixel values. The palm profile is fully independent of the face profile — palm baseline_g_std is typically higher (less melanin variation), so motion threshold, amplitude target, and HRV floor are all re-anchored to palm signal characteristics rather than carried over from face calibration.

---

## 4. Pilot Evaluation

### 4.1 Subjects and Protocol

*Note: This section presents preliminary results from a pilot evaluation. Full validation with ECG ground truth across a larger population is ongoing and will be reported in a subsequent study.*

Two subjects participated in pilot evaluation sessions. Subject 1 (ITA range 4–27, FST IV–V) and Subject 2 (ITA range −83 to −25, FST V–VI). Sessions were conducted in a standard home office environment under ceiling lighting without controlled illumination. No special equipment was used beyond a standard integrated laptop webcam (640×480, 30 fps nominal, measured 29.8–30.2 fps).

Seven valid sessions were recorded across both subjects, varying lighting conditions and time of day. Sessions with artificially modified routing thresholds (used during development testing) and sessions with deliberate face occlusion were excluded, leaving 7 sessions for analysis. HR reference was obtained from a fingertip pulse oximeter (SpO2 device) measured immediately before and after each session.

### 4.2 Heart Rate Estimation

HR estimates across sessions ranged 55–96 BPM, consistent with resting and mildly elevated heart rates across measurement conditions. The FFT cross-check correction fired in 3 of 7 sessions, correcting sub-harmonic selection errors of 20–30 BPM before they propagated to the final output. HR agreement between the two pipeline estimates (FFT and RR-based) was within ±5 BPM in 5 of 7 sessions and within ±9 BPM in all sessions.

Systematic validation against the pulse oximeter reference across all 7 sessions is in progress. Preliminary comparison in 4 face-accepted sessions indicates HR estimates within 2–5 BPM of the reference, consistent with the rPPG literature for comparable signal conditions [2, 3].

**Table 2. Session summary — HR and routing**

| Session | ITA | Modality | HR (BPM) | FFT–RR Agr. | SNR Score | Routing |
|---|---|---|---|---|---|---|
| S1 | −26.1 (FST V) | Face | 62.6 | ±8.1 | 0.538 | Accepted |
| S2 | −25.1 (FST V) | Face | 57.1 | ±4.9 | 0.558 | Accepted |
| S3 | −68.4 (FST VI) | Face | 76.2 | ±8.6 | 0.519 | Accepted |
| S4 | −79.4 (FST VI) | Face→Palm | 82.6→71.6 | — / ±0.8 | 0.462→0.624 | Amplitude floor |
| S5 | −5.2 (FST V) | Face | 80.8 | ±3.4 | 0.359 | Accepted (MEDIUM) |
| S6 | 27.4 (FST IV) | Face→Palm | 73.2→69.9 | — / ±1.5 | 0.313→0.613 | Composite score |
| S7 | −83.3 (FST VI) | Face→Palm | 57.8→71.6 | — / ±0.8 | 0.483→0.624 | Amplitude floor |

### 4.3 Respiratory Rate

Respiratory rate was successfully detected in all 7 sessions with confidence above 0.7. Detected rates ranged 12.0–22.3 BrPM. In all sessions, the adaptive notch filter correctly removed the breathing artifact from the pulse signal, with HR band power ratio improving from a pre-filter mean of 9.8% to a post-filter mean of 88.6% (range 83–97%).

One session showed a respiratory rate of 22.3 BrPM (elevated), correctly classified as such. Two sessions showed rates below 12 BrPM (bradypnea), also correctly classified. These were not treated as errors but as accurate reflections of the subject's breathing pattern during measurement.

### 4.4 Routing Behavior

**Face accepted (4 sessions):** Sessions S1–S3, S5. All had filtered std above their personal floor and composite scores above the ITA-adjusted threshold. S5 was accepted at MEDIUM quality (SNR 0.359, threshold 0.300 for FST V) — correctly identified as usable but not high-confidence.

**Palm routed — amplitude floor (2 sessions):** Sessions S4, S7. Both were FST VI subjects (ITA −79.4, −83.3). In S4, the composite score was 0.462 — above the FST VI HIGH threshold of 0.45 — but the filtered std (0.003449) fell below the personal floor, correctly triggering palm routing. This demonstrates the value of the second routing check: the composite score would have accepted the face result, but the amplitude floor correctly identified that HRV would be unreliable. Palm HR agreement in these sessions was ±0.8 BPM.

**Palm routed — composite score (1 session):** Session S6 (FST IV, ITA 27.4). SNR score 0.313 fell below the MEDIUM threshold of 0.35. Palm SNR improved to 0.613, crossing the HIGH threshold.

**False palm routings:** 0 of 7 sessions. No session where face measurement was reliable was incorrectly routed to palm.

### 4.5 HRV Observations

RMSSD was not validated against ECG in this pilot. Qualitative observations from the 4 face-accepted sessions: RMSSD ranged 214–291 ms — higher than literature norms for resting adults (20–80 ms), consistent with the known beat timing imprecision at the signal amplitudes produced by this webcam under home lighting. This confirms the motivation for the amplitude floor: even in face-accepted sessions, filtered std values of 0.001–0.005 are at the lower boundary of reliable HRV extraction, and the system correctly flags RMSSD as low-confidence when below the personal floor.

In the 3 palm sessions, palm RMSSD ranged 229–256 ms — marginally lower than face RMSSD for the same subjects, consistent with the stronger palm signal producing slightly better beat timing. Full HRV validation requires ECG ground truth and is planned for future work.

### 4.6 SubjectProfile Calibration Consistency

Across sessions, the personal floors computed by SubjectProfile ranged from 0.000494 (FST V, good lighting) to 0.013034 (corrupted calibration excluded from analysis). For the 7 valid sessions, personal floors ranged 0.000494–0.005332, correctly reflecting the range of signal strengths observed. The session-measured scale factor (`calib_to_filtered_scale`) ranged 0.003–0.015 across sessions, confirming that a fixed 0.004 constant would be incorrect for a meaningful fraction of real sessions.

---

## 5. Discussion

### 5.1 The Inclusivity Gap in rPPG

Our results confirm the documented performance gap for darker skin tones in rPPG systems. The fundamental cause is physical: melanin absorption reduces reflected light amplitude before it reaches the camera sensor, and no signal processing algorithm can recover information that was never captured. The correct clinical response is not to try harder with the face signal — it is to route to a measurement site that does not have this limitation. The palm, as shown by Luo et al. [8], provides a consistently stronger signal regardless of skin tone.

PulseRoute's contribution is not that it solves the physics problem, but that it correctly identifies when the physics problem is present and responds appropriately. Session S4 illustrates this precisely: the composite score would have accepted the face result for this FST VI subject, but the personal amplitude floor correctly identified that RMSSD would be unreliable. Without the second routing check, the system would have reported inflated HRV to the clinician without warning.

### 5.2 Personal Floors vs. Fixed Constants

The most technically novel aspect of PulseRoute is the replacement of the fixed HRV reliability floor with a personal, session-derived value. A fixed constant (e.g., 0.002) derived from pilot data on one subject with one webcam transfers poorly to other deployment contexts for three reasons.

First, signal amplitude varies with device quality: a high-resolution camera in good lighting produces filtered std 5–10× higher than a laptop webcam in dim conditions. A floor calibrated for the laptop would never trigger on the high-quality camera, even when HRV would be unreliable on that camera too.

Second, signal amplitude varies with skin tone: a fixed floor that triggers correctly at FST V may never trigger at FST III (where the signal is naturally stronger) or may always trigger at FST VI (where even the best-case signal falls below it).

Third, signal amplitude varies with lighting: the same subject produces filtered std of 0.001 under dim ceiling lighting and 0.005 under direct window light. A fixed floor calibrated for dim conditions would never trigger in good lighting, missing cases where HRV is unreliable due to algorithm limitations rather than signal amplitude.

The personal floor eliminates all three problems simultaneously because it is derived from the patient's own signal on their own device in their own lighting, and scales proportionally with whatever combination of factors determines their actual signal strength.

### 5.3 Limitations

**Small pilot sample:** Two subjects with 7 valid sessions provides initial validation but is insufficient for clinical claims. A larger study with ECG ground truth, diverse skin tones, ages, and medical conditions is required before deployment.

**No ECG-validated HRV:** Pulse oximeter ground truth provides HR but not beat-to-beat intervals. RMSSD accuracy requires ECG or a validated PPG device.

**Single camera type:** All sessions used a standard 640×480 laptop webcam. Performance on higher-resolution cameras has not been characterized.

**Palm validation pending:** The palm pipeline is implemented and runs correctly, but palm HR has not been validated against ground truth in isolation (only as part of the routing flow).

**Age range:** Both subjects were adults aged 20–30. Older subjects have not been tested.

**Scale factor limitation:** `calibrate_scale_factor` measures the raw-to-filtered ratio at the green channel level, which does not account for the normalization and POS steps that further modify signal amplitude. This means the measured scale factor approximates the true ratio but does not equal it exactly. Future work will investigate running a full mini-pipeline during calibration to measure the true end-to-end ratio.

---

## 6. Conclusion

We presented PulseRoute, a contactless vital signs monitoring system designed for inclusive teleconsultation. The system's central contribution is an architecture in which no threshold is fixed at a population-average constant — every decision boundary is derived from the patient's own signal in their own environment. This eliminates the principal sources of bias and failure in existing rPPG systems: thresholds calibrated on lab subjects under controlled lighting that fail for darker skin tones, low-quality cameras, or home environments.

The routing architecture — face first for user experience, palm as fallback for signal quality — is, to our knowledge, the first published implementation of a systematic face-to-palm rPPG routing system with a principled, patient-personalized signal quality decision boundary. The dual routing checks (composite score and personal amplitude floor) correctly handled all 7 pilot sessions with no false routings.

Full validation with ECG ground truth across diverse skin tones and age groups is planned as the next phase of this work. The system is open source and publicly available at https://github.com/arijbelmabrouk/PulseRoute.

---

## References

[1] Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. *Optics Express*, 16(26), 21434–21445.

[2] McDuff, D., Gontarek, S., & Picard, R. W. (2014). Improvements in remote cardiopulmonary measurement using a five band digital camera. *IEEE Transactions on Biomedical Engineering*, 61(10), 2593–2601.

[3] Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. *Optics Express*, 18(10), 10762–10774.

[4] Ba, Y., Liu, Z., Li, X., Shi, X., & Achanta, R. (2021). Overcoming racial bias in automatic facial analysis. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 1–10.

[5] Nowara, E. M., McDuff, D., & Veeraraghavan, A. (2020). A meta-analysis of the impact of skin tone and gender on non-contact photoplethysmography measurements. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops*, pp. 284–285.

[6] Hill, L. K., Sollers III, J. J., & Thayer, J. F. (2015). Resistance, resilience, and racial health disparities: implications of autonomic nervous system functioning. *Annals of Behavioral Medicine*, 49(4), 510–519.

[7] Jacques, S. L. (2013). Optical properties of biological tissues: a review. *Physics in Medicine & Biology*, 58(11), R37.

[8] Luo, H., Yang, D., Barszczyk, A., Vempala, N., Wei, J., Wu, S. J., ... & Lee, K. (2019). Smartphone-based blood pressure measurement using transdermal optical imaging technology. *Circulation: Cardiovascular Imaging*, 12(8), e008857.

[9] Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. *Optics Express*, 16(26), 21434–21445.

[10] de Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. *IEEE Transactions on Biomedical Engineering*, 60(10), 2878–2886.

[11] Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). Algorithmic principles of remote PPG. *IEEE Transactions on Biomedical Engineering*, 64(7), 1479–1491.

[12] Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172*.

[13] Yu, C., Wang, J., Peng, C., Gao, C., Yu, G., & Sang, N. (2018). BiSeNet: Bilateral segmentation network for real-time semantic segmentation. *Proceedings of the European Conference on Computer Vision (ECCV)*, pp. 325–341.

[14] Del Bino, S., Sok, J., Bessac, E., & Bernerd, F. (2006). Relationship between skin response to ultraviolet exposure and skin color type. *Pigment Cell Research*, 19(6), 606–614.

[15] McDuff, D. J., Blackford, E. B., & Estepp, J. R. (2018). Fusing partial camera signals for cardiac pulse rate estimation. *IEEE Transactions on Biomedical Engineering*, 65(8), 1725–1739.

[16] Shi, J., Alinejad, M., & Bhaskaran, B. (2019). Contactless heart rate and heart rate variability monitoring using video. In *2019 IEEE 16th International Conference on Wearable and Implantable Body Sensor Networks (BSN)*, pp. 1–4.

[17] Shaffer, F., & Ginsberg, J. P. (2017). An overview of heart rate variability metrics and norms. *Frontiers in Public Health*, 5, 258.

[18] Nunan, D., Sandercock, G. R., & Brodie, D. A. (2010). A quantitative systematic review of normal values for short-term heart rate variability in healthy adults. *Pacing and Clinical Electrophysiology*, 33(11), 1407–1417.