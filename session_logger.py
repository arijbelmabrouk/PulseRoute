"""
session_logger.py — PulseRoute Session Logger

Writes one CSV row per measurement session to logs/sessions.csv.

Usage:
    logger = SessionLogger()
    logger.set_patient_id("P-001")
    logger.set_mode("auto")
    logger.set_profile(profile, face_state)
    logger.set_face_results(hr_results, snr_score, ...)
    logger.set_routing(route_palm, reason)
    logger.set_palm_results(hr_results, snr_score, ...)
    logger.set_final(modality, hr_results, rr_bpm)
    logger.write()

    # On failure:
    logger.set_failure("Palm not detected")
    logger.write()
"""

import csv
import os
import sys
import time
from datetime import datetime


# ─────────────────────────────────────────
# Log file location
# ─────────────────────────────────────────

def _get_log_dir():
    if getattr(sys, 'frozen', False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, 'logs')


LOG_DIR  = _get_log_dir()
LOG_FILE = os.path.join(LOG_DIR, 'sessions.csv')

CSV_FIELDS = [
    'timestamp',
    'patient_id',          # ← NEW: links session to patient
    'patient_age',         # ← NEW: patient age for session context
    'mode',
    'ita',
    'fitzpatrick',
    'profile_valid',
    'calibration_scale_factor',
    'fps_measured',
    'face_snr_score',
    'face_quality_level',
    'face_hr_bpm',
    'face_rr_bpm',
    'face_rmssd',
    'face_hr_reliable',
    'face_confidence',
    'routing_decision',
    'routing_reason',
    'palm_hr_bpm',
    'palm_rr_bpm',
    'palm_snr_score',
    'palm_hr_reliable',
    'palm_confidence',
    'final_modality',
    'final_hr_bpm',
    'final_rr_bpm',
    'hrv_available',
    'hrv_fps_message',
    'failure_reason',
    'session_duration_sec',
]


# ─────────────────────────────────────────
# SessionLogger
# ─────────────────────────────────────────

class SessionLogger:
    """
    Accumulates session data and writes one CSV row
    at the end of the session (success or failure).
    """

    def __init__(self):
        self._start = time.time()
        self._row   = {field: '' for field in CSV_FIELDS}
        self._row['timestamp'] = datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S'
        )
        self._ensure_log_file()

    # ── Setters ──────────────────────────────────────

    def set_patient_id(self, patient_id: str):
        """Link this session to a patient name/ID."""
        self._row['patient_id'] = str(patient_id).strip()

    def set_patient_age(self, patient_age):
        """Record patient age for the current session."""
        if patient_age is None:
            self._row['patient_age'] = ''
        else:
            self._row['patient_age'] = str(patient_age).strip()

    def set_mode(self, mode: str):
        """'auto' or 'palm'"""
        self._row['mode'] = mode

    def set_profile(self, profile, face_state=None):
        self._row['ita']         = _fmt(profile.ita, 1)
        self._row['fitzpatrick'] = profile.fitzpatrick or ''
        self._row['profile_valid'] = (
            'True' if profile.is_valid else 'False'
        )
        self._row['calibration_scale_factor'] = _fmt(
            getattr(profile, 'calib_to_filtered_scale', None), 6
        )

    def set_fps(self, fps: float):
        self._row['fps_measured'] = _fmt(fps, 1)

    def set_face_results(self, hr_results: dict,
                          snr_score: float,
                          quality_level: str,
                          rr_bpm):
        self._row['face_snr_score']     = _fmt(snr_score, 4)
        self._row['face_quality_level'] = quality_level or ''
        self._row['face_hr_bpm']        = _fmt(
            hr_results.get('final_hr'), 1
        )
        self._row['face_rr_bpm']        = _fmt(rr_bpm, 1)
        self._row['face_rmssd']         = _fmt(
            hr_results.get('rmssd'), 1
        )
        self._row['face_hr_reliable']   = str(
            hr_results.get('hr_reliable', True)
        )
        self._row['face_confidence']    = _fmt(
            hr_results.get('confidence'), 3
        )

    def set_routing(self, route_palm, reason: str = ''):
        if route_palm is None:
            self._row['routing_decision'] = 'DIRECT_PALM'
        else:
            self._row['routing_decision'] = (
                'PALM' if route_palm else 'FACE'
            )
        self._row['routing_reason'] = reason or ''

    def set_palm_results(self, hr_results: dict,
                          snr_score: float,
                          rr_bpm):
        self._row['palm_hr_bpm']      = _fmt(
            hr_results.get('final_hr'), 1
        )
        self._row['palm_rr_bpm']      = _fmt(rr_bpm, 1)
        self._row['palm_snr_score']   = _fmt(snr_score, 4)
        self._row['palm_hr_reliable'] = str(
            hr_results.get('hr_reliable', True)
        )
        self._row['palm_confidence']  = _fmt(
            hr_results.get('confidence'), 3
        )

    def set_final(self, modality: str,
                   hr_results: dict,
                   rr_bpm):
        self._row['final_modality'] = modality.upper()
        self._row['final_hr_bpm']   = _fmt(
            hr_results.get('final_hr'), 1
        )
        self._row['final_rr_bpm']   = _fmt(rr_bpm, 1)
        self._row['hrv_available']  = str(
            hr_results.get('hrv_available', False)
        )
        self._row['hrv_fps_message'] = hr_results.get(
            'hrv_fps_message', ''
        )

    def set_failure(self, reason: str):
        self._row['failure_reason'] = reason

    # ── Write ────────────────────────────────────────

    def write(self):
        """
        Write the session row to the CSV.
        Silently swallows write errors — logging must never
        crash the pipeline.
        """
        self._row['session_duration_sec'] = _fmt(
            time.time() - self._start, 1
        )
        try:
            with open(LOG_FILE, 'a', newline='',
                      encoding='utf-8') as f:
                writer = csv.DictWriter(
                    f, fieldnames=CSV_FIELDS
                )
                writer.writerow(self._row)
        except Exception as e:
            print(f"[logger] Warning — could not write log: {e}")

    # ── Internal ─────────────────────────────────────

    def _ensure_log_file(self):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            if not os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'w', newline='',
                          encoding='utf-8') as f:
                    writer = csv.DictWriter(
                        f, fieldnames=CSV_FIELDS
                    )
                    writer.writeheader()
        except Exception as e:
            print(
                f"[logger] Warning — could not create log file: {e}"
            )


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def _fmt(value, decimals: int) -> str:
    if value is None:
        return ''
    try:
        return f'{float(value):.{decimals}f}'
    except (TypeError, ValueError):
        return str(value)