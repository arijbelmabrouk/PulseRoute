"""
Step 12 — Pipeline Event Publisher

Called from run_web.py after each step to push metrics
to the FastAPI server, which broadcasts to the dashboard.
"""

import base64
import json
import urllib.request
import urllib.error

SERVER_URL       = "http://localhost:8000/api/event"
FRAME_SERVER_URL = "http://localhost:8000/api/frame"


def publish(event_type: str, data: dict) -> bool:
    """
    POST an event to the dashboard server.
    Never raises — the pipeline must never crash because
    of a display failure.
    """
    payload = {"event": event_type, **data}
    body    = json.dumps(payload).encode("utf-8")
    req     = urllib.request.Request(
        SERVER_URL,
        data    = body,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"  [Step 12] Dashboard publish failed: {exc}")
        return False


def publish_frame(frame_b64: str) -> bool:
    """
    POST an annotated JPEG frame (base64-encoded) to the
    dashboard server, which broadcasts it to the patient
    WebSocket so the patient sees their live camera feed
    with ROI overlay.

    Called from the pipeline every N frames during signal
    extraction (Step 3 / Step 3b).

    Parameters
    ----------
    frame_b64 : str
        Base64-encoded JPEG bytes of the annotated frame.

    Returns
    -------
    bool
        True if server acknowledged, False on any error.
    """
    body = json.dumps({"frame": frame_b64}).encode("utf-8")
    req  = urllib.request.Request(
        FRAME_SERVER_URL,
        data    = body,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.status == 200
    except Exception as exc:
        print(f"  [Step 12] Frame publish failed: {exc}")
        return False


# ── Convenience wrappers ───────────────────────────────

def publish_status(status: str, progress: int = 0,
                   message: str = ""):
    publish("status", {
        "status":   status,
        "progress": progress,
        "message":  message,
    })


def publish_step(step: int, data: dict):
    publish("step_complete", {
        "step":  step,
        "steps": {str(step): data},
    })


def publish_motion(rejected_frames: int, total_frames: int,
                   motion_g: float = 0.0):
    publish("motion", {
        "rejected_frames": rejected_frames,
        "total_frames":    total_frames,
        "motion_g":        motion_g,
        "motion_pct":      round(
            rejected_frames / max(total_frames, 1) * 100, 1
        ),
    })


def publish_routing(route_palm: bool, reason: str,
                    snr_score: float,
                    std_floor_triggered: bool):
    publish("routing_decision", {
        "route_palm":          route_palm,
        "routing_reason":      reason,
        "snr_score":           snr_score,
        "std_floor_triggered": std_floor_triggered,
    })


def publish_final(hr_results: dict, snr_score: float,
                  quality_level: str, route_palm: bool,
                  ita: float, fitzpatrick: str,
                  rr_bpm, snr_report: dict):
    publish("pipeline_done", {
        "status":   "complete",
        "progress": 100,
        "final": {
            "hr_bpm":          hr_results.get("final_hr"),
            "rmssd":           hr_results.get("rmssd"),
            "hrv_overall":     hr_results.get("hrv_overall"),
            "confidence":      hr_results.get("confidence"),
            "confidence_level": hr_results.get("confidence_level"),
            "rr_bpm":          rr_bpm,
            "snr_score":       round(snr_score, 4),
            "quality_level":   quality_level,
            "route_palm":      route_palm,
            "ita":             round(ita, 1),
            "fitzpatrick":     fitzpatrick,
            "std_floor_triggered": snr_report.get(
                "std_floor_triggered", False
            ),
            "routing_reason": (
                "Signal too weak for HRV — palm recommended"
                if snr_report.get("std_floor_triggered")
                else (
                    "SNR below threshold"
                    if route_palm else "Face accepted"
                )
            ),
        },
    })