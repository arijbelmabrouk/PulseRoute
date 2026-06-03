"""
Step 12 — Pipeline Event Publisher
Called from run_web.py after each step to push metrics
to the FastAPI server, which broadcasts to the dashboard.

Usage in run_web.py:
    from step12_display.publisher import publish

    publish("step_complete", {
        "step": 9,
        "hr_bpm": hr_results['final_hr'],
        "rmssd": hr_results['rmssd'],
    })
"""

import json
import urllib.request
import urllib.error

SERVER_URL = "http://localhost:8000/api/event"


def publish(event_type: str, data: dict) -> bool:
    """
    POST an event to the dashboard server.

    Parameters
    ----------
    event_type : str
        Short identifier for the event, e.g. "step_complete",
        "motion_detected", "routing_decision", "pipeline_done".
    data : dict
        Payload merged into the event. Always includes
        `event` key set to event_type.

    Returns
    -------
    bool
        True if the server acknowledged, False on any error.
        Errors are printed but never raise — the pipeline
        must never crash because of a display failure.
    """
    payload = {"event": event_type, **data}
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.status == 200
    except Exception as exc:
        # Dashboard unreachable — log and continue silently
        print(f"  [Step 12] Dashboard publish failed: {exc}")
        return False


# ── Convenience wrappers used in run_web.py ────────────

def publish_status(status: str, progress: int = 0, message: str = ""):
    """Update the top-level pipeline status bar."""
    publish("status", {
        "status": status,
        "progress": progress,
        "message": message,
    })


def publish_step(step: int, data: dict):
    """Mark a step as complete with its results."""
    publish("step_complete", {
        "step": step,
        "steps": {str(step): data},
    })


def publish_motion(rejected_frames: int, total_frames: int,
                   motion_g: float = 0.0):
    """Broadcast motion rejection event."""
    publish("motion", {
        "rejected_frames": rejected_frames,
        "total_frames": total_frames,
        "motion_g": motion_g,
        "motion_pct": round(rejected_frames / max(total_frames, 1) * 100, 1),
    })


def publish_routing(route_palm: bool, reason: str,
                    snr_score: float, std_floor_triggered: bool):
    """Broadcast routing decision after Step 11."""
    publish("routing_decision", {
        "route_palm": route_palm,
        "routing_reason": reason,
        "snr_score": snr_score,
        "std_floor_triggered": std_floor_triggered,
    })


def publish_final(hr_results: dict, snr_score: float,
                  quality_level: str, route_palm: bool,
                  ita: float, fitzpatrick: str,
                  rr_bpm, snr_report: dict):
    """Broadcast final results after all steps complete."""
    publish("pipeline_done", {
        "status": "complete",
        "progress": 100,
        "final": {
            "hr_bpm":         hr_results.get("final_hr"),
            "rmssd":          hr_results.get("rmssd"),
            "hrv_overall":    hr_results.get("hrv_overall"),
            "confidence":     hr_results.get("confidence"),
            "confidence_level": hr_results.get("confidence_level"),
            "rr_bpm":         rr_bpm,
            "snr_score":      round(snr_score, 4),
            "quality_level":  quality_level,
            "route_palm":     route_palm,
            "ita":            round(ita, 1),
            "fitzpatrick":    fitzpatrick,
            "std_floor_triggered": snr_report.get("std_floor_triggered", False),
            "routing_reason": (
                "Signal too weak for HRV — palm recommended"
                if snr_report.get("std_floor_triggered")
                else ("SNR below threshold" if route_palm else "Face accepted")
            ),
        },
    })
