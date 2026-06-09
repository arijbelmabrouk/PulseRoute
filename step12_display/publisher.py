"""
Step 12 — Pipeline Event Publisher

Key change in this version:
  publish_frame() is now NON-BLOCKING.

  Previously it called urllib.request.urlopen() directly,
  which blocked the camera capture loop for 20-50ms per
  frame. At 30fps (33ms/frame) this halved the live feed
  framerate and caused the "frozen feed" symptom.

  Now: frames are dropped into a thread-safe queue.
  A single background daemon thread drains the queue
  and does the HTTP POST. The capture loop never waits.

  Queue size raised to 10 (was 3). During heavy BiSeNet
  processing the main thread holds the GIL for long
  stretches. A larger queue prevents frames from being
  dropped before the publisher thread gets a chance to
  drain them. This fixes the symptom where all Step 2
  frames arrived at the server only after Step 2 ended.

  publish_frame() also calls time.sleep(0) after enqueue.
  This is a GIL yield hint — it tells the CPython
  scheduler to switch to the publisher daemon thread
  immediately rather than waiting for the next bytecode
  tick. Cost: ~0 microseconds. Benefit: Step 2 frames
  start arriving at the browser within the first second
  of Phase 1 instead of after Phase 2 completes.
"""

import base64
import json
import queue
import threading
import time
import urllib.request
import urllib.error

SERVER_URL       = "http://localhost:8000/api/event"
FRAME_SERVER_URL = "http://localhost:8000/api/frame"

# ── Background frame publisher ─────────────────────────
# Queue size 10: large enough to buffer frames during
# heavy BiSeNet inference without dropping everything.
_frame_queue: queue.Queue = queue.Queue(maxsize=10)
_publisher_started = False
_publisher_lock    = threading.Lock()


def _frame_publisher_thread():
    """
    Background daemon thread.
    Drains _frame_queue and POSTs frames to /api/frame.
    Runs forever — killed automatically when main exits.
    """
    while True:
        try:
            frame_b64 = _frame_queue.get(timeout=2.0)
        except queue.Empty:
            continue

        body = json.dumps({"frame": frame_b64}).encode("utf-8")
        req  = urllib.request.Request(
            FRAME_SERVER_URL,
            data    = body,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=1) as _:
                pass
        except Exception:
            pass  # silent — never crash pipeline for a frame

        _frame_queue.task_done()


def _ensure_publisher_running():
    """Start the background thread once, lazily."""
    global _publisher_started
    with _publisher_lock:
        if not _publisher_started:
            t = threading.Thread(
                target=_frame_publisher_thread,
                daemon=True,
                name="frame-publisher"
            )
            t.start()
            _publisher_started = True


# ── Public API ─────────────────────────────────────────

def publish(event_type: str, data: dict) -> bool:
    """
    POST an event to the dashboard server.
    Blocking — called infrequently (once per step).
    Never raises — pipeline must never crash on display failure.
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
    Enqueue an annotated JPEG frame for background publishing.

    NON-BLOCKING — returns immediately. The background
    thread handles the actual HTTP POST so the camera
    capture loop is never stalled.

    If the queue is full (publisher fell behind), the
    oldest pending frame is discarded and the new frame
    takes its place — keeping the feed current.

    time.sleep(0) after enqueue is a CPython GIL yield.
    It tells the scheduler to switch to the publisher
    daemon thread immediately. This is critical during
    Step 2 where BiSeNet inference holds the GIL for
    long periods and the publisher thread never gets
    scheduled otherwise.

    Parameters
    ----------
    frame_b64 : str
        Base64-encoded JPEG bytes of the annotated frame.

    Returns
    -------
    bool  Always True (enqueue never blocks or fails).
    """
    _ensure_publisher_running()

    # If queue is full, drop the oldest pending frame
    # so the newest frame gets sent instead.
    if _frame_queue.full():
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            pass

    try:
        _frame_queue.put_nowait(frame_b64)
    except queue.Full:
        pass  # race condition guard — safe to drop

    # Yield GIL to publisher thread immediately.
    # Without this, the publisher thread only runs
    # between Python bytecode ticks, which during
    # heavy numpy/torch work can be hundreds of ms apart.
    time.sleep(0)

    return True


# ── Convenience wrappers (unchanged) ──────────────────

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


def publish_recording_progress(progress: int,
                                message: str = "",
                                active: bool = True):
    publish("recording_progress", {
        "recording_active":   active,
        "recording_progress": progress,
        "recording_message":  message,
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
            "hr_bpm":           hr_results.get("final_hr"),
            "rmssd":            hr_results.get("rmssd"),
            "hrv_overall":      hr_results.get("hrv_overall"),
            "confidence":       hr_results.get("confidence"),
            "confidence_level": hr_results.get("confidence_level"),
            "rr_bpm":           rr_bpm,
            "snr_score":        round(snr_score, 4),
            "quality_level":    quality_level,
            "route_palm":       route_palm,
            "ita":              round(ita, 1),
            "fitzpatrick":      fitzpatrick,
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