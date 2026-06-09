"""
Step 12 — Web Dashboard Backend

FastAPI + WebSocket server.
Separate patient and doctor WebSocket connections.
Pipeline spawned as subprocess on /api/start.

Run:
    uvicorn step12_display.server:app --reload --port 8000
"""

import asyncio
import csv
import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional, Set

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="PulseRoute Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────

shared_state: Dict[str, Any] = {
    "status":    "idle",
    "progress":  0,
    "steps":     {},
    "patient_id": "",
    "patient_connection_count": 0,
}

patient_connections: Set[WebSocket] = set()
doctor_connections:  Set[WebSocket] = set()

# Running pipeline subprocess
_pipeline_proc: Optional[subprocess.Popen] = None

# Project root (two levels up from this file)
_SERVER_FILE   = os.path.abspath(__file__)
_STEP12_DIR    = os.path.dirname(_SERVER_FILE)
_PROJECT_ROOT  = os.path.dirname(_STEP12_DIR)
_LOG_FILE      = os.path.join(_PROJECT_ROOT, "logs", "sessions.csv")

# ─────────────────────────────────────────
# Broadcast helpers
# ─────────────────────────────────────────

async def _send(ws: WebSocket, msg: str) -> bool:
    try:
        await ws.send_text(msg)
        return True
    except Exception:
        return False


async def broadcast_event(data: Dict[str, Any]):
    """Merge data into shared_state, send to all connections."""
    global shared_state
    shared_state = {
        **shared_state,
        **data,
        "steps": {
            **shared_state.get("steps", {}),
            **data.get("steps", {}),
        }
    }
    msg = json.dumps(data)
    dead: Set[WebSocket] = set()
    for ws in list(patient_connections | doctor_connections):
        ok = await _send(ws, msg)
        if not ok:
            dead.add(ws)
    patient_connections.difference_update(dead)
    doctor_connections.difference_update(dead)


async def broadcast_frame(frame_b64: str):
    """Send annotated JPEG frame to patient connections only."""
    msg = json.dumps({"type": "frame", "frame": frame_b64})
    dead: Set[WebSocket] = set()
    for ws in list(patient_connections):
        ok = await _send(ws, msg)
        if not ok:
            dead.add(ws)
    patient_connections.difference_update(dead)


# ─────────────────────────────────────────
# Internal event queue (pipeline → server)
# ─────────────────────────────────────────

event_queue: asyncio.Queue = asyncio.Queue()


async def _queue_dispatcher():
    while True:
        event = await event_queue.get()
        if event.get("type") == "frame":
            await broadcast_frame(event.get("frame", ""))
        else:
            await broadcast_event(event)


@app.on_event("startup")
async def startup():
    asyncio.create_task(_queue_dispatcher())


# ─────────────────────────────────────────
# WebSocket endpoints
# ─────────────────────────────────────────

async def _ws_keepalive(ws: WebSocket,
                         connections: Set[WebSocket],
                         is_patient: bool = False):
    """Accept, send current state, keep alive."""
    await ws.accept()
    connections.add(ws)

    if is_patient:
        await broadcast_event({
            "patient_connection_count": len(patient_connections),
        })

    try:
        await ws.send_text(json.dumps(shared_state))
    except Exception:
        connections.discard(ws)
        if is_patient:
            await broadcast_event({
                "patient_connection_count": len(patient_connections),
            })
        return
    try:
        while True:
            await asyncio.sleep(20)
            ok = await _send(ws, '{"ping":1}')
            if not ok:
                break
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        connections.discard(ws)
        if is_patient:
            await broadcast_event({
                "patient_connection_count": len(patient_connections),
            })


@app.websocket("/ws/patient")
async def patient_ws(websocket: WebSocket):
    await _ws_keepalive(websocket, patient_connections, is_patient=True)


@app.websocket("/ws/doctor")
async def doctor_ws(websocket: WebSocket):
    await _ws_keepalive(websocket, doctor_connections)


# ─────────────────────────────────────────
# Pipeline event ingestion (from publisher)
# ─────────────────────────────────────────

@app.post("/api/event")
async def receive_event(payload: Dict[str, Any]):
    await event_queue.put(payload)
    return {"ok": True}


@app.post("/api/frame")
async def receive_frame(payload: Dict[str, Any]):
    """Receive annotated JPEG frame from pipeline."""
    await event_queue.put({
        "type":  "frame",
        "frame": payload.get("frame", ""),
    })
    return {"ok": True}


# ─────────────────────────────────────────
# Patient ID registration
# ─────────────────────────────────────────

@app.post("/api/patient-id")
async def set_patient_id(payload: Dict[str, Any]):
    """
    Patient submits their name/ID from the patient page.
    Stored in shared_state and broadcast to the doctor.
    """
    patient_id = str(payload.get("patient_id", "")).strip()
    patient_mode = str(payload.get("mode", "auto")).strip()
    shared_state["patient_id"]   = patient_id
    shared_state["patient_mode"] = patient_mode
    await broadcast_event({
        "patient_id":   patient_id,
        "patient_mode": patient_mode,
    })
    return {"ok": True}


# ─────────────────────────────────────────
# Pipeline start
# ─────────────────────────────────────────

@app.post("/api/start")
async def start_pipeline(payload: Dict[str, Any]):
    """
    Doctor clicks Start.
    Spawns run_web.py as a subprocess with mode and
    patient_id passed via environment variables.
    Kills any existing pipeline process first.
    """
    global _pipeline_proc

    # Kill existing process if still running
    if _pipeline_proc is not None:
        try:
            if _pipeline_proc.poll() is None:
                _pipeline_proc.terminate()
        except Exception:
            pass
        _pipeline_proc = None

    # Mode: use payload override, else patient's stored choice, else auto
    mode       = str(payload.get("mode") or
                     shared_state.get("patient_mode", "auto")).strip()
    patient_id = str(payload.get("patient_id", "")
                     or shared_state.get("patient_id", "")).strip()

    # Reset state before starting
    fresh: Dict[str, Any] = {
        "status":     "starting",
        "progress":   0,
        "steps":      {},
        "patient_id": patient_id,
    }
    shared_state.clear()
    shared_state.update(fresh)
    await broadcast_event(fresh)

    # Build env for subprocess
    env = os.environ.copy()
    env["PULSEROUTE_MODE"]       = mode
    env["PULSEROUTE_PATIENT_ID"] = patient_id
    env["PULSEROUTE_HEADLESS"]   = "true"

    script = os.path.join(_PROJECT_ROOT, "run_web.py")

    try:
        _pipeline_proc = subprocess.Popen(
            [sys.executable, script],
            env=env,
            cwd=_PROJECT_ROOT,
        )
        return {"ok": True, "pid": _pipeline_proc.pid}
    except Exception as exc:
        await broadcast_event({
            "status":  "failed",
            "message": f"Failed to start pipeline: {exc}",
        })
        return {"ok": False, "error": str(exc)}


# ─────────────────────────────────────────
# Session history
# ─────────────────────────────────────────

@app.get("/api/history")
async def get_history(patient_id: str = Query("")):
    """
    Read sessions.csv and return rows newest-first.
    If patient_id is provided, filter to that patient.
    Returns an empty list if the log file doesn't exist.
    """
    if not os.path.exists(_LOG_FILE):
        return []

    expected_fields = [
        'timestamp', 'patient_id', 'patient_age', 'mode', 'ita',
        'fitzpatrick', 'profile_valid', 'calibration_scale_factor',
        'fps_measured', 'face_snr_score', 'face_quality_level',
        'face_hr_bpm', 'face_rr_bpm', 'face_rmssd', 'face_hr_reliable',
        'face_confidence', 'routing_decision', 'routing_reason',
        'palm_hr_bpm', 'palm_rr_bpm', 'palm_snr_score',
        'palm_hr_reliable', 'palm_confidence', 'final_modality',
        'final_hr_bpm', 'final_rr_bpm', 'hrv_available',
        'hrv_fps_message', 'failure_reason', 'session_duration_sec',
    ]

    rows = []
    try:
        with open(_LOG_FILE, newline='', encoding="utf-8") as f:
            reader = list(csv.reader(f))

        if not reader:
            return []

        header = reader[0]
        if all(field in expected_fields for field in header):
            records = [dict(zip(header, row)) for row in reader[1:]]
        else:
            records = [dict(zip(expected_fields, row)) for row in reader]

        for row in records:
            pid = row.get("patient_id", "").strip()
            if patient_id and pid != patient_id.strip():
                continue
            rows.append(row)
    except Exception as exc:
        return {"error": str(exc)}

    return list(reversed(rows))   # newest first


# ─────────────────────────────────────────
# State and reset
# ─────────────────────────────────────────

@app.get("/api/state")
async def get_state():
    return shared_state


@app.get("/api/patient-connections")
async def get_patient_connections():
    return {
        "count": len(patient_connections),
        "ready": len(patient_connections) > 0,
    }


@app.post("/api/reset")
async def reset_state():
    """Kill pipeline and reset to idle."""
    global _pipeline_proc
    if _pipeline_proc is not None:
        try:
            if _pipeline_proc.poll() is None:
                _pipeline_proc.terminate()
        except Exception:
            pass
        _pipeline_proc = None

    fresh = {
        "status":     "idle",
        "progress":   0,
        "steps":      {},
        "patient_id": shared_state.get("patient_id", ""),
    }
    shared_state.clear()
    shared_state.update(fresh)
    await broadcast_event(fresh)
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(
        "step12_display.server:app",
        host="0.0.0.0", port=8000, reload=True
    )