"""
Step 12 — Web Dashboard Backend
FastAPI + WebSocket server that receives pipeline events
from run_web.py and broadcasts them to connected clients.

Run:
    uvicorn step12_display.server:app --reload --port 8000
"""

import asyncio
import json
from typing import Any, Dict, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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

# ── Connection manager ─────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()
        self.latest_state: Dict[str, Any] = {
            "status": "idle",
            "progress": 0,
            "steps": {}
        }

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)
        # Send current snapshot immediately so late joiners
        # see the current state
        try:
            await ws.send_text(json.dumps(self.latest_state))
        except Exception:
            self.active.discard(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, data: Dict[str, Any]):
        # Merge into latest state
        self.latest_state = {
            **self.latest_state,
            **data,
            "steps": {
                **self.latest_state.get("steps", {}),
                **data.get("steps", {}),
            }
        }
        msg = json.dumps(data)
        dead: Set[WebSocket] = set()
        for ws in list(self.active):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()

# ── Internal event queue ───────────────────────────────
event_queue: asyncio.Queue = asyncio.Queue()


async def _queue_dispatcher():
    while True:
        event = await event_queue.get()
        await manager.broadcast(event)


@app.on_event("startup")
async def startup():
    asyncio.create_task(_queue_dispatcher())


# ── WebSocket endpoints ────────────────────────────────
@app.websocket("/ws/patient")
async def patient_ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Just keep the connection alive with a ping
            # every 20 seconds; ignore anything the client sends
            await asyncio.sleep(20)
            try:
                await websocket.send_text('{"ping":1}')
            except Exception:
                break
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        manager.disconnect(websocket)


@app.websocket("/ws/doctor")
async def doctor_ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(20)
            try:
                await websocket.send_text('{"ping":1}')
            except Exception:
                break
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        manager.disconnect(websocket)


# ── REST endpoint for pipeline to publish events ───────
@app.post("/api/event")
async def receive_event(payload: Dict[str, Any]):
    await event_queue.put(payload)
    return {"ok": True}


@app.get("/api/state")
async def get_state():
    return manager.latest_state


@app.post("/api/reset")
async def reset_state():
    manager.latest_state = {"status": "idle", "progress": 0, "steps": {}}
    await manager.broadcast(manager.latest_state)
    return {"ok": True}


if __name__ == "__main__":
    uvicorn.run(
        "step12_display.server:app",
        host="0.0.0.0", port=8000, reload=True
    )
