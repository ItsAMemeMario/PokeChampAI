"""WebSocket endpoint for live battle dashboard streaming."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.deps import get_session_store
from app.services.ws_hub import snapshot_payload, ws_hub

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws")
async def battle_websocket(websocket: WebSocket) -> None:
    await ws_hub.connect(websocket)
    store = get_session_store()
    try:
        await ws_hub.send(
            websocket,
            {"type": "snapshot", "payload": snapshot_payload(store)},
        )
        while True:
            # Keep the connection alive; clients may send pings / no-ops.
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.debug("Client disconnected from /ws")
    except Exception:
        logger.exception("WebSocket error on /ws")
    finally:
        ws_hub.disconnect(websocket)
