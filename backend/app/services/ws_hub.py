"""WebSocket connection hub for live battle dashboard updates."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket
from pydantic import BaseModel

from app.schema.battle_log import BattleLogEvent
from app.services.session import SessionStore

logger = logging.getLogger(__name__)


def _dump(model: BaseModel | None) -> dict[str, Any] | None:
    if model is None:
        return None
    return model.model_dump(mode="json", by_alias=True)


def _dump_event(event: BattleLogEvent) -> dict[str, Any]:
    return event.model_dump(mode="json")


def session_payload(store: SessionStore) -> dict[str, Any]:
    return {
        "phase": store.phase.value,
        "turn_number": store.turn_number,
        "cv_running": store.cv_running,
        "team_loaded": store.team_loaded,
        "adb_connected": store.adb_connected,
    }


def flatten_battle_logs(
    store: SessionStore,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Return recent battle log events newest-last, capped at ``limit``.

    Includes pre-turn lead-in events from ``battle_logs[0]``.
    """
    events: list[BattleLogEvent] = []
    for turn_logs in store.battle_logs:
        events.extend(turn_logs)
    if limit > 0 and len(events) > limit:
        events = events[-limit:]
    return [_dump_event(event) for event in events]


def snapshot_payload(store: SessionStore) -> dict[str, Any]:
    return {
        "session": session_payload(store),
        "game_state": _dump(store.game_state),
        "battle_logs": flatten_battle_logs(store),
        "opponent_team_species": store.opponent_team_species,
        "player_selected_species": store.player_selected_species,
        "team_preview_suggestion": _dump(store.team_preview_suggestion),
        "turn_suggestion": _dump(store.turn_suggestion),
    }


class WsHub:
    """Fan-out JSON messages to connected dashboard clients."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)
        logger.debug("WebSocket connected (%d clients)", len(self._connections))

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)
        logger.debug("WebSocket disconnected (%d clients)", len(self._connections))

    async def send(self, websocket: WebSocket, message: dict[str, Any]) -> None:
        await websocket.send_json(message)

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self._connections:
            return
        dead: list[WebSocket] = []
        for websocket in tuple(self._connections):
            try:
                await websocket.send_json(message)
            except Exception:
                logger.debug("Dropping dead WebSocket client", exc_info=True)
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)

    def publish(self, message: dict[str, Any]) -> None:
        """Schedule a broadcast from sync or async / worker-thread context."""
        if self._loop is None or not self._connections:
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            self._loop.create_task(self.broadcast(message))
        else:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)


ws_hub = WsHub()


def publish_snapshot(store: SessionStore) -> None:
    ws_hub.publish({"type": "snapshot", "payload": snapshot_payload(store)})


def publish_session(store: SessionStore) -> None:
    ws_hub.publish({"type": "session", "payload": session_payload(store)})


def publish_phase(store: SessionStore) -> None:
    ws_hub.publish(
        {
            "type": "phase",
            "payload": {
                "phase": store.phase.value,
                "turn_number": store.turn_number,
            },
        }
    )


def publish_state(store: SessionStore) -> None:
    ws_hub.publish({"type": "state", "payload": _dump(store.game_state)})


def publish_log(event: BattleLogEvent) -> None:
    ws_hub.publish({"type": "log", "payload": _dump_event(event)})


def publish_log_patched(turn: int, index: int, event: BattleLogEvent) -> None:
    ws_hub.publish(
        {
            "type": "log_patched",
            "payload": {
                "turn": turn,
                "index": index,
                "event": _dump_event(event),
            },
        }
    )


def publish_team_preview(store: SessionStore) -> None:
    ws_hub.publish(
        {
            "type": "team_preview",
            "payload": {
                "opponent_species": store.opponent_team_species,
                "player_selected_species": store.player_selected_species,
                "suggestion": _dump(store.team_preview_suggestion),
            },
        }
    )


def publish_turn_suggestion(store: SessionStore) -> None:
    ws_hub.publish(
        {
            "type": "turn_suggestion",
            "payload": _dump(store.turn_suggestion),
        }
    )
