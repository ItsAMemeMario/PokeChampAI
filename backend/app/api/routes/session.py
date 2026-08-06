from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_session_store
from app.services.cv_runner import start_cv, stop_cv
from app.services.session import BattlePhase, SessionStore
from app.services.ws_hub import publish_session, publish_snapshot

router = APIRouter(prefix="/api/session", tags=["session"])


class SessionStatus(BaseModel):
    phase: BattlePhase
    turn_number: int
    cv_running: bool
    team_loaded: bool
    adb_connected: bool


def _session_status(store: SessionStore) -> SessionStatus:
    return SessionStatus(
        phase=store.phase,
        turn_number=store.turn_number,
        cv_running=store.cv_running,
        team_loaded=store.team_loaded,
        adb_connected=store.adb_connected,
    )


@router.get("", response_model=SessionStatus)
async def get_session(
    store: SessionStore = Depends(get_session_store),
) -> SessionStatus:
    return _session_status(store)


@router.post("/start", response_model=SessionStatus)
async def start_session(
    store: SessionStore = Depends(get_session_store),
) -> SessionStatus:
    try:
        store.start_monitoring()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    start_cv(store)
    publish_snapshot(store)
    return _session_status(store)


@router.post("/stop", response_model=SessionStatus)
async def stop_session(
    store: SessionStore = Depends(get_session_store),
) -> SessionStatus:
    store.stop_monitoring()
    await stop_cv()
    publish_session(store)
    return _session_status(store)
