"""REST endpoints for current game state and battle logs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict

from app.api.deps import get_session_store
from app.schema.gamestate import GameState
from app.services.session import SessionStore
from app.services.ws_hub import flatten_battle_logs

router = APIRouter(tags=["state"])


class GameStateResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    game_state: GameState | None


class BattleLogsResponse(BaseModel):
    events: list[dict[str, Any]]


@router.get("/api/state", response_model=GameStateResponse)
async def get_state(
    store: SessionStore = Depends(get_session_store),
) -> GameStateResponse:
    return GameStateResponse(game_state=store.game_state)


@router.get("/api/logs", response_model=BattleLogsResponse)
async def get_logs(
    limit: int = Query(100, ge=1, le=500),
    store: SessionStore = Depends(get_session_store),
) -> BattleLogsResponse:
    return BattleLogsResponse(events=flatten_battle_logs(store, limit=limit))
