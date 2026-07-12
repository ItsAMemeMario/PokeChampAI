from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_session_store
from app.schema.team import PlayerTeam, parse_team
from app.services.session import SessionStore

router = APIRouter(prefix="/api/team", tags=["team"])


class TeamSubmitRequest(BaseModel):
    pokepaste: str


@router.post("", response_model=PlayerTeam)
async def submit_team(
    body: TeamSubmitRequest,
    store: SessionStore = Depends(get_session_store),
) -> PlayerTeam:
    try:
        team = parse_team(body.pokepaste)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    store.set_team(team)
    return team


@router.get("", response_model=PlayerTeam)
async def get_team(
    store: SessionStore = Depends(get_session_store),
) -> PlayerTeam:
    if store.player_team is None:
        raise HTTPException(status_code=404, detail="No team saved")
    return store.player_team
