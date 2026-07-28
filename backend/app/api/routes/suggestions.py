from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_session_store
from app.schema.suggestions import TeamPreviewSuggestion, TurnSuggestion
from app.services.session import SessionStore

router = APIRouter(prefix="/api/suggestions", tags=["suggestions"])


class TeamPreviewSuggestionResponse(BaseModel):
    opponent_species: list[str] | None
    player_selected_species: list[str] | None
    suggestion: TeamPreviewSuggestion | None


class TurnSuggestionResponse(BaseModel):
    suggestion: TurnSuggestion | None


@router.get("/team-preview", response_model=TeamPreviewSuggestionResponse)
async def get_team_preview_suggestion(
    store: SessionStore = Depends(get_session_store),
) -> TeamPreviewSuggestionResponse:
    if not store.team_loaded:
        raise HTTPException(status_code=400, detail="No player team saved")
    return TeamPreviewSuggestionResponse(
        opponent_species=store.opponent_team_species,
        player_selected_species=store.player_selected_species,
        suggestion=store.team_preview_suggestion,
    )


@router.get("/turn", response_model=TurnSuggestionResponse)
async def get_turn_suggestion(
    store: SessionStore = Depends(get_session_store),
) -> TurnSuggestionResponse:
    if not store.team_loaded:
        raise HTTPException(status_code=400, detail="No player team saved")
    return TurnSuggestionResponse(suggestion=store.turn_suggestion)
