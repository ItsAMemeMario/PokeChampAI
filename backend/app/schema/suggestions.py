from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

from app.schema.common import Pokemon


class TeamPreviewSuggestion(BaseModel):
    predicted_opponent_bring: list[str]
    predicted_opponent_lead_pair: tuple[str, str]
    suggested_player_bring: list[str]
    suggested_player_lead_pair: tuple[str, str]
    reasoning: str

    @field_validator("predicted_opponent_bring", "suggested_player_bring")
    @classmethod
    def exactly_four_species(cls, value: list[str]) -> list[str]:
        if len(value) != 4:
            raise ValueError(f"Expected exactly 4 species, got {len(value)}")
        return value

    @field_validator("predicted_opponent_lead_pair")
    @classmethod
    def opponent_lead_pair_matches_bring(cls, value: tuple[str, str], info) -> tuple[str, str]:
        bring = info.data.get("predicted_opponent_bring")
        if bring and (value[0] not in bring or value[1] not in bring):
            raise ValueError("predicted_opponent_lead_pair must match the first two entries of predicted_opponent_bring")
        return value

    @field_validator("suggested_player_lead_pair")
    @classmethod
    def player_lead_pair_matches_bring(cls, value: tuple[str, str], info) -> tuple[str, str]:
        bring = info.data.get("suggested_player_bring")
        if bring and (value[0] not in bring or value[1] not in bring):
            raise ValueError("suggested_player_lead_pair must match the first two entries of suggested_player_bring")
        return value


class Move(BaseModel):
    actor: Pokemon
    mega: bool
    move: str
    targets: list[Pokemon]

class Switch(BaseModel):
    switch_out: Pokemon
    switch_in: Pokemon

class TurnAction(BaseModel):
    action: Move | Switch
    reasoning: str


class TurnSuggestion(BaseModel):
    turn_number: int
    actions: list[TurnAction]
    overall_reasoning: str

    @field_validator("actions")
    @classmethod
    def one_action_per_slot(cls, value: list[TurnAction]) -> list[TurnAction]:
        if len(value) != 2:
            raise ValueError(f"Expected exactly 2 actions (one per active slot), got {len(value)}")
        slots = {action.slot for action in value}
        if slots != {1, 2}:
            raise ValueError("actions must include one entry for slot 1 and one for slot 2")
        return value
