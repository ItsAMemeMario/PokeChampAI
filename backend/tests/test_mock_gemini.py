"""Tests for MockGeminiService (no API key)."""

from __future__ import annotations

import numpy as np
import pytest

from app.schema.common import Pokemon
from app.schema.gamestate import (
    ActivePokemon,
    BenchedPokemon,
    FieldState,
    GameState,
    Hazards,
    SideState,
    StatStages,
)
from app.schema.suggestions import Move, Switch
from app.schema.team import PlayerPokemon, PlayerTeam
from app.services.gemini import create_gemini_service
from app.services.mock_gemini import (
    FILLER_TEXT,
    MOCK_OPPONENT_SPECIES,
    MockGeminiService,
)


def _sample_team() -> PlayerTeam:
    return PlayerTeam(
        pokemon=[
            PlayerPokemon(
                species="Sinistcha",
                item="Sitrus Berry",
                ability="Hospitality",
                evs={"hp": 252},
                nature="Bold",
                moves=["Matcha Gotcha", "Rage Powder", "Trick Room", "Protect"],
            ),
            PlayerPokemon(
                species="Staraptor",
                item="Staraptite",
                ability="Intimidate",
                evs={"atk": 252},
                nature="Jolly",
                moves=["Brave Bird", "Close Combat", "U-turn", "Protect"],
            ),
            PlayerPokemon(
                species="Garchomp",
                item="Lum Berry",
                ability="Rough Skin",
                evs={"atk": 252},
                nature="Jolly",
                moves=["Earthquake", "Outrage", "Rock Slide", "Protect"],
            ),
            PlayerPokemon(
                species="Incineroar",
                item="Wide Lens",
                ability="Intimidate",
                evs={"hp": 252},
                nature="Careful",
                moves=["Fake Out", "Flare Blitz", "Knock Off", "Parting Shot"],
            ),
            PlayerPokemon(
                species="Charizard",
                item="Charizardite Y",
                ability="Blaze",
                evs={"spa": 252},
                nature="Timid",
                moves=["Heat Wave", "Solar Beam", "Air Slash", "Protect"],
            ),
            PlayerPokemon(
                species="Sneasler",
                item="Focus Sash",
                ability="Poison Touch",
                evs={"atk": 252},
                nature="Jolly",
                moves=["Dire Claw", "Close Combat", "Fake Out", "Protect"],
            ),
        ]
    )


def _active(species: str) -> ActivePokemon:
    return ActivePokemon(species=species, hp_percentage=100, stat_stages=StatStages())


def _side(
    s1: ActivePokemon | None,
    s2: ActivePokemon | None,
    *,
    benched: list[BenchedPokemon] | None = None,
) -> SideState:
    return SideState(
        slot_1=s1,
        slot_2=s2,
        benched=benched or [],
        hazards=Hazards(spikes=0, toxic_spikes=0, stealth_rocks=0),
    )


def _game_state() -> GameState:
    return GameState(
        turn_number=2,
        field=FieldState(),
        player=_side(
            _active("Sinistcha"),
            _active("Staraptor"),
            benched=[
                BenchedPokemon(species="Garchomp", hp_percentage=100),
                BenchedPokemon(species="Incineroar", hp_percentage=100),
            ],
        ),
        opponent=_side(_active("Scizor"), _active("Hatterene")),
    )


def test_create_gemini_service_returns_mock_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    service = create_gemini_service()
    assert isinstance(service, MockGeminiService)
    assert service.interaction_id == "mock-interaction"


def test_create_gemini_service_preserves_interaction_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    service = create_gemini_service(interaction_id="int_prior")
    assert isinstance(service, MockGeminiService)
    assert service.interaction_id == "int_prior"


@pytest.mark.asyncio
async def test_mock_identify_opponent_species() -> None:
    service = MockGeminiService()
    species = await service.identify_opponent_species(np.zeros((10, 10, 3), dtype=np.uint8))
    assert species == MOCK_OPPONENT_SPECIES


@pytest.mark.asyncio
async def test_mock_suggest_team_preview_first_four() -> None:
    service = MockGeminiService()
    team = _sample_team()
    opponent = [
        "Blaziken",
        "Scizor",
        "Milotic",
        "Aerodactyl",
        "Grimmsnarl",
        "Whimsicott",
    ]
    suggestion = await service.suggest_team_preview(team, opponent)

    assert suggestion.suggested_player_bring == [
        "Sinistcha",
        "Staraptor",
        "Garchomp",
        "Incineroar",
    ]
    assert suggestion.suggested_player_lead_pair == ("Sinistcha", "Staraptor")
    assert suggestion.predicted_opponent_bring == [
        "Blaziken",
        "Scizor",
        "Milotic",
        "Aerodactyl",
    ]
    assert suggestion.predicted_opponent_lead_pair == ("Blaziken", "Scizor")
    assert suggestion.reasoning == FILLER_TEXT


@pytest.mark.asyncio
async def test_mock_suggest_turn_random_legal_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force one move and one switch across the two slots.
    choices = iter(["move", "switch"])
    monkeypatch.setattr(
        "app.services.mock_gemini.random.choice",
        lambda seq: next(choices) if seq == ["move", "switch"] else seq[0],
    )

    service = MockGeminiService()
    suggestion = await service.suggest_turn(
        _game_state(),
        _sample_team(),
        [],
        turn_number=2,
        opponent_team_species=list(MOCK_OPPONENT_SPECIES),
    )

    assert suggestion.turn_number == 2
    assert suggestion.overall_reasoning == FILLER_TEXT
    assert len(suggestion.actions) == 2

    first = suggestion.actions[0].action
    second = suggestion.actions[1].action
    assert isinstance(first, Move)
    assert first.actor == Pokemon(species="Sinistcha", side="player", slot=1)
    assert first.move in {
        "Matcha Gotcha",
        "Rage Powder",
        "Trick Room",
        "Protect",
    }
    assert first.targets == [Pokemon(species="Scizor", side="opponent", slot=1)]
    assert suggestion.actions[0].reasoning == FILLER_TEXT

    assert isinstance(second, Switch)
    assert second.switch_out == Pokemon(species="Staraptor", side="player", slot=2)
    assert second.switch_in.species in {"Garchomp", "Incineroar"}
    assert second.switch_in.side == "player"
    assert second.switch_in.slot == 2
