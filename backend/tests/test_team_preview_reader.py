"""Tests for opponent team preview cropping and vision pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

from app.cv.regions import default_assets_dir, load_regions
from app.cv.team_preview_reader import (
    crop_opponent_sprite_slots,
    crop_opponent_team_preview,
    read_opponent_team_preview,
    stack_sprite_slots,
)
from app.schema.suggestions import TeamPreviewSuggestion
from app.schema.team import OpponentTeamPreview, PlayerPokemon, PlayerTeam


def _load_asset(name: str) -> np.ndarray:
    path = default_assets_dir() / name
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


@pytest.fixture
def region_config():
    return load_regions()


def test_crop_opponent_team_preview_dimensions(region_config) -> None:
    image = _load_asset("team_preview.png")
    crop = crop_opponent_team_preview(image, region_config)
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0
    assert crop.ndim == 3


def test_crop_opponent_sprite_slots_count_and_shape(region_config) -> None:
    image = _load_asset("team_preview.png")
    slots = crop_opponent_sprite_slots(image, region_config)
    assert len(slots) == 6
    for slot in slots:
        assert slot.ndim == 3
        assert slot.shape[0] > 0
        assert slot.shape[1] > 0


def test_stack_sprite_slots_combines_vertical(region_config) -> None:
    image = _load_asset("team_preview.png")
    slots = crop_opponent_sprite_slots(image, region_config)
    stacked = stack_sprite_slots(slots)
    total_height = sum(slot.shape[0] for slot in slots)
    assert stacked.shape[0] == total_height
    assert stacked.shape[2] == 3


@pytest.mark.asyncio
async def test_read_opponent_team_preview_calls_gemini(region_config) -> None:
    image = _load_asset("team_preview.png")
    expected_species = [
        "Blaziken",
        "Scizor",
        "Sinistcha",
        "Milotic",
        "Aerodactyl",
        "Grimmsnarl",
    ]
    mock_gemini = MagicMock()
    mock_gemini.identify_opponent_species = AsyncMock(return_value=expected_species)

    result = await read_opponent_team_preview(image, region_config, gemini=mock_gemini)

    assert isinstance(result, OpponentTeamPreview)
    assert result.species == expected_species
    mock_gemini.identify_opponent_species.assert_awaited_once()
    vision_input = mock_gemini.identify_opponent_species.await_args.args[0]
    assert isinstance(vision_input, np.ndarray)
    assert vision_input.ndim == 3


@pytest.mark.asyncio
async def test_cv_team_preview_pipeline_stores_suggestion() -> None:
    from app.services.cv_runner import _process_team_preview_entry
    from app.services.session import SessionStore

    image = _load_asset("team_preview.png")
    store = SessionStore()
    store.player_team = PlayerTeam(
        pokemon=[
            PlayerPokemon(
                species=f"Mon{i}",
                item="Sitrus Berry",
                ability="Ability",
                evs={"hp": 0},
                nature="Timid",
                moves=["Move1", "Move2", "Move3", "Move4"],
            )
            for i in range(6)
        ]
    )

    mock_gemini = MagicMock()
    mock_gemini.interaction_id = "int_preview"
    opponent_species = [
        "Blaziken",
        "Scizor",
        "Sinistcha",
        "Milotic",
        "Aerodactyl",
        "Grimmsnarl",
    ]
    mock_gemini.identify_opponent_species = AsyncMock(return_value=opponent_species)
    mock_gemini.suggest_team_preview = AsyncMock(
        return_value=TeamPreviewSuggestion(
            predicted_opponent_bring=["Blaziken", "Scizor", "Sinistcha", "Milotic"],
            predicted_opponent_lead_pair=("Blaziken", "Scizor"),
            suggested_player_bring=["Mon0", "Mon1", "Mon2", "Mon3"],
            suggested_player_lead_pair=("Mon0", "Mon1"),
            reasoning="Test reasoning",
        )
    )

    import app.services.cv_runner as cv_runner_module

    original = cv_runner_module.GeminiService
    mock_service_cls = MagicMock(return_value=mock_gemini)
    cv_runner_module.GeminiService = mock_service_cls
    try:
        await _process_team_preview_entry(store, image)
    finally:
        cv_runner_module.GeminiService = original

    assert store.opponent_team_species == opponent_species
    assert store.team_preview_suggestion is not None
    assert store.team_preview_suggestion.suggested_player_lead_pair == ("Mon0", "Mon1")
    assert store.gemini_interaction_id == "int_preview"
    mock_gemini.suggest_team_preview.assert_awaited_once()
    mock_service_cls.assert_called_with(interaction_id=None)
