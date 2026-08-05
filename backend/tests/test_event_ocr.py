"""Integration tests for event OCR with frame diffing."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from app.cv.event_ocr import EventOcrEngine, _region_changed, _region_has_content
from app.cv.regions import crop_region, default_assets_dir, load_regions
from app.schema.battle_log import TurnStartEvent
from app.schema.team import PlayerPokemon, PlayerTeam
from app.services.cv_runner import _process_battle_animation_events
from app.services.session import SessionStore


def _sample_team() -> PlayerTeam:
    return PlayerTeam(
        pokemon=[
            PlayerPokemon(
                species="Sinistcha",
                item="Sitrus Berry",
                ability="Hospitality",
                evs={"hp": 252},
                nature="Bold",
                moves=["Matcha Gotcha"],
            ),
            PlayerPokemon(
                species="Staraptor",
                item="Staraptite",
                ability="Intimidate",
                evs={"atk": 252},
                nature="Jolly",
                moves=["Brave Bird"],
            ),
            PlayerPokemon(
                species="Garchomp",
                item="Lum Berry",
                ability="Rough Skin",
                evs={"atk": 252},
                nature="Jolly",
                moves=["Earthquake"],
            ),
            PlayerPokemon(
                species="Incineroar",
                item="Wide Lens",
                ability="Intimidate",
                evs={"hp": 252},
                nature="Careful",
                moves=["Fake Out"],
            ),
            PlayerPokemon(
                species="Charizard",
                item="Charizardite Y",
                ability="Blaze",
                evs={"spa": 252},
                nature="Timid",
                moves=["Heat Wave"],
            ),
            PlayerPokemon(
                species="Sneasler",
                item="Focus Sash",
                ability="Poison Touch",
                evs={"atk": 252},
                nature="Jolly",
                moves=["Dire Claw"],
            ),
        ]
    )


def _load_asset(name: str) -> np.ndarray:
    path = default_assets_dir() / name
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


@pytest.fixture
def region_config():
    return load_regions()


def test_region_changed_detects_difference(region_config) -> None:
    from app.cv.event_ocr import _downscale_gray

    ability = crop_region(
        _load_asset("player_slot_1_banner.png"),
        region_config.get("player_slot_1_banner"),
    )
    item = crop_region(
        _load_asset("player_slot_2_banner.png"),
        region_config.get("player_slot_2_banner"),
    )
    assert _region_changed(item, None) is True
    assert _region_changed(item, _downscale_gray(ability)) is True


def test_region_changed_ignores_identical_frame(region_config) -> None:
    crop = crop_region(
        _load_asset("player_slot_1_banner.png"),
        region_config.get("player_slot_1_banner"),
    )
    from app.cv.event_ocr import _downscale_gray

    prev = _downscale_gray(crop)
    assert _region_changed(crop, prev) is False


def test_region_has_content_on_active_slot_banner(region_config) -> None:
    crop = crop_region(
        _load_asset("player_slot_1_banner.png"),
        region_config.get("player_slot_1_banner"),
    )
    assert _region_has_content(crop, "player_slot_1_banner") is True

    empty = crop_region(
        _load_asset("player_slot_1_banner.png"),
        region_config.get("player_slot_2_banner"),
    )
    assert _region_has_content(empty, "player_slot_2_banner") is False


@patch("app.cv.event_ocr._ocr_text")
def test_event_ocr_engine_emits_once_per_region(mock_ocr, region_config) -> None:
    mock_ocr.side_effect = lambda crop, mode="banner": "Staraptor's Intimidate"
    engine = EventOcrEngine()
    image = _load_asset("player_slot_1_banner.png")

    first = engine.process_frame(image, region_config)
    second = engine.process_frame(image, region_config)

    assert len(first) == 1
    assert first[0].type == "ability_triggered"
    assert first[0].actor.slot == 1
    assert second == []


@patch("app.cv.event_ocr._ocr_text")
def test_event_ocr_engine_re_emits_after_region_clears(mock_ocr, region_config) -> None:
    mock_ocr.side_effect = lambda crop, mode="banner": "Staraptor's Intimidate"
    engine = EventOcrEngine()
    ability_image = _load_asset("player_slot_1_banner.png")
    empty_image = np.zeros_like(ability_image)

    first = engine.process_frame(ability_image, region_config)
    engine.process_frame(empty_image, region_config)
    second = engine.process_frame(ability_image, region_config)

    assert len(first) == 1
    assert len(second) == 1


@patch("app.cv.event_ocr._ocr_text")
def test_process_battle_animation_events_appends_to_session(mock_ocr, region_config) -> None:
    mock_ocr.return_value = "Hatterene's Sitrus Berry"

    store = SessionStore()
    store.set_team(_sample_team())
    store.turn_number = 1
    store.append_battle_log(TurnStartEvent(raw_text="Turn 1", turn_number=1))
    engine = EventOcrEngine()
    image = _load_asset("opponent_slot_2_banner.png")

    _process_battle_animation_events(store, image, engine, region_config)

    assert len(store.battle_logs[1]) >= 2
    assert store.battle_logs[1][1].type == "item_used"
    assert store.battle_logs[1][1].pokemon.side == "opponent"
    assert store.battle_logs[1][1].pokemon.slot == 2


def test_event_ocr_on_reference_screenshots(region_config) -> None:
    """End-to-end OCR on calibration assets (requires EasyOCR)."""
    engine = EventOcrEngine()
    expectations = {
        "player_slot_1_banner.png": ("ability_triggered", "player", 1),
        "player_slot_2_banner.png": ("item_used", "player", 2),
        "opponent_slot_1_banner.png": ("ability_triggered", "opponent", 1),
        "opponent_slot_2_banner.png": ("item_used", "opponent", 2),
        "battle_text.png": ("mega_evolution", None, None),
    }

    for asset, (expected_type, expected_side, expected_slot) in expectations.items():
        events = engine.process_frame(_load_asset(asset), region_config)
        print(events)
        engine.reset()
        assert events, f"Expected events from {asset}"
        match = next((event for event in events if event.type == expected_type), None)
        assert match is not None, f"{asset}: got {[event.type for event in events]}"
        if expected_side is not None:
            pokemon = getattr(match, "pokemon", None) or getattr(match, "actor", None)
            assert pokemon is not None
            assert pokemon.side == expected_side
            assert pokemon.slot == expected_slot
