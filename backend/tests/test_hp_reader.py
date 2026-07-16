"""Tests for slot-card HP reader: parse, stability gate, GameState deltas."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from app.cv.hp_reader import (
    HPReader,
    SlotCardRead,
    parse_slot_card_text,
    read_slot_card,
    _region_has_content,
)
from app.cv.regions import crop_region, default_assets_dir, load_regions
from app.schema.gamestate import (
    ActivePokemon,
    FieldState,
    GameState,
    Hazards,
    SideState,
    StatStages,
)
from app.services.cv_runner import (
    _process_hp_action_selection_snapshot,
    _process_hp_animation_frame,
)
from app.services.session import SessionStore


def _active(species: str, hp: int) -> ActivePokemon:
    return ActivePokemon(
        species=species,
        hp_percentage=hp,
        stat_stages=StatStages(),
    )


def _game_state(
    *,
    player_slot_1: ActivePokemon | None = None,
    player_slot_2: ActivePokemon | None = None,
    opponent_slot_1: ActivePokemon | None = None,
    opponent_slot_2: ActivePokemon | None = None,
) -> GameState:
    empty_side = lambda s1, s2: SideState(
        slot_1=s1,
        slot_2=s2,
        benched=[],
        hazards=Hazards(spikes=0, toxic_spikes=0, stealth_rocks=0),
    )
    return GameState(
        turn_number=1,
        field=FieldState(),
        player=empty_side(player_slot_1, player_slot_2),
        opponent=empty_side(opponent_slot_1, opponent_slot_2),
    )


def _load_asset(name: str) -> np.ndarray:
    path = default_assets_dir() / name
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


@pytest.fixture
def region_config():
    return load_regions()


def test_parse_player_numeric_hp() -> None:
    reading = parse_slot_card_text("Staraptor 162 / 162", "player")
    assert reading is not None
    assert reading.species == "Staraptor"
    assert reading.hp_pct == 100
    assert reading.raw_text == "Staraptor 162 / 162"

    damaged = parse_slot_card_text("Sinistcha 82/178", "player")
    assert damaged is not None
    assert damaged.species == "Sinistcha"
    assert damaged.hp_pct == 46  # round(82/178*100)
    assert damaged.raw_text == "Sinistcha 82 / 178"

    # Slash often OCR'd as "7".
    glued = parse_slot_card_text("Staraptor 1627162", "player")
    assert glued == SlotCardRead(
        species="Staraptor",
        hp_pct=100,
        raw_text="Staraptor 162 / 162",
    )

    # Trailing italic "l" often OCR'd as "/".
    trailing_l = parse_slot_card_text("Grimmsnar/ 199/199", "player")
    assert trailing_l == SlotCardRead(
        species="Grimmsnarl",
        hp_pct=100,
        raw_text="Grimmsnarl 199 / 199",
    )


def test_parse_opponent_percent_hp() -> None:
    reading = parse_slot_card_text("Hatterene 47%", "opponent")
    assert reading is not None
    assert reading.species == "Hatterene"
    assert reading.hp_pct == 47
    assert reading.raw_text == "Hatterene 47%"


def test_ocr_action_selection_slot_cards(region_config) -> None:
    """End-to-end EasyOCR on action_selection.png slot cards."""
    image = _load_asset("action_selection.png")
    expected = {
        "player_slot_1_card": SlotCardRead(
            species="Staraptor",
            hp_pct=100,
            raw_text="Staraptor 162 / 162",
        ),
        "player_slot_2_card": SlotCardRead(
            species="Grimmsnarl",
            hp_pct=100,
            raw_text="Grimmsnarl 199 / 199",
        ),
        "opponent_slot_1_card": SlotCardRead(
            species="Milotic",
            hp_pct=100,
            raw_text="Milotic 100%",
        ),
        "opponent_slot_2_card": SlotCardRead(
            species="Scizor",
            hp_pct=100,
            raw_text="Scizor 100%",
        ),
    }

    for region_name, expected_reading in expected.items():
        reading = read_slot_card(image, region_config, region_name)
        assert reading == expected_reading, f"{region_name}: got {reading!r}"


def test_region_has_content_on_action_selection_cards(region_config) -> None:
    image = _load_asset("action_selection.png")
    for name in (
        "player_slot_1_card",
        "player_slot_2_card",
        "opponent_slot_1_card",
        "opponent_slot_2_card",
    ):
        crop = crop_region(image, region_config.get(name))
        assert _region_has_content(crop) is True, name

    empty = np.zeros((94, 210, 3), dtype=np.uint8)
    assert _region_has_content(empty) is False


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_stability_gate_emits_delta_vs_game_state(mock_ocr, region_config) -> None:
    """Start on inter-frame change; commit after 2 stable frames; delta vs GameState."""
    mock_ocr.return_value = "Sinistcha 178/178"
    image = _load_asset("action_selection.png")
    # Mask all but player_slot_1_card so only one tracker advances.
    masked = image.copy()
    for name in (
        "player_slot_2_card",
        "opponent_slot_1_card",
        "opponent_slot_2_card",
    ):
        x, y, w, h = region_config.get(name)
        masked[y : y + h, x : x + w] = 0

    game_state = _game_state(player_slot_1=_active("Sinistcha", 100))
    reader = HPReader()

    # Frame 1: establish baseline at 100%.
    assert reader.process_animation_frame(masked, region_config, game_state) == []

    # Frame 2: HP drops — start gate (stable_frames=1), no emit yet.
    mock_ocr.return_value = "Sinistcha 82/178"
    assert reader.process_animation_frame(masked, region_config, game_state) == []

    # Frame 3: same damaged value — commit (stable_frames=2).
    events = reader.process_animation_frame(masked, region_config, game_state)
    assert len(events) == 1
    event = events[0]
    assert event.type == "hp_change"
    assert event.pokemon.species == "Sinistcha"
    assert event.pokemon.side == "player"
    assert event.pokemon.slot == 1
    # Delta vs GameState (100), not vs previous frame OCR baseline.
    assert event.hp_pct_change == 46 - 100

    # Lingering same value must not re-emit.
    assert reader.process_animation_frame(masked, region_config, game_state) == []


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_stability_gate_resets_when_value_keeps_changing(mock_ocr, region_config) -> None:
    mock_ocr.return_value = "Hatterene 100%"
    image = _load_asset("action_selection.png")
    masked = image.copy()
    for name in (
        "player_slot_1_card",
        "player_slot_2_card",
        "opponent_slot_1_card",
    ):
        x, y, w, h = region_config.get(name)
        masked[y : y + h, x : x + w] = 0

    game_state = _game_state(opponent_slot_2=_active("Hatterene", 100))
    reader = HPReader()

    assert reader.process_animation_frame(masked, region_config, game_state) == []
    mock_ocr.return_value = "Hatterene 80%"
    assert reader.process_animation_frame(masked, region_config, game_state) == []
    # Still animating — candidate resets to 60%, stable_frames=1 (no commit yet).
    mock_ocr.return_value = "Hatterene 60%"
    assert reader.process_animation_frame(masked, region_config, game_state) == []
    # Second consecutive 60% frame commits (stable_frames >= 2).
    events = reader.process_animation_frame(masked, region_config, game_state)
    assert len(events) == 1
    assert events[0].hp_pct_change == 60 - 100


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_empty_region_resets_tracker(mock_ocr, region_config) -> None:
    mock_ocr.return_value = "Sinistcha 178/178"
    image = _load_asset("action_selection.png")
    masked = image.copy()
    for name in (
        "player_slot_2_card",
        "opponent_slot_1_card",
        "opponent_slot_2_card",
    ):
        x, y, w, h = region_config.get(name)
        masked[y : y + h, x : x + w] = 0

    game_state = _game_state(player_slot_1=_active("Sinistcha", 100))
    reader = HPReader()
    reader.process_animation_frame(masked, region_config, game_state)
    mock_ocr.return_value = "Sinistcha 82/178"
    reader.process_animation_frame(masked, region_config, game_state)

    empty = np.zeros_like(masked)
    assert reader.process_animation_frame(empty, region_config, game_state) == []
    # Tracker reset — need a fresh baseline before another start gate.
    mock_ocr.return_value = "Sinistcha 82/178"
    assert reader.process_animation_frame(masked, region_config, game_state) == []
    assert reader.process_animation_frame(masked, region_config, game_state) == []


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_action_selection_snapshot_emits_drift(mock_ocr, region_config) -> None:
    mock_ocr.return_value = "Milotic 85%"
    image = _load_asset("action_selection.png")
    masked = image.copy()
    for name in (
        "player_slot_1_card",
        "player_slot_2_card",
        "opponent_slot_2_card",
    ):
        x, y, w, h = region_config.get(name)
        masked[y : y + h, x : x + w] = 0

    game_state = _game_state(opponent_slot_1=_active("Milotic", 100))
    reader = HPReader()
    events = reader.read_action_selection_snapshot(masked, region_config, game_state)

    assert len(events) == 1
    assert events[0].pokemon.species == "Milotic"
    assert events[0].pokemon.side == "opponent"
    assert events[0].pokemon.slot == 1
    assert events[0].hp_pct_change == 85 - 100


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_cv_runner_appends_hp_events(mock_ocr, region_config) -> None:
    mock_ocr.return_value = "Sinistcha 178/178"
    image = _load_asset("action_selection.png")
    masked = image.copy()
    for name in (
        "player_slot_2_card",
        "opponent_slot_1_card",
        "opponent_slot_2_card",
    ):
        x, y, w, h = region_config.get(name)
        masked[y : y + h, x : x + w] = 0

    store = SessionStore()
    store.game_state = _game_state(player_slot_1=_active("Sinistcha", 100))
    reader = HPReader()

    _process_hp_animation_frame(store, masked, reader, region_config)
    mock_ocr.return_value = "Sinistcha 82/178"
    _process_hp_animation_frame(store, masked, reader, region_config)
    _process_hp_animation_frame(store, masked, reader, region_config)

    assert len(store.battle_logs) == 1
    assert store.battle_logs[0].type == "hp_change"
    assert store.battle_logs[0].hp_pct_change == 46 - 100


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_cv_runner_snapshot_helper(mock_ocr, region_config) -> None:
    mock_ocr.return_value = "Staraptor 81/162"
    image = _load_asset("action_selection.png")
    # Zero other cards so a single mock return is enough.
    masked = image.copy()
    for name in (
        "player_slot_2_card",
        "opponent_slot_1_card",
        "opponent_slot_2_card",
    ):
        x, y, w, h = region_config.get(name)
        masked[y : y + h, x : x + w] = 0

    store = SessionStore()
    store.game_state = _game_state(player_slot_1=_active("Staraptor", 100))
    reader = HPReader()
    _process_hp_action_selection_snapshot(store, masked, reader, region_config)

    assert len(store.battle_logs) == 1
    assert store.battle_logs[0].hp_pct_change == 50 - 100  # round(81/162*100)=50
