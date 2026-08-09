"""Tests for slot-card HP reader: parse, stability gate, GameState deltas."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from app.cv.hp_reader import (
    HPReader,
    SlotCardRead,
    _HP_BAR_IN_CARD_DEFAULT,
    _crop_hp_bar,
    _downscale_bar_gray,
    _hp_bar_changed,
    _hp_bar_in_card,
    parse_slot_card_text,
    read_slot_card,
    _slot_card_visible,
)
from app.cv.regions import RegionConfig, crop_region, default_assets_dir, load_regions
from app.schema.gamestate import (
    ActivePokemon,
    FieldState,
    GameState,
    Hazards,
    SideState,
    StatStages,
)
from app.services.cv_runner import (
    _emit_turn_start_on_action_selection_entry,
    _process_hp_action_selection_snapshot,
    _process_hp_animation_frame,
)
from app.schema.battle_log import TurnStartEvent
from app.services.session import SessionStore

try:
    import torch

    _CUDA_AVAILABLE = bool(torch.cuda.is_available())
except Exception:  # pragma: no cover
    _CUDA_AVAILABLE = False


def _open_turn(store: SessionStore, turn: int = 1) -> None:
    store.turn_number = turn
    store.append_battle_log(
        TurnStartEvent(raw_text=f"Turn {turn}", turn_number=turn)
    )


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


def _with_bar_motion(
    image: np.ndarray,
    region_config,
    card_name: str,
    salt: int,
) -> np.ndarray:
    """Perturb in-card HP-bar pixels so the motion gate fires without killing visibility."""
    out = image.copy()
    cx, cy, _cw, _ch = region_config.get(card_name)
    bx, by, bw, bh = _hp_bar_in_card(region_config)
    x, y = cx + bx, cy + by
    bar = out[y : y + bh, x : x + bw].astype(np.int16)
    # Alternate brighter/darker so consecutive salts differ after 4× downscale.
    delta = 35 if salt % 2 else -35
    out[y : y + bh, x : x + bw] = np.clip(bar + delta, 0, 255).astype(np.uint8)
    return out


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


def test_parse_slot_card_snaps_species_to_known() -> None:
    typo = parse_slot_card_text(
        "Garchmp 100 / 100",
        "player",
        player_species=["Garchomp", "Sinistcha", "Staraptor", "Incineroar"],
    )
    assert typo is not None
    assert typo.species == "Garchomp"

    form = parse_slot_card_text(
        "Arcanine 80 / 100",
        "player",
        player_species=["Arcanine-Hisui", "Sinistcha", "Staraptor", "Garchomp"],
    )
    assert form is not None
    assert form.species == "Arcanine-Hisui"

    opponent = parse_slot_card_text(
        "Arcanine 47%",
        "opponent",
        player_species=["Arcanine-Hisui", "Sinistcha", "Staraptor", "Garchomp"],
        opponent_species=["Arcanine", "Scizor", "Hatterene", "Milotic", "Blaziken", "Amoonguss"],
    )
    assert opponent is not None
    assert opponent.species == "Arcanine"


def test_parse_opponent_percent_hp() -> None:
    reading = parse_slot_card_text("Hatterene 47%", "opponent")
    assert reading is not None
    assert reading.species == "Hatterene"
    assert reading.hp_pct == 47
    assert reading.raw_text == "Hatterene 47%"


def _mask_other_cards(image: np.ndarray, region_config, keep: str) -> np.ndarray:
    masked = image.copy()
    for name in (
        "player_slot_1_card",
        "player_slot_2_card",
        "opponent_slot_1_card",
        "opponent_slot_2_card",
    ):
        if name == keep:
            continue
        x, y, w, h = region_config.get(name)
        masked[y : y + h, x : x + w] = 0
    return masked


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required for EasyOCR")
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


def test_slot_card_visible_on_action_selection_cards(region_config) -> None:
    image = _load_asset("action_selection.png")
    for name in (
        "player_slot_1_card",  # green-highlight border during action selection
        "player_slot_2_card",
        "opponent_slot_1_card",
        "opponent_slot_2_card",
    ):
        crop = crop_region(image, region_config.get(name))
        assert _slot_card_visible(crop) is True, name

    empty = np.zeros((94, 210, 3), dtype=np.uint8)
    assert _slot_card_visible(empty) is False


def test_hp_bar_in_card_uses_calibrated_offset(region_config) -> None:
    assert _hp_bar_in_card(region_config) == (17, 48, 195, 26)
    empty_config = RegionConfig(resolution=(1600, 900), regions={})
    assert _hp_bar_in_card(empty_config) == _HP_BAR_IN_CARD_DEFAULT


def test_hp_bar_changed_detects_motion(region_config) -> None:
    image = _load_asset("action_selection.png")
    card = crop_region(image, region_config.get("player_slot_1_card"))
    bar = _crop_hp_bar(card, _hp_bar_in_card(region_config))
    prev = _downscale_bar_gray(bar)

    assert _hp_bar_changed(bar, None) is True
    assert _hp_bar_changed(bar, prev) is False

    moved = bar.copy()
    moved[:] = np.clip(moved.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    assert _hp_bar_changed(moved, prev) is True


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_animation_skips_ocr_when_template_misses(mock_ocr, region_config) -> None:
    """Invisible / empty card crop must not OCR."""
    mock_ocr.return_value = "Sinistcha 178/178"
    image = _load_asset("action_selection.png")
    empty = np.zeros_like(image)
    reader = HPReader()
    assert reader.process_animation_frame(empty, region_config, None) == []
    mock_ocr.assert_not_called()


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_animation_skips_ocr_when_bar_idle(mock_ocr, region_config) -> None:
    """Unchanged HP bar must not OCR again while not tracking."""
    mock_ocr.return_value = "Sinistcha 178/178"
    image = _load_asset("action_selection.png")
    masked = _mask_other_cards(image, region_config, "player_slot_1_card")
    reader = HPReader()
    card = "player_slot_1_card"

    reader.process_animation_frame(
        _with_bar_motion(masked, region_config, card, 1),
        region_config,
        None,
    )
    assert mock_ocr.call_count == 1

    # Settling from nudged → clean bar is one more motion sample.
    reader.process_animation_frame(masked, region_config, None)
    assert mock_ocr.call_count == 2

    # Identical frames afterward: no motion, tracking closed → skip OCR.
    reader.process_animation_frame(masked, region_config, None)
    reader.process_animation_frame(masked, region_config, None)
    assert mock_ocr.call_count == 2


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_animation_ocrs_when_visible_and_bar_moves(mock_ocr, region_config) -> None:
    mock_ocr.return_value = "Sinistcha 178/178"
    image = _load_asset("action_selection.png")
    masked = _mask_other_cards(image, region_config, "player_slot_1_card")
    reader = HPReader()
    card = "player_slot_1_card"

    reader.process_animation_frame(
        _with_bar_motion(masked, region_config, card, 1),
        region_config,
        None,
    )
    reader.process_animation_frame(
        _with_bar_motion(masked, region_config, card, 2),
        region_config,
        None,
    )
    assert mock_ocr.call_count == 2


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_action_selection_snapshot_ocrs_without_bar_motion(mock_ocr, region_config) -> None:
    """Snapshot is template-gated only (no motion requirement)."""
    mock_ocr.return_value = "Staraptor 162/162"
    image = _load_asset("action_selection.png")
    masked = _mask_other_cards(image, region_config, "player_slot_1_card")
    reader = HPReader()
    reader.read_action_selection_snapshot(masked, region_config, None)
    assert mock_ocr.call_count == 1


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_stability_gate_emits_delta_vs_game_state(mock_ocr, region_config) -> None:
    """Start on inter-frame change; commit after 2 stable frames; delta vs GameState."""
    mock_ocr.return_value = "Sinistcha 178/178"
    image = _load_asset("action_selection.png")
    masked = _mask_other_cards(image, region_config, "player_slot_1_card")

    game_state = _game_state(player_slot_1=_active("Sinistcha", 100))
    reader = HPReader()
    card = "player_slot_1_card"

    # Frame 1: establish baseline at 100% (first bar sample counts as motion).
    assert (
        reader.process_animation_frame(
            _with_bar_motion(masked, region_config, card, 1),
            region_config,
            game_state,
        )
        == []
    )

    # Frame 2: HP drops — start gate (stable_frames=1), no emit yet.
    mock_ocr.return_value = "Sinistcha 82/178"
    assert (
        reader.process_animation_frame(
            _with_bar_motion(masked, region_config, card, 2),
            region_config,
            game_state,
        )
        == []
    )

    # Frame 3: same damaged value — commit (stable_frames=2); tracking keeps OCR alive.
    events = reader.process_animation_frame(masked, region_config, game_state)
    assert len(events) == 1
    event = events[0]
    assert event.type == "hp_change"
    assert event.pokemon.species == "Sinistcha"
    assert event.pokemon.side == "player"
    assert event.pokemon.slot == 1
    # Delta vs GameState (100), not vs previous frame OCR baseline.
    assert event.hp_pct_change == 46 - 100

    # Lingering same value must not re-emit (no bar motion + tracking closed).
    assert reader.process_animation_frame(masked, region_config, game_state) == []


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_stability_gate_resets_when_value_keeps_changing(mock_ocr, region_config) -> None:
    mock_ocr.return_value = "Hatterene 100%"
    image = _load_asset("action_selection.png")
    masked = _mask_other_cards(image, region_config, "opponent_slot_2_card")

    game_state = _game_state(opponent_slot_2=_active("Hatterene", 100))
    reader = HPReader()
    card = "opponent_slot_2_card"

    assert (
        reader.process_animation_frame(
            _with_bar_motion(masked, region_config, card, 1),
            region_config,
            game_state,
        )
        == []
    )
    mock_ocr.return_value = "Hatterene 80%"
    assert (
        reader.process_animation_frame(
            _with_bar_motion(masked, region_config, card, 2),
            region_config,
            game_state,
        )
        == []
    )
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
    masked = _mask_other_cards(image, region_config, "player_slot_1_card")

    game_state = _game_state(player_slot_1=_active("Sinistcha", 100))
    reader = HPReader()
    card = "player_slot_1_card"
    reader.process_animation_frame(
        _with_bar_motion(masked, region_config, card, 1),
        region_config,
        game_state,
    )
    mock_ocr.return_value = "Sinistcha 82/178"
    reader.process_animation_frame(
        _with_bar_motion(masked, region_config, card, 2),
        region_config,
        game_state,
    )

    empty = np.zeros_like(masked)
    assert reader.process_animation_frame(empty, region_config, game_state) == []
    # Tracker reset — need a fresh baseline before another start gate.
    mock_ocr.return_value = "Sinistcha 82/178"
    assert (
        reader.process_animation_frame(
            _with_bar_motion(masked, region_config, card, 3),
            region_config,
            game_state,
        )
        == []
    )
    assert reader.process_animation_frame(masked, region_config, game_state) == []


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_action_selection_snapshot_emits_drift(mock_ocr, region_config) -> None:
    mock_ocr.return_value = "Milotic 85%"
    image = _load_asset("action_selection.png")
    masked = _mask_other_cards(image, region_config, "opponent_slot_1_card")

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
    masked = _mask_other_cards(image, region_config, "player_slot_1_card")

    store = SessionStore()
    store.game_state = _game_state(player_slot_1=_active("Sinistcha", 100))
    _open_turn(store)
    reader = HPReader()
    card = "player_slot_1_card"

    _process_hp_animation_frame(
        store,
        _with_bar_motion(masked, region_config, card, 1),
        reader,
        region_config,
    )
    mock_ocr.return_value = "Sinistcha 82/178"
    _process_hp_animation_frame(
        store,
        _with_bar_motion(masked, region_config, card, 2),
        reader,
        region_config,
    )
    _process_hp_animation_frame(store, masked, reader, region_config)

    assert len(store.battle_logs[1]) == 2  # turn_start + hp_change
    assert store.battle_logs[1][1].type == "hp_change"
    assert store.battle_logs[1][1].hp_pct_change == 46 - 100


@patch("app.cv.hp_reader._ocr_slot_card_text")
def test_cv_runner_snapshot_helper(mock_ocr, region_config) -> None:
    mock_ocr.return_value = "Staraptor 81/162"
    image = _load_asset("action_selection.png")
    masked = _mask_other_cards(image, region_config, "player_slot_1_card")

    store = SessionStore()
    store.game_state = _game_state(player_slot_1=_active("Staraptor", 100))
    _open_turn(store)
    reader = HPReader()
    _process_hp_action_selection_snapshot(store, masked, reader, region_config)

    assert len(store.battle_logs[1]) == 2
    assert store.battle_logs[1][1].hp_pct_change == 50 - 100  # round(81/162*100)=50


def test_emit_turn_start_on_action_selection_entry() -> None:
    store = SessionStore()
    store.game_state = _game_state(player_slot_1=_active("Sinistcha", 100))
    assert store.turn_number == 0

    _emit_turn_start_on_action_selection_entry(store)
    assert store.turn_number == 1
    assert store.game_state.turn_number == 1
    assert store.battle_logs[0] == []
    assert len(store.battle_logs[1]) == 1
    assert store.battle_logs[1][0].type == "turn_start"
    assert store.battle_logs[1][0].turn_number == 1

    _emit_turn_start_on_action_selection_entry(store)
    assert store.turn_number == 2
    assert len(store.battle_logs) == 3
    assert store.battle_logs[2][0].type == "turn_start"
    assert store.battle_logs[2][0].turn_number == 2
