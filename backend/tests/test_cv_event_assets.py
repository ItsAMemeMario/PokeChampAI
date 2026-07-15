"""End-to-end event OCR tests for backend/tests/assets/cv/events screenshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.cv.event_ocr import EventOcrEngine
from app.cv.phase_detector import has_battle_ended
from app.cv.regions import config_for_image, load_regions
from app.schema.battle_log import BattleLogEvent


def _events_assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets" / "cv" / "events"


def _load_event_asset(name: str) -> np.ndarray:
    path = _events_assets_dir() / name
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _events_of_type(events: list[BattleLogEvent], event_type: str) -> list[BattleLogEvent]:
    return [event for event in events if event.type == event_type]


@pytest.fixture
def region_config():
    return load_regions()


@pytest.fixture
def event_engine() -> EventOcrEngine:
    return EventOcrEngine()


def test_charizard_mega(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(_load_event_asset("charizard_mega.png"), region_config)
    megas = _events_of_type(events, "mega_evolution")
    assert len(megas) == 1
    assert megas[0].pokemon.species == "Charizard"
    assert megas[0].pokemon.side == "player"


def test_forfeit_detected_as_battle_end(event_engine: EventOcrEngine, region_config) -> None:
    image = _load_event_asset("forfeit.png")
    display_config = config_for_image(region_config, image)
    assert has_battle_ended(image, display_config) is True
    # Forfeit is a phase signal, not a typed battle-log event.
    assert event_engine.process_frame(image, region_config) == []


def test_grimmsnarl_faints(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(_load_event_asset("grimmsnarl_faints.png"), region_config)
    faints = _events_of_type(events, "faint")
    assert len(faints) == 1
    assert faints[0].pokemon.species == "Grimmsnarl"
    assert faints[0].pokemon.side == "player"


def test_grimmsnarl_uses_swagger(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("grimmsnarl_uses_swagger.png"), region_config
    )
    moves = _events_of_type(events, "move_used")
    assert len(moves) == 1
    assert moves[0].actor.species == "Grimmsnarl"
    assert moves[0].actor.side == "player"
    assert moves[0].move == "Swagger"


def test_opponent_sends_out_charizard(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("opponent_sends_out_charizard.png"), region_config
    )
    switch_ins = _events_of_type(events, "switch_in")
    assert len(switch_ins) == 1
    assert switch_ins[0].pokemon.species == "Charizard"
    assert switch_ins[0].pokemon.side == "opponent"


def test_opponent_sends_out_staraptor_and_whimsicott(
    event_engine: EventOcrEngine, region_config
) -> None:
    events = event_engine.process_frame(
        _load_event_asset("opponent_sends_out_staraptor_and_whimsicott.png"),
        region_config,
    )
    switch_ins = _events_of_type(events, "switch_in")
    assert len(switch_ins) == 2
    assert switch_ins[0].pokemon.species == "Staraptor"
    assert switch_ins[0].pokemon.side == "opponent"
    assert switch_ins[0].pokemon.slot == 1
    assert switch_ins[1].pokemon.species == "Whimsicott"
    assert switch_ins[1].pokemon.side == "opponent"
    assert switch_ins[1].pokemon.slot == 2


def test_opposing_tailwind_blows(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("opposing_tailwind_blows.png"), region_config
    )
    sides = _events_of_type(events, "side_condition")
    assert len(sides) == 1
    assert sides[0].condition == "tailwind"
    assert sides[0].side == "opponent"


def test_player_sends_out_charizard(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("player_sends_out_charizard.png"), region_config
    )
    switch_ins = _events_of_type(events, "switch_in")
    assert len(switch_ins) == 1
    assert switch_ins[0].pokemon.species == "Charizard"
    assert switch_ins[0].pokemon.side == "player"


def test_player_sends_out_sneasler_and_grimmsnarl(
    event_engine: EventOcrEngine, region_config
) -> None:
    events = event_engine.process_frame(
        _load_event_asset("player_sends_out_sneasler_and_grimmsnarl.png"),
        region_config,
    )
    switch_ins = _events_of_type(events, "switch_in")
    assert len(switch_ins) == 2
    assert switch_ins[0].pokemon.species == "Sneasler"
    assert switch_ins[0].pokemon.side == "player"
    assert switch_ins[0].pokemon.slot == 1
    assert switch_ins[1].pokemon.species == "Grimmsnarl"
    assert switch_ins[1].pokemon.side == "player"
    assert switch_ins[1].pokemon.slot == 2


def test_player_withdraws_sneasler(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("player_withdraws_sneasler.png"), region_config
    )
    switch_outs = _events_of_type(events, "switch_out")
    assert len(switch_outs) == 1
    assert switch_outs[0].pokemon.species == "Sneasler"
    assert switch_outs[0].pokemon.side == "player"


def test_staraptor_attack_rises(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("staraptor_attack_rises.png"), region_config
    )
    stats = _events_of_type(events, "stat_change")
    assert len(stats) == 1
    assert stats[0].pokemon.species == "Staraptor"
    assert stats[0].pokemon.side == "opponent"
    assert stats[0].stat == "atk"
    assert stats[0].stages_delta == 2


def test_staraptor_confused(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("staraptor_confused.png"), region_config
    )
    volatiles = _events_of_type(events, "volatile_applied")
    assert len(volatiles) == 1
    assert volatiles[0].pokemon.species == "Staraptor"
    assert volatiles[0].pokemon.side == "opponent"
    assert volatiles[0].volatile == "confused"


def test_staraptor_faints(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("staraptor_faints.png"), region_config
    )
    faints = _events_of_type(events, "faint")
    assert len(faints) == 1
    assert faints[0].pokemon.species == "Staraptor"
    assert faints[0].pokemon.side == "opponent"


def test_staraptor_intimidate_and_stat_drop(
    event_engine: EventOcrEngine, region_config
) -> None:
    events = event_engine.process_frame(
        _load_event_asset("staraptor_intimidate+player_pokemon_stat_drop.png"),
        region_config,
    )

    abilities = _events_of_type(events, "ability_triggered")
    assert len(abilities) == 1
    assert abilities[0].actor.species == "Staraptor"
    assert abilities[0].actor.side == "opponent"
    assert abilities[0].actor.slot == 1
    assert abilities[0].ability == "Intimidate"

    stats = _events_of_type(events, "stat_change")
    assert len(stats) == 2
    by_species = {event.pokemon.species: event for event in stats}
    assert set(by_species) == {"Sneasler", "Grimmsnarl"}
    assert by_species["Sneasler"].pokemon.side == "player"
    assert by_species["Sneasler"].pokemon.slot == 1
    assert by_species["Sneasler"].stat == "atk"
    assert by_species["Sneasler"].stages_delta == -1
    assert by_species["Grimmsnarl"].pokemon.side == "player"
    assert by_species["Grimmsnarl"].pokemon.slot == 2
    assert by_species["Grimmsnarl"].stat == "atk"
    assert by_species["Grimmsnarl"].stages_delta == -1


def test_staraptor_uses_protect(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("staraptor_uses_protect.png"), region_config
    )
    moves = _events_of_type(events, "move_used")
    assert len(moves) == 1
    assert moves[0].actor.species == "Staraptor"
    assert moves[0].actor.side == "opponent"
    assert moves[0].move == "Protect"


def test_staraptor_uses_steel_wing(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("staraptor_uses_steel_wing.png"), region_config
    )
    moves = _events_of_type(events, "move_used")
    assert len(moves) == 1
    assert moves[0].actor.species == "Staraptor"
    assert moves[0].actor.side == "opponent"
    assert moves[0].move == "Steel Wing"


def test_sunny_weather_starts(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("sunny_weather_starts.png"), region_config
    )

    abilities = _events_of_type(events, "ability_triggered")
    assert len(abilities) == 1
    assert abilities[0].actor.species == "Charizard"
    assert abilities[0].actor.side == "player"
    assert abilities[0].actor.slot == 1
    assert abilities[0].ability == "Drought"

    weather = _events_of_type(events, "weather_change")
    assert len(weather) == 1
    assert weather[0].weather == "sunny"


def test_whimsicott_item_focus_sash(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("whimsicott_item_focus_sash.png"), region_config
    )
    items = _events_of_type(events, "item_used")
    assert len(items) == 1
    assert items[0].pokemon.species == "Whimsicott"
    assert items[0].pokemon.side == "opponent"
    assert items[0].pokemon.slot == 2
    assert items[0].item == "Focus Sash"


def test_whimsicott_uses_dazzling_gleam(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("whimsicott_uses_dazzling_gleam.png"), region_config
    )
    moves = _events_of_type(events, "move_used")
    assert len(moves) == 1
    assert moves[0].actor.species == "Whimsicott"
    assert moves[0].actor.side == "opponent"
    assert moves[0].move == "Dazzling Gleam"


def test_whimsicott_uses_tailwind(event_engine: EventOcrEngine, region_config) -> None:
    events = event_engine.process_frame(
        _load_event_asset("whimsicott_uses_tailwind.png"), region_config
    )
    moves = _events_of_type(events, "move_used")
    assert len(moves) == 1
    assert moves[0].actor.species == "Whimsicott"
    assert moves[0].actor.side == "opponent"
    assert moves[0].move == "Tailwind"


def test_all_event_assets_are_covered() -> None:
    """Guardrail: every PNG under assets/cv/events has a dedicated assertion test."""
    covered = {
        "charizard_mega.png",
        "forfeit.png",
        "grimmsnarl_faints.png",
        "grimmsnarl_uses_swagger.png",
        "opponent_sends_out_charizard.png",
        "opponent_sends_out_staraptor_and_whimsicott.png",
        "opposing_tailwind_blows.png",
        "player_sends_out_charizard.png",
        "player_sends_out_sneasler_and_grimmsnarl.png",
        "player_withdraws_sneasler.png",
        "staraptor_attack_rises.png",
        "staraptor_confused.png",
        "staraptor_faints.png",
        "staraptor_intimidate+player_pokemon_stat_drop.png",
        "staraptor_uses_protect.png",
        "staraptor_uses_steel_wing.png",
        "sunny_weather_starts.png",
        "whimsicott_item_focus_sash.png",
        "whimsicott_uses_dazzling_gleam.png",
        "whimsicott_uses_tailwind.png",
    }
    on_disk = {path.name for path in _events_assets_dir().glob("*.png")}
    assert on_disk == covered
