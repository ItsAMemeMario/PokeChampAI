"""Unit tests for battle event text parsing."""

from __future__ import annotations

import pytest

from app.cv.event_parser import (
    is_known_item,
    normalize_ocr_text,
    parse_battle_text,
    parse_side_banner,
)


def test_normalize_ocr_text_fixes_apostrophe_noise() -> None:
    assert normalize_ocr_text("Sinistcha'$ Sitrus Berry") == "Sinistcha's Sitrus Berry"
    assert normalize_ocr_text("Sinistcha' $ Sitrus Berry") == "Sinistcha's Sitrus Berry"
    assert normalize_ocr_text("Whimsicott' s Focus Sash") == "Whimsicott's Focus Sash"
    assert normalize_ocr_text("Gol Charizardl") == "Go! Charizard!"
    assert normalize_ocr_text("The qpposing Staraptor faintedl") == (
        "The opposing Staraptor fainted!"
    )
    assert normalize_ocr_text("Whimsicott' s Focus Sash") == "Whimsicott's Focus Sash"
    assert normalize_ocr_text("Whimsicott $ Focus Sash") == "Whimsicott's Focus Sash"
    assert normalize_ocr_text("Gol Charizardl") == "Go! Charizard!"
    assert normalize_ocr_text("Bluesent out Garchomp!") == "Blue sent out Garchomp!"
    assert normalize_ocr_text("Go! Foo ard Bar!") == "Go! Foo and Bar!"
    assert normalize_ocr_text("The 0pposing Garchomp") == "The opposing Garchomp"


def test_parse_side_banner_ability() -> None:
    event = parse_side_banner(
        "Staraptor's Intimidate =",
        "player",
        slot=1,
    )
    assert event is not None
    assert event.type == "ability_triggered"
    assert event.actor.species == "Staraptor"
    assert event.actor.slot == 1
    assert event.ability == "Intimidate"


def test_parse_side_banner_item_from_known_item_name() -> None:
    event = parse_side_banner(
        "Sinistcha's Sitrus Berry",
        "player",
        slot=2,
    )
    assert event is not None
    assert event.type == "item_used"
    assert event.pokemon.species == "Sinistcha"
    assert event.pokemon.slot == 2
    assert event.item == "Sitrus Berry"


def test_parse_side_banner_opponent_ability() -> None:
    event = parse_side_banner("Aerodactyl's Unnerve", "opponent", slot=1)
    assert event is not None
    assert event.type == "ability_triggered"
    assert event.actor.species == "Aerodactyl"
    assert event.actor.slot == 1
    assert event.ability == "Unnerve"


def test_parse_side_banner_opponent_item_via_common_list() -> None:
    event = parse_side_banner("Hatterene's Sitrus Berry", "opponent", slot=2)
    assert event is not None
    assert event.type == "item_used"
    assert event.pokemon.species == "Hatterene"
    assert event.pokemon.slot == 2
    assert event.item == "Sitrus Berry"


def test_is_known_item_checks_regulation_mb_list() -> None:
    assert is_known_item("Sitrus Berry") is True
    assert is_known_item("Leftovers") is True
    assert is_known_item("Intimidate") is False
    assert is_known_item("Assault Vest") is False
    assert is_known_item("Rocky Helmet") is False
    assert is_known_item("Staraptite") is True


def test_parse_battle_text_mega_evolution() -> None:
    events = parse_battle_text(
        "The opposing Scizor's Scizorite is reacting to the Trainer's Omni Ring!"
    )
    assert any(event.type == "mega_evolution" for event in events)
    mega = next(event for event in events if event.type == "mega_evolution")
    assert mega.pokemon.species == "Scizor"
    assert mega.pokemon.side == "opponent"
    assert mega.variant == "regular"


def test_parse_battle_text_mega_evolution_xy_form() -> None:
    events = parse_battle_text(
        "Charizard's Charizardite Y is reacting to Trainer's Omni Ring!"
    )
    assert len(events) == 1
    assert events[0].type == "mega_evolution"
    assert events[0].pokemon.species == "Charizard"
    assert events[0].pokemon.side == "player"
    assert events[0].variant == "Y"


def test_parse_battle_text_stat_change_one_player_multi_stat() -> None:
    events = parse_battle_text("Blastoise's Attack, Sp. Atk, and Speed rose sharply!")
    assert len(events) == 3
    assert all(event.type == "stat_change" for event in events)
    assert all(event.pokemon.species == "Blastoise" for event in events)
    assert all(event.pokemon.side == "player" for event in events)
    assert {event.stat for event in events} == {"atk", "spa", "spe"}
    assert all(event.stages_delta == 2 for event in events)


def test_parse_battle_text_stat_change_two_player() -> None:
    events = parse_battle_text("Manectric and Mamoswine's Attack rose!")
    assert len(events) == 2
    assert all(event.type == "stat_change" for event in events)
    assert all(event.stat == "atk" for event in events)
    assert all(event.stages_delta == 1 for event in events)
    by_species = {event.pokemon.species: event for event in events}
    assert by_species["Manectric"].pokemon.side == "player"
    assert by_species["Mamoswine"].pokemon.side == "player"


def test_parse_battle_text_stat_change_one_opponent() -> None:
    events = parse_battle_text("The opposing Garchomp's speed harshly fell!")
    assert len(events) == 1
    assert events[0].type == "stat_change"
    assert events[0].pokemon.species == "Garchomp"
    assert events[0].pokemon.side == "opponent"
    assert events[0].stat == "spe"
    assert events[0].stages_delta == -2


def test_parse_battle_text_stat_change_compound_opponents() -> None:
    events = parse_battle_text(
        "The opposing Garchomp and the opposing Sylveon's Attack fell!"
    )
    assert len(events) == 2
    assert all(event.type == "stat_change" for event in events)
    species = {event.pokemon.species for event in events}
    assert species == {"Garchomp", "Sylveon"}
    assert all(event.pokemon.side == "opponent" for event in events)
    assert all(event.stat == "atk" for event in events)
    assert all(event.stages_delta == -1 for event in events)


def test_parse_battle_text_stat_change_with_ocr_noise() -> None:
    events = parse_battle_text("The opposing Palafin and the opposing Tyrantrum Attack telll")
    assert len(events) == 2
    assert all(event.stages_delta == -1 for event in events)
    assert all(event.stat == "atk" for event in events)
    assert {event.pokemon.species for event in events} == {"Palafin", "Tyrantrum"}



def test_parse_battle_text_move_used() -> None:
    events = parse_battle_text("Sinistcha used Matcha Gotcha!")
    assert len(events) == 1
    assert events[0].type == "move_used"
    assert events[0].actor.species == "Sinistcha"
    assert events[0].move == "Matcha Gotcha"


def test_parse_battle_text_faint() -> None:
    events = parse_battle_text("The opposing Grimmsnarl fainted!")
    assert len(events) == 1
    assert events[0].type == "faint"
    assert events[0].pokemon.species == "Grimmsnarl"
    assert events[0].pokemon.side == "opponent"


def test_parse_battle_text_move_failed() -> None:
    events = parse_battle_text("But it failed!")
    assert len(events) == 1
    assert events[0].type == "move_failed"
    # Actor/move are filled later by the battle log completer.
    assert events[0].actor is None
    assert events[0].move == ""


def test_parse_player_manual_switch_out() -> None:
    events = parse_battle_text("Blastoise, come back!")
    assert len(events) == 1
    assert events[0].type == "switch_out"
    assert events[0].pokemon.species == "Blastoise"
    assert events[0].pokemon.side == "player"


def test_parse_player_manual_switch_in() -> None:
    events = parse_battle_text("Go! Manectric!")
    assert len(events) == 1
    assert events[0].type == "switch_in"
    assert events[0].pokemon.species == "Manectric"
    assert events[0].pokemon.side == "player"


def test_parse_opponent_manual_switch_out() -> None:
    events = parse_battle_text("Blue withdrew Garchomp!")
    assert len(events) == 1
    assert events[0].type == "switch_out"
    assert events[0].pokemon.species == "Garchomp"
    assert events[0].pokemon.side == "opponent"


def test_parse_opponent_manual_switch_in() -> None:
    events = parse_battle_text("Blue sent out Sylveon!")
    assert len(events) == 1
    assert events[0].type == "switch_in"
    assert events[0].pokemon.species == "Sylveon"
    assert events[0].pokemon.side == "opponent"


def test_parse_self_switch_out_player() -> None:
    events = parse_battle_text("Incineroar went back to Ash!")
    assert len(events) == 1
    assert events[0].type == "switch_out"
    assert events[0].pokemon.species == "Incineroar"
    assert events[0].pokemon.side == "player"


def test_parse_self_switch_out_opponent() -> None:
    events = parse_battle_text("The opposing Landorus went back to Blue!")
    assert len(events) == 1
    assert events[0].type == "switch_out"
    assert events[0].pokemon.species == "Landorus"
    assert events[0].pokemon.side == "opponent"


def test_parse_dragged_out_player() -> None:
    events = parse_battle_text("Rillaboom got dragged out!")
    assert len(events) == 1
    assert events[0].type == "switch_in"
    assert events[0].pokemon.species == "Rillaboom"
    assert events[0].pokemon.side == "player"


def test_parse_dragged_out_opponent() -> None:
    events = parse_battle_text("The opposing Whimsicott got dragged out!")
    assert len(events) == 1
    assert events[0].type == "switch_in"
    assert events[0].pokemon.species == "Whimsicott"
    assert events[0].pokemon.side == "opponent"


def test_parse_player_dual_lead_switch_in() -> None:
    events = parse_battle_text("Go! Incineroar and Rillaboom!")
    assert len(events) == 1
    assert events[0].type == "lead_in"
    assert events[0].side == "player"
    assert events[0].slot_1.species == "Incineroar"
    assert events[0].slot_1.slot == 1
    assert events[0].slot_2.species == "Rillaboom"
    assert events[0].slot_2.slot == 2


def test_parse_opponent_dual_lead_switch_in() -> None:
    events = parse_battle_text("Blue sent out Garchomp and Sylveon!")
    assert len(events) == 1
    assert events[0].type == "lead_in"
    assert events[0].side == "opponent"
    assert events[0].slot_1.species == "Garchomp"
    assert events[0].slot_1.slot == 1
    assert events[0].slot_2.species == "Sylveon"
    assert events[0].slot_2.slot == 2


def test_parse_volatile_taunt() -> None:
    events = parse_battle_text("Incineroar fell for the taunt!")
    assert len(events) == 1
    assert events[0].type == "volatile_applied"
    assert events[0].pokemon.species == "Incineroar"
    assert events[0].pokemon.side == "player"
    assert events[0].volatile == "taunted"


def test_parse_volatile_encore_opponent() -> None:
    events = parse_battle_text("The opposing Garchomp must do an encore!")
    assert len(events) == 1
    assert events[0].type == "volatile_applied"
    assert events[0].pokemon.species == "Garchomp"
    assert events[0].pokemon.side == "opponent"
    assert events[0].volatile == "encore"


def test_parse_volatile_confused() -> None:
    events = parse_battle_text("The opposing Whimsicott became confused!")
    assert len(events) == 1
    assert events[0].type == "volatile_applied"
    assert events[0].pokemon.species == "Whimsicott"
    assert events[0].pokemon.side == "opponent"
    assert events[0].volatile == "confused"


def test_parse_volatile_confused_cured() -> None:
    events = parse_battle_text("The opposing Whimsicott snapped out of its confusion!")
    assert len(events) == 1
    assert events[0].type == "volatile_cured"
    assert events[0].pokemon.species == "Whimsicott"
    assert events[0].pokemon.side == "opponent"
    assert events[0].volatile == "confused"


def test_parse_status_applied_showdown_starts() -> None:
    cases = [
        ("Incineroar was burned!", "brn", "player"),
        ("The opposing Garchomp is paralyzed! It may be unable to move!", "par", "opponent"),
        ("Rillaboom was poisoned!", "psn", "player"),
        ("The opposing Whimsicott was badly poisoned!", "tox", "opponent"),
        ("Landorus was frozen solid!", "frz", "player"),
        ("The opposing Musharna fell asleep!", "slp", "opponent"),
    ]
    for text, status, side in cases:
        events = parse_battle_text(text)
        assert len(events) == 1, text
        assert events[0].type == "status_applied", text
        assert events[0].status == status, text
        assert events[0].pokemon.side == side, text


def test_parse_status_cured_showdown_ends() -> None:
    cases = [
        ("Incineroar's burn was healed!", "brn", "player"),
        ("The opposing Garchomp was cured of paralysis!", "par", "opponent"),
        ("Rillaboom was cured of its poisoning!", "psn", "player"),
        ("Landorus thawed out!", "frz", "player"),
        ("The opposing Musharna woke up!", "slp", "opponent"),
    ]
    for text, status, side in cases:
        events = parse_battle_text(text)
        assert len(events) == 1, text
        assert events[0].type == "status_cured", text
        assert events[0].status == status, text
        assert events[0].pokemon.side == side, text


def test_parse_trick_room_start_and_end() -> None:
    start = parse_battle_text("The opposing Musharna twisted the dimensions!")
    assert len(start) == 1
    assert start[0].type == "trick_room_start"

    end = parse_battle_text("The twisted dimensions returned to normal!")
    assert len(end) == 1
    assert end[0].type == "trick_room_end"


def test_parse_weather_start_and_end() -> None:
    starts = [
        ("The sunlight turned harsh!", "sunny"),
        ("It started to rain!", "rain"),
        ("A sandstorm kicked up!", "sandstorm"),
        ("It started to snow!", "snow"),
    ]
    for text, weather in starts:
        events = parse_battle_text(text)
        assert len(events) == 1, text
        assert events[0].type == "weather_start", text
        assert events[0].weather == weather, text

    ends = [
        ("The harsh sunlight faded.", "sunny"),
        ("The rain stopped.", "rain"),
        ("The sandstorm subsided.", "sandstorm"),
        ("The snow stopped.", "snow"),
    ]
    for text, weather in ends:
        events = parse_battle_text(text)
        assert len(events) == 1, text
        assert events[0].type == "weather_end", text
        assert events[0].weather == weather, text


def test_parse_terrain_start_and_end() -> None:
    starts = [
        ("An electric current ran across the battlefield!", "electric_terrain"),
        ("Grass grew to cover the battlefield!", "grassy_terrain"),
        ("Mist swirled around the battlefield!", "misty_terrain"),
        ("The battlefield got weird!", "psychic_terrain"),
    ]
    for text, terrain in starts:
        events = parse_battle_text(text)
        assert len(events) == 1, text
        assert events[0].type == "terrain_start", text
        assert events[0].terrain == terrain, text

    ends = [
        ("The electricity disappeared from the battlefield.", "electric_terrain"),
        ("The grass disappeared from the battlefield.", "grassy_terrain"),
        ("The mist disappeared from the battlefield.", "misty_terrain"),
        ("The weirdness disappeared from the battlefield!", "psychic_terrain"),
    ]
    for text, terrain in ends:
        events = parse_battle_text(text)
        assert len(events) == 1, text
        assert events[0].type == "terrain_end", text
        assert events[0].terrain == terrain, text


def test_parse_side_condition_tailwind() -> None:
    events = parse_battle_text("A tailwind started blowing on your side!")
    assert len(events) == 1
    assert events[0].type == "side_condition"
    assert events[0].condition == "tailwind"
    assert events[0].side == "player"

    events = parse_battle_text("A tailwind started blowing on the opposing side!")
    assert events[0].side == "opponent"
    assert events[0].condition == "tailwind"


def test_parse_side_condition_screens() -> None:
    events = parse_battle_text(
        "Reflect made your side stronger against physical moves!"
    )
    assert events[0].type == "side_condition"
    assert events[0].condition == "reflect"
    assert events[0].side == "player"

    events = parse_battle_text(
        "Light Screen made the opponent's side stronger against special moves!"
    )
    assert events[0].condition == "light_screen"
    assert events[0].side == "opponent"

    events = parse_battle_text(
        "Aurora Veil made your side stronger against physical and special moves!"
    )
    assert events[0].condition == "aurora_veil"
    assert events[0].side == "player"


def test_parse_side_condition_hazards() -> None:
    events = parse_battle_text(
        "Spikes were scattered on the ground all around your side!"
    )
    assert events[0].condition == "spikes"
    assert events[0].side == "player"

    events = parse_battle_text(
        "Toxic spikes were scattered on the ground all around the opposing side!"
    )
    assert events[0].condition == "toxic_spikes"
    assert events[0].side == "opponent"

    events = parse_battle_text(
        "Pointed stones float in the air around the opposing team!"
    )
    assert events[0].condition == "stealth_rocks"
    assert events[0].side == "opponent"


def _field_path(event: object, path: str) -> object:
    value = event
    for part in path.split("."):
        value = getattr(value, part)
    return value


@pytest.mark.parametrize(
    ("text", "player_species", "opponent_species", "expected"),
    [
        (
            "The opposing Garchomp can't use Earthquake because of gravity!",
            ("Incineroar",),
            ("Garchomp",),
            {
                "type": "move_failed",
                "reason": "gravity",
                "actor.species": "Garchomp",
                "actor.side": "opponent",
                "move": "Earthquake",
            },
        ),
        (
            "The opposing Garchomp copied Incineroar's stat changes!",
            ("Incineroar",),
            ("Garchomp",),
            {
                "type": "stat_stage_operation",
                "operation": "copy",
                "pokemon.species": "Garchomp",
                "pokemon.side": "opponent",
                "target.species": "Incineroar",
                "target.side": "player",
            },
        ),
        (
            "Incineroar stole the opposing Garchomp's Sitrus Berry!",
            ("Incineroar",),
            ("Garchomp",),
            {
                "type": "held_item_changed",
                "change": "stolen",
                "pokemon.species": "Incineroar",
                "source.species": "Garchomp",
                "source.side": "opponent",
                "item": "Sitrus Berry",
            },
        ),
        (
            "Coba Berry only allows the use of Wish!",
            ("Garchomp",),
            ("Incineroar",),
            {
                "type": "move_availability_changed",
                "restriction": "forced_move",
                "source_item": "Coba Berry",
                "move": "Wish",
                "clears_on_switch": True,
            },
        ),
        (
            "It doesn't affect the opposing Garchomp...",
            ("Incineroar",),
            ("Garchomp",),
            {
                "type": "move_outcome",
                "outcome": "immune",
                "target.species": "Garchomp",
                "target.side": "opponent",
            },
        ),
        (
            "The Pokémon was hit 3 time(s)!",
            ("Incineroar",),
            ("Garchomp",),
            {
                "type": "move_outcome",
                "outcome": "hit_count",
                "count": 3,
            },
        ),
        (
            "Go! Garchomp▽\nand Incineroar!",
            ("Garchomp", "Incineroar"),
            ("Whimsicott",),
            {
                "type": "lead_in",
                "side": "player",
                "slot_1.species": "Garchomp",
                "slot_2.species": "Incineroar",
            },
        ),
    ],
)
def test_parse_catalog_texts_into_structured_events(
    text: str,
    player_species: tuple[str, ...],
    opponent_species: tuple[str, ...],
    expected: dict[str, object],
) -> None:
    """Source-backed templates retain their typed fields through dispatch."""
    events = parse_battle_text(
        text,
        player_species=player_species,
        opponent_species=opponent_species,
    )

    assert len(events) == 1, text
    for path, value in expected.items():
        assert _field_path(events[0], path) == value, f"{text}: {path}"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Choose a move!",
        "The opposing Garchomp used!",
        "It doesn't affect...",
    ],
)
def test_parse_battle_text_rejects_unknown_and_incomplete_messages(text: str) -> None:
    assert parse_battle_text(
        text,
        player_species=("Incineroar",),
        opponent_species=("Garchomp",),
    ) == []
