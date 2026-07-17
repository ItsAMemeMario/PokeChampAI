"""Unit tests for battle event text parsing."""

from __future__ import annotations

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
    assert is_known_item("Staraptorite") is True


def test_parse_battle_text_mega_evolution_with_ocr_errors() -> None:
    events = parse_battle_text("The opposing Scizor has Mega Evolvea ito Mega Scizorl")
    assert any(event.type == "mega_evolution" for event in events)
    mega = next(event for event in events if event.type == "mega_evolution")
    assert mega.pokemon.species == "Scizor"
    assert mega.pokemon.side == "opponent"


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
    assert len(events) == 2
    assert all(event.type == "switch_in" for event in events)
    assert events[0].pokemon.species == "Incineroar"
    assert events[0].pokemon.side == "player"
    assert events[0].pokemon.slot == 1
    assert events[1].pokemon.species == "Rillaboom"
    assert events[1].pokemon.side == "player"
    assert events[1].pokemon.slot == 2


def test_parse_opponent_dual_lead_switch_in() -> None:
    events = parse_battle_text("Blue sent out Garchomp and Sylveon!")
    assert len(events) == 2
    assert all(event.type == "switch_in" for event in events)
    assert events[0].pokemon.species == "Garchomp"
    assert events[0].pokemon.side == "opponent"
    assert events[0].pokemon.slot == 1
    assert events[1].pokemon.species == "Sylveon"
    assert events[1].pokemon.side == "opponent"
    assert events[1].pokemon.slot == 2


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
