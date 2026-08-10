"""Catalog + fixed-template dispatcher smoke tests."""

from __future__ import annotations

from app.cv.battle_text_catalog import catalog_templates, normalize_catalog_text
from app.cv.event_parser import match_fixed_catalog_template, parse_battle_text


def test_catalog_has_fixed_and_legacy_entries() -> None:
    fixed = catalog_templates(matcher="fixed")
    legacy = catalog_templates(matcher="legacy")
    tokenized = catalog_templates(matcher="tokenized")
    assert len(fixed) >= 40
    assert len(legacy) >= 5
    assert len(tokenized) >= 5
    assert any(t.id.startswith("side.stealth_rocks") for t in fixed)
    assert any(t.id == "outcome.super_effective" for t in fixed)


def test_normalize_catalog_text_strips_layout_markers() -> None:
    assert normalize_catalog_text("Go! Incineroar▽\nand Rillaboom!") == (
        "Go! Incineroar and Rillaboom!"
    )
    assert normalize_catalog_text("It’s super effective!") == (
        "It's super effective!"
    )


def test_parse_move_outcome_champions_fixed() -> None:
    events = parse_battle_text("It’s super effective!")
    assert len(events) == 1
    assert events[0].type == "move_outcome"
    assert events[0].outcome == "super_effective"


def test_parse_perish_song_and_fairy_lock() -> None:
    perish = parse_battle_text(
        "All Pokémon that heard the song will faint in three turns!"
    )
    assert perish[0].type == "perish_song_started"
    assert perish[0].turns_remaining == 3

    lock = parse_battle_text(
        "No one will be able to leave the battlefield during the next turn!"
    )
    assert lock[0].type == "switch_lock_started"


def test_parse_field_effects_and_weather_suppression() -> None:
    gravity = parse_battle_text("Gravity intensified!")
    assert gravity[0].type == "field_effect_changed"
    assert gravity[0].effect == "gravity"
    assert gravity[0].action == "start"

    suppress = parse_battle_text("The effects of the weather disappeared.")
    assert suppress[0].type == "field_effect_changed"
    assert suppress[0].effect == "weather_suppression"


def test_parse_side_condition_end_and_safeguard_sticky_web() -> None:
    end = parse_battle_text("Your side’s Reflect wore off!")
    assert end[0].type == "side_condition"
    assert end[0].condition == "reflect"
    assert end[0].action == "end"
    assert end[0].side == "player"

    safeguard = parse_battle_text("Your side became cloaked in a mystical veil!")
    assert safeguard[0].condition == "safeguard"
    assert safeguard[0].action == "start"

    web = parse_battle_text(
        "A sticky web has been laid out on the ground on the opposing side!"
    )
    assert web[0].condition == "sticky_web"
    assert web[0].side == "opponent"


def test_parse_champions_stealth_rocks_on_side() -> None:
    events = parse_battle_text(
        "Pointed stones float in the air on the opposing side!"
    )
    assert events[0].condition == "stealth_rocks"
    assert events[0].side == "opponent"


def test_parse_move_failed_reasons_from_catalog() -> None:
    failed = parse_battle_text("But it failed!")
    assert failed[0].type == "move_failed"
    assert failed[0].reason == "failed"

    no_pp = parse_battle_text("But there was no PP left for the move!")
    assert no_pp[0].type == "move_failed"
    assert no_pp[0].reason == "no_pp"


def test_fixed_match_rejects_empty() -> None:
    assert match_fixed_catalog_template("") is None
