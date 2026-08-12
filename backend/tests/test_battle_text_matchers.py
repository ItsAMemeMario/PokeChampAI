"""Regression tests for token-first positional fuzzy matching."""

from __future__ import annotations

from app.cv.event_parser import parse_battle_text


def test_token_first_match_survives_noise_in_short_middle_literal() -> None:
    events = parse_battle_text(
        "the opposing Pangoro c%p@ed Forretress's stat changes!",
        player_species=("Chimecho", "Forretress", "Rhyperior", "Scizor"),
        opponent_species=(
            "Absol",
            "Pangoro",
            "Reuniclus",
            "Excadrill",
            "Tinkaton",
            "Rotom",
        ),
    )

    assert len(events) == 1
    event = events[0]
    assert event.type == "stat_stage_operation"
    assert event.operation == "copy"
    assert event.pokemon is not None
    assert event.target is not None
    assert event.pokemon.species == "Pangoro"
    assert event.pokemon.side == "opponent"
    assert event.target.species == "Forretress"


def test_short_damaged_move_is_snapped_from_its_position() -> None:
    events = parse_battle_text(
        "Coba Berry only allows the use of Wis%!",
        player_species=("Hawlucha",),
        opponent_species=("Annihilape",),
    )

    assert len(events) == 1
    event = events[0]
    assert event.type == "move_availability_changed"
    assert event.restriction == "forced_move"
    assert event.source_item == "Coba Berry"
    assert event.move == "Wish"


def test_noisy_fixed_template_does_not_become_toxic_spikes() -> None:
    events = parse_battle_text(
        "S&ikes were scattered on the ground all around1your side!"
    )

    assert len(events) == 1
    event = events[0]
    assert event.type == "side_condition"
    assert event.condition == "spikes"
    assert event.side == "player"
