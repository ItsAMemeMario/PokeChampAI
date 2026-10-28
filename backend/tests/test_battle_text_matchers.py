"""Regression tests for token-first positional fuzzy matching."""

from __future__ import annotations

import pytest

from app.cv.battle_text_matchers import (
    emit_tokenized_match,
    match_tokenized_catalog_template,
)
from app.cv.event_parser import normalize_ocr_text, parse_battle_text


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


@pytest.mark.parametrize(
    ("template_id", "text", "expected_type"),
    [
        (
            "outcome.hit_count",
            "The Pokemon was hit 3 time(s)!",
            "move_outcome",
        ),
        (
            "outcome.immune_named",
            "It doesn't affect the opposing Garchomp...",
            "move_outcome",
        ),
        (
            "outcome.miss_avoided",
            "The opposing Garchomp avoided the attack!",
            "move_outcome",
        ),
        (
            "stat_op.clear_one",
            "The opposing Garchomp's stat changes were removed!",
            "stat_stage_operation",
        ),
        (
            "stat_op.invert",
            "The opposing Garchomp's stat changes were inverted!",
            "stat_stage_operation",
        ),
        (
            "stat_op.copy",
            "The opposing Garchomp copied Incineroar's stat changes!",
            "stat_stage_operation",
        ),
        (
            "stat_op.swap_all",
            "The opposing Garchomp switched stat changes with its target!",
            "stat_stage_operation",
        ),
        (
            "stat_op.swap_offensive",
            (
                "The opposing Garchomp switched all changes to its Attack and "
                "Sp. Atk with its target!"
            ),
            "stat_stage_operation",
        ),
        (
            "stat_op.swap_defensive",
            (
                "The opposing Garchomp switched all changes to its Defense and "
                "Sp. Def with its target!"
            ),
            "stat_stage_operation",
        ),
        (
            "item.frisked",
            "The opposing Garchomp was frisked, revealing its Coba Berry!",
            "held_item_changed",
        ),
        (
            "item.weakened_move",
            "Coba Berry weakened Earthquake's power!",
            "held_item_changed",
        ),
        (
            "item.obtained",
            "The opposing Garchomp obtained one Coba Berry.",
            "held_item_changed",
        ),
        (
            "item.stolen",
            "Incineroar stole the opposing Garchomp's Coba Berry!",
            "held_item_changed",
        ),
        (
            "item.ate",
            "The opposing Garchomp ate its Coba Berry!",
            "held_item_changed",
        ),
        (
            "item.lost",
            "The opposing Garchomp lost its Coba Berry!",
            "held_item_changed",
        ),
        (
            "item.used",
            "The opposing Garchomp used its Coba Berry!",
            "held_item_changed",
        ),
        (
            "item.weakened_damage",
            "Coba Berry weakened damage to the opposing Garchomp!",
            "held_item_changed",
        ),
        (
            "avail.cooldown_move",
            "Protect can't be used twice in a row!",
            "move_availability_changed",
        ),
        (
            "avail.forced_can_only",
            "The opposing Garchomp can only use Earthquake!",
            "move_availability_changed",
        ),
        (
            "avail.forced_item",
            "Coba Berry only allows the use of Wish!",
            "move_availability_changed",
        ),
        (
            "fail.flinch",
            "The opposing Garchomp flinched and couldn't move!",
            "move_failed",
        ),
        (
            "fail.par_cant_move",
            "The opposing Garchomp is paralyzed! It can't move!",
            "move_failed",
        ),
        (
            "fail.freeze",
            "The opposing Garchomp is frozen solid!",
            "move_failed",
        ),
        (
            "fail.sleep",
            "The opposing Garchomp is fast asleep.",
            "move_failed",
        ),
        (
            "fail.recharge",
            "The opposing Garchomp must recharge!",
            "move_failed",
        ),
        (
            "fail.gravity_block",
            "The opposing Garchomp can't use Earthquake because of gravity!",
            "move_failed",
        ),
    ],
)
def test_tokenized_catalog_templates_match_clean_source_text(
    template_id: str,
    text: str,
    expected_type: str,
) -> None:
    """Each tokenized template has a deterministic source-text regression."""
    match = match_tokenized_catalog_template(
        normalize_ocr_text(text),
        player_species=("Incineroar",),
        opponent_species=("Garchomp",),
        include_ids=(template_id,),
    )

    assert match is not None, template_id
    assert match.template.id == template_id
    events = emit_tokenized_match(text, match)
    assert len(events) == 1
    assert events[0].type == expected_type


def test_tokenized_matcher_rejects_empty_text() -> None:
    assert match_tokenized_catalog_template("") is None


def test_status_and_item_variants_do_not_cross_match() -> None:
    paralysis = parse_battle_text("Garchomp is paralyzed! It may be unable to move!")
    assert len(paralysis) == 1
    assert paralysis[0].type == "status_applied"
    assert paralysis[0].status == "par"

    move = parse_battle_text("Sinistcha used Protect!")
    item = parse_battle_text("Sinistcha used its Sitrus Berry!")
    assert move[0].type == "move_used"
    assert item[0].type == "held_item_changed"
