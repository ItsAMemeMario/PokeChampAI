"""Champions-first battle-text template catalog.

Canonical strings come from the Champout ``btl_std.json`` dump. Showdown
``default.ts`` strings are explicit fallbacks only. Tokenized templates are
declared here for the dispatcher; variable-token matching lands in the matcher
layer (``event_parser`` / later todo work).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

TokenName = Literal[
    "POKEMON",
    "MOVE",
    "ITEM",
    "ABILITY",
    "TRAINER",
    "TARGET",
    "SOURCE",
    "STAT",
    "TYPE",
    "NUMBER",
    "SIDE",
    "SPECIES",
]

MatcherKind = Literal["fixed", "tokenized", "legacy"]


@dataclass(frozen=True)
class BattleTextTemplate:
    """One declarative message pattern → event shape."""

    id: str
    # Logical family used by the emitter (often the BattleLogEvent.type).
    event_kind: str
    champions: tuple[str, ...] = ()
    showdown: tuple[str, ...] = ()
    tokens: tuple[TokenName, ...] = ()
    # Large fixed substring for tokenized templates (RapidFuzz anchor).
    fixed_anchor: str | None = None
    priority: int = 100
    matcher: MatcherKind = "fixed"
    multi_event: bool = False
    # Static fields merged into the emitted event (side, outcome, …).
    static: Mapping[str, Any] = field(default_factory=dict)
    # Existing regex/handler name when matcher == "legacy".
    legacy_handler: str | None = None


def _t(
    template_id: str,
    event_kind: str,
    *,
    champions: tuple[str, ...] | str = (),
    showdown: tuple[str, ...] | str = (),
    tokens: tuple[TokenName, ...] = (),
    fixed_anchor: str | None = None,
    priority: int = 100,
    matcher: MatcherKind = "fixed",
    multi_event: bool = False,
    static: Mapping[str, Any] | None = None,
    legacy_handler: str | None = None,
) -> BattleTextTemplate:
    if isinstance(champions, str):
        champions = (champions,)
    if isinstance(showdown, str):
        showdown = (showdown,)
    return BattleTextTemplate(
        id=template_id,
        event_kind=event_kind,
        champions=champions,
        showdown=showdown,
        tokens=tokens,
        fixed_anchor=fixed_anchor,
        priority=priority,
        matcher=matcher,
        multi_event=multi_event,
        static=static or {},
        legacy_handler=legacy_handler,
    )


def normalize_catalog_text(text: str) -> str:
    """Normalize Champions / Showdown / OCR layout before template compare."""
    cleaned = text.replace("\u2019", "'").replace("\u2018", "'")
    cleaned = cleaned.replace("\u2026", "...")
    cleaned = cleaned.replace("**", "")
    # Champout layout markers and hard line breaks.
    cleaned = cleaned.replace("▽", " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Catalog entries (priority: lower runs earlier among fixed matches)
# ---------------------------------------------------------------------------

_MOVE_OUTCOMES: tuple[BattleTextTemplate, ...] = (
    _t(
        "outcome.extremely_effective",
        "move_outcome",
        champions="It's extremely effective!",
        priority=20,
        static={"outcome": "extremely_effective"},
    ),
    _t(
        "outcome.super_effective",
        "move_outcome",
        champions="It's super effective!",
        priority=20,
        static={"outcome": "super_effective"},
    ),
    _t(
        "outcome.resisted",
        "move_outcome",
        champions="It's not very effective...",
        priority=20,
        static={"outcome": "resisted"},
    ),
    _t(
        "outcome.mostly_ineffective",
        "move_outcome",
        champions="It's mostly ineffective...",
        priority=20,
        static={"outcome": "mostly_ineffective"},
    ),
    _t(
        "outcome.critical",
        "move_outcome",
        champions="A critical hit!",
        priority=20,
        static={"outcome": "critical"},
    ),
    _t(
        "outcome.ohko",
        "move_outcome",
        champions="It's a one-hit KO!",
        priority=20,
        static={"outcome": "ohko"},
    ),
    _t(
        "outcome.hit_count",
        "move_outcome",
        # Dump form is truncated ("hit {0} !"); accept both dump-shaped and full.
        champions=(
            "The Pokémon was hit [NUMBER]!",
            "The Pokémon was hit [NUMBER] time(s)!",
            "The Pokemon was hit [NUMBER]!",
            "The Pokemon was hit [NUMBER] time(s)!",
        ),
        tokens=("NUMBER",),
        fixed_anchor="The Pokemon was hit",
        matcher="tokenized",
        priority=25,
        static={"outcome": "hit_count"},
    ),
    _t(
        "outcome.immune_named",
        "move_outcome",
        showdown=(
            "It doesn't affect [POKEMON]...",
            "It had no effect!",
            "[POKEMON] is unaffected!",
        ),
        tokens=("POKEMON",),
        fixed_anchor="doesn't affect",
        matcher="tokenized",
        priority=30,
        static={"outcome": "immune"},
    ),
    _t(
        "outcome.miss_avoided",
        "move_outcome",
        showdown=(
            "[POKEMON] avoided the attack!",
            "[SOURCE]'s attack missed!",
        ),
        tokens=("POKEMON", "SOURCE"),
        matcher="tokenized",
        priority=30,
        static={"outcome": "miss"},
    ),
)

_FIELD_EFFECTS: tuple[BattleTextTemplate, ...] = (
    _t(
        "field.gravity.start",
        "field_effect_changed",
        champions="Gravity intensified!",
        priority=20,
        static={"effect": "gravity", "action": "start"},
    ),
    _t(
        "field.gravity.end",
        "field_effect_changed",
        champions="Gravity returned to normal!",
        priority=20,
        static={"effect": "gravity", "action": "end"},
    ),
    _t(
        "field.magic_room.start",
        "field_effect_changed",
        champions=(
            "It created a bizarre area in which Pokemon's held items lose their effects!",
            "It created a bizarre area in which Pokémon's held items lose their effects!",
        ),
        priority=20,
        static={"effect": "magic_room", "action": "start"},
    ),
    _t(
        "field.magic_room.end",
        "field_effect_changed",
        champions=(
            "Magic Room wore off, and held items' effects returned to normal!",
        ),
        priority=20,
        static={"effect": "magic_room", "action": "end"},
    ),
    _t(
        "field.wonder_room.start",
        "field_effect_changed",
        champions=(
            "It created a bizarre area in which Defense and Sp. Def stats are swapped!",
        ),
        priority=20,
        static={"effect": "wonder_room", "action": "start"},
    ),
    _t(
        "field.wonder_room.end",
        "field_effect_changed",
        champions=(
            "Wonder Room wore off, and Defense and Sp. Def stats returned to normal!",
        ),
        priority=20,
        static={"effect": "wonder_room", "action": "end"},
    ),
    _t(
        "field.weather_suppression",
        "field_effect_changed",
        champions="The effects of the weather disappeared.",
        priority=15,
        static={"effect": "weather_suppression", "action": "start"},
    ),
)

_PERISH_AND_LOCK: tuple[BattleTextTemplate, ...] = (
    _t(
        "perish_song.started",
        "perish_song_started",
        champions=(
            "All Pokemon that heard the song will faint in three turns!",
            "All Pokémon that heard the song will faint in three turns!",
        ),
        priority=20,
        static={"turns_remaining": 3},
    ),
    _t(
        "switch_lock.fairy_lock",
        "switch_lock_started",
        champions=(
            "No one will be able to leave the battlefield during the next turn!",
        ),
        priority=20,
        static={"scope": "all_active"},
    ),
)

_STAT_OPS: tuple[BattleTextTemplate, ...] = (
    _t(
        "stat_op.clear_all",
        "stat_stage_operation",
        champions="All stat changes were eliminated!",
        priority=20,
        static={"operation": "clear_all"},
    ),
    _t(
        "stat_op.clear_one",
        "stat_stage_operation",
        showdown="[POKEMON]'s stat changes were removed!",
        tokens=("POKEMON",),
        fixed_anchor="stat changes were removed",
        matcher="tokenized",
        priority=40,
        static={"operation": "clear_one"},
    ),
    _t(
        "stat_op.invert",
        "stat_stage_operation",
        showdown="[POKEMON]'s stat changes were inverted!",
        tokens=("POKEMON",),
        fixed_anchor="stat changes were inverted",
        matcher="tokenized",
        priority=40,
        static={"operation": "invert"},
    ),
    _t(
        "stat_op.copy",
        "stat_stage_operation",
        showdown="[POKEMON] copied [TARGET]'s stat changes!",
        tokens=("POKEMON", "TARGET"),
        fixed_anchor="copied",
        matcher="tokenized",
        priority=40,
        static={"operation": "copy"},
    ),
    _t(
        "stat_op.swap_all",
        "stat_stage_operation",
        showdown="[POKEMON] switched stat changes with its target!",
        tokens=("POKEMON",),
        fixed_anchor="switched stat changes with its target",
        matcher="tokenized",
        priority=40,
        static={"operation": "swap_all"},
    ),
    _t(
        "stat_op.swap_offensive",
        "stat_stage_operation",
        showdown=(
            "[POKEMON] switched all changes to its Attack and Sp. Atk with its target!"
        ),
        tokens=("POKEMON",),
        fixed_anchor="Attack and Sp. Atk with its target",
        matcher="tokenized",
        priority=40,
        static={"operation": "swap_offensive"},
    ),
    _t(
        "stat_op.swap_defensive",
        "stat_stage_operation",
        showdown=(
            "[POKEMON] switched all changes to its Defense and Sp. Def with its target!"
        ),
        tokens=("POKEMON",),
        fixed_anchor="Defense and Sp. Def with its target",
        matcher="tokenized",
        priority=40,
        static={"operation": "swap_defensive"},
    ),
)

_HELD_ITEMS: tuple[BattleTextTemplate, ...] = (
    _t(
        "item.teatime",
        "held_item_changed",
        champions="It's teatime! Everyone dug in to their Berries!",
        priority=20,
        static={"change": "consumed"},
    ),
    _t(
        "item.frisked",
        "held_item_changed",
        champions="[POKEMON] was frisked, revealing its [ITEM]!",
        tokens=("POKEMON", "ITEM"),
        fixed_anchor="was frisked, revealing",
        matcher="tokenized",
        priority=40,
        static={"change": "revealed"},
    ),
    _t(
        "item.weakened_move",
        "held_item_changed",
        champions="[ITEM] weakened [MOVE]'s power!",
        tokens=("ITEM", "MOVE"),
        fixed_anchor="weakened",
        matcher="tokenized",
        priority=45,
        static={"change": "activated"},
    ),
    _t(
        "item.obtained",
        "held_item_changed",
        showdown="[POKEMON] obtained one [ITEM].",
        tokens=("POKEMON", "ITEM"),
        fixed_anchor="obtained one",
        matcher="tokenized",
        priority=40,
        static={"change": "obtained"},
    ),
    _t(
        "item.stolen",
        "held_item_changed",
        showdown="[POKEMON] stole [SOURCE]'s [ITEM]!",
        tokens=("POKEMON", "SOURCE", "ITEM"),
        fixed_anchor="stole",
        matcher="tokenized",
        priority=40,
        static={"change": "stolen"},
    ),
    _t(
        "item.ate",
        "held_item_changed",
        showdown="[POKEMON] ate its [ITEM]!",
        tokens=("POKEMON", "ITEM"),
        fixed_anchor="ate its",
        matcher="tokenized",
        priority=40,
        static={"change": "consumed"},
    ),
    _t(
        "item.lost",
        "held_item_changed",
        showdown="[POKEMON] lost its [ITEM]!",
        tokens=("POKEMON", "ITEM"),
        fixed_anchor="lost its",
        matcher="tokenized",
        priority=40,
        static={"change": "lost"},
    ),
    _t(
        "item.used",
        "held_item_changed",
        showdown="[POKEMON] used its [ITEM]!",
        tokens=("POKEMON", "ITEM"),
        fixed_anchor="used its",
        matcher="tokenized",
        priority=40,
        static={"change": "activated"},
    ),
    _t(
        "item.weakened_damage",
        "held_item_changed",
        showdown=(
            "[ITEM] weakened damage to [POKEMON]!",
            "[ITEM] weakened the damage to [POKEMON]!",
        ),
        tokens=("ITEM", "POKEMON"),
        fixed_anchor="weakened",
        matcher="tokenized",
        priority=45,
        static={"change": "activated"},
    ),
)

_FAIL_AND_AVAIL: tuple[BattleTextTemplate, ...] = (
    _t(
        "fail.but_it_failed",
        "move_failed",
        champions="But it failed!",
        priority=20,
        static={"reason": "failed"},
    ),
    _t(
        "fail.no_pp",
        "move_failed",
        champions="But there was no PP left for the move!",
        priority=20,
        static={"reason": "no_pp"},
    ),
    _t(
        "fail.insufficient_hp_sub",
        "move_failed",
        champions=(
            "But it does not have enough HP left to make a substitute!",
        ),
        priority=20,
        static={"reason": "insufficient_hp"},
    ),
    _t(
        "fail.unusable",
        "move_failed",
        champions="This move can't be used!",
        priority=20,
        static={"reason": "unusable"},
    ),
    _t(
        "avail.cooldown_move",
        "move_availability_changed",
        champions="[MOVE] can't be used twice in a row!",
        tokens=("MOVE",),
        fixed_anchor="can't be used twice in a row",
        matcher="tokenized",
        priority=40,
        static={"restriction": "cooldown", "clears_on_switch": False},
    ),
    _t(
        "avail.forced_can_only",
        "move_availability_changed",
        champions="[POKEMON] can only use [MOVE]!",
        tokens=("POKEMON", "MOVE"),
        fixed_anchor="can only use",
        matcher="tokenized",
        priority=40,
        static={"restriction": "forced_move", "clears_on_switch": True},
    ),
    _t(
        "avail.forced_item",
        "move_availability_changed",
        champions="[ITEM] only allows the use of [MOVE]!",
        tokens=("ITEM", "MOVE"),
        fixed_anchor="only allows the use of",
        matcher="tokenized",
        priority=40,
        static={"restriction": "forced_move", "clears_on_switch": True},
    ),
    _t(
        "fail.flinch",
        "move_failed",
        showdown="[POKEMON] flinched and couldn't move!",
        tokens=("POKEMON",),
        fixed_anchor="flinched and couldn't move",
        matcher="tokenized",
        priority=40,
        static={"reason": "flinch"},
    ),
    _t(
        "fail.par_cant_move",
        "move_failed",
        showdown="[POKEMON] is paralyzed! It can't move!",
        tokens=("POKEMON",),
        fixed_anchor="It can't move",
        matcher="tokenized",
        priority=40,
        static={"reason": "paralysis"},
    ),
    _t(
        "fail.freeze",
        "move_failed",
        showdown="[POKEMON] is frozen solid!",
        tokens=("POKEMON",),
        fixed_anchor="is frozen solid",
        matcher="tokenized",
        priority=50,
        static={"reason": "freeze"},
    ),
    _t(
        "fail.sleep",
        "move_failed",
        showdown="[POKEMON] is fast asleep.",
        tokens=("POKEMON",),
        fixed_anchor="is fast asleep",
        matcher="tokenized",
        priority=40,
        static={"reason": "sleep"},
    ),
    _t(
        "fail.recharge",
        "move_failed",
        showdown="[POKEMON] must recharge!",
        tokens=("POKEMON",),
        fixed_anchor="must recharge",
        matcher="tokenized",
        priority=40,
        static={"reason": "recharge"},
    ),
    _t(
        "fail.gravity_block",
        "move_failed",
        showdown="[POKEMON] can't use [MOVE] because of gravity!",
        tokens=("POKEMON", "MOVE"),
        fixed_anchor="because of gravity",
        matcher="tokenized",
        priority=40,
        static={"reason": "gravity"},
    ),
)

_SIDE_CONDITIONS: tuple[BattleTextTemplate, ...] = (
    # Reflect
    _t(
        "side.reflect.start.player",
        "side_condition",
        champions="Reflect made your side stronger against physical moves!",
        priority=20,
        static={"condition": "reflect", "side": "player", "action": "start"},
    ),
    _t(
        "side.reflect.start.opponent",
        "side_condition",
        champions=(
            "Reflect made the opposing side stronger against physical moves!",
        ),
        showdown=(
            "Reflect made the opponent's side stronger against physical moves!",
        ),
        priority=20,
        static={"condition": "reflect", "side": "opponent", "action": "start"},
    ),
    _t(
        "side.reflect.end.player",
        "side_condition",
        champions="Your side's Reflect wore off!",
        priority=20,
        static={"condition": "reflect", "side": "player", "action": "end"},
    ),
    _t(
        "side.reflect.end.opponent",
        "side_condition",
        champions="The opposing side's Reflect wore off!",
        priority=20,
        static={"condition": "reflect", "side": "opponent", "action": "end"},
    ),
    # Light Screen
    _t(
        "side.light_screen.start.player",
        "side_condition",
        champions="Light Screen made your side stronger against special moves!",
        priority=20,
        static={
            "condition": "light_screen",
            "side": "player",
            "action": "start",
        },
    ),
    _t(
        "side.light_screen.start.opponent",
        "side_condition",
        champions=(
            "Light Screen made the opposing side stronger against special moves!",
        ),
        showdown=(
            "Light Screen made the opponent's side stronger against special moves!",
        ),
        priority=20,
        static={
            "condition": "light_screen",
            "side": "opponent",
            "action": "start",
        },
    ),
    _t(
        "side.light_screen.end.player",
        "side_condition",
        champions="Your side's Light Screen wore off!",
        priority=20,
        static={
            "condition": "light_screen",
            "side": "player",
            "action": "end",
        },
    ),
    _t(
        "side.light_screen.end.opponent",
        "side_condition",
        champions="The opposing side's Light Screen wore off!",
        priority=20,
        static={
            "condition": "light_screen",
            "side": "opponent",
            "action": "end",
        },
    ),
    # Aurora Veil
    _t(
        "side.aurora_veil.start.player",
        "side_condition",
        champions=(
            "Aurora Veil made your side stronger against physical and special moves!",
        ),
        priority=20,
        static={
            "condition": "aurora_veil",
            "side": "player",
            "action": "start",
        },
    ),
    _t(
        "side.aurora_veil.start.opponent",
        "side_condition",
        champions=(
            "Aurora Veil made the opposing side stronger against physical and special moves!",
        ),
        priority=20,
        static={
            "condition": "aurora_veil",
            "side": "opponent",
            "action": "start",
        },
    ),
    _t(
        "side.aurora_veil.end.player",
        "side_condition",
        champions="Your side's Aurora Veil wore off!",
        priority=20,
        static={
            "condition": "aurora_veil",
            "side": "player",
            "action": "end",
        },
    ),
    _t(
        "side.aurora_veil.end.opponent",
        "side_condition",
        champions="The opposing side's Aurora Veil wore off!",
        priority=20,
        static={
            "condition": "aurora_veil",
            "side": "opponent",
            "action": "end",
        },
    ),
    # Safeguard (mystical veil)
    _t(
        "side.safeguard.start.player",
        "side_condition",
        champions="Your side became cloaked in a mystical veil!",
        priority=20,
        static={"condition": "safeguard", "side": "player", "action": "start"},
    ),
    _t(
        "side.safeguard.start.opponent",
        "side_condition",
        champions="The opposing side became cloaked in a mystical veil!",
        priority=20,
        static={
            "condition": "safeguard",
            "side": "opponent",
            "action": "start",
        },
    ),
    _t(
        "side.safeguard.end.player",
        "side_condition",
        champions="Your side is no longer protected by the mystical veil!",
        priority=20,
        static={"condition": "safeguard", "side": "player", "action": "end"},
    ),
    _t(
        "side.safeguard.end.opponent",
        "side_condition",
        champions=(
            "The opposing side is no longer protected by the mystical veil!",
        ),
        priority=20,
        static={"condition": "safeguard", "side": "opponent", "action": "end"},
    ),
    # Tailwind
    _t(
        "side.tailwind.start.player",
        "side_condition",
        champions="A tailwind started blowing on your side!",
        priority=20,
        static={"condition": "tailwind", "side": "player", "action": "start"},
    ),
    _t(
        "side.tailwind.start.opponent",
        "side_condition",
        champions="A tailwind started blowing on the opposing side!",
        priority=20,
        static={"condition": "tailwind", "side": "opponent", "action": "start"},
    ),
    _t(
        "side.tailwind.end.player",
        "side_condition",
        champions="Your side's tailwind petered out!",
        priority=20,
        static={"condition": "tailwind", "side": "player", "action": "end"},
    ),
    _t(
        "side.tailwind.end.opponent",
        "side_condition",
        champions="The opposing side's tailwind petered out!",
        priority=20,
        static={"condition": "tailwind", "side": "opponent", "action": "end"},
    ),
    # Spikes
    _t(
        "side.spikes.start.player",
        "side_condition",
        champions="Spikes were scattered on the ground all around your side!",
        priority=20,
        static={"condition": "spikes", "side": "player", "action": "start"},
    ),
    _t(
        "side.spikes.start.opponent",
        "side_condition",
        champions=(
            "Spikes were scattered on the ground all around the opposing side!",
        ),
        priority=20,
        static={"condition": "spikes", "side": "opponent", "action": "start"},
    ),
    _t(
        "side.spikes.end.player",
        "side_condition",
        champions="The spikes disappeared from the ground around your side!",
        priority=20,
        static={"condition": "spikes", "side": "player", "action": "end"},
    ),
    _t(
        "side.spikes.end.opponent",
        "side_condition",
        champions=(
            "The spikes disappeared from the ground around the opposing side!",
        ),
        priority=20,
        static={"condition": "spikes", "side": "opponent", "action": "end"},
    ),
    # Toxic Spikes
    _t(
        "side.toxic_spikes.start.player",
        "side_condition",
        champions=(
            "Toxic spikes were scattered on the ground all around your side!",
        ),
        priority=20,
        static={
            "condition": "toxic_spikes",
            "side": "player",
            "action": "start",
        },
    ),
    _t(
        "side.toxic_spikes.start.opponent",
        "side_condition",
        champions=(
            "Toxic spikes were scattered on the ground all around the opposing side!",
        ),
        priority=20,
        static={
            "condition": "toxic_spikes",
            "side": "opponent",
            "action": "start",
        },
    ),
    _t(
        "side.toxic_spikes.end.player",
        "side_condition",
        champions=(
            "The toxic spikes disappeared from the ground around your side!",
        ),
        priority=20,
        static={
            "condition": "toxic_spikes",
            "side": "player",
            "action": "end",
        },
    ),
    _t(
        "side.toxic_spikes.end.opponent",
        "side_condition",
        champions=(
            "The toxic spikes disappeared from the ground around the opposing side!",
        ),
        priority=20,
        static={
            "condition": "toxic_spikes",
            "side": "opponent",
            "action": "end",
        },
    ),
    # Stealth Rock — Champions "on [SIDE]", Showdown "around [SIDE] team"
    _t(
        "side.stealth_rocks.start.player",
        "side_condition",
        champions="Pointed stones float in the air on your side!",
        showdown="Pointed stones float in the air around your team!",
        priority=20,
        static={
            "condition": "stealth_rocks",
            "side": "player",
            "action": "start",
        },
    ),
    _t(
        "side.stealth_rocks.start.opponent",
        "side_condition",
        champions="Pointed stones float in the air on the opposing side!",
        showdown="Pointed stones float in the air around the opposing team!",
        priority=20,
        static={
            "condition": "stealth_rocks",
            "side": "opponent",
            "action": "start",
        },
    ),
    _t(
        "side.stealth_rocks.end.player",
        "side_condition",
        champions="The pointed stones disappeared from your side!",
        priority=20,
        static={
            "condition": "stealth_rocks",
            "side": "player",
            "action": "end",
        },
    ),
    _t(
        "side.stealth_rocks.end.opponent",
        "side_condition",
        champions="The pointed stones disappeared from the opposing side!",
        priority=20,
        static={
            "condition": "stealth_rocks",
            "side": "opponent",
            "action": "end",
        },
    ),
    # Sticky Web
    _t(
        "side.sticky_web.start.player",
        "side_condition",
        champions="A sticky web has been laid out on the ground on your side!",
        priority=20,
        static={"condition": "sticky_web", "side": "player", "action": "start"},
    ),
    _t(
        "side.sticky_web.start.opponent",
        "side_condition",
        champions=(
            "A sticky web has been laid out on the ground on the opposing side!",
        ),
        priority=20,
        static={
            "condition": "sticky_web",
            "side": "opponent",
            "action": "start",
        },
    ),
    _t(
        "side.sticky_web.end.player",
        "side_condition",
        champions=(
            "The sticky web has disappeared from the ground on your side!",
        ),
        priority=20,
        static={"condition": "sticky_web", "side": "player", "action": "end"},
    ),
    _t(
        "side.sticky_web.end.opponent",
        "side_condition",
        champions=(
            "The sticky web has disappeared from the ground on the opposing side!",
        ),
        priority=20,
        static={
            "condition": "sticky_web",
            "side": "opponent",
            "action": "end",
        },
    ),
)

_WEATHER: tuple[BattleTextTemplate, ...] = (
    _t(
        "weather.sun.start",
        "weather_start",
        champions="The sunlight turned harsh!",
        priority=20,
        static={"weather": "sunny"},
    ),
    _t(
        "weather.rain.start",
        "weather_start",
        champions="It started to rain!",
        priority=20,
        static={"weather": "rain"},
    ),
    _t(
        "weather.sand.start",
        "weather_start",
        champions="A sandstorm kicked up!",
        priority=20,
        static={"weather": "sandstorm"},
    ),
    _t(
        "weather.snow.start",
        "weather_start",
        champions="It started to snow!",
        priority=20,
        static={"weather": "snow"},
    ),
    _t(
        "weather.sun.end",
        "weather_end",
        champions="The harsh sunlight faded.",
        priority=20,
        static={"weather": "sunny"},
    ),
    _t(
        "weather.rain.end",
        "weather_end",
        champions="The rain stopped.",
        priority=20,
        static={"weather": "rain"},
    ),
    _t(
        "weather.sand.end",
        "weather_end",
        champions="The sandstorm subsided.",
        priority=20,
        static={"weather": "sandstorm"},
    ),
    _t(
        "weather.snow.end",
        "weather_end",
        champions="The snow stopped.",
        priority=20,
        static={"weather": "snow"},
    ),
)

_TERRAIN: tuple[BattleTextTemplate, ...] = (
    _t(
        "terrain.electric.start",
        "terrain_start",
        champions="An electric current ran across the battlefield!",
        priority=20,
        static={"terrain": "electric_terrain"},
    ),
    _t(
        "terrain.grassy.start",
        "terrain_start",
        champions="Grass grew to cover the battlefield!",
        priority=20,
        static={"terrain": "grassy_terrain"},
    ),
    _t(
        "terrain.misty.start",
        "terrain_start",
        champions="Mist swirled around the battlefield!",
        priority=20,
        static={"terrain": "misty_terrain"},
    ),
    _t(
        "terrain.psychic.start",
        "terrain_start",
        champions="The battlefield got weird!",
        priority=20,
        static={"terrain": "psychic_terrain"},
    ),
    _t(
        "terrain.electric.end",
        "terrain_end",
        champions="The electricity disappeared from the battlefield.",
        priority=20,
        static={"terrain": "electric_terrain"},
    ),
    _t(
        "terrain.grassy.end",
        "terrain_end",
        champions="The grass disappeared from the battlefield.",
        priority=20,
        static={"terrain": "grassy_terrain"},
    ),
    _t(
        "terrain.misty.end",
        "terrain_end",
        champions="The mist disappeared from the battlefield.",
        priority=20,
        static={"terrain": "misty_terrain"},
    ),
    _t(
        "terrain.psychic.end",
        "terrain_end",
        champions="The weirdness disappeared from the battlefield!",
        priority=20,
        static={"terrain": "psychic_terrain"},
    ),
)

_TRICK_ROOM: tuple[BattleTextTemplate, ...] = (
    _t(
        "trick_room.end",
        "trick_room_end",
        champions="The twisted dimensions returned to normal!",
        priority=20,
    ),
    _t(
        "trick_room.start",
        "trick_room_start",
        # Champions dump lacks a fixed start string; keep Showdown/OCR form.
        showdown="[POKEMON] twisted the dimensions!",
        tokens=("POKEMON",),
        fixed_anchor="twisted the dimensions",
        matcher="legacy",
        legacy_handler="trick_room",
        priority=60,
    ),
)

_LEGACY_FLOW: tuple[BattleTextTemplate, ...] = (
    _t(
        "legacy.mega",
        "mega_evolution",
        champions=(
            "[POKEMON]'s [ITEM] is reacting to [TRAINER]'s Omni Ring!",
            "[POKEMON] is reacting to [TRAINER]'s Omni Ring!",
        ),
        showdown=(
            "[POKEMON] has Mega Evolved into Mega [SPECIES]!",
        ),
        tokens=("POKEMON", "ITEM", "TRAINER", "SPECIES"),
        fixed_anchor="Omni Ring",
        matcher="legacy",
        legacy_handler="mega_evolution",
        priority=70,
    ),
    _t(
        "legacy.move_used",
        "move_used",
        champions="[POKEMON] used [MOVE]!",
        tokens=("POKEMON", "MOVE"),
        fixed_anchor=" used ",
        matcher="legacy",
        legacy_handler="move_used",
        priority=80,
    ),
    _t(
        "legacy.faint",
        "faint",
        champions="[POKEMON] fainted!",
        tokens=("POKEMON",),
        fixed_anchor="fainted",
        matcher="legacy",
        legacy_handler="faint",
        priority=70,
    ),
    _t(
        "legacy.switch",
        "switch",
        champions=(
            "Go! [POKEMON]!",
            "Go! [POKEMON] and [POKEMON]!",
            "[TRAINER] sent out [POKEMON]!",
            "[TRAINER] sent out [POKEMON] and [POKEMON]!",
            "[POKEMON], come back!",
            "[TRAINER] withdrew [POKEMON]!",
        ),
        showdown=(
            "[POKEMON] got dragged out!",
            "[POKEMON] went back to [TRAINER]!",
        ),
        tokens=("POKEMON", "TRAINER"),
        matcher="legacy",
        legacy_handler="switch",
        multi_event=True,
        priority=75,
    ),
    _t(
        "legacy.stat_change",
        "stat_change",
        champions="[POKEMON]'s [STAT] rose!",
        tokens=("POKEMON", "STAT", "ITEM"),
        matcher="legacy",
        legacy_handler="stat_change",
        multi_event=True,
        priority=85,
    ),
    _t(
        "legacy.status",
        "status",
        showdown=(
            "[POKEMON] was burned!",
            "[POKEMON] is paralyzed! It may be unable to move!",
            "[POKEMON] was poisoned!",
            "[POKEMON] was badly poisoned!",
            "[POKEMON] fell asleep!",
            "[POKEMON] was frozen solid!",
            "[POKEMON]'s burn was healed!",
            "[POKEMON] was cured of paralysis!",
            "[POKEMON] was cured of its poisoning!",
            "[POKEMON] woke up!",
            "[POKEMON] thawed out!",
        ),
        tokens=("POKEMON", "ITEM", "MOVE"),
        matcher="legacy",
        legacy_handler="status",
        priority=90,
    ),
    _t(
        "legacy.volatile",
        "volatile",
        showdown=(
            "[POKEMON] fell for the taunt!",
            "[POKEMON] must do an encore!",
            "[POKEMON] became confused!",
            "[POKEMON] became confused due to fatigue!",
            "[POKEMON] snapped out of its confusion!",
        ),
        tokens=("POKEMON", "ITEM"),
        matcher="legacy",
        legacy_handler="volatile",
        priority=90,
    ),
    _t(
        "legacy.side_banner",
        "side_banner",
        champions=("[POKEMON]'s [ABILITY]", "[POKEMON]'s [ITEM]"),
        tokens=("POKEMON", "ABILITY", "ITEM"),
        matcher="legacy",
        legacy_handler="side_banner",
        priority=50,
    ),
)

BATTLE_TEXT_TEMPLATES: tuple[BattleTextTemplate, ...] = (
    *_MOVE_OUTCOMES,
    *_FIELD_EFFECTS,
    *_PERISH_AND_LOCK,
    *_STAT_OPS,
    *_HELD_ITEMS,
    *_FAIL_AND_AVAIL,
    *_SIDE_CONDITIONS,
    *_WEATHER,
    *_TERRAIN,
    *_TRICK_ROOM,
    *_LEGACY_FLOW,
)


def catalog_templates(
    *,
    matcher: MatcherKind | None = None,
) -> tuple[BattleTextTemplate, ...]:
    """Return templates sorted by priority (ascending)."""
    items = BATTLE_TEXT_TEMPLATES
    if matcher is not None:
        items = tuple(t for t in items if t.matcher == matcher)
    return tuple(sorted(items, key=lambda t: (t.priority, t.id)))


def template_candidate_strings(template: BattleTextTemplate) -> tuple[str, ...]:
    """Normalized Champions + Showdown strings used for fixed matching."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in (*template.champions, *template.showdown):
        # Skip unresolved token slots for fixed matching.
        if "[" in raw and "]" in raw:
            continue
        normalized = normalize_catalog_text(raw)
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return tuple(out)
