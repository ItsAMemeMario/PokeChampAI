"""Regulation M-B doubles spread-move targeting for the battle log completer.

Only unambiguous geometric targets are listed, and only moves legal in
Pokemon Champions Regulation M-B (MetaVGC allowed-move pool).

Source: https://metavgc.com/guides/pokemon-champions-regulation-m-b-legal-pokemon-items-moves

Do not invent targets for single-target moves from move name alone.
"""

from __future__ import annotations

from typing import Literal

# Hits every adjacent Pokemon except the user (both foes + ally).
ALL_ADJACENT_MOVES: frozenset[str] = frozenset(
    {
        "Boomburst",
        "Brutal Swing",
        "Bulldoze",
        "Discharge",
        "Earthquake",
        "Explosion",
        "Lava Plume",
        "Misty Explosion",
        "Muddy Water",
        "Parabolic Charge",
        "Petal Blizzard",
        "Self-Destruct",
        "Sludge Wave",
        "Sparkling Aria",
        "Surf",
        "Teeter Dance",
    }
)

# Hits both adjacent opposing Pokemon only.
ALL_FOES_MOVES: frozenset[str] = frozenset(
    {
        "Air Cutter",
        "Blizzard",
        "Breaking Swipe",
        "Burning Jealousy",
        "Clanging Scales",
        "Dazzling Gleam",
        "Electroweb",
        "Eruption",
        "Heat Wave",
        "Hyper Voice",
        "Icy Wind",
        "Make It Rain",
        "Matcha Gotcha",
        "Mortal Spin",
        "Rock Slide",
        "Snarl",
        "Struggle Bug",
        "Water Spout",
    }
)

SpreadKind = Literal["all_adjacent", "all_foes"]


def normalize_move_name(move: str) -> str:
    """Canonicalize OCR / display move names for lookup."""
    return " ".join(move.strip().split())


def spread_kind(move: str) -> SpreadKind | None:
    """Return spread geometry for ``move``, or None if not a known spread move."""
    name = normalize_move_name(move)
    # Title-case lookup with a case-insensitive fallback.
    if name in ALL_ADJACENT_MOVES:
        return "all_adjacent"
    if name in ALL_FOES_MOVES:
        return "all_foes"
    lowered = {m.lower(): m for m in ALL_ADJACENT_MOVES | ALL_FOES_MOVES}
    canonical = lowered.get(name.lower())
    if canonical is None:
        return None
    if canonical in ALL_ADJACENT_MOVES:
        return "all_adjacent"
    return "all_foes"
