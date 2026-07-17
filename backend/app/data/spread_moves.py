"""Known doubles spread-move targeting used by the battle log completer.

Only unambiguous geometric targets are listed. Do not invent targets for
single-target moves from move name alone.
"""

from __future__ import annotations

from typing import Literal

# Hits every adjacent Pokemon except the user (both foes + ally).
ALL_ADJACENT_MOVES: frozenset[str] = frozenset(
    {
        "Earthquake",
        "Bulldoze",
        "Magnitude",
        "Surf",
        "Muddy Water",
        "Discharge",
        "Lava Plume",
        "Sludge Wave",
        "Petal Blizzard",
        "Parabolic Charge",
        "Brutal Swing",
        "Sparkling Aria",
        "Misty Explosion",
        "Explosion",
        "Self-Destruct",
        "Selfdestruct",
        "Teeter Dance",
    }
)

# Hits both adjacent opposing Pokemon only.
ALL_FOES_MOVES: frozenset[str] = frozenset(
    {
        "Rock Slide",
        "Icy Wind",
        "Dazzling Gleam",
        "Heat Wave",
        "Blizzard",
        "Hyper Voice",
        "Snarl",
        "Origin Pulse",
        "Precipice Blades",
        "Disarming Voice",
        "Electroweb",
        "Glaciate",
        "Eruption",
        "Water Spout",
        "Boomburst",
        "Struggle Bug",
        "Incinerate",
        "Frost Breath",
        "Diamond Storm",
        "Make It Rain",
        "Astral Barrage",
        "Bleakwind Storm",
        "Wildbolt Storm",
        "Sandsear Storm",
        "Springtide Storm",
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
