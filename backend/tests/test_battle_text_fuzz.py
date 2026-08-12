"""Black-box fuzz for battle-text parsing.

Generates strings from the catalog, mutates arbitrary characters to random
printable ASCII (including move/item/ability token values), and asserts only
through ``parse_battle_text``. If noise breaks matching, that is a parser bug
— not a reason to narrow the mutator.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import Any

from app.cv.battle_text_catalog import (
    BattleTextTemplate,
    catalog_templates,
    normalize_catalog_text,
)
from app.cv.event_parser import parse_battle_text
from app.data.abilities import REGULATION_MB_ABILITIES
from app.data.items import REGULATION_MB_ITEMS
from app.data.learnsets import REGULATION_MB_LEARNSETS
from app.data.moves import REGULATION_MB_MOVES
from app.data.species import REGULATION_MB_SPECIES

N_ITER = 200
MUTATE_PERCENTAGE_MIN = 2.0
MUTATE_PERCENTAGE_MAX = 5.0
FUZZ_SEED = 20260812

PLAYER_TEAM_SIZE = 4
OPPONENT_TEAM_SIZE = 6

_TOKEN_RE = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")
_OPPOSING_PREFIX_RE = re.compile(r"^(?:the\s+)?opposing\s+", re.IGNORECASE)

_SPECIES_POOL = tuple(sorted(REGULATION_MB_SPECIES))
_ITEM_POOL = tuple(sorted(REGULATION_MB_ITEMS))
_ABILITY_POOL = tuple(sorted(REGULATION_MB_ABILITIES))
_ALL_MOVES = tuple(sorted(REGULATION_MB_MOVES))
_STAT_DISPLAY = (
    "Attack",
    "Defense",
    "Sp. Atk",
    "Sp. Def",
    "Speed",
    "accuracy",
    "evasion",
)
_STAT_TO_KEY = {
    "Attack": "atk",
    "Defense": "def",
    "Sp. Atk": "spa",
    "Sp. Def": "spd",
    "Speed": "spe",
    "accuracy": "accuracy",
    "evasion": "evasion",
}
_TRAINER_POOL = ("Blue", "Red", "Larry", "Iris", "Cynthia")
_TYPE_POOL = ("Fire", "Water", "Grass", "Electric", "Fairy", "Dragon")
_PRINTABLE_ASCII = "".join(chr(code) for code in range(0x20, 0x7F))


@dataclass(frozen=True)
class FuzzTeams:
    """Per-iteration species sets used for filling and for snap pools."""

    player: tuple[str, ...]
    opponent: tuple[str, ...]


def _form_family(species: str) -> str:
    """Collapse forme variants onto one family (Arcanine / Arcanine-Hisui)."""
    return species.split("-", 1)[0]


def _pick_team(rng: random.Random, size: int) -> tuple[str, ...]:
    """Sample ``size`` species with distinct form families."""
    chosen: list[str] = []
    used_families: set[str] = set()
    candidates = list(_SPECIES_POOL)
    rng.shuffle(candidates)
    for species in candidates:
        family = _form_family(species)
        if family in used_families:
            continue
        chosen.append(species)
        used_families.add(family)
        if len(chosen) >= size:
            break
    assert len(chosen) == size, f"could not pick {size} distinct-form species"
    return tuple(chosen)


def _init_teams(rng: random.Random) -> FuzzTeams:
    return FuzzTeams(
        player=_pick_team(rng, PLAYER_TEAM_SIZE),
        opponent=_pick_team(rng, OPPONENT_TEAM_SIZE),
    )


def _learnset_moves(species: str) -> tuple[str, ...]:
    moves = REGULATION_MB_LEARNSETS.get(species)
    if moves:
        return tuple(sorted(moves))
    return _ALL_MOVES


def _usable_patterns(template: BattleTextTemplate) -> list[str]:
    """Complete battle-line patterns (skip banner stubs and unresolved slots for fixed)."""
    if template.legacy_handler == "side_banner":
        return []
    patterns: list[str] = []
    for raw in (*template.champions, *template.showdown):
        if template.matcher == "fixed" and "[" in raw:
            continue
        if template.matcher != "fixed" and "[" not in raw:
            cleaned = normalize_catalog_text(raw)
            if cleaned.endswith(("!", ".", "...", "?")):
                patterns.append(raw)
            continue
        cleaned = normalize_catalog_text(raw)
        if cleaned.endswith(("!", ".", "...", "?")):
            patterns.append(raw)
    return patterns


def _fuzzable_templates() -> list[BattleTextTemplate]:
    """Every catalog template with a complete battle-line pattern.

    Includes fixed, legacy, and tokenized entries. Failures mean the parser did
    not recover the catalog event under printable-ASCII noise — fix the parser.
    """
    return [t for t in catalog_templates() if _usable_patterns(t)]


def _expected_event_type(template: BattleTextTemplate, pattern: str) -> str:
    """Map catalog event_kind (+ pattern) onto the BattleLogEvent.type discriminator."""
    kind = template.event_kind
    lowered = pattern.casefold()

    if kind == "switch":
        if " and [" in pattern or " and [POKEMON]" in pattern:
            if "go!" in lowered or "sent out" in lowered:
                return "lead_in"
        if "come back" in lowered or "withdrew" in lowered or "went back" in lowered:
            return "switch_out"
        return "switch_in"

    if kind == "status":
        if any(
            needle in lowered
            for needle in ("healed", "cured", "woke", "thawed", "defrosted")
        ):
            return "status_cured"
        return "status_applied"

    if kind == "volatile":
        if "snapped out" in lowered:
            return "volatile_cured"
        return "volatile_applied"

    return kind


def _strip_opposing(raw: str) -> str:
    return _OPPOSING_PREFIX_RE.sub("", raw).strip()


def _pick_species_fill(
    teams: FuzzTeams,
    rng: random.Random,
    *,
    side_hint: str | None,
    used_on_side: dict[str, set[str]],
) -> str:
    """Pick a team species. Free-side opponent fills use ``the opposing`` prefix.

    Switch/lead templates with ``static.side`` already encode side in the wording,
    so those fills stay bare species names from that side's team.
    """
    if side_hint == "player":
        side = "player"
        with_prefix = False
    elif side_hint == "opponent":
        side = "opponent"
        with_prefix = False
    else:
        side = "opponent" if rng.random() < 0.45 else "player"
        with_prefix = side == "opponent"

    pool = list(teams.player if side == "player" else teams.opponent)
    unused = [s for s in pool if s not in used_on_side[side]]
    species = rng.choice(unused or pool)
    used_on_side[side].add(species)
    if with_prefix:
        return f"the opposing {species}"
    return species


def _fill_pattern(
    pattern: str,
    template: BattleTextTemplate,
    teams: FuzzTeams,
    rng: random.Random,
) -> tuple[str, dict[str, list[str]]]:
    """Fill every ``[TOKEN]`` occurrence using the per-iteration teams."""
    pattern = (
        normalize_catalog_text(pattern)
        .replace("Pokémon", "Pokemon")
        .replace("é", "e")
    )
    side_hint = template.static.get("side")
    if isinstance(side_hint, str):
        side_hint_value: str | None = side_hint
    else:
        side_hint_value = None

    values: dict[str, list[str]] = {}
    used_on_side: dict[str, set[str]] = {"player": set(), "opponent": set()}
    # Owning Pokémon for subsequent [MOVE] tokens in the same pattern.
    owner_species: str | None = None

    def repl(match: re.Match[str]) -> str:
        nonlocal owner_species
        name = match.group(1)

        if name in {"POKEMON", "TARGET", "SOURCE", "SPECIES"}:
            value = _pick_species_fill(
                teams,
                rng,
                side_hint=side_hint_value,
                used_on_side=used_on_side,
            )
            if name == "POKEMON" and owner_species is None:
                owner_species = _strip_opposing(value)
        elif name == "MOVE":
            if owner_species is not None:
                move_pool = _learnset_moves(owner_species)
            else:
                move_pool = _ALL_MOVES
            value = rng.choice(move_pool)
        elif name == "ITEM":
            value = rng.choice(_ITEM_POOL)
        elif name == "ABILITY":
            value = rng.choice(_ABILITY_POOL)
        elif name == "STAT":
            value = rng.choice(_STAT_DISPLAY)
        elif name == "NUMBER":
            value = str(rng.randint(1, 5))
        elif name == "TRAINER":
            value = rng.choice(_TRAINER_POOL)
        elif name == "TYPE":
            value = rng.choice(_TYPE_POOL)
        elif name == "SIDE":
            value = rng.choice(("your", "the opposing"))
        else:
            value = "X"

        values.setdefault(name, []).append(value)
        return value

    filled = _TOKEN_RE.sub(repl, pattern)
    return filled, values


def _mutate_text(text: str, rng: random.Random) -> str:
    """Mutate ``MUTATE_PERCENTAGE_*`` of characters to random printable ASCII."""
    if not text:
        return text
    pct = rng.uniform(MUTATE_PERCENTAGE_MIN, MUTATE_PERCENTAGE_MAX)
    n_mut = max(1, int(round(len(text) * pct / 100.0)))
    n_mut = min(n_mut, len(text))
    chars = list(text)
    for pos in rng.sample(range(len(chars)), n_mut):
        replacement = rng.choice(_PRINTABLE_ASCII)
        while replacement == chars[pos] and len(_PRINTABLE_ASCII) > 1:
            replacement = rng.choice(_PRINTABLE_ASCII)
        chars[pos] = replacement
    return "".join(chars)


def _assert_static_fields(event: Any, static: dict[str, Any], *, ctx: str) -> None:
    for key, expected in static.items():
        if hasattr(event, key):
            assert getattr(event, key) == expected, (
                f"{ctx}: static {key}={getattr(event, key)!r} != {expected!r}"
            )
            continue
        if key == "side":
            nested = [
                getattr(event, attr, None)
                for attr in ("pokemon", "slot_1", "slot_2", "actor", "source")
            ]
            pokes = [p for p in nested if p is not None and hasattr(p, "side")]
            assert pokes, f"{ctx}: missing field side"
            assert all(p.side == expected for p in pokes), (
                f"{ctx}: nested side {[p.side for p in pokes]!r} != {expected!r}"
            )
            continue
        assert False, f"{ctx}: missing field {key}"


def _assert_token_fields(
    event: Any,
    values: dict[str, list[str]],
    *,
    ctx: str,
) -> None:
    found_species: list[str] = []
    for attr in ("pokemon", "actor", "target", "source", "slot_1", "slot_2"):
        poke = getattr(event, attr, None)
        if poke is not None and hasattr(poke, "species"):
            found_species.append(poke.species)

    for token in ("POKEMON", "TARGET", "SOURCE"):
        for raw in values.get(token, ()):
            species = _strip_opposing(raw)
            assert species in found_species, (
                f"{ctx}: expected species {species!r} in {found_species!r}"
            )

    for raw in values.get("MOVE", ()):
        if hasattr(event, "move") and getattr(event, "move"):
            assert event.move == raw, ctx
    for raw in values.get("ITEM", ()):
        if hasattr(event, "item") and getattr(event, "item"):
            assert event.item == raw, ctx
    for raw in values.get("ABILITY", ()):
        if hasattr(event, "ability") and getattr(event, "ability"):
            assert event.ability == raw, ctx
    for raw in values.get("NUMBER", ()):
        if hasattr(event, "count") and event.count is not None:
            assert event.count == int(raw), ctx
    for raw in values.get("STAT", ()):
        if hasattr(event, "stat") and getattr(event, "stat"):
            assert event.stat == _STAT_TO_KEY[raw], ctx


def test_battle_text_catalog_fuzz_black_box() -> None:
    rng = random.Random(FUZZ_SEED)
    templates = _fuzzable_templates()
    assert templates, "expected fuzzable catalog templates"

    logging.disable(logging.INFO)
    try:
        for i in range(N_ITER):
            teams = _init_teams(rng)
            template = rng.choice(templates)
            pattern = rng.choice(_usable_patterns(template))
            filled, values = _fill_pattern(pattern, template, teams, rng)
            mutated = _mutate_text(filled, rng)
            expected_type = _expected_event_type(template, pattern)

            # Keep the selected teams as the snap pools for this iteration.
            events = parse_battle_text(
                mutated,
                player_species=teams.player,
                opponent_species=teams.opponent,
            )
            ctx = (
                f"iter={i} id={template.id} matcher={template.matcher} "
                f"player={teams.player!r} opponent={teams.opponent!r} "
                f"pattern={pattern!r} filled={filled!r} mutated={mutated!r} "
                f"values={values!r}"
            )
            assert events, f"{ctx}: expected at least one event"
            event = events[0]
            assert event.type == expected_type, (
                f"{ctx}: type {event.type!r} != {expected_type!r}"
            )
            _assert_static_fields(event, dict(template.static), ctx=ctx)
            _assert_token_fields(event, values, ctx=ctx)
    finally:
        logging.disable(logging.NOTSET)
