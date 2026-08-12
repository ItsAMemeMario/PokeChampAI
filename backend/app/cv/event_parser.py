"""Parse OCR text from side banners and battle text into BattleLogEvent objects."""

from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterable

from rapidfuzz import fuzz, process

from app.cv.battle_text_catalog import (
    BattleTextTemplate,
    catalog_templates,
    normalize_catalog_text,
    template_candidate_strings,
)
from app.cv.battle_text_matchers import (
    emit_tokenized_match,
    match_tokenized_catalog_template,
)
from app.schema.battle_log import (
    AbilityTriggeredEvent,
    BattleLogEvent,
    FieldEffectChangedEvent,
    HeldItemChangedEvent,
    MoveAvailabilityChangedEvent,
    MoveFailedEvent,
    MoveOutcomeEvent,
    ItemUsedEvent,
    PerishSongStartedEvent,
    SideConditionEvent,
    StatChangeEvent,
    StatStageOperationEvent,
    SwitchLockStartedEvent,
    TerrainEndEvent,
    TerrainStartEvent,
    TrickRoomStartEvent,
    TrickRoomEndEvent,
    WeatherEndEvent,
    WeatherStartEvent,
)
from app.data.abilities import REGULATION_MB_ABILITIES
from app.data.items import REGULATION_MB_ITEMS, is_regulation_mb_item
from app.schema.common import Pokemon, Side, Slot
from app.util.event_identity import semantic_key
from app.util.legal_snap import snap_to_legal

logger = logging.getLogger(__name__)

# Fixed-template RapidFuzz gates.
_FIXED_SCORE_CUTOFF = 88.0
# A small number of random substitutions can make neighboring full templates
# score within a couple of points.  Reject only true ties after positional
# rescoring instead of dropping a recoverable event.
_FIXED_AMBIGUITY_MARGIN = 0.25
_FIXED_WORD_RE = re.compile(r"[A-Za-z0-9]+")

_PLAYER_SPECIES: ContextVar[tuple[str, ...]] = ContextVar(
    "event_parser_player_species",
    default=(),
)
_OPPONENT_SPECIES: ContextVar[tuple[str, ...]] = ContextVar(
    "event_parser_opponent_species",
    default=(),
)

_STAT_ALIASES: dict[str, str] = {
    "attack": "atk",
    "atk": "atk",
    "defense": "def",
    "defence": "def",
    "def": "def",
    "sp. atk": "spa",
    "sp atk": "spa",
    "spa": "spa",
    "sp. def": "spd",
    "sp def": "spd",
    "spd": "spd",
    "speed": "spe",
    "spe": "spe",
    "accuracy": "accuracy",
    "evasion": "evasion",
}

_SIDE_BANNER_RE = re.compile(
    r"^(?P<species>.+?)['\u2019$]s?\s+(?P<name>.+?)(?:\s*[=!.]+)?$",
    re.IGNORECASE,
)
_STAT_TAIL_RE = re.compile(
    r"(?P<source>.+?)['\u2019]\s*s\s+(?P<clause>.+)$",
    re.IGNORECASE | re.DOTALL,
)
# Fallback when OCR drops the possessive apostrophe.
_STAT_TAIL_NO_APOSTROPHE_RE = re.compile(
    r"(?P<source>.+?)\s+"
    r"(?P<clause>"
    r"(?:(?:attack|defense|defence|sp\.?\s*atk|sp\.?\s*def|speed|accuracy|evasion|atk|def|spa|spd|spe)"
    r"(?:\s*,\s*|\s+and\s+)?)+"
    r"\s*(?:(?:sharply|harshly|severely|drastically)\s+)?"
    r"(?:fell|rose|fall|\{?ell+|tell+|ros\w*)"
    r"(?:\s+(?:sharply|harshly|severely|drastically))?"
    r".*)",
    re.IGNORECASE | re.DOTALL,
)
_STAT_NAME_RE = re.compile(
    r"attack|defense|defence|sp\.?\s*atk|sp\.?\s*def|speed|accuracy|evasion|atk|def|spa|spd|spe",
    re.IGNORECASE,
)
_STAT_DIRECTION_RE = re.compile(
    r"(?:(?:sharply|harshly|severely|drastically)\s+)?"
    # "fell" is often OCR'd as "tell", "{ell", or similar.
    r"(?:fell|rose|fall|\{?ell+|tell+|ros\w*)"
    r"(?:\s+(?:sharply|harshly|severely|drastically))?",
    re.IGNORECASE,
)
_OPPONENT_PREFIX_RE = re.compile(r"^(?:the\s+)?opposing\s+", re.IGNORECASE)


@dataclass(frozen=True)
class _FixedMatch:
    template: BattleTextTemplate
    score: float
    matched_text: str


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR quirks before pattern matching."""
    cleaned = normalize_catalog_text(text)
    # EasyOCR often mangles "'s" as "'$", "' $", "' s", "' 5", or bare "$".
    cleaned = re.sub(r"['\u2019]\s*[\$5sS]\s+", "'s ", cleaned)
    cleaned = re.sub(r"['\u2019]\s*\$", "'s", cleaned)
    cleaned = re.sub(r"['\u2019]\s+s\b", "'s", cleaned)
    cleaned = re.sub(r"(\w)\s+\$\s+", r"\1's ", cleaned)
    # "Go!" is often OCR'd as "Gol".
    cleaned = re.sub(r"\bGol\b", "Go!", cleaned, flags=re.IGNORECASE)
    # "opposing" / "the opposing" frequently OCR-glued or misspelled.
    cleaned = re.sub(r"(?i)\bthe\s*[oq0w]?pposing\b", "The opposing", cleaned)
    cleaned = re.sub(r"(?i)\b[oq0w]pposing\b", "opposing", cleaned)
    cleaned = re.sub(r"(?i)\bsent\s*out\b", "sent out", cleaned)
    cleaned = re.sub(r"(?i)(\w)sent out\b", r"\1 sent out", cleaned)
    # Dual leads: "and" is often OCR'd as "ard".
    cleaned = re.sub(r"(?i)\s+ard\s+", " and ", cleaned)
    # Trailing "!" OCR'd as l/i/I/1 (e.g. "Swaggerl", "Charizardl").
    cleaned = re.sub(r"([A-Za-z])[lIi1]\s*$", r"\1!", cleaned)
    cleaned = cleaned.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _fixed_template_index() -> list[tuple[str, BattleTextTemplate]]:
    """Build non-token catalog candidates for whole-message fuzzy matching."""
    pairs: list[tuple[str, BattleTextTemplate]] = []
    for template in catalog_templates():
        for candidate in template_candidate_strings(template):
            pairs.append((candidate, template))
    return pairs


_FIXED_INDEX: list[tuple[str, BattleTextTemplate]] | None = None


def _get_fixed_index() -> list[tuple[str, BattleTextTemplate]]:
    global _FIXED_INDEX
    if _FIXED_INDEX is None:
        _FIXED_INDEX = _fixed_template_index()
    return _FIXED_INDEX


def match_fixed_catalog_template(normalized: str) -> _FixedMatch | None:
    """Best unique fixed-template match, or None if weak/ambiguous/no hit."""
    if not normalized:
        return None

    index = _get_fixed_index()
    if not index:
        return None

    choices = [text for text, _ in index]
    # Exact case-insensitive win.
    lowered = normalized.casefold()
    for text, template in index:
        if text.casefold() == lowered:
            return _FixedMatch(template=template, score=100.0, matched_text=text)

    raw_matches = process.extract(
        normalized,
        choices,
        scorer=fuzz.WRatio,
        score_cutoff=_FIXED_SCORE_CUTOFF,
        limit=10,
    )
    if not raw_matches:
        return None

    # Whole-message ratio alone makes an omitted descriptive word look close
    # to a single OCR substitution (for example "spikes" vs "toxic spikes").
    # Re-score in word positions; this is still full-template matching, not an
    # anchor lookup, and keeps noisy fixed messages distinguishable.
    matches = sorted(
        (
            (
                text,
                _fixed_positional_score(normalized, text, base_score=float(score)),
                index,
            )
            for text, score, index in raw_matches
        ),
        key=lambda match: (-match[1], match[2]),
    )
    best_text, best_score, best_idx = matches[0]
    best_template = index[best_idx][1]

    for other_text, other_score, other_idx in matches[1:]:
        other_template = index[other_idx][1]
        if other_template.id == best_template.id:
            continue
        if best_score - other_score <= _FIXED_AMBIGUITY_MARGIN:
            logger.info(
                "Ambiguous fixed catalog match for %r: %s (%.1f) vs %s (%.1f)",
                normalized,
                best_template.id,
                best_score,
                other_template.id,
                other_score,
            )
            return None

    return _FixedMatch(
        template=best_template,
        score=float(best_score),
        matched_text=best_text,
    )


def _fixed_positional_score(text: str, candidate: str, *, base_score: float) -> float:
    """Penalize long candidate words that are absent from the OCR line.

    Keep the primary score as a whole-message ratio. Character substitutions
    can split or join words, making strict word-by-word alignment unreliable;
    absence of a descriptive word (``toxic`` versus ``spikes``), however, is a
    reliable tie-breaker.
    """
    penalty = 0.0
    lowered = text.casefold()
    for word in _FIXED_WORD_RE.findall(candidate.casefold()):
        if len(word) < 4:
            continue
        word_score = fuzz.partial_ratio(word, lowered)
        if word_score < 55.0:
            penalty += (55.0 - word_score) * 0.35
    return base_score - penalty


def _emit_fixed_template(
    raw_text: str,
    match: _FixedMatch,
) -> list[BattleLogEvent]:
    """Build BattleLogEvent(s) from a fixed catalog hit."""
    template = match.template
    static: dict[str, Any] = dict(template.static)
    kind = template.event_kind

    if kind == "move_outcome":
        return [
            MoveOutcomeEvent(
                raw_text=raw_text,
                outcome=static["outcome"],
            )
        ]
    if kind == "field_effect_changed":
        return [
            FieldEffectChangedEvent(
                raw_text=raw_text,
                effect=static["effect"],
                action=static["action"],
            )
        ]
    if kind == "perish_song_started":
        return [
            PerishSongStartedEvent(
                raw_text=raw_text,
                turns_remaining=static.get("turns_remaining", 3),
            )
        ]
    if kind == "switch_lock_started":
        return [
            SwitchLockStartedEvent(
                raw_text=raw_text,
                scope=static.get("scope", "all_active"),
            )
        ]
    if kind == "stat_stage_operation":
        return [
            StatStageOperationEvent(
                raw_text=raw_text,
                operation=static["operation"],
            )
        ]
    if kind == "held_item_changed":
        return [
            HeldItemChangedEvent(
                raw_text=raw_text,
                change=static["change"],
            )
        ]
    if kind == "move_failed":
        return [
            MoveFailedEvent(
                raw_text=raw_text,
                reason=static.get("reason", "failed"),
            )
        ]
    if kind == "move_availability_changed":
        return [
            MoveAvailabilityChangedEvent(
                raw_text=raw_text,
                restriction=static["restriction"],
                clears_on_switch=static.get("clears_on_switch"),
            )
        ]
    if kind == "side_condition":
        return [
            SideConditionEvent(
                raw_text=raw_text,
                side=static["side"],
                condition=static["condition"],
                action=static.get("action", "start"),
            )
        ]
    if kind == "weather_start":
        return [WeatherStartEvent(raw_text=raw_text, weather=static["weather"])]
    if kind == "weather_end":
        return [WeatherEndEvent(raw_text=raw_text, weather=static["weather"])]
    if kind == "terrain_start":
        return [TerrainStartEvent(raw_text=raw_text, terrain=static["terrain"])]
    if kind == "terrain_end":
        return [TerrainEndEvent(raw_text=raw_text, terrain=static["terrain"])]
    if kind == "trick_room_end":
        return [TrickRoomEndEvent(raw_text=raw_text)]
    if kind == "trick_room_start":
        return [TrickRoomStartEvent(raw_text=raw_text)]

    logger.warning("No emitter for fixed catalog kind %s (%s)", kind, template.id)
    return []


def _clean_species_raw(species: str) -> str:
    species = species.strip().rstrip("!.,").strip()
    if species.lower().startswith("the opposing "):
        species = species[len("the opposing ") :].strip()
    if species.lower().startswith("opposing "):
        species = species[len("opposing ") :].strip()
    return species


def _side_known_species(
    side: Side,
    *,
    player_species: Iterable[str] | None = None,
    opponent_species: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return the side-specific known list used to resolve OCR species names."""
    if side == "player":
        if player_species is not None:
            return tuple(player_species)
        return _PLAYER_SPECIES.get()
    if opponent_species is not None:
        return tuple(opponent_species)
    return _OPPONENT_SPECIES.get()


def _pokemon(
    species: str,
    side: Side,
    *,
    slot: Slot = 1,
    player_species: Iterable[str] | None = None,
    opponent_species: Iterable[str] | None = None,
) -> Pokemon:
    cleaned = _clean_species_raw(species)
    known = _side_known_species(
        side,
        player_species=player_species,
        opponent_species=opponent_species,
    )
    snapped = snap_to_legal(cleaned, known) or cleaned
    return Pokemon(species=snapped, side=side, slot=slot)


def _snap_item(name: str) -> str:
    return snap_to_legal(name, REGULATION_MB_ITEMS) or name.strip()


def _snap_ability(name: str) -> str:
    return snap_to_legal(name, REGULATION_MB_ABILITIES) or name.strip()


def is_known_item(name: str) -> bool:
    """Return True if ``name`` is a Regulation M-B legal held item (after snap)."""
    snapped = snap_to_legal(name, REGULATION_MB_ITEMS)
    if snapped is not None:
        return True
    return is_regulation_mb_item(name)


def parse_side_banner(
    text: str,
    side: Side,
    *,
    slot: Slot = 1,
    effect_text: str = "",
    player_species: Iterable[str] | None = None,
    opponent_species: Iterable[str] | None = None,
) -> BattleLogEvent | None:
    """Parse a per-slot banner OCR string into an ability or item event.

    Ability vs item is decided by name lookup against Regulation M-B legal items.
    """
    normalized = normalize_ocr_text(text)
    match = _SIDE_BANNER_RE.match(normalized)
    if not match:
        return None

    logger.info("Parsing side banner")
    species = match.group("species").strip()
    name = match.group("name").strip()
    pokemon = _pokemon(
        species,
        side,
        slot=slot,
        player_species=player_species,
        opponent_species=opponent_species,
    )

    if is_known_item(name):
        return ItemUsedEvent(
            raw_text=text,
            pokemon=pokemon,
            item=_snap_item(name),
        )

    return AbilityTriggeredEvent(
        raw_text=text,
        actor=pokemon,
        ability=_snap_ability(name),
        effect_text=effect_text,
    )


def _stat_delta(direction: str) -> int:
    lowered = direction.lower()
    if (
        "fell" in lowered
        or "fall" in lowered
        or "tell" in lowered
        or "ell" in lowered
    ):
        if "severely" in lowered:
            return -3
        if "harshly" in lowered or "sharply" in lowered:
            return -2
        return -1
    if "rose" in lowered or "ros" in lowered:
        if "drastically" in lowered:
            return 3
        if "sharply" in lowered:
            return 2
        return 1
    return 0


def _resolve_stat_key(stat_raw: str) -> str | None:
    normalized = re.sub(r"\s+", " ", stat_raw.strip().lower())
    if normalized in _STAT_ALIASES:
        return _STAT_ALIASES[normalized]
    for alias, canonical in _STAT_ALIASES.items():
        if alias in normalized:
            return canonical
    return None


def _parse_stat_subjects(source: str) -> list[tuple[str, Side]]:
    """Split the subject side of '...\'s Attack fell!' into (species, side) pairs."""
    subjects: list[tuple[str, Side]] = []
    for fragment in re.split(r"\s+and\s+", source, flags=re.IGNORECASE):
        fragment = fragment.strip().rstrip(",").strip()
        if not fragment:
            continue
        side: Side = "opponent" if _OPPONENT_PREFIX_RE.search(fragment) else "player"
        species = _OPPONENT_PREFIX_RE.sub("", fragment).strip()
        if species:
            subjects.append((species, side))
    return subjects


def _parse_stat_clause(clause: str) -> list[tuple[str, int]]:
    """Parse 'Attack, Sp. Atk, and Speed rose sharply!' into (stat, delta) pairs."""
    direction_match = None
    for match in _STAT_DIRECTION_RE.finditer(clause):
        direction_match = match
    if direction_match is None:
        return []

    delta = _stat_delta(direction_match.group(0))
    if delta == 0:
        return []

    stats_region = clause[: direction_match.start()]
    results: list[tuple[str, int]] = []
    seen: set[str] = set()
    for match in _STAT_NAME_RE.finditer(stats_region):
        stat_key = _resolve_stat_key(match.group(0))
        if stat_key is None or stat_key in seen:
            continue
        seen.add(stat_key)
        results.append((stat_key, delta))
    return results


def _parse_stat_changes(text: str) -> list[StatChangeEvent]:
    """Parse player/opponent single and dual subjects with one or more stats.

    Partition on possessive ``'s``: names before, stat changes after.
    """
    match = _STAT_TAIL_RE.search(text) or _STAT_TAIL_NO_APOSTROPHE_RE.search(text)
    if match is None:
        return []

    logger.info("Parsing stat changes")
    subjects = _parse_stat_subjects(match.group("source"))
    clause_stats = _parse_stat_clause(match.group("clause"))
    if not subjects or not clause_stats:
        return []

    events: list[StatChangeEvent] = []
    for index, (species, side) in enumerate(subjects):
        slot: Slot = 1 if index == 0 else 2
        pokemon = _pokemon(species, side, slot=slot)
        for stat_key, delta in clause_stats:
            events.append(
                StatChangeEvent(
                    raw_text=text,
                    pokemon=pokemon,
                    stat=stat_key,  # type: ignore[arg-type]
                    stages_delta=delta,
                )
            )
    return events


def parse_battle_text(
    text: str,
    *,
    player_species: Iterable[str] | None = None,
    opponent_species: Iterable[str] | None = None,
) -> list[BattleLogEvent]:
    """Parse bottom battle-text OCR into one or more typed events.

    Dispatch order:
    1. Whole-message matching for patterns with no dynamic tokens.
    2. Token-first positional fuzzy matching for every variable template.
    3. The multi-subject stat-clause parser.
    """
    logger.info("Parsing battle text: %s", text)
    normalized = normalize_ocr_text(text)
    logger.info("Normalized OCR text: %s", normalized)
    if not normalized:
        return []

    player_token = None
    opponent_token = None
    player_snapshot: tuple[str, ...] | None = None
    opponent_snapshot: tuple[str, ...] | None = None
    if player_species is not None:
        player_snapshot = tuple(player_species)
        player_token = _PLAYER_SPECIES.set(player_snapshot)
    if opponent_species is not None:
        opponent_snapshot = tuple(opponent_species)
        opponent_token = _OPPONENT_SPECIES.set(opponent_snapshot)
    try:
        fixed = match_fixed_catalog_template(normalized)
        if fixed is not None:
            logger.info(
                "Fixed catalog match: %s (score=%.1f)",
                fixed.template.id,
                fixed.score,
            )
            return _dedupe_events(_emit_fixed_template(text, fixed))

        tokenized = match_tokenized_catalog_template(
            normalized,
            player_species=player_snapshot,
            opponent_species=opponent_snapshot,
        )
        if tokenized is not None:
            logger.info(
                "Token-first catalog match: %s (score=%.1f)",
                tokenized.template.id,
                tokenized.score,
            )
            emitted = emit_tokenized_match(text, tokenized)
            if emitted:
                return _dedupe_events(emitted)

        # One Champions line can describe multiple subjects and/or multiple
        # stats. It has no one-to-one token layout, so retain this specialized
        # multi-event parser after generic template matching.
        events: list[BattleLogEvent] = _parse_stat_changes(normalized)

        logger.info(
            "Parsing complete, added events: %s",
            [getattr(e, "type", type(e).__name__) for e in events],
        )
        return _dedupe_events(events)
    finally:
        if player_token is not None:
            _PLAYER_SPECIES.reset(player_token)
        if opponent_token is not None:
            _OPPONENT_SPECIES.reset(opponent_token)


def _dedupe_events(events: Iterable[BattleLogEvent]) -> list[BattleLogEvent]:
    seen: set[tuple] = set()
    unique: list[BattleLogEvent] = []
    for event in events:
        key = semantic_key(event, include_raw_text=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return unique
