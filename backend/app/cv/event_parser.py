"""Parse OCR text from side banners and battle text into BattleLogEvent objects."""

from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from typing import Iterable

from app.schema.battle_log import (
    AbilityTriggeredEvent,
    BattleLogEvent,
    FaintEvent,
    LeadInEvent,
    MegaEvolutionEvent,
    MoveFailedEvent,
    MoveUsedEvent,
    ItemUsedEvent,
    SideConditionEvent,
    StatChangeEvent,
    StatusAppliedEvent,
    SwitchInEvent,
    SwitchOutEvent,
    TerrainChangeEvent,
    TrickRoomChangeEvent,
    VolatileAppliedEvent,
    WeatherChangeEvent,
)
from app.data.abilities import REGULATION_MB_ABILITIES
from app.data.items import REGULATION_MB_ITEMS, is_regulation_mb_item
from app.data.moves import REGULATION_MB_MOVES
from app.schema.common import Pokemon, Side, Slot
from app.util.legal_snap import snap_to_legal

logger = logging.getLogger(__name__)

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

_OPPOSING = r"(?:the\s+opposing\s+)?"
_SIDE_PHRASE = r"(?:your|the\s+opposing|the\s+opponent['\u2019]?s)"

_SIDE_BANNER_RE = re.compile(
    r"^(?P<species>.+?)['\u2019$]s?\s+(?P<name>.+?)(?:\s*[=!.]+)?$",
    re.IGNORECASE,
)
_MEGA_EVOLUTION_RE = re.compile(
    _OPPOSING + r"(?P<species>.+?)\s+[a-zA-Z]+ite\s+(?P<mega_form>(X|Y|Z)?)\s*is\s+react",
    re.IGNORECASE,
)
_MOVE_USED_RE = re.compile(
    # Take the remainder of the line as the move; trailing "!" is often OCR'd as "l".
    _OPPOSING + r"(?P<species>.+?)\s+use[d]?\s+(?P<move>.+?)\s*$",
    re.IGNORECASE,
)
_FAINT_RE = re.compile(
    _OPPOSING + r"(?P<species>.+?)\s+faint",
    re.IGNORECASE,
)
_MOVE_FAILED_RE = re.compile(
    r"^But\s+it\s+failed\s*!?\s*$",
    re.IGNORECASE,
)
_SWITCH_RE = re.compile(
    r"(?:"
    # Dual lead switch-ins (must precede single switch-in patterns)
    # "Gol" is rewritten to "Go!" in normalize_ocr_text; require "Go!" so "Gotcha" is not a hit.
    r"Go!\s*(?P<player_dual_1>.+?)\s+and\s+(?P<player_dual_2>.+?)"
    r"|(?P<trainer_dual>.+?)\s+sent\s+out\s+(?P<opponent_dual_1>.+?)\s+and\s+(?P<opponent_dual_2>.+?)"
    # Switch-out (comma often OCR'd as ";")
    r"|(?P<player_out>.+?)[,;]\s*come\s+back"
    r"|(?P<trainer_out>.+?)\s+withdrew\s+(?P<opponent_out>.+?)"
    r"|" + _OPPOSING + r"(?P<self_out>.+?)\s+went\s+back\s+to\s+(?:.+?)"
    # Single switch-in
    r"|Go!\s*(?P<player_in>.+?)"
    r"|(?P<trainer_in>.+?)\s+sent\s+out\s+(?P<opponent_in>.+?)"
    r"|" + _OPPOSING + r"(?P<dragged>.+?)\s+got\s+dragged\s+out"
    # Do not treat bare "l" as a terminator — it steals the final letter of Grimmsnarl et al.
    # normalize_ocr_text converts trailing OCR "l" into "!".
    r")\s*[!.]?\s*$",
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
_STATUS_RE = re.compile(
    _OPPOSING + r"(?P<species>.+?)\s+"
    r"(?:"
    r"was\s+burned"
    r"|is\s+paralyzed,\s+so\s+it\s+may\s+be\s+unable\s+to\s+move"
    r"|was\s+badly\s+poisoned"
    r"|was\s+poisoned"
    r"|was\s+frozen\s+solid"
    r"|fell\s+asleep"
    r")",
    re.IGNORECASE,
)
_VOLATILE_RE = re.compile(
    _OPPOSING + r"(?P<species>.+?)\s+"
    r"(?:"
    r"fell\s+for\s+the\s+taunt"
    r"|must\s+do\s+an\s+encore"
    r"|became\s+confused"
    r")",
    re.IGNORECASE,
)
_SIDE_CONDITION_RE = re.compile(
    r"(?:"
    r"A\s+tailwind\s+started\s+blowing\s+on\s+" + _SIDE_PHRASE + r"\s+side"
    r"|Reflect\s+made\s+" + _SIDE_PHRASE + r"\s+side\s+stronger\s+against\s+physical\s+moves"
    r"|Light\s+Screen\s+made\s+" + _SIDE_PHRASE + r"\s+side\s+stronger\s+against\s+special\s+moves"
    r"|Aurora\s+Veil\s+made\s+" + _SIDE_PHRASE + r"\s+side\s+stronger\s+against\s+physical\s+and\s+special\s+moves"
    r"|Toxic\s+spikes\s+were\s+scattered\s+on\s+the\s+ground\s+all\s+around\s+" + _SIDE_PHRASE + r"\s+side"
    r"|Spikes\s+were\s+scattered\s+on\s+the\s+ground\s+all\s+around\s+" + _SIDE_PHRASE + r"\s+side"
    r"|Pointed\s+stones\s+float\s+in\s+the\s+air\s+around\s+" + _SIDE_PHRASE + r"\s+team"
    r")",
    re.IGNORECASE,
)


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR quirks before pattern matching."""
    cleaned = text.replace("\u2019", "'")
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


def _side_from_text(text: str) -> Side:
    lowered = text.lower()
    return "opponent" if ("opposing" in lowered or "opponent" in lowered) else "player"


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


def _snap_move(name: str) -> str:
    return snap_to_legal(name, REGULATION_MB_MOVES) or name.strip()


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


def _parse_weather(text: str) -> WeatherChangeEvent | None:
    lowered = text.lower()
    if "sunlight" in lowered:
        logger.info("Parsing weather")
        if "faded" in lowered:
            return WeatherChangeEvent(raw_text=text, weather="none")
        return WeatherChangeEvent(raw_text=text, weather="sunny")
    if re.search(r"\brain\b", lowered):
        logger.info("Parsing weather")
        if "stopped" in lowered:
            return WeatherChangeEvent(raw_text=text, weather="none")
        return WeatherChangeEvent(raw_text=text, weather="rain")
    if "sandstorm" in lowered or "picked up" in lowered:
        logger.info("Parsing weather")
        if "subsided" in lowered:
            return WeatherChangeEvent(raw_text=text, weather="none")
        return WeatherChangeEvent(raw_text=text, weather="sandstorm")
    if re.search(r"\bsnow\b", lowered):
        logger.info("Parsing weather")
        if "stopped" in lowered:
            return WeatherChangeEvent(raw_text=text, weather="none")
        return WeatherChangeEvent(raw_text=text, weather="snow")
    return None


def _parse_terrain(text: str) -> TerrainChangeEvent | None:
    lowered = text.lower()
    if "battlefield" in lowered:
        logger.info("Parsing terrain")
        if "electric" in lowered or "current" in lowered:
            return TerrainChangeEvent(raw_text=text, terrain="electric_terrain")
        if "grass" in lowered or "grew" in lowered:
            return TerrainChangeEvent(raw_text=text, terrain="grassy_terrain")
        if "mist" in lowered or "swirled" in lowered:
            return TerrainChangeEvent(raw_text=text, terrain="misty_terrain")
        if "weird" in lowered:
            return TerrainChangeEvent(raw_text=text, terrain="psychic_terrain")
        if "disappeared" in lowered:
            return TerrainChangeEvent(raw_text=text, terrain="none")
    return None


def _parse_trick_room(text: str) -> TrickRoomChangeEvent | None:
    lowered = text.lower()
    if "twisted" in lowered or "dimensions" in lowered:
        logger.info("Parsing trick room")
        if "returned" in lowered or "normal" in lowered:
            return TrickRoomChangeEvent(raw_text=text, active=False)
        return TrickRoomChangeEvent(raw_text=text, active=True)
    return None


def _parse_side_condition(text: str) -> SideConditionEvent | None:
    match = _SIDE_CONDITION_RE.search(text)
    if not match:
        return None
    logger.info("Parsing side condition")
    lowered = text.lower()
    side = _side_from_text(text)
    if "aurora veil" in lowered:
        condition = "aurora_veil"
    elif "light screen" in lowered:
        condition = "light_screen"
    elif "reflect" in lowered:
        condition = "reflect"
    elif "tailwind" in lowered or "blow" in lowered:
        condition = "tailwind"
    elif "toxic spikes" in lowered:
        condition = "toxic_spikes"
    elif "spikes" in lowered:
        condition = "spikes"
    elif "pointed stones" in lowered:
        condition = "stealth_rocks"
    else:
        return None
    return SideConditionEvent(raw_text=text, side=side, condition=condition)  # type: ignore[arg-type]


def _parse_status(text: str) -> StatusAppliedEvent | None:
    match = _STATUS_RE.search(text)
    if not match:
        return None
    logger.info("Parsing status")
    species = match.group("species").strip()
    lowered = text.lower()
    side = _side_from_text(text)
    if "burned" in lowered:
        status = "brn"
    elif "paraly" in lowered and ("may" in lowered or "unable" in lowered):
        status = "par"
    elif "badly poison" in lowered:
        status = "tox"
    elif "poisoned" in lowered:
        status = "psn"
    elif "frozen" in lowered and "was" in lowered:
        status = "frz"
    elif "sleep" in lowered and "fell" in lowered:
        status = "slp"
    else:
        return None
    return StatusAppliedEvent(
        raw_text=text,
        pokemon=_pokemon(species, side),
        status=status,  # type: ignore[arg-type]
    )


def _parse_volatile(text: str) -> VolatileAppliedEvent | None:
    match = _VOLATILE_RE.search(text)
    if not match:
        return None
    logger.info("Parsing volatile")
    species = match.group("species").strip()
    lowered = text.lower()
    side = _side_from_text(text)
    if "fell for the taunt" in lowered:
        volatile = "taunted"
    elif "must do an encore" in lowered:
        volatile = "encore"
    elif "became confused" in lowered:
        volatile = "confused"
    else:
        return None
    return VolatileAppliedEvent(
        raw_text=text,
        pokemon=_pokemon(species, side),
        volatile=volatile,  # type: ignore[arg-type]
    )


def _parse_switch(text: str) -> list[BattleLogEvent]:
    match = _SWITCH_RE.search(text)
    if not match:
        return []
    logger.info("Parsing switch")
    lowered = text.lower()
    if match.group("player_dual_1") is not None:
        return [
            LeadInEvent(
                raw_text=text,
                side="player",
                slot_1=_pokemon(match.group("player_dual_1"), "player", slot=1),
                slot_2=_pokemon(match.group("player_dual_2"), "player", slot=2),
            )
        ]
    if match.group("opponent_dual_1") is not None:
        return [
            LeadInEvent(
                raw_text=text,
                side="opponent",
                slot_1=_pokemon(match.group("opponent_dual_1"), "opponent", slot=1),
                slot_2=_pokemon(match.group("opponent_dual_2"), "opponent", slot=2),
            )
        ]
    if match.group("player_out") is not None or "come back" in lowered:
        return [
            SwitchOutEvent(
                raw_text=text,
                pokemon=_pokemon(match.group("player_out"), "player"),
            )
        ]
    if match.group("opponent_out") is not None or "withdrew" in lowered:
        return [
            SwitchOutEvent(
                raw_text=text,
                pokemon=_pokemon(match.group("opponent_out"), "opponent"),
            )
        ]
    if match.group("self_out") is not None or "went back to" in lowered:
        return [
            SwitchOutEvent(
                raw_text=text,
                pokemon=_pokemon(match.group("self_out"), _side_from_text(text)),
            )
        ]
    if match.group("player_in") is not None:
        return [
            SwitchInEvent(
                raw_text=text,
                pokemon=_pokemon(match.group("player_in"), "player"),
            )
        ]
    if match.group("opponent_in") is not None or "sent out" in lowered:
        return [
            SwitchInEvent(
                raw_text=text,
                pokemon=_pokemon(match.group("opponent_in"), "opponent"),
            )
        ]
    if match.group("dragged") is not None or "got dragged out" in lowered:
        return [
            SwitchInEvent(
                raw_text=text,
                pokemon=_pokemon(match.group("dragged"), _side_from_text(text)),
            )
        ]
    return []


def _parse_mega_evolution(text: str) -> MegaEvolutionEvent | None:
    match = _MEGA_EVOLUTION_RE.search(text)
    if not match:
        return None
    logger.info("Parsing mega evolution")
    species = match.group("species").strip()
    species = re.sub(r"['\u2019]s$", "", species).strip()
    mega_form = (match.group("mega_form") or "").strip().upper()
    variant = mega_form if mega_form in {"X", "Y", "Z"} else "regular"
    return MegaEvolutionEvent(
        raw_text=text,
        pokemon=_pokemon(species, _side_from_text(text)),
        variant=variant,  # type: ignore[arg-type]
    )


def _parse_move_used(text: str) -> MoveUsedEvent | None:
    match = _MOVE_USED_RE.search(text)
    if not match:
        return None
    logger.info("Parsing move used")
    # Trailing bang / OCR-as-l is already normalized to "!" before this runs.
    move = match.group("move").strip().rstrip("!.,").strip()
    if not move:
        return None
    return MoveUsedEvent(
        raw_text=text,
        actor=_pokemon(match.group("species"), _side_from_text(text)),
        move=_snap_move(move),
        targets=[],
    )


def _parse_faint(text: str) -> FaintEvent | None:
    match = _FAINT_RE.search(text)
    if not match:
        return None
    logger.info("Parsing faint")
    return FaintEvent(
        raw_text=text,
        pokemon=_pokemon(match.group("species"), _side_from_text(text)),
    )


def _parse_move_failed(text: str) -> MoveFailedEvent | None:
    if _MOVE_FAILED_RE.match(text) is None:
        return None
    logger.info("Parsing move failed")
    # Actor resolved later by most recent move used
    return MoveFailedEvent(raw_text=text)


def parse_battle_text(
    text: str,
    *,
    player_species: Iterable[str] | None = None,
    opponent_species: Iterable[str] | None = None,
) -> list[BattleLogEvent]:
    """Parse bottom battle-text OCR into one or more typed events."""
    logger.info("Parsing battle text: %s", text)
    normalized = normalize_ocr_text(text)
    logger.info("Normalized OCR text: %s", normalized)
    if not normalized:
        return []

    player_token = None
    opponent_token = None
    if player_species is not None:
        player_token = _PLAYER_SPECIES.set(tuple(player_species))
    if opponent_species is not None:
        opponent_token = _OPPONENT_SPECIES.set(tuple(opponent_species))
    try:
        events: list[BattleLogEvent] = []

        for multi_parser in (_parse_stat_changes, _parse_switch):
            events.extend(multi_parser(normalized))

        for single_parser in (
            _parse_mega_evolution,
            _parse_move_used,
            _parse_move_failed,
            _parse_faint,
            _parse_status,
            _parse_volatile,
            _parse_weather,
            _parse_terrain,
            _parse_trick_room,
            _parse_side_condition,
        ):
            event = single_parser(normalized)
            if event is not None:
                events.append(event)

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


def _event_dedupe_key(event: BattleLogEvent) -> tuple:
    key: list[object] = [event.type, event.raw_text]
    if isinstance(event, LeadInEvent):
        key.extend([event.side, event.slot_1.species, event.slot_2.species])
        return tuple(key)
    pokemon = getattr(event, "pokemon", None) or getattr(event, "actor", None)
    if pokemon is not None:
        key.extend([pokemon.species, pokemon.side, pokemon.slot])
    if event.type == "stat_change":
        key.extend([event.stat, event.stages_delta])
    if event.type in {"move_used", "move_failed"}:
        key.append(event.move)
    return tuple(key)


def _dedupe_events(events: Iterable[BattleLogEvent]) -> list[BattleLogEvent]:
    seen: set[tuple] = set()
    unique: list[BattleLogEvent] = []
    for event in events:
        key = _event_dedupe_key(event)
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return unique
