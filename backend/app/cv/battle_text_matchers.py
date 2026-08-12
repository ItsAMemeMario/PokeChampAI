"""Token-first, positional fuzzy matching for battle-text templates.

The matcher intentionally does not use ``BattleTextTemplate.fixed_anchor``.
It first localizes legal dynamic values (Pokemon, moves, items, and so on)
inside the OCR line.  It then scores every literal segment only in the gap
before, between, or after those token spans.  This keeps short literals such
as ``used`` from becoming global anchors while still tolerating noise anywhere
in a message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

from rapidfuzz import fuzz, process

from app.cv.battle_text_catalog import (
    BattleTextTemplate,
    catalog_templates,
    normalize_catalog_text,
)
from app.data.abilities import REGULATION_MB_ABILITIES
from app.data.items import REGULATION_MB_ITEMS
from app.data.learnsets import REGULATION_MB_LEARNSETS
from app.data.moves import REGULATION_MB_MOVES
from app.data.species import REGULATION_MB_SPECIES
from app.schema.battle_log import (
    BattleLogEvent,
    FaintEvent,
    FieldEffectChangedEvent,
    HeldItemChangedEvent,
    LeadInEvent,
    MegaEvolutionEvent,
    MoveAvailabilityChangedEvent,
    MoveFailedEvent,
    MoveOutcomeEvent,
    MoveUsedEvent,
    PerishSongStartedEvent,
    StatChangeEvent,
    StatStageOperationEvent,
    StatusAppliedEvent,
    StatusCuredEvent,
    SwitchInEvent,
    SwitchLockStartedEvent,
    SwitchOutEvent,
    TrickRoomStartEvent,
    VolatileAppliedEvent,
    VolatileCuredEvent,
)
from app.schema.common import Pokemon, Side, Slot


_TOKEN_RE = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")

_MAX_ENTITY_CANDIDATES = 10
_MAX_PATTERN_STATES = 1_200
_TOKENIZED_SCORE_CUTOFF = 68.0
# Positional literal scores make different templates comparable. A one-character
# OCR mutation can legitimately make close variants differ by roughly one
# point, so reject only an effectively exact tie.
_AMBIGUITY_MARGIN = 0.25

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

_STAT_DISPLAY_NAMES: dict[str, str] = {
    "atk": "Attack",
    "def": "Defense",
    "spa": "Sp. Atk",
    "spd": "Sp. Def",
    "spe": "Speed",
    "accuracy": "Accuracy",
    "evasion": "Evasion",
}

_TYPE_NAMES = (
    "Normal",
    "Fire",
    "Water",
    "Electric",
    "Grass",
    "Ice",
    "Fighting",
    "Poison",
    "Ground",
    "Flying",
    "Psychic",
    "Bug",
    "Rock",
    "Ghost",
    "Dragon",
    "Dark",
    "Steel",
    "Fairy",
)


@dataclass(frozen=True)
class TokenSpan:
    """One legal token value localized within normalized OCR text."""

    name: str
    value: str | None
    start: int
    end: int
    score: float
    side: Side | None = None


@dataclass(frozen=True)
class TokenizedMatch:
    """A template and its ordered, snapped token spans."""

    template: BattleTextTemplate
    pattern: str
    spans: tuple[TokenSpan, ...]
    score: float

    @property
    def tokens(self) -> Mapping[str, tuple[str | None, ...]]:
        values: dict[str, list[str | None]] = {}
        for span in self.spans:
            values.setdefault(span.name, []).append(span.value)
        return {name: tuple(values_) for name, values_ in values.items()}


def render_tokenized_template(match: TokenizedMatch) -> str:
    """Render the matched catalog pattern with snapped canonical token values.

    ``raw_text`` is an OCR artifact.  Once a template match is accepted, the
    event log should instead retain the catalog sentence with its resolved
    Pokémon, move, item, and other dynamic values.  This makes downstream
    completion, logs, and Gemini context deterministic without inventing
    values for unresolved tokens.
    """
    pattern = normalize_catalog_text(match.pattern)
    spans = iter(match.spans)

    def replace(token_match: re.Match[str]) -> str:
        token_name = token_match.group(1)
        span = next(spans, None)
        if span is None or span.name != token_name:
            # A malformed match must not silently substitute an unrelated
            # value. This is defensive only: `_match_pattern` builds spans in
            # the same order as token occurrences.
            return token_match.group(0)
        return _render_token_value(
            pattern,
            token_name,
            span,
            token_offset=token_match.start(),
        )

    return normalize_catalog_text(_TOKEN_RE.sub(replace, pattern))


def _render_token_value(
    pattern: str,
    token_name: str,
    span: TokenSpan,
    *,
    token_offset: int,
) -> str:
    value = span.value
    if value is None:
        # The parser may emit an event with an unrecoverable number. Preserve
        # that uncertainty explicitly rather than retaining malformed OCR or
        # fabricating a value.
        return "?"

    if token_name == "STAT":
        return _STAT_DISPLAY_NAMES.get(value, value)

    if token_name in {"POKEMON", "TARGET", "SOURCE"}:
        if span.side == "opponent" and _needs_opposing_prefix(pattern, token_name):
            article = "The" if _token_starts_sentence(pattern, token_offset) else "the"
            return f"{article} opposing {value}"
    return value


def _token_starts_sentence(pattern: str, token_offset: int) -> bool:
    before = pattern[:token_offset].rstrip()
    return not before or before[-1] in ".!?"


def _needs_opposing_prefix(pattern: str, token_name: str) -> bool:
    """Whether a template itself does not already identify the opponent side."""
    lowered = pattern.casefold()
    if token_name != "POKEMON":
        return True
    # The trainer grammar already establishes that the named Pokémon is on
    # the opponent's side. Adding a prefix would produce "sent out The
    # opposing ..." rather than a canonical battle message.
    if "sent out [pokemon]" in lowered or "withdrew [pokemon]" in lowered:
        return False
    # "Go!" always describes the player side.
    return "go! [pokemon]" not in lowered


def match_tokenized_catalog_template(
    normalized: str,
    *,
    player_species: Iterable[str] | None = None,
    opponent_species: Iterable[str] | None = None,
    include_ids: tuple[str, ...] | None = None,
    exclude_id_prefixes: tuple[str, ...] = ("banner.",),
) -> TokenizedMatch | None:
    """Return the strongest non-fixed catalog match for ``normalized``.

    Matching is token-first.  Legal token candidates are localized before
    fixed text is considered, and every fixed segment is compared only in the
    region bounded by its surrounding token spans.
    """
    if not normalized:
        return None

    locator = _TokenLocator(
        normalized,
        player_species=player_species,
        opponent_species=opponent_species,
    )
    candidates: list[TokenizedMatch] = []
    for template in catalog_templates():
        if include_ids is not None and template.id not in include_ids:
            continue
        if any(template.id.startswith(prefix) for prefix in exclude_id_prefixes):
            continue
        for pattern in _template_patterns_with_tokens(template):
            match = _match_pattern(locator, template, pattern)
            if match is not None:
                candidates.append(match)

    if not candidates:
        return None

    candidates.sort(key=lambda match: (-match.score, match.template.priority, match.template.id))
    best = candidates[0]
    for other in candidates[1:]:
        if other.template.id == best.template.id:
            continue
        if best.score - other.score <= _AMBIGUITY_MARGIN:
            return None
        break
    return best


def emit_tokenized_match(raw_text: str, match: TokenizedMatch) -> list[BattleLogEvent]:
    """Construct typed events from a successful tokenized catalog match."""
    template = match.template
    static = dict(template.static)
    tokens = _spans_by_name(match.spans)
    kind = template.event_kind
    canonical_text = render_tokenized_template(match)

    pokemon = _pokemon_from_span(_first(tokens, "POKEMON"))
    source = _pokemon_from_span(_first(tokens, "SOURCE"))
    target = _pokemon_from_span(_first(tokens, "TARGET"))
    species = _pokemon_from_span(_first(tokens, "SPECIES"))
    move = _value(_first(tokens, "MOVE"))
    item = _value(_first(tokens, "ITEM"))

    if kind == "move_outcome":
        count = _int_value(_first(tokens, "NUMBER"))
        return [
            MoveOutcomeEvent(
                raw_text=canonical_text,
                outcome=static["outcome"],
                target=pokemon or target,
                count=count,
            )
        ]

    if kind == "stat_stage_operation":
        return [
            StatStageOperationEvent(
                raw_text=canonical_text,
                operation=static["operation"],
                pokemon=pokemon,
                target=target,
            )
        ]

    if kind == "held_item_changed":
        return [
            HeldItemChangedEvent(
                raw_text=canonical_text,
                change=static["change"],
                pokemon=pokemon,
                item=item,
                source=source,
                associated_move=move,
            )
        ]

    if kind == "move_failed":
        return [
            MoveFailedEvent(
                raw_text=canonical_text,
                actor=pokemon,
                move=move or "",
                reason=static.get("reason", "failed"),
            )
        ]

    if kind == "move_availability_changed":
        return [
            MoveAvailabilityChangedEvent(
                raw_text=canonical_text,
                restriction=static["restriction"],
                pokemon=pokemon,
                move=move,
                source_item=item,
                clears_on_switch=static.get("clears_on_switch"),
            )
        ]

    if kind == "field_effect_changed":
        return [
            FieldEffectChangedEvent(
                raw_text=canonical_text,
                effect=static["effect"],
                action=static["action"],
                source=pokemon or source,
            )
        ]

    if kind == "perish_song_started":
        return [
            PerishSongStartedEvent(
                raw_text=canonical_text,
                turns_remaining=static.get("turns_remaining", 3),
                source=pokemon,
            )
        ]

    if kind == "switch_lock_started":
        return [
            SwitchLockStartedEvent(
                raw_text=canonical_text,
                scope=static.get("scope", "all_active"),
                source=pokemon,
            )
        ]

    if kind == "move_used":
        if pokemon is None or move is None:
            return []
        return [
            MoveUsedEvent(
                raw_text=canonical_text,
                actor=pokemon,
                move=move,
                targets=[],
            )
        ]

    if kind == "faint":
        if pokemon is None:
            return []
        return [FaintEvent(raw_text=canonical_text, pokemon=pokemon)]

    if kind == "mega_evolution":
        if pokemon is None:
            return []
        variant = _mega_variant(item, species.species if species else None)
        return [
            MegaEvolutionEvent(
                raw_text=canonical_text,
                pokemon=pokemon,
                variant=variant,
            )
        ]

    if kind == "trick_room_start":
        return [TrickRoomStartEvent(raw_text=canonical_text)]

    if kind == "stat_change":
        stat = _value(_first(tokens, "STAT"))
        if pokemon is None or stat is None:
            return []
        delta = _stat_delta_from_pattern(match.pattern)
        if delta == 0:
            return []
        return [
            StatChangeEvent(
                raw_text=canonical_text,
                pokemon=pokemon,
                stat=stat,  # type: ignore[arg-type]
                stages_delta=delta,
            )
        ]

    if kind == "status":
        if pokemon is None:
            return []
        status, cured = _status_from_pattern(match.pattern)
        if status is None:
            return []
        event_cls = StatusCuredEvent if cured else StatusAppliedEvent
        return [
            event_cls(
                raw_text=canonical_text,
                pokemon=pokemon,
                status=status,  # type: ignore[arg-type]
            )
        ]

    if kind == "volatile":
        if pokemon is None:
            return []
        volatile, cured = _volatile_from_pattern(match.pattern)
        if volatile is None:
            return []
        event_cls = VolatileCuredEvent if cured else VolatileAppliedEvent
        return [
            event_cls(
                raw_text=canonical_text,
                pokemon=pokemon,
                volatile=volatile,  # type: ignore[arg-type]
            )
        ]

    if kind == "switch":
        return _emit_switch(canonical_text, match.pattern, tokens)

    return []


class _TokenLocator:
    """Caches token-localization candidates for one normalized OCR message."""

    def __init__(
        self,
        text: str,
        *,
        player_species: Iterable[str] | None,
        opponent_species: Iterable[str] | None,
    ) -> None:
        self.text = text
        self.lowered = text.casefold()
        self._player_species_provided = player_species is not None
        self._opponent_species_provided = opponent_species is not None
        self.player_species = tuple(player_species or REGULATION_MB_SPECIES)
        self.opponent_species = tuple(opponent_species or REGULATION_MB_SPECIES)
        self._cache: dict[tuple[str, str | None, str | None], tuple[TokenSpan, ...]] = {}
        self._move_base_candidates: tuple[TokenSpan, ...] | None = None
        self._species_base_candidates: dict[Side | None, tuple[TokenSpan, ...]] = {}

    def candidates(
        self,
        token: str,
        *,
        side_hint: Side | None,
        owner_species: str | None,
    ) -> tuple[TokenSpan, ...]:
        key = (token, side_hint, owner_species)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if token in {"POKEMON", "TARGET", "SOURCE", "SPECIES"}:
            result = self._species_candidates(token, side_hint)
        elif token == "MOVE":
            result = self._move_candidates(owner_species)
        elif token == "ITEM":
            result = self._legal_candidates(token, tuple(REGULATION_MB_ITEMS))
        elif token == "ABILITY":
            result = self._legal_candidates(token, tuple(REGULATION_MB_ABILITIES))
        elif token == "STAT":
            result = self._stat_candidates()
        elif token == "NUMBER":
            result = self._number_candidates()
        elif token == "TRAINER":
            result = self._word_candidates(token)
        elif token == "TYPE":
            result = self._legal_candidates(token, _TYPE_NAMES)
        elif token == "SIDE":
            result = self._side_candidates()
        else:
            result = ()

        # Trainer and damaged numeric values have no legal-name score to rank
        # their location. Keep every positional candidate for the surrounding
        # literals to resolve instead of dropping a late-line trainer name.
        deduped = _dedupe_spans(
            result,
            limit=None if token in {"NUMBER", "TRAINER"} else _MAX_ENTITY_CANDIDATES,
        )
        self._cache[key] = deduped
        return deduped

    def _move_candidates(self, owner_species: str | None) -> tuple[TokenSpan, ...]:
        """Reuse one broad move search, then apply the owner's learnset."""
        if self._move_base_candidates is None:
            self._move_base_candidates = tuple(
                self._pool_candidates(
                    "MOVE",
                    tuple(REGULATION_MB_MOVES),
                    side=None,
                    limit=80,
                    truncate=False,
                )
            )

        if owner_species is None:
            return tuple(_best_spans(self._move_base_candidates))

        legal_moves = set(_learnset_moves(owner_species))
        owned = [span for span in self._move_base_candidates if span.value in legal_moves]
        if owned:
            return tuple(_best_spans(owned))

        # Rare short/damaged names may fall outside the broad top candidates.
        # Only then pay for a focused learnset search.
        return self._legal_candidates("MOVE", tuple(legal_moves))

    def _species_candidates(
        self,
        token: str,
        side_hint: Side | None,
    ) -> tuple[TokenSpan, ...]:
        base = self._species_base_candidates.get(side_hint)
        if base is None:
            base = self._locate_species(side_hint)
            self._species_base_candidates[side_hint] = base
        if token == "POKEMON":
            return base
        return tuple(
            TokenSpan(
                name=token,
                value=span.value,
                start=span.start,
                end=span.end,
                score=span.score,
                side=span.side,
            )
            for span in base
        )

    def _locate_species(self, side_hint: Side | None) -> tuple[TokenSpan, ...]:
        """Locate side-aware species once, independent of token role."""
        token = "POKEMON"
        result: list[TokenSpan] = []
        if side_hint in {"player", "opponent"}:
            pool = self.player_species if side_hint == "player" else self.opponent_species
            for span in self._pool_candidates(token, pool, side=side_hint):
                result.append(span)
            if (
                side_hint == "player"
                and not self._player_species_provided
                or side_hint == "opponent"
                and not self._opponent_species_provided
            ):
                result.extend(self._unknown_species_candidates(token, side_hint))
            return tuple(_best_spans(result))

        # A player name is a bare species. An opponent name includes the textual
        # prefix where available, so the token span carries side evidence.
        for span in self._pool_candidates(token, self.player_species, side="player"):
            if _near_opposing_prefix(self.lowered, span.start):
                result.append(
                    TokenSpan(
                        name=span.name,
                        value=span.value,
                        start=span.start,
                        end=span.end,
                        score=max(0.0, span.score - 18.0),
                        side=span.side,
                    )
                )
            else:
                result.append(span)

        for species in self.opponent_species:
            query = f"the opposing {species}"
            alignment = fuzz.partial_ratio_alignment(query.casefold(), self.lowered)
            if alignment.score < _entity_floor(query):
                continue
            result.append(
                TokenSpan(
                    name=token,
                    value=species,
                    start=alignment.dest_start,
                    end=alignment.dest_end,
                    score=float(alignment.score),
                    side="opponent",
                )
            )
        if not self._player_species_provided:
            result.extend(self._unknown_species_candidates(token, "player"))
        if not self._opponent_species_provided:
            result.extend(self._unknown_species_candidates(token, "opponent"))
        return tuple(_best_spans(result))

    def _unknown_species_candidates(
        self,
        token: str,
        side: Side,
    ) -> tuple[TokenSpan, ...]:
        """Fallback spans for callers without a discovered team roster.

        Known team lists are always authoritative.  This fallback preserves
        parser behavior for fixture text whose historical species is outside
        Regulation M-B, such as Landorus or Rillaboom.
        """
        candidates: list[TokenSpan] = []
        if side == "opponent":
            for prefix in re.finditer(
                r"(?i)(?:the\s+)?opposing\s+",
                self.text,
            ):
                tail = self.text[prefix.end() :]
                for match in _WORD_RE.finditer(tail):
                    start = prefix.end() + match.start()
                    end = prefix.end() + match.end()
                    candidates.append(
                        TokenSpan(
                            name=token,
                            value=match.group(0),
                            start=prefix.start(),
                            end=end,
                            score=96.0,
                            side="opponent",
                        )
                    )
                    # The first word after the prefix is the only plausible
                    # name for a one-Pokemon template.
                    break
        else:
            for match in _WORD_RE.finditer(self.text):
                if _near_opposing_prefix(self.lowered, match.start()):
                    continue
                candidates.append(
                    TokenSpan(
                        name=token,
                        value=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        score=96.0,
                        side="player",
                    )
                )
        return tuple(candidates)

    def _legal_candidates(
        self,
        token: str,
        pool: Sequence[str],
    ) -> tuple[TokenSpan, ...]:
        return tuple(self._pool_candidates(token, pool, side=None))

    def _pool_candidates(
        self,
        token: str,
        pool: Sequence[str],
        *,
        side: Side | None,
        limit: int | None = None,
        truncate: bool = True,
    ) -> list[TokenSpan]:
        if not pool:
            return []
        extracted = process.extract(
            self.lowered,
            pool,
            # ``partial_ratio`` is directional for a long OCR line and a
            # short legal value. Score the legal value *inside* the line;
            # reversing these arguments can hide a damaged short move such as
            # ``Wish`` -> ``Wis%`` behind unrelated literal text.
            scorer=_value_in_text_score,
            limit=min(limit or _MAX_ENTITY_CANDIDATES, len(pool)),
        )
        result: list[TokenSpan] = []
        for value, _score, _index in extracted:
            alignment = fuzz.partial_ratio_alignment(value.casefold(), self.lowered)
            if alignment.score < _entity_floor(value):
                continue
            result.append(
                TokenSpan(
                    name=token,
                    value=value,
                    start=alignment.dest_start,
                    end=alignment.dest_end,
                    score=float(alignment.score),
                    side=side,
                )
            )
        return _best_spans(result) if truncate else result

    def _stat_candidates(self) -> tuple[TokenSpan, ...]:
        result: list[TokenSpan] = []
        for alias, canonical in _STAT_ALIASES.items():
            alignment = fuzz.partial_ratio_alignment(alias.casefold(), self.lowered)
            if alignment.score < _entity_floor(alias):
                continue
            result.append(
                TokenSpan(
                    name="STAT",
                    value=canonical,
                    start=alignment.dest_start,
                    end=alignment.dest_end,
                    score=float(alignment.score),
                )
            )
        return tuple(_best_spans(result))

    def _number_candidates(self) -> tuple[TokenSpan, ...]:
        exact = [
            TokenSpan(
                name="NUMBER",
                value=match.group(0),
                start=match.start(),
                end=match.end(),
                score=100.0,
            )
            for match in re.finditer(r"\d+", self.text)
        ]
        if exact:
            return tuple(exact)

        # A one-character count can be irrecoverably replaced by arbitrary
        # noise. Keep its positional slot so the event can still be emitted
        # with an unresolved count rather than inventing a number.
        return tuple(
            TokenSpan(
                name="NUMBER",
                value=None,
                start=index,
                end=index + 1,
                score=35.0,
            )
            for index, character in enumerate(self.text)
            if not character.isspace()
        )

    def _word_candidates(self, token: str) -> tuple[TokenSpan, ...]:
        spans: list[TokenSpan] = []
        # Preserve ordinary word boundaries so ``Cynthia's`` can leave ``'s``
        # for the following literal.
        for match in _WORD_RE.finditer(self.text):
            spans.append(
                TokenSpan(
                    name=token,
                    value=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    score=78.0,
                )
            )
        # A trainer is not a legal-name snap target. Also accept a complete
        # non-whitespace chunk so ``Iris`` -> ``^ris`` remains localized.
        for match in re.finditer(r"\S+", self.text):
            spans.append(
                TokenSpan(
                    name=token,
                    value=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    score=78.0,
                )
            )
        return tuple(spans)

    def _side_candidates(self) -> tuple[TokenSpan, ...]:
        values = ("your", "the opposing", "opposing", "opponent's")
        result: list[TokenSpan] = []
        for value in values:
            alignment = fuzz.partial_ratio_alignment(value.casefold(), self.lowered)
            if alignment.score < 60.0:
                continue
            result.append(
                TokenSpan(
                    name="SIDE",
                    value=value,
                    start=alignment.dest_start,
                    end=alignment.dest_end,
                    score=float(alignment.score),
                )
            )
        return tuple(_best_spans(result))


def _match_pattern(
    locator: _TokenLocator,
    template: BattleTextTemplate,
    pattern: str,
) -> TokenizedMatch | None:
    pattern = normalize_catalog_text(pattern)
    pieces = _TOKEN_RE.split(pattern)
    token_names = [pieces[index] for index in range(1, len(pieces), 2)]
    if not token_names:
        return None
    literals = [pieces[index] for index in range(0, len(pieces), 2)]

    best: TokenizedMatch | None = None
    states = 0

    def walk(
        index: int,
        previous_end: int,
        spans: list[TokenSpan],
        literal_scores: list[tuple[float, int]],
    ) -> None:
        nonlocal best, states
        if states >= _MAX_PATTERN_STATES:
            return
        states += 1

        if index == len(token_names):
            tail_score = _literal_score(literals[-1], locator.text[previous_end:])
            if tail_score is None:
                return
            scored = [*literal_scores, tail_score]
            score = _combined_score(spans, scored)
            if score < _TOKENIZED_SCORE_CUTOFF:
                return
            candidate = TokenizedMatch(
                template=template,
                pattern=pattern,
                spans=tuple(spans),
                score=score,
            )
            if best is None or candidate.score > best.score:
                best = candidate
            return

        token = token_names[index]
        owner_species = _owner_species(spans)
        side_hint = _side_hint(template, pattern, token, spans)
        candidates = locator.candidates(
            token,
            side_hint=side_hint,
            owner_species=owner_species,
        )
        for candidate in candidates:
            if candidate.start < previous_end:
                continue
            literal_score = _literal_score(
                literals[index],
                locator.text[previous_end : candidate.start],
            )
            if literal_score is None:
                continue
            walk(
                index + 1,
                candidate.end,
                [*spans, candidate],
                [*literal_scores, literal_score],
            )

    walk(0, 0, [], [])
    return best


def _template_patterns_with_tokens(template: BattleTextTemplate) -> tuple[str, ...]:
    if template.legacy_handler == "side_banner":
        return ()
    patterns: list[str] = []
    for pattern in (*template.champions, *template.showdown):
        if _TOKEN_RE.search(pattern):
            patterns.append(pattern)
    return tuple(patterns)


def _side_hint(
    template: BattleTextTemplate,
    pattern: str,
    token: str,
    spans: Sequence[TokenSpan],
) -> Side | None:
    if token not in {"POKEMON", "TARGET", "SOURCE", "SPECIES"}:
        return None

    static_side = template.static.get("side")
    if static_side in {"player", "opponent"}:
        return static_side  # type: ignore[return-value]

    lowered = pattern.casefold()
    if "go!" in lowered or "come back" in lowered:
        return "player"
    if "sent out" in lowered or "withdrew" in lowered:
        return "opponent"
    if token == "SPECIES":
        owner = _first_named(spans, "POKEMON")
        if owner is not None:
            return owner.side
    return None


def _literal_score(expected: str, observed: str) -> tuple[float, int] | None:
    """Score one literal only inside its token-bounded text region."""
    # ``_match_pattern`` normalizes the entire template and OCR line before
    # slicing. Avoid re-running the catalog normalizer for every candidate
    # state; the boundaries merely need their surrounding whitespace removed.
    expected_normalized = expected.strip()
    observed_normalized = observed.strip()
    semantic_length = len(re.sub(r"[^a-z0-9]+", "", expected_normalized.casefold()))

    if not expected_normalized:
        if observed_normalized:
            return None
        return (100.0, 0)

    if semantic_length == 0:
        # Punctuation is easy to destroy and has no event semantics, but a
        # punctuation-only suffix cannot absorb a second Pokemon or sentence.
        nonspace = observed_normalized.strip()
        allowed_length = max(1, len(expected_normalized.strip()))
        if len(nonspace) > allowed_length:
            return None
        return (100.0, 0)

    score = float(fuzz.ratio(expected_normalized.casefold(), observed_normalized.casefold()))
    if semantic_length <= 2:
        floor = 45.0
    elif semantic_length <= 5:
        floor = 65.0
    elif semantic_length <= 8:
        # Two substitutions can land in a six-to-eight-character connector
        # (for example ``copied`` -> ``c%p@ed``) while both surrounding
        # entity tokens remain decisive.
        floor = 60.0
    else:
        floor = 70.0
    if score < floor:
        return None
    return (score, semantic_length)


def _combined_score(
    spans: Sequence[TokenSpan],
    literal_scores: Sequence[tuple[float, int]],
) -> float:
    token_score = sum(span.score for span in spans) / len(spans)
    literal_weight = sum(weight for _score, weight in literal_scores)
    literal_score = (
        sum(score * weight for score, weight in literal_scores) / literal_weight
        if literal_weight
        else 100.0
    )
    # Unknown trainer / destroyed one-character number spans carry low scores,
    # but substantial fixed text in the correct position can still identify
    # the event safely.
    return (token_score * 0.55) + (literal_score * 0.45)


def _entity_floor(value: str) -> float:
    length = len(re.sub(r"[^A-Za-z0-9]+", "", value))
    if length <= 3:
        return 55.0
    if length <= 5:
        return 62.0
    if length <= 8:
        return 70.0
    return 75.0


def _value_in_text_score(
    text: str,
    value: str,
    *,
    score_cutoff: float | None = None,
    **_kwargs: Any,
) -> float:
    return float(
        fuzz.partial_ratio(
            value.casefold(),
            text.casefold(),
            score_cutoff=score_cutoff,
        )
    )


def _best_spans(spans: Sequence[TokenSpan]) -> list[TokenSpan]:
    ordered = sorted(spans, key=lambda span: (-span.score, span.start, span.end, span.value or ""))
    return ordered[:_MAX_ENTITY_CANDIDATES]


def _dedupe_spans(
    spans: Sequence[TokenSpan],
    *,
    limit: int | None = _MAX_ENTITY_CANDIDATES,
) -> tuple[TokenSpan, ...]:
    unique: dict[tuple[str, str | None, int, int, Side | None], TokenSpan] = {}
    for span in spans:
        key = (span.name, span.value, span.start, span.end, span.side)
        existing = unique.get(key)
        if existing is None or span.score > existing.score:
            unique[key] = span
    ordered = sorted(
        unique.values(),
        key=lambda span: (-span.score, span.start, span.end, span.value or ""),
    )
    return tuple(ordered if limit is None else ordered[:limit])


def _near_opposing_prefix(text: str, start: int) -> bool:
    prefix = text[max(0, start - 18) : start]
    return fuzz.partial_ratio("opposing", prefix) >= 65.0


def _learnset_moves(species: str | None) -> tuple[str, ...]:
    if species is None:
        return tuple(REGULATION_MB_MOVES)
    moves = REGULATION_MB_LEARNSETS.get(species)
    return tuple(moves) if moves else tuple(REGULATION_MB_MOVES)


def _spans_by_name(spans: Sequence[TokenSpan]) -> Mapping[str, tuple[TokenSpan, ...]]:
    grouped: dict[str, list[TokenSpan]] = {}
    for span in spans:
        grouped.setdefault(span.name, []).append(span)
    return {name: tuple(values) for name, values in grouped.items()}


def _first(
    grouped: Mapping[str, tuple[TokenSpan, ...]],
    name: str,
) -> TokenSpan | None:
    values = grouped.get(name, ())
    return values[0] if values else None


def _at(
    grouped: Mapping[str, tuple[TokenSpan, ...]],
    name: str,
    index: int,
) -> TokenSpan | None:
    values = grouped.get(name, ())
    return values[index] if len(values) > index else None


def _first_named(spans: Sequence[TokenSpan], name: str) -> TokenSpan | None:
    return next((span for span in spans if span.name == name), None)


def _owner_species(spans: Sequence[TokenSpan]) -> str | None:
    owner = _first_named(spans, "POKEMON")
    return owner.value if owner is not None else None


def _value(span: TokenSpan | None) -> str | None:
    return span.value if span is not None else None


def _int_value(span: TokenSpan | None) -> int | None:
    if span is None or span.value is None or not span.value.isdigit():
        return None
    value = int(span.value)
    return value if value >= 1 else None


def _pokemon_from_span(
    span: TokenSpan | None,
    *,
    side: Side | None = None,
    slot: Slot = 1,
) -> Pokemon | None:
    if span is None or span.value is None:
        return None
    return Pokemon(species=span.value, side=side or span.side or "player", slot=slot)


def _mega_variant(item: str | None, species: str | None) -> Literal["regular", "X", "Y", "Z"]:
    for value in (item, species):
        if not value:
            continue
        token = value.strip().upper().split()[-1]
        if token in {"X", "Y", "Z"}:
            return token  # type: ignore[return-value]
    return "regular"


def _stat_delta_from_pattern(pattern: str) -> int:
    lowered = pattern.casefold()
    if "fell" in lowered or "fall" in lowered:
        if "severely" in lowered:
            return -3
        if "harshly" in lowered or "sharply" in lowered:
            return -2
        return -1
    if "rose" in lowered:
        if "drastically" in lowered:
            return 3
        if "sharply" in lowered:
            return 2
        return 1
    return 0


def _status_from_pattern(pattern: str) -> tuple[str | None, bool]:
    lowered = pattern.casefold()
    if "burn" in lowered:
        return "brn", "healed" in lowered
    if "paraly" in lowered:
        return "par", "cured" in lowered
    if "badly poison" in lowered:
        return "tox", False
    if "poison" in lowered:
        return "psn", "cured" in lowered
    if "frozen" in lowered or "thawed" in lowered:
        return "frz", "thawed" in lowered
    if "asleep" in lowered or "woke" in lowered:
        return "slp", "woke" in lowered
    return None, False


def _volatile_from_pattern(pattern: str) -> tuple[str | None, bool]:
    lowered = pattern.casefold()
    if "taunt" in lowered:
        return "taunted", False
    if "encore" in lowered:
        return "encore", False
    if "confusion" in lowered or "confused" in lowered:
        return "confused", "snapped out" in lowered
    return None, False


def _emit_switch(
    raw_text: str,
    pattern: str,
    grouped: Mapping[str, tuple[TokenSpan, ...]],
) -> list[BattleLogEvent]:
    lowered = pattern.casefold()
    pokemon = _first(grouped, "POKEMON")
    pokemon2 = _at(grouped, "POKEMON", 1)
    if pokemon is None:
        return []

    if " and [pokemon]" in lowered and ("go!" in lowered or "sent out" in lowered):
        side: Side = "player" if "go!" in lowered else "opponent"
        first = _pokemon_from_span(pokemon, side=side, slot=1)
        second = _pokemon_from_span(pokemon2, side=side, slot=2)
        if first is None or second is None:
            return []
        return [
            LeadInEvent(
                raw_text=raw_text,
                side=side,
                slot_1=first,
                slot_2=second,
            )
        ]

    if "come back" in lowered:
        resolved = _pokemon_from_span(pokemon, side="player")
        return [SwitchOutEvent(raw_text=raw_text, pokemon=resolved)] if resolved else []
    if "withdrew" in lowered:
        resolved = _pokemon_from_span(pokemon, side="opponent")
        return [SwitchOutEvent(raw_text=raw_text, pokemon=resolved)] if resolved else []
    if "went back" in lowered:
        resolved = _pokemon_from_span(pokemon)
        return [SwitchOutEvent(raw_text=raw_text, pokemon=resolved)] if resolved else []
    if "go!" in lowered:
        resolved = _pokemon_from_span(pokemon, side="player")
        return [SwitchInEvent(raw_text=raw_text, pokemon=resolved)] if resolved else []
    if "sent out" in lowered:
        resolved = _pokemon_from_span(pokemon, side="opponent")
        return [SwitchInEvent(raw_text=raw_text, pokemon=resolved)] if resolved else []
    if "dragged out" in lowered:
        resolved = _pokemon_from_span(pokemon)
        return [SwitchInEvent(raw_text=raw_text, pokemon=resolved)] if resolved else []
    return []
