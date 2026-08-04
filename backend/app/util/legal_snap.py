"""RapidFuzz snapping of OCR / vision strings onto Regulation M-B legal names."""

from __future__ import annotations

from collections.abc import Iterable

from rapidfuzz import fuzz, process

DEFAULT_SCORE_CUTOFF = 80.0


def normalize_name(value: str) -> str:
    """Collapse whitespace for comparison."""
    return " ".join(value.strip().split())


def snap_to_legal(
    raw: str,
    legal: Iterable[str],
    *,
    score_cutoff: float = DEFAULT_SCORE_CUTOFF,
) -> str | None:
    """Return the best legal name for ``raw``, or None if below ``score_cutoff``.

    Exact case-insensitive matches win immediately. Otherwise uses RapidFuzz
    WRatio against the legal pool.
    """
    cleaned = normalize_name(raw)
    if not cleaned:
        return None

    choices = list(legal)
    if not choices:
        return None

    lowered = {c.lower(): c for c in choices}
    exact = lowered.get(cleaned.lower())
    if exact is not None:
        return exact

    match = process.extractOne(
        cleaned,
        choices,
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff,
    )
    if match is None:
        return None
    return match[0]


def prefer_known_species(
    raw: str,
    known: Iterable[str],
    legal: Iterable[str],
    *,
    score_cutoff: float = DEFAULT_SCORE_CUTOFF,
) -> str:
    """Snap ``raw`` preferring already-known species (form disambiguation).

    Example: OCR ``Arcanine`` with known ``Arcanine-Hisui`` resolves to the
    Hisui form rather than base Arcanine when both are legal.
    """
    cleaned = normalize_name(raw)
    if not cleaned:
        return raw

    known_list = [normalize_name(s) for s in known if normalize_name(s)]
    legal_list = list(legal)

    if known_list:
        exact_known = snap_to_legal(cleaned, known_list, score_cutoff=100)
        if exact_known is not None:
            return exact_known

        # Form preference: known entries whose base name equals the OCR string.
        base = cleaned.lower()
        form_hits = [
            s
            for s in known_list
            if s.lower() == base or s.lower().startswith(base + "-")
        ]
        if len(form_hits) == 1:
            return form_hits[0]
        if len(form_hits) > 1:
            snapped = snap_to_legal(cleaned, form_hits, score_cutoff=score_cutoff)
            if snapped is not None:
                return snapped
            return form_hits[0]

        known_snap = snap_to_legal(cleaned, known_list, score_cutoff=score_cutoff)
        if known_snap is not None:
            return known_snap

    legal_snap = snap_to_legal(cleaned, legal_list, score_cutoff=score_cutoff)
    return legal_snap if legal_snap is not None else cleaned
