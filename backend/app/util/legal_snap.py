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
