"""Read HP from active slot cards with a 2-frame stability gate."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.cv.event_ocr import _ocr_text
from app.cv.regions import RegionConfig, config_for_image, crop_region
from app.schema.battle_log import HPChangeEvent
from app.schema.common import Pokemon, Side, Slot
from app.schema.gamestate import GameState

logger = logging.getLogger(__name__)

_SLOT_CARD_REGIONS: tuple[tuple[str, Side, Slot], ...] = (
    ("player_slot_1_card", "player", 1),
    ("player_slot_2_card", "player", 2),
    ("opponent_slot_1_card", "opponent", 1),
    ("opponent_slot_2_card", "opponent", 2),
)

_STABLE_FRAMES_REQUIRED = 2
_REGION_CONTENT_STD_MIN = 20.0
_REGION_CONTENT_MEAN_MIN = 15.0

# Player cards: "162 / 162" or "162/162"
_PLAYER_HP_RE = re.compile(r"(?P<current>\d+)\s*/\s*(?P<max>\d+)")
# Opponent cards: "47%" or "100 %"
_OPPONENT_HP_RE = re.compile(r"(?P<pct>\d+)\s*%")
# Strip gender / level remnants left by OCR near species names.
_SPECIES_NOISE_RE = re.compile(
    r"[\u2640\u2642♀♂]|^\s*lv\.?\s*\d+\s*|\s*lv\.?\s*\d+\s*$",
    re.IGNORECASE,
)


@dataclass
class SlotCardRead:
    species: str
    hp_pct: int
    raw_text: str


def read_slot_card(
    image: np.ndarray,
    config: RegionConfig,
    region_name: str,
) -> SlotCardRead | None:
    """OCR a single calibrated slot-card region into a structured reading."""
    side, _slot = _side_slot_for_region(region_name)
    display_config = config_for_image(config, image)
    crop = crop_region(image, display_config.get(region_name))
    if not _region_has_content(crop):
        return None
    return parse_slot_card_text(_ocr_slot_card_text(crop), side)


def _side_slot_for_region(region_name: str) -> tuple[Side, Slot]:
    for name, side, slot in _SLOT_CARD_REGIONS:
        if name == region_name:
            return side, slot
    raise KeyError(f"Not a slot card region: {region_name}")


@dataclass
class HPReadTracker:
    """Per-slot card tracker for animation-phase HP debouncing."""

    species: str | None = None
    prev_frame_hp_pct: int | None = None
    candidate_hp_pct: int | None = None
    stable_frames: int = 0
    tracking: bool = False
    committed: bool = False

    def reset(self) -> None:
        self.species = None
        self.prev_frame_hp_pct = None
        self.candidate_hp_pct = None
        self.stable_frames = 0
        self.tracking = False
        self.committed = False


@dataclass
class HPReader:
    """Poll slot cards during battle animation; snapshot on action selection."""

    _trackers: dict[str, HPReadTracker] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear per-slot trackers (e.g. when leaving battle animation)."""
        self._trackers.clear()

    def process_animation_frame(
        self,
        image: np.ndarray,
        config: RegionConfig,
        game_state: GameState | None,
    ) -> list[HPChangeEvent]:
        """Mode 1: poll visible slot cards; emit after a 2-frame stable read."""
        display_config = config_for_image(config, image)
        events: list[HPChangeEvent] = []

        for region_name, side, slot in _SLOT_CARD_REGIONS:
            crop = crop_region(image, display_config.get(region_name))
            tracker = self._trackers.setdefault(region_name, HPReadTracker())

            if not _region_has_content(crop):
                tracker.reset()
                continue

            reading = parse_slot_card_text(_ocr_slot_card_text(crop), side)
            if reading is None:
                continue

            tracker.species = reading.species
            event = self._update_tracker(
                tracker,
                reading,
                side=side,
                slot=slot,
                game_state=game_state,
            )
            if event is not None:
                events.append(event)

        return events

    def read_action_selection_snapshot(
        self,
        image: np.ndarray,
        config: RegionConfig,
        game_state: GameState | None,
    ) -> list[HPChangeEvent]:
        """Mode 2: authoritative 4-slot read at turn boundary (no stability gate)."""
        display_config = config_for_image(config, image)
        events: list[HPChangeEvent] = []

        for region_name, side, slot in _SLOT_CARD_REGIONS:
            crop = crop_region(image, display_config.get(region_name))
            if not _region_has_content(crop):
                continue

            reading = parse_slot_card_text(_ocr_slot_card_text(crop), side)
            if reading is None:
                continue

            # Snapshot becomes the new baseline for animation tracking.
            tracker = self._trackers.setdefault(region_name, HPReadTracker())
            tracker.species = reading.species
            tracker.prev_frame_hp_pct = reading.hp_pct
            tracker.candidate_hp_pct = reading.hp_pct
            tracker.stable_frames = 0
            tracker.tracking = False
            tracker.committed = True

            event = _maybe_hp_change_event(
                reading,
                side=side,
                slot=slot,
                game_state=game_state,
            )
            if event is not None:
                events.append(event)

        return events

    def _update_tracker(
        self,
        tracker: HPReadTracker,
        reading: SlotCardRead,
        *,
        side: Side,
        slot: Slot,
        game_state: GameState | None,
    ) -> HPChangeEvent | None:
        hp_pct = reading.hp_pct
        event: HPChangeEvent | None = None

        # Re-open tracking if HP moves again after a prior commit on the same card.
        if tracker.committed and tracker.prev_frame_hp_pct is not None:
            if hp_pct != tracker.prev_frame_hp_pct:
                tracker.tracking = True
                tracker.committed = False
                tracker.candidate_hp_pct = hp_pct
                tracker.stable_frames = 1

        # Start gate: begin only when consecutive frames differ.
        elif (
            not tracker.tracking
            and tracker.prev_frame_hp_pct is not None
            and hp_pct != tracker.prev_frame_hp_pct
        ):
            tracker.tracking = True
            tracker.candidate_hp_pct = hp_pct
            tracker.stable_frames = 1
            tracker.committed = False

        elif tracker.tracking and not tracker.committed:
            if tracker.candidate_hp_pct is None or hp_pct != tracker.candidate_hp_pct:
                tracker.candidate_hp_pct = hp_pct
                tracker.stable_frames = 1
            else:
                tracker.stable_frames += 1

            # Commit gate: unchanged for 2 frames at 3 FPS.
            if tracker.stable_frames >= _STABLE_FRAMES_REQUIRED:
                final_hp_pct = tracker.candidate_hp_pct
                assert final_hp_pct is not None
                event = _maybe_hp_change_event(
                    SlotCardRead(
                        species=reading.species,
                        hp_pct=final_hp_pct,
                        raw_text=reading.raw_text,
                    ),
                    side=side,
                    slot=slot,
                    game_state=game_state,
                )
                tracker.committed = True

        tracker.prev_frame_hp_pct = hp_pct
        return event


def parse_slot_card_text(text: str, side: Side) -> SlotCardRead | None:
    """Parse OCR text from a player (current/max) or opponent (%) slot card."""
    cleaned = _normalize_slot_ocr_text(text, side)
    if not cleaned:
        return None

    if side == "player":
        match = _PLAYER_HP_RE.search(cleaned)
        if match is None:
            return None
        current = int(match.group("current"))
        maximum = int(match.group("max"))
        if maximum <= 0:
            return None
        hp_pct = int(round(100 * current / maximum))
        hp_pct = max(0, min(100, hp_pct))
        species = _extract_species(cleaned, match.start(), match.end())
        if not species:
            return None
        # Canonical raw_text: UI usually shows "162 / 162" with spaces.
        raw_text = f"{species} {current} / {maximum}"
    else:
        match = _OPPONENT_HP_RE.search(cleaned)
        if match is None:
            return None
        hp_pct = max(0, min(100, int(match.group("pct"))))
        species = _extract_species(cleaned, match.start(), match.end())
        if not species:
            return None
        raw_text = f"{species} {hp_pct}%"

    return SlotCardRead(species=species, hp_pct=hp_pct, raw_text=raw_text)


def _normalize_slot_ocr_text(text: str, side: Side) -> str:
    """Clean OCR noise common on Champions slot cards."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    if side == "player":
        # Thin "/" is often read as "7", gluing current/max into one token (1627162).
        cleaned = re.sub(r"\b(\d{2,3})7(\d{2,3})\b", r"\1 / \2", cleaned)
    else:
        # Collapse "100 %" spacing; repair common 100% OCR tails (1005/100e).
        cleaned = re.sub(r"\b(\d{1,3})\s+%", r"\1%", cleaned)
        cleaned = re.sub(r"\b(100)[5eEoOsSgG]\b", r"100%", cleaned)
        if not re.search(r"\d\s*%", cleaned):
            cleaned = re.sub(r"\b(\d{1,3})\s*$", r"\1%", cleaned)

    return cleaned


def _ocr_slot_card_text(crop_rgb: np.ndarray) -> str:
    """OCR a slot-card crop (grayscale upscale; Otsu destroys thin '/' and names)."""
    import easyocr

    reader = getattr(_ocr_slot_card_text, "_reader", None)
    if reader is None:
        # Reuse event_ocr's singleton when available to avoid loading EasyOCR twice.
        reader = getattr(_ocr_text, "_reader", None)
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        _ocr_text._reader = reader  # type: ignore[attr-defined]
    _ocr_slot_card_text._reader = reader  # type: ignore[attr-defined]
    prepared = _preprocess_slot_card_for_ocr(crop_rgb)
    lines = reader.readtext(prepared, detail=0, paragraph=True)
    return " ".join(lines).strip()


def _preprocess_slot_card_for_ocr(crop_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(upscaled, cv2.COLOR_GRAY2RGB)


def lookup_active_hp(
    game_state: GameState | None,
    side: Side,
    slot: Slot,
    species: str | None = None,
) -> int | None:
    """Return GameState HP % for the active mon in ``side``/``slot`` (or by species)."""
    if game_state is None:
        return None

    side_state = game_state.player if side == "player" else game_state.opponent
    by_slot = side_state.slot_1 if slot == 1 else side_state.slot_2
    if by_slot is not None:
        if species is None or _species_matches(by_slot.species, species):
            return by_slot.hp_percentage

    if species:
        for candidate in (side_state.slot_1, side_state.slot_2):
            if candidate is not None and _species_matches(candidate.species, species):
                return candidate.hp_percentage
    return None


def _maybe_hp_change_event(
    reading: SlotCardRead,
    *,
    side: Side,
    slot: Slot,
    game_state: GameState | None,
) -> HPChangeEvent | None:
    game_hp = lookup_active_hp(game_state, side, slot, reading.species)
    if game_hp is None:
        logger.debug(
            "Skipping HP event for %s slot %s (%s): no GameState HP",
            side,
            slot,
            reading.species,
        )
        return None

    hp_pct_change = reading.hp_pct - game_hp
    if hp_pct_change == 0:
        return None

    return HPChangeEvent(
        pokemon=Pokemon(species=reading.species, side=side, slot=slot),
        hp_pct_change=hp_pct_change,
        raw_text=reading.raw_text,
    )


def _extract_species(text: str, hp_start: int, hp_end: int) -> str:
    before = text[:hp_start].strip(" -:·|")
    after = text[hp_end:].strip(" -:·|")
    candidate = before or after
    candidate = _SPECIES_NOISE_RE.sub(" ", candidate)
    candidate = " ".join(candidate.split()).strip(" .:!-")
    # Prefer the last token cluster before HP (name sits above the bar).
    if before:
        parts = before.split()
        # Drop stray short OCR tokens at the start.
        while parts and len(parts[0]) <= 1:
            parts.pop(0)
        candidate = " ".join(parts).strip(" .:!-")
    return candidate


def _species_matches(known: str, observed: str) -> bool:
    a = known.casefold().replace("-", " ").replace("'", "").strip()
    b = observed.casefold().replace("-", " ").replace("'", "").strip()
    return a == b or a in b or b in a


def _region_has_content(crop_rgb: np.ndarray) -> bool:
    """Brightness/content gate — slot cards are dark panels with light text."""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    return (
        float(gray.std()) >= _REGION_CONTENT_STD_MIN
        and float(gray.mean()) >= _REGION_CONTENT_MEAN_MIN
    )
