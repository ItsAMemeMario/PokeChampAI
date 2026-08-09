"""Read HP from active slot cards with a 2-frame stability gate."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from app.cv.ocr_reader import map_parallel, read_text
from app.cv.regions import RegionConfig, config_for_image, crop_region, cv_templates_dir
from app.schema.battle_log import HPChangeEvent
from app.schema.common import Pokemon, Side, Slot
from app.schema.gamestate import GameState
from app.data.species import REGULATION_MB_SPECIES
from app.util.legal_snap import snap_to_legal

logger = logging.getLogger(__name__)

# (card region, side, slot) — HP bar is cropped via a shared offset inside each card.
_SLOT_CARD_REGIONS: tuple[tuple[str, Side, Slot], ...] = (
    ("player_slot_1_card", "player", 1),
    ("player_slot_2_card", "player", 2),
    ("opponent_slot_1_card", "opponent", 1),
    ("opponent_slot_2_card", "opponent", 2),
)
# Fallback if config lacks hp_bar regions; normally derived from calibrated JSON.
_HP_BAR_IN_CARD_DEFAULT = (17, 48, 195, 26)  # x, y, w, h relative to card

_STABLE_FRAMES_REQUIRED = 2
# Grayscale template match (one template covers normal + green-highlight borders).
_SLOT_CARD_TEMPLATE_SCORE_MIN = 0.42
_EMPTY_CROP_MEAN_MAX = 5.0
# HP-bar motion on the in-card bar crop only.
_HP_BAR_DIFF_DOWNSCALE = 4
_HP_BAR_DIFF_MEAN_MIN = 3.0
# Player cards: purple/blue name banner — HSV keeps white AA strokes.
_SLOT_CARD_SAT_MAX = 50
_SLOT_CARD_VALUE_MIN = 180
_SLOT_CARD_OCR_UPSCALE = 2.0
# Opponent cards: pink/magenta name banner — gray threshold + stronger upscale.
_OPPONENT_SLOT_OCR_UPSCALE = 3.0
_OPPONENT_SLOT_GRAY_THRESH = 180

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
    *,
    player_species: Iterable[str] | None = None,
    opponent_species: Iterable[str] | None = None,
) -> SlotCardRead | None:
    """OCR a single calibrated slot-card region into a structured reading."""
    side, _slot = _side_slot_for_region(region_name)
    display_config = config_for_image(config, image)
    crop = crop_region(image, display_config.get(region_name))
    if not _slot_card_visible(crop):
        return None
    return parse_slot_card_text(
        _ocr_slot_card_text(crop, side),
        side,
        player_species=player_species,
        opponent_species=opponent_species,
    )


def _side_slot_for_region(region_name: str) -> tuple[Side, Slot]:
    for card_name, side, slot in _SLOT_CARD_REGIONS:
        if card_name == region_name:
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
    _prev_bar_gray: dict[str, np.ndarray] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear per-slot trackers (e.g. when leaving battle animation)."""
        self._trackers.clear()
        self._prev_bar_gray.clear()

    def process_animation_frame(
        self,
        image: np.ndarray,
        config: RegionConfig,
        game_state: GameState | None,
        *,
        player_species: Iterable[str] | None = None,
        opponent_species: Iterable[str] | None = None,
    ) -> list[HPChangeEvent]:
        """Mode 1: OCR only when card is visible and HP bar is animating (or mid-track)."""
        display_config = config_for_image(config, image)
        bar_in_card = _hp_bar_in_card(display_config)
        events: list[HPChangeEvent] = []
        ocr_jobs: list[tuple[str, Side, Slot, np.ndarray]] = []

        for card_name, side, slot in _SLOT_CARD_REGIONS:
            crop = crop_region(image, display_config.get(card_name))
            tracker = self._trackers.setdefault(card_name, HPReadTracker())

            if not _slot_card_visible(crop):
                tracker.reset()
                self._prev_bar_gray.pop(card_name, None)
                continue

            bar_crop = _crop_hp_bar(crop, bar_in_card)
            prev_gray = self._prev_bar_gray.get(card_name)
            bar_moving = _hp_bar_changed(bar_crop, prev_gray)
            self._prev_bar_gray[card_name] = _downscale_bar_gray(bar_crop)

            # OCR while the bar is moving, or while the stability gate is open.
            if not bar_moving and not tracker.tracking:
                continue

            ocr_jobs.append((card_name, side, slot, crop))

        texts = map_parallel(
            lambda item: _ocr_slot_card_text(item[0], item[1]),
            [(crop, side) for _n, side, _s, crop in ocr_jobs],
        )
        for (card_name, side, slot, _crop), text in zip(ocr_jobs, texts, strict=True):
            tracker = self._trackers.setdefault(card_name, HPReadTracker())
            reading = parse_slot_card_text(
                text,
                side,
                player_species=player_species,
                opponent_species=opponent_species,
            )
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
        *,
        player_species: Iterable[str] | None = None,
        opponent_species: Iterable[str] | None = None,
    ) -> list[HPChangeEvent]:
        """Mode 2: authoritative 4-slot read at turn boundary (template gate only)."""
        display_config = config_for_image(config, image)
        bar_in_card = _hp_bar_in_card(display_config)
        events: list[HPChangeEvent] = []
        ocr_jobs: list[tuple[str, Side, Slot, np.ndarray]] = []

        for card_name, side, slot in _SLOT_CARD_REGIONS:
            crop = crop_region(image, display_config.get(card_name))
            if not _slot_card_visible(crop):
                continue
            ocr_jobs.append((card_name, side, slot, crop))

        texts = map_parallel(
            lambda item: _ocr_slot_card_text(item[0], item[1]),
            [(crop, side) for _n, side, _s, crop in ocr_jobs],
        )
        for (card_name, side, slot, crop), text in zip(ocr_jobs, texts, strict=True):
            reading = parse_slot_card_text(
                text,
                side,
                player_species=player_species,
                opponent_species=opponent_species,
            )
            if reading is None:
                continue

            # Snapshot becomes the new baseline for animation tracking.
            tracker = self._trackers.setdefault(card_name, HPReadTracker())
            tracker.species = reading.species
            tracker.prev_frame_hp_pct = reading.hp_pct
            tracker.candidate_hp_pct = reading.hp_pct
            tracker.stable_frames = 0
            tracker.tracking = False
            tracker.committed = True

            self._prev_bar_gray[card_name] = _downscale_bar_gray(
                _crop_hp_bar(crop, bar_in_card)
            )

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
                tracker.tracking = False

        tracker.prev_frame_hp_pct = hp_pct
        return event


def parse_slot_card_text(
    text: str,
    side: Side,
    *,
    player_species: Iterable[str] | None = None,
    opponent_species: Iterable[str] | None = None,
) -> SlotCardRead | None:
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
        species = _snap_slot_species(
            species,
            side,
            player_species=player_species,
            opponent_species=opponent_species,
        )
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
        species = _snap_slot_species(
            species,
            side,
            player_species=player_species,
            opponent_species=opponent_species,
        )
        raw_text = f"{species} {hp_pct}%"

    return SlotCardRead(species=species, hp_pct=hp_pct, raw_text=raw_text)


def _snap_slot_species(
    species: str,
    side: Side,
    *,
    player_species: Iterable[str] | None = None,
    opponent_species: Iterable[str] | None = None,
) -> str:
    """Snap OCR species to the side's known list (bring-4 / opponent-6)."""
    known: Iterable[str]
    if side == "player":
        known = player_species or REGULATION_MB_SPECIES
    else:
        known = opponent_species or REGULATION_MB_SPECIES
    return snap_to_legal(species, known) or species


def _normalize_slot_ocr_text(text: str, side: Side) -> str:
    """Clean OCR noise common on Champions slot cards."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""

    if side == "player":
        # Thin "/" is often read as "7", gluing current/max into one token (1627162).
        cleaned = re.sub(r"\b(\d{2,3})7(\d{2,3})\b", r"\1 / \2", cleaned)
    else:
        # Italic "%" is often read as "*" (e.g. "46*" → "46%").
        cleaned = re.sub(r"\b(\d{1,3})\s*\*", r"\1%", cleaned)
        # Collapse "100 %" spacing; repair common 100% OCR tails (1005/100e).
        cleaned = re.sub(r"\b(\d{1,3})\s+%", r"\1%", cleaned)
        cleaned = re.sub(r"\b(100)[5eEoOsSgG]\b", r"100%", cleaned)
        if not re.search(r"\d\s*%", cleaned):
            cleaned = re.sub(r"\b(\d{1,3})\s*$", r"\1%", cleaned)

    return cleaned


def _ocr_slot_card_text(crop_rgb: np.ndarray, side: Side | None = None) -> str:
    """OCR a slot-card crop with a side-specific preprocess."""
    prepared = _preprocess_slot_card_for_ocr(crop_rgb, side=side)
    lines = read_text(prepared, detail=0, paragraph=True)
    return " ".join(lines).strip()


def _preprocess_slot_card_for_ocr(
    crop_rgb: np.ndarray,
    *,
    side: Side | None = None,
) -> np.ndarray:
    """Side-specific mask/threshold, then upscale for EasyOCR.

    Player name banners are purple/blue; opponent banners are pink/magenta.
    One HSV gate cannot serve both: the player gate starves white strokes on pink
    (live Musharna → ``40m1"``), while gray-threshold damages player HP fractions.
    """
    if side == "opponent":
        return _preprocess_opponent_slot_card(crop_rgb)
    return _preprocess_player_slot_card(crop_rgb)


def _preprocess_player_slot_card(crop_rgb: np.ndarray) -> np.ndarray:
    """HSV low-sat + bright mask for purple/blue player name banners."""
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    keep = (hsv[:, :, 1] <= _SLOT_CARD_SAT_MAX) & (hsv[:, :, 2] >= _SLOT_CARD_VALUE_MIN)

    masked = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
    masked[keep] = 255
    upscaled = cv2.resize(
        masked,
        None,
        fx=_SLOT_CARD_OCR_UPSCALE,
        fy=_SLOT_CARD_OCR_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    return cv2.cvtColor(upscaled, cv2.COLOR_GRAY2RGB)


def _preprocess_opponent_slot_card(crop_rgb: np.ndarray) -> np.ndarray:
    """Gray threshold + 3× upscale for pink/magenta opponent name banners."""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    upscaled = cv2.resize(
        gray,
        None,
        fx=_OPPONENT_SLOT_OCR_UPSCALE,
        fy=_OPPONENT_SLOT_OCR_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    _, thresh = cv2.threshold(
        upscaled, _OPPONENT_SLOT_GRAY_THRESH, 255, cv2.THRESH_BINARY
    )
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


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
    # Italic trailing "l" is often OCR'd as "/" (e.g. Grimmsnar/ → Grimmsnarl).
    # Apply before stripping "/", or the letter is lost.
    if candidate.endswith("/"):
        candidate = candidate[:-1] + "l"
    return candidate.strip(" .:!-")


def _species_matches(known: str, observed: str) -> bool:
    a = known.casefold().replace("-", " ").replace("'", "").strip()
    b = observed.casefold().replace("-", " ").replace("'", "").strip()
    return a == b or a in b or b in a


@lru_cache(maxsize=1)
def _slot_card_template() -> np.ndarray | None:
    """Averaged grayscale of normal + highlighted cards (action_selection.png)."""
    path = cv_templates_dir() / "slot_card.png"
    if not path.is_file():
        return None
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _hp_bar_in_card(config: RegionConfig) -> tuple[int, int, int, int]:
    """Shared card-relative HP bar rect, taken from player_slot_1 calibration."""
    try:
        cx, cy, _cw, _ch = config.get("player_slot_1_card")
        bx, by, bw, bh = config.get("player_slot_1_hp_bar")
        return (bx - cx, by - cy, bw, bh)
    except KeyError:
        return _HP_BAR_IN_CARD_DEFAULT


def _crop_hp_bar(card_rgb: np.ndarray, bar_in_card: tuple[int, int, int, int]) -> np.ndarray:
    """Crop the HP bar from a slot-card crop using the shared relative rect."""
    x, y, w, h = bar_in_card
    return card_rgb[y : y + h, x : x + w].copy()


def _slot_card_visible(crop_rgb: np.ndarray) -> bool:
    """True when the shared slot-card template matches (highlight or normal)."""
    template = _slot_card_template()
    if template is None:
        return False
    # matchTemplate is unstable on near-empty buffers.
    if float(crop_rgb.mean()) <= _EMPTY_CROP_MEAN_MAX:
        return False

    prepared = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    if prepared.shape != template.shape:
        template = cv2.resize(template, (prepared.shape[1], prepared.shape[0]))
    score = cv2.matchTemplate(
        prepared.astype(np.float32),
        template.astype(np.float32),
        cv2.TM_CCOEFF_NORMED,
    )[0, 0]
    return float(score) >= _SLOT_CARD_TEMPLATE_SCORE_MIN


def _downscale_bar_gray(crop_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    return cv2.resize(
        gray,
        (
            max(1, width // _HP_BAR_DIFF_DOWNSCALE),
            max(1, height // _HP_BAR_DIFF_DOWNSCALE),
        ),
        interpolation=cv2.INTER_AREA,
    )


def _hp_bar_changed(crop_rgb: np.ndarray, previous_gray: np.ndarray | None) -> bool:
    """Motion gate on the HP-bar ROI only (not the full animated background)."""
    current_gray = _downscale_bar_gray(crop_rgb)
    if previous_gray is None:
        return True
    if previous_gray.shape != current_gray.shape:
        return True
    diff = cv2.absdiff(previous_gray, current_gray)
    return float(diff.mean()) >= _HP_BAR_DIFF_MEAN_MIN
