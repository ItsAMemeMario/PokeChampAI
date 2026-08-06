"""OCR event regions with frame diffing during battle animation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import easyocr
import numpy as np

from app.cv.event_parser import parse_battle_text, parse_side_banner
from app.cv.regions import RegionConfig, config_for_image, crop_region
from app.schema.battle_log import BattleLogEvent
from app.schema.common import Side, Slot

logger = logging.getLogger(__name__)

# Per-slot text-only banners (icons excluded from calibrated crops).
_EVENT_REGIONS: tuple[tuple[str, Side | None, Slot | None], ...] = (
    ("player_slot_1_banner", "player", 1),
    ("player_slot_2_banner", "player", 2),
    ("opponent_slot_1_banner", "opponent", 1),
    ("opponent_slot_2_banner", "opponent", 2),
    ("battle_text", None, None),
)

_BANNER_REGION_NAMES = frozenset(name for name, _, _ in _EVENT_REGIONS if name != "battle_text")

_REGION_CONTENT_STD_MIN = 35.0
_REGION_CONTENT_MEAN_MIN = 25.0
_BANNER_DARK_RATIO_MIN = 80.0
_BANNER_BRIGHT_RATIO_MIN = 3.0
_FRAME_DIFF_MEAN_MIN = 4.0
_DIFF_DOWNSCALE = 4


@dataclass
class EventOcrEngine:
    """Stateful OCR over per-slot banners and battle text with frame diffing."""

    _previous_frames: dict[str, np.ndarray | None] = field(default_factory=dict)
    _last_emitted_text: dict[str, str] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear diff state when leaving battle animation."""
        self._previous_frames.clear()
        self._last_emitted_text.clear()

    def process_frame(
        self,
        image: np.ndarray,
        config: RegionConfig,
        *,
        player_species: list[str] | None = None,
        opponent_species: list[str] | None = None,
    ) -> list[BattleLogEvent]:
        """OCR changed event regions and return parsed battle log events."""
        display_config = config_for_image(config, image)
        events: list[BattleLogEvent] = []

        for region_name, side, slot in _EVENT_REGIONS:
            logger.info("Processing region: %s", region_name)
            crop = crop_region(image, display_config.get(region_name))
            if not _region_has_content(crop, region_name):
                logger.info("Region %s has no content", region_name)
                self._previous_frames.pop(region_name, None)
                self._last_emitted_text.pop(region_name, None)
                continue

            prev = self._previous_frames.get(region_name)
            if not _region_changed(crop, prev):
                continue

            self._previous_frames[region_name] = _downscale_gray(crop)
            mode = "battle_text" if region_name == "battle_text" else "banner"
            text = _ocr_text(crop, mode=mode)
            logger.info("Raw OCRed text: %s", text)
            if not text:
                continue

            if text == self._last_emitted_text.get(region_name):
                continue

            region_events = self._parse_region(
                region_name,
                side,
                slot,
                text,
                player_species=player_species,
                opponent_species=opponent_species,
            )
            if not region_events:
                logger.debug("Unparsed OCR in %s: %r", region_name, text)
                continue

            self._last_emitted_text[region_name] = text
            events.extend(region_events)

        return events

    def _parse_region(
        self,
        region_name: str,
        side: Side | None,
        slot: Slot | None,
        text: str,
        *,
        player_species: list[str] | None = None,
        opponent_species: list[str] | None = None,
    ) -> list[BattleLogEvent]:
        if region_name == "battle_text":
            return parse_battle_text(
                text,
                player_species=player_species,
                opponent_species=opponent_species,
            )

        assert side is not None and slot is not None
        event = parse_side_banner(
            text,
            side,
            slot=slot,
            player_species=player_species,
            opponent_species=opponent_species,
        )
        return [event] if event is not None else []


def _region_has_content(crop_rgb: np.ndarray, region_name: str) -> bool:
    if region_name in _BANNER_REGION_NAMES:
        return _banner_has_content(crop_rgb)
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    return float(gray.std()) >= _REGION_CONTENT_STD_MIN and float(gray.mean()) >= _REGION_CONTENT_MEAN_MIN


def _banner_has_content(crop_rgb: np.ndarray) -> bool:
    """Detect white text on a dark text-only ability/item banner crop."""
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    dark_ratio = float((gray < 90).mean()) * 100.0
    bright_ratio = float((gray > 200).mean()) * 100.0
    return dark_ratio >= _BANNER_DARK_RATIO_MIN and bright_ratio >= _BANNER_BRIGHT_RATIO_MIN


def _downscale_gray(crop_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape[:2]
    return cv2.resize(
        gray,
        (max(1, width // _DIFF_DOWNSCALE), max(1, height // _DIFF_DOWNSCALE)),
        interpolation=cv2.INTER_AREA,
    )


def _region_changed(crop_rgb: np.ndarray, previous_gray: np.ndarray | None) -> bool:
    current_gray = _downscale_gray(crop_rgb)
    if previous_gray is None:
        return True
    if previous_gray.shape != current_gray.shape:
        return True
    diff = cv2.absdiff(previous_gray, current_gray)
    return float(diff.mean()) >= _FRAME_DIFF_MEAN_MIN


def _preprocess_for_ocr(crop_rgb: np.ndarray, *, mode: str = "banner") -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if mode == "battle_text":
        # White battle-message text on a dark translucent bar — Otsu often fragments it.
        _, thresh = cv2.threshold(upscaled, 180, 255, cv2.THRESH_BINARY)
    else:
        _, thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


def _ocr_text(crop_rgb: np.ndarray, *, mode: str = "banner") -> str:
    logger.info("OCRing text in %s", mode)
    reader = getattr(_ocr_text, "_reader", None)
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        _ocr_text._reader = reader  # type: ignore[attr-defined]
    prepared = _preprocess_for_ocr(crop_rgb, mode=mode)
    lines = reader.readtext(prepared, detail=0, paragraph=True)
    return " ".join(lines).strip()
