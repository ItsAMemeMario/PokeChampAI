"""Detect player bring-4 from team-preview selection-order badges."""

from __future__ import annotations

import logging
import re

import cv2
import numpy as np

from app.cv.ocr_reader import read_text
from app.cv.regions import RegionConfig, config_for_image, crop_region
from app.schema.team import PlayerTeam

logger = logging.getLogger(__name__)

_SLOT_COUNT = 6
_ORDER_BADGE_WIDTH_RATIO = 0.25
_ORDER_DIGIT_RE = re.compile(r"[1-6]")


def crop_player_team_selection(image: np.ndarray, config: RegionConfig) -> np.ndarray:
    """Crop the player's left team-preview / selection column."""
    display_config = config_for_image(config, image)
    return crop_region(image, display_config.get("player_team_selection"))


def split_player_selection_slots(image: np.ndarray, config: RegionConfig) -> list[np.ndarray]:
    """Vertically split ``player_team_selection`` into six equal panel crops (top → bottom)."""
    column = crop_player_team_selection(image, config)
    height, _ = column.shape[:2]
    slot_height = max(1, height // _SLOT_COUNT)
    slots: list[np.ndarray] = []
    for index in range(_SLOT_COUNT):
        y1 = index * slot_height
        y2 = height if index == _SLOT_COUNT - 1 else (index + 1) * slot_height
        slots.append(column[y1:y2].copy())
    return slots


def _order_badge_crop(slot_rgb: np.ndarray) -> np.ndarray:
    """Leftmost strip of a panel where the selection-order number badge appears."""
    width = slot_rgb.shape[1]
    badge_width = max(1, int(round(width * _ORDER_BADGE_WIDTH_RATIO)))
    return slot_rgb[:, :badge_width]


def _preprocess_order_badge_for_ocr(crop_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    upscaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


def _ocr_order_badge_text(crop_rgb: np.ndarray) -> str:
    prepared = _preprocess_order_badge_for_ocr(crop_rgb)
    lines = read_text(prepared, detail=0, paragraph=True)
    return " ".join(lines).strip()


def _parse_selection_order(text: str) -> int | None:
    """Extract a selection order digit (1–6) from OCR text."""
    match = _ORDER_DIGIT_RE.search(text)
    if match is None:
        return None
    return int(match.group(0))


def read_selection_orders(image: np.ndarray, config: RegionConfig) -> list[int | None]:
    """
    OCR the leftmost 25% of each panel for the selection-order badge.

    Returns a length-6 list: order digit when present, ``None`` when absent/unreadable.
    """
    slots = split_player_selection_slots(image, config)
    orders: list[int | None] = []
    for slot in slots:
        badge = _order_badge_crop(slot)
        text = _ocr_order_badge_text(badge)
        orders.append(_parse_selection_order(text))
    return orders


def read_player_selected_species(
    image: np.ndarray,
    config: RegionConfig,
    player_team: PlayerTeam,
) -> list[str]:
    """
    Map panels with selection-order badges to species via the player's pokepaste order.

    Returns selected species ordered by badge number (1 → N), typically 4.
    """
    if len(player_team.pokemon) != _SLOT_COUNT:
        raise ValueError(
            f"Player team must have {_SLOT_COUNT} Pokemon, got {len(player_team.pokemon)}"
        )

    orders = read_selection_orders(image, config)
    ranked = [
        (order, index, player_team.pokemon[index].species)
        for index, order in enumerate(orders)
        if order is not None
    ]
    ranked.sort(key=lambda item: (item[0], item[1]))
    species_ordered = [item[2] for item in ranked]
    logger.info("Player team selection: %s (orders=%s)", species_ordered, orders)
    return species_ordered
