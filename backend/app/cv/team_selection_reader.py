"""Detect which of the player's six preview panels are selected (bring-4)."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from app.cv.regions import RegionConfig, config_for_image, crop_region
from app.schema.team import PlayerTeam

logger = logging.getLogger(__name__)

_SLOT_COUNT = 6
# Bright mint/lime highlight on selected panels (tuned on assets/cv/team_selection.png).
_SELECTED_GREEN_HSV_LOW = np.array([35, 25, 150], dtype=np.uint8)
_SELECTED_GREEN_HSV_HIGH = np.array([95, 180, 255], dtype=np.uint8)
_SELECTED_GREEN_RATIO_MIN = 15.0


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


def _slot_green_ratio(slot_rgb: np.ndarray) -> float:
    hsv = cv2.cvtColor(slot_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, _SELECTED_GREEN_HSV_LOW, _SELECTED_GREEN_HSV_HIGH)
    return float(mask.mean())


def detect_selected_slot_mask(image: np.ndarray, config: RegionConfig) -> list[bool]:
    """
    Return a length-6 boolean mask: True when the panel has the green selected highlight.

    Index 0 is the top panel (first Pokémon in the player's pokepaste order).
    """
    slots = split_player_selection_slots(image, config)
    return [_slot_green_ratio(slot) >= _SELECTED_GREEN_RATIO_MIN for slot in slots]


def read_player_selected_species(
    image: np.ndarray,
    config: RegionConfig,
    player_team: PlayerTeam,
) -> list[str]:
    """
    Map green-highlighted panels to species using the player's submitted team order.

    Returns selected species in top-to-bottom (pokepaste) order — typically 4.
    """
    selected = detect_selected_slot_mask(image, config)
    if len(player_team.pokemon) != _SLOT_COUNT:
        raise ValueError(
            f"Player team must have {_SLOT_COUNT} Pokemon, got {len(player_team.pokemon)}"
        )
    species = [
        player_team.pokemon[index].species
        for index, is_selected in enumerate(selected)
        if is_selected
    ]
    logger.info(
        "Player team selection: %s (mask=%s)",
        species,
        selected,
    )
    return species
