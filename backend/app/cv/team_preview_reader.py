"""Read opponent team preview sprites from the right column via Gemini vision."""

from __future__ import annotations

import logging

import numpy as np

from app.cv.regions import RegionConfig, config_for_image, crop_region
from app.schema.team import OpponentTeamPreview
from app.services.gemini import GeminiService

logger = logging.getLogger(__name__)

# Fraction of each preview panel width used for the sprite (left side; type icons on right).
_PREVIEW_SLOT_COUNT = 6
_PREVIEW_SPRITE_HEIGHT = 95


def crop_opponent_team_preview(image: np.ndarray, config: RegionConfig) -> np.ndarray:
    """Crop the full opponent team-preview column (6 red panels, no species names)."""
    display_config = config_for_image(config, image)
    return crop_region(image, display_config.get("opponent_team_preview"))


def crop_opponent_sprite_slots(image: np.ndarray, config: RegionConfig) -> list[np.ndarray]:
    """
    Split the opponent preview column into six sprite-only crops (top to bottom).

    Each panel shows a sprite on the left and type/gender icons on the right.
    Species names are not shown on the opponent side during team preview.
    """
    column = crop_opponent_team_preview(image, config)
    height, _ = column.shape[:2]
    slot_height = max(1, height // _PREVIEW_SLOT_COUNT)

    slots: list[np.ndarray] = []
    for index in range(_PREVIEW_SLOT_COUNT):
        y1 = index * slot_height
        y2 = y1 + _PREVIEW_SPRITE_HEIGHT
        slots.append(column[y1:y2, :].copy())
    return slots


def stack_sprite_slots(slots: list[np.ndarray]) -> np.ndarray:
    """Vertically stack sprite crops into a single image for vision input."""
    if not slots:
        raise ValueError("At least one sprite slot is required")
    max_width = max(slot.shape[1] for slot in slots)
    padded = []
    for slot in slots:
        if slot.shape[1] < max_width:
            pad = np.zeros((slot.shape[0], max_width - slot.shape[1], 3), dtype=slot.dtype)
            padded.append(np.concatenate([slot, pad], axis=1))
        else:
            padded.append(slot)
    return np.concatenate(padded, axis=0)


async def read_opponent_team_preview(
    image: np.ndarray,
    config: RegionConfig,
    *,
    gemini: GeminiService | None = None,
) -> OpponentTeamPreview:
    """
    Crop opponent sprite panels and identify all six species via Gemini vision.

    Returns species in top-to-bottom order matching the on-screen preview column.
    """
    slots = crop_opponent_sprite_slots(image, config)
    vision_input = stack_sprite_slots(slots)
    service = gemini or GeminiService()
    species = await service.identify_opponent_species(vision_input)
    logger.info("Opponent team preview identified: %s", species)
    return OpponentTeamPreview(species=species)
