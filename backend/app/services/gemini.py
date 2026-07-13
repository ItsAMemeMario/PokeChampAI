"""Gemini API integration for team preview vision and suggestions."""

from __future__ import annotations

import json
import logging
import os
from io import BytesIO

import numpy as np
from google import genai
from google.genai import types
from PIL import Image

from app.schema.suggestions import TeamPreviewSuggestion
from app.schema.team import OpponentTeamPreview, PlayerTeam

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
_VISION_IDENTIFY_SCHEMA = OpponentTeamPreview.model_json_schema()


class GeminiService:
    """Thin wrapper around google-genai for battle-assist prompts."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        client: genai.Client | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        if client is None and not resolved_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self._model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self._client = client or genai.Client(api_key=resolved_key)

    @staticmethod
    def _rgb_to_png_bytes(image: np.ndarray) -> bytes:
        pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue()

    async def identify_opponent_species(self, image: np.ndarray) -> list[str]:
        """Identify six opponent species from team-preview sprite crops."""
        prompt = (
            "You are analyzing a Pokemon Champions team preview screen crop. "
            "The image shows six opponent Pokemon sprites stacked vertically, top to bottom. "
            "No species names are visible — identify each Pokemon from its sprite and any "
            "visible type icons. Return exactly six species names in top-to-bottom order "
            "using standard English species names (e.g. 'Scizor', not 'Mega Scizor')."
        )
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=self._rgb_to_png_bytes(image),
                    mime_type="image/png",
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=_VISION_IDENTIFY_SCHEMA,
            ),
        )
        parsed = OpponentTeamPreview.model_validate_json(response.text or "{}")
        return parsed.species

    async def suggest_team_preview(
        self,
        player_team: PlayerTeam,
        opponent_species: list[str],
    ) -> TeamPreviewSuggestion:
        """Suggest bring-4 and lead pairs given the player's team and opponent's six."""
        player_payload = [mon.model_dump() for mon in player_team.pokemon]
        prompt = (
            "You are a Pokemon Champions Regulation M-B doubles VGC expert.\n"
            f"Player team: {json.dumps(player_payload)}\n"
            f"Opponent team (6): {json.dumps(opponent_species)}\n"
            "Predict which 4 the opponent will bring and suggest which 4 the player should "
            "bring, including selection order (first two entries in each bring list are the "
            "predicted lead pair). Return JSON matching the TeamPreviewSuggestion schema."
        )
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=TeamPreviewSuggestion.model_json_schema(),
            ),
        )
        return TeamPreviewSuggestion.model_validate_json(response.text or "{}")
