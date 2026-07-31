"""Detect battle UI phase from a single screenshot frame."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import easyocr

from app.cv.regions import (
    RegionConfig,
    config_for_image,
    crop_region,
    default_assets_dir,
    load_regions,
)
from app.services.session import BattlePhase

logger = logging.getLogger(__name__)

# Thresholds tuned on assets/cv reference screenshots at 1600x900.
_FIGHT_PURPLE_RATIO_MIN = 70.0
_FIGHT_TEMPLATE_SCORE_MIN = 0.55
_STANDBY_TEMPLATE_SCORE_MIN = 0.55
_TEAM_PREVIEW_RED_RATIO_MIN = 55.0
_BATTLE_UI_STD_MIN = 35.0
_BATTLE_UI_MEAN_MIN = 25.0
# Same near-gray + bright mask as HP slot cards (white "Communicating..." text).
_STANDBY_CHROMA_MAX = 25
_STANDBY_BRIGHT_MIN = 160

_BATTLE_UI_REGIONS = (
    "player_slot_1_card",
    "player_slot_2_card",
    "opponent_slot_1_card",
    "opponent_slot_2_card",
    "player_slot_1_banner",
    "player_slot_2_banner",
    "opponent_slot_1_banner",
    "opponent_slot_2_banner",
    "battle_text",
)

_BATTLE_END_TEXTS = ("forfeit", "forteit", "you defeated", "you lost to", "has ended")
_TEAM_SELECTION_STANDBY_MARKERS = ("preparing",)


@dataclass(frozen=True)
class PhaseTransition:
    previous: BattlePhase
    current: BattlePhase

    @property
    def changed(self) -> bool:
        return self.previous != self.current

    @property
    def entered_action_selection(self) -> bool:
        return (
            self.current == BattlePhase.ACTION_SELECTION
            and self.previous != BattlePhase.ACTION_SELECTION
        )

    @property
    def entered_team_preview(self) -> bool:
        return (
            self.current == BattlePhase.TEAM_PREVIEW
            and self.previous != BattlePhase.TEAM_PREVIEW
        )

    @property
    def entered_team_selected(self) -> bool:
        return (
            self.current == BattlePhase.TEAM_SELECTED
            and self.previous != BattlePhase.TEAM_SELECTED
        )

    @property
    def entered_battle(self) -> bool:
        """True when leaving team preview / selection into in-match animation."""
        return self.current == BattlePhase.BATTLE_ANIMATION and self.previous in (
            BattlePhase.TEAM_SELECTED,
            BattlePhase.TEAM_PREVIEW,
        )

    @property
    def entered_battle_animation(self) -> bool:
        return (
            self.current == BattlePhase.BATTLE_ANIMATION
            and self.previous != BattlePhase.BATTLE_ANIMATION
        )


@lru_cache(maxsize=1)
def _fight_button_template() -> np.ndarray | None:
    """Load grayscale FIGHT button template from assets/cv."""
    path = default_assets_dir() / "fight_button_template.png"
    if not path.is_file():
        return None
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


@lru_cache(maxsize=1)
def _standby_screen_template() -> np.ndarray | None:
    """Load near-gray/bright 'Communicating...' template from assets/cv."""
    path = default_assets_dir() / "standby_screen_template.png"
    if not path.is_file():
        return None
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _purple_ratio(hsv: np.ndarray) -> float:
    mask = cv2.inRange(hsv, np.array([100, 30, 60]), np.array([160, 255, 255]))
    return float(mask.mean())


def _red_ratio(hsv: np.ndarray) -> float:
    mask_low = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([15, 255, 255]))
    mask_high = cv2.inRange(hsv, np.array([165, 80, 80]), np.array([180, 255, 255]))
    combined = cv2.bitwise_or(mask_low, mask_high)
    return float(combined.mean())


def _preprocess_near_gray_bright(crop_rgb: np.ndarray) -> np.ndarray:
    """Keep near-gray bright pixels (white UI text); drop colored background."""
    r = crop_rgb[:, :, 0].astype(np.int16)
    g = crop_rgb[:, :, 1].astype(np.int16)
    b = crop_rgb[:, :, 2].astype(np.int16)
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    brightness = (r + g + b) // 3
    keep = (chroma <= _STANDBY_CHROMA_MAX) & (brightness >= _STANDBY_BRIGHT_MIN)
    masked = np.zeros(crop_rgb.shape[:2], dtype=np.uint8)
    masked[keep] = 255
    return masked


def _preprocess_prompt_for_ocr(crop_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


def _ocr_text(crop_rgb: np.ndarray) -> str:
    reader = getattr(_ocr_text, "_reader", None)
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        _ocr_text._reader = reader  # type: ignore[attr-defined]
    prepared = _preprocess_prompt_for_ocr(crop_rgb)
    lines = reader.readtext(prepared, detail=0, paragraph=True)
    return " ".join(lines).strip()


def _ocr_prompt_mentions_select_four(crop_rgb: np.ndarray) -> bool:
    text = _ocr_text(crop_rgb)
    return "select" in text and "4" in text


def _ocr_prompt_indicates_end(crop_rgb: np.ndarray) -> bool:
    text = _ocr_text(crop_rgb).lower()
    return any(marker in text for marker in _BATTLE_END_TEXTS)


def _ocr_prompt_indicates_team_selection_standby(crop_rgb: np.ndarray) -> bool:
    text = _ocr_text(crop_rgb).lower()
    return any(marker in text for marker in _TEAM_SELECTION_STANDBY_MARKERS)


def is_team_preview(image: np.ndarray, config: RegionConfig) -> bool:
    """Return True when the team preview screen is visible."""
    prompt_crop = crop_region(image, config.get("team_preview_prompt"))
    if _ocr_prompt_mentions_select_four(prompt_crop):
        return True

    preview_crop = crop_region(image, config.get("opponent_team_preview"))
    hsv = cv2.cvtColor(preview_crop, cv2.COLOR_RGB2HSV)
    return _red_ratio(hsv) >= _TEAM_PREVIEW_RED_RATIO_MIN

def has_battle_ended(image: np.ndarray, config: RegionConfig) -> bool:
    """Return True when the battle has ended."""
    prompt_crop = crop_region(image, config.get("battle_text"))
    return _ocr_prompt_indicates_end(prompt_crop)


def is_standby_screen_visible(image: np.ndarray, config: RegionConfig) -> bool:
    """Return True when the center standby screen shows 'Communicating...'."""
    crop = crop_region(image, config.get("standby_screen"))
    prepared = _preprocess_near_gray_bright(crop)

    template = _standby_screen_template()
    if template is None:
        return False

    if prepared.shape != template.shape:
        template = cv2.resize(template, (prepared.shape[1], prepared.shape[0]))
    score = cv2.matchTemplate(prepared, template, cv2.TM_CCOEFF_NORMED)[0, 0]
    return float(score) >= _STANDBY_TEMPLATE_SCORE_MIN


def is_team_selection_standby_visible(image: np.ndarray, config: RegionConfig) -> bool:
    """Return True when center text shows 'Preparing for Battle' after both sides lock in."""
    crop = crop_region(image, config.get("team_selection_standby_text"))
    return _ocr_prompt_indicates_team_selection_standby(crop)


def is_fight_button_visible(image: np.ndarray, config: RegionConfig) -> bool:
    """Return True when the bottom-right FIGHT action button is visible."""
    crop = crop_region(image, config.get("fight_button"))
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    template = _fight_button_template()
    if template is not None:
        if gray.shape != template.shape:
            template = cv2.resize(template, (gray.shape[1], gray.shape[0]))
        score = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)[0, 0]
        return float(score) >= _FIGHT_TEMPLATE_SCORE_MIN

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    return _purple_ratio(hsv) >= _FIGHT_PURPLE_RATIO_MIN


def has_battle_ui(image: np.ndarray, config: RegionConfig) -> bool:
    """Return True when in-battle HUD elements are visible in event regions."""
    for name in _BATTLE_UI_REGIONS:
        crop = crop_region(image, config.get(name))
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        if float(gray.std()) >= _BATTLE_UI_STD_MIN and float(gray.mean()) >= _BATTLE_UI_MEAN_MIN:
            return True
    return False


def detect_phase(
    image: np.ndarray,
    config: RegionConfig,
    *,
    in_match: bool = False,
    saw_team_preview: bool = False,
    current_phase: BattlePhase = BattlePhase.IDLE,
) -> BattlePhase:
    """
    Classify the current frame using priority-ordered checks.

    1. team_selected — "Preparing for Battle" after brings are locked
    2. team_preview — center prompt text or opponent preview column
    3. battle_animation — standby screen ("Communicating...")
    4. action_selection — FIGHT button visible (entry signal; PhaseDetector latches sub-menus)
    5. battle_animation — in-match without FIGHT or standby
    6. idle — fallback
    """
    if is_team_selection_standby_visible(image, config):
        return BattlePhase.TEAM_SELECTED
    if is_team_preview(image, config):
        return BattlePhase.TEAM_PREVIEW
    if is_standby_screen_visible(image, config):
        return BattlePhase.BATTLE_ANIMATION
    if is_fight_button_visible(image, config):
        return BattlePhase.ACTION_SELECTION
    if current_phase == BattlePhase.ACTION_SELECTION:
        return BattlePhase.ACTION_SELECTION
    if current_phase == BattlePhase.TEAM_SELECTED:
        return BattlePhase.TEAM_SELECTED
    if in_match or saw_team_preview or has_battle_ui(image, config):
        return BattlePhase.BATTLE_ANIMATION
    return BattlePhase.IDLE


class PhaseDetector:
    """Stateful phase detector for consecutive screenshot frames."""

    def __init__(self, config: RegionConfig | Path | str | None = None) -> None:
        if config is None:
            self._config = load_regions()
        elif isinstance(config, RegionConfig):
            self._config = config
        else:
            self._config = load_regions(config)

        self._phase = BattlePhase.IDLE
        self._in_match = False
        self._saw_team_preview = False

    @property
    def phase(self) -> BattlePhase:
        return self._phase

    def reset(self) -> None:
        self._phase = BattlePhase.IDLE
        self._in_match = False
        self._saw_team_preview = False

    def detect(self, image: np.ndarray) -> BattlePhase:
        """Update internal state and return the detected phase."""
        transition = self.detect_transition(image)
        return transition.current

    def detect_transition(self, image: np.ndarray) -> PhaseTransition:
        """Detect phase and return the transition from the previous frame."""
        previous = self._phase
        display_config = config_for_image(self._config, image)

        current = previous
        match previous:
            case BattlePhase.IDLE:
                # Detect new match
                if is_team_preview(image, display_config):
                    current = BattlePhase.TEAM_PREVIEW
                elif is_team_selection_standby_visible(image, display_config):
                    # Missed early preview frames; both sides already standing by.
                    current = BattlePhase.TEAM_SELECTED
            case BattlePhase.TEAM_PREVIEW:
                # Both sides locked in → "Preparing for Battle"
                if is_team_selection_standby_visible(image, display_config):
                    current = BattlePhase.TEAM_SELECTED
                # Fallback if preview UI disappears without the standby text (skipped frames)
                elif not is_team_preview(image, display_config):
                    current = BattlePhase.BATTLE_ANIMATION
            case BattlePhase.TEAM_SELECTED:
                # Battle begins when "Preparing for Battle" leaves the screen
                if not is_team_selection_standby_visible(image, display_config):
                    current = BattlePhase.BATTLE_ANIMATION
            case BattlePhase.BATTLE_ANIMATION:
                # Turn starts when FIGHT button is visible
                if is_fight_button_visible(image, display_config):
                    current = BattlePhase.ACTION_SELECTION
                # Battle ended
                elif has_battle_ended(image, display_config):
                    current = BattlePhase.IDLE
            case BattlePhase.ACTION_SELECTION:
                # Actions selected, animation begins
                if is_standby_screen_visible(image, display_config):
                    current = BattlePhase.BATTLE_ANIMATION

        self._phase = current
        return PhaseTransition(previous=previous, current=current)
