"""Regression tests for phase detection on reference screenshots."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.cv.phase_detector import (
    PhaseDetector,
    has_battle_ended,
    is_action_selection_standby_visible,
    is_team_preview,
    is_team_selection_standby_visible,
)
from app.cv.regions import config_for_image, default_assets_dir, load_regions
from app.services.session import BattlePhase


def _load_asset(name: str) -> np.ndarray:
    path = default_assets_dir() / name
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _advance_to_battle_animation(detector: PhaseDetector) -> None:
    detector.detect(_load_asset("team_preview.png"))
    detector.detect(_load_asset("team_selection.png"))
    detector.detect(_load_asset("battle_text.png"))


def _advance_to_action_selection(detector: PhaseDetector) -> None:
    _advance_to_battle_animation(detector)
    detector.detect(_load_asset("action_selection.png"))


@pytest.fixture
def region_config():
    return load_regions()


def test_is_team_preview_template_match(region_config) -> None:
    """Near-gray template match detects team preview prompt without OCR."""
    preview = _load_asset("team_preview.png")
    preview_config = config_for_image(region_config, preview)
    assert is_team_preview(preview, preview_config) is True

    selection = _load_asset("team_selection.png")
    selection_config = config_for_image(region_config, selection)
    assert is_team_preview(selection, selection_config) is False

    action = _load_asset("action_selection.png")
    action_config = config_for_image(region_config, action)
    assert is_team_preview(action, action_config) is False


def test_is_team_selection_standby_template_match(region_config) -> None:
    """Near-gray template match detects Preparing for Battle without OCR."""
    image = _load_asset("team_selection.png")
    display_config = config_for_image(region_config, image)
    assert is_team_selection_standby_visible(image, display_config) is True

    preview = _load_asset("team_preview.png")
    preview_config = config_for_image(region_config, preview)
    assert is_team_selection_standby_visible(preview, preview_config) is False

    action = _load_asset("action_selection.png")
    action_config = config_for_image(region_config, action)
    assert is_team_selection_standby_visible(action, action_config) is False


def test_is_action_selection_standby_template_match(region_config) -> None:
    """Near-gray template match detects Communicating... without OCR."""
    standby = _load_asset("standby.png")
    standby_config = config_for_image(region_config, standby)
    assert is_action_selection_standby_visible(standby, standby_config) is True

    action = _load_asset("action_selection.png")
    action_config = config_for_image(region_config, action)
    assert is_action_selection_standby_visible(action, action_config) is False

    battle = _load_asset("battle_text.png")
    battle_config = config_for_image(region_config, battle)
    assert is_action_selection_standby_visible(battle, battle_config) is False


def test_action_selection_poll_interval_is_five_fps() -> None:
    from app.services.cv_runner import _ACTION_SELECTION_POLL_SEC, _poll_interval

    assert abs(_ACTION_SELECTION_POLL_SEC - 0.2) < 1e-9
    assert abs(_poll_interval(BattlePhase.ACTION_SELECTION) - 0.2) < 1e-9


def test_phase_detector_strict_idle_to_team_preview_only(region_config) -> None:
    detector = PhaseDetector(region_config)

    transition = detector.detect_transition(_load_asset("action_selection.png"))
    assert transition.previous == BattlePhase.IDLE
    assert transition.current == BattlePhase.IDLE

    transition = detector.detect_transition(_load_asset("battle_text.png"))
    assert transition.current == BattlePhase.IDLE


def test_phase_detector_stateful_flow(region_config) -> None:
    detector = PhaseDetector(region_config)

    transition = detector.detect_transition(_load_asset("team_preview.png"))
    assert transition.previous == BattlePhase.IDLE
    assert transition.current == BattlePhase.TEAM_PREVIEW
    assert transition.entered_team_preview is True

    transition = detector.detect_transition(_load_asset("team_selection.png"))
    assert transition.previous == BattlePhase.TEAM_PREVIEW
    assert transition.current == BattlePhase.TEAM_SELECTED
    assert transition.entered_team_selected is True

    transition = detector.detect_transition(_load_asset("battle_text.png"))
    assert transition.previous == BattlePhase.TEAM_SELECTED
    assert transition.current == BattlePhase.BATTLE_ANIMATION
    assert transition.entered_battle_animation is True
    assert transition.entered_battle is True

    transition = detector.detect_transition(_load_asset("action_selection.png"))
    assert transition.previous == BattlePhase.BATTLE_ANIMATION
    assert transition.current == BattlePhase.ACTION_SELECTION
    assert transition.entered_action_selection is True
    assert transition.entered_battle_animation is False

    transition = detector.detect_transition(_load_asset("standby.png"))
    assert transition.previous == BattlePhase.ACTION_SELECTION
    assert transition.current == BattlePhase.BATTLE_ANIMATION
    assert transition.entered_battle_animation is True
    assert transition.entered_battle is False

    transition = detector.detect_transition(_load_asset("battle_text.png"))
    assert transition.previous == BattlePhase.BATTLE_ANIMATION
    assert transition.current == BattlePhase.BATTLE_ANIMATION
    assert transition.entered_action_selection is False
    assert transition.entered_battle_animation is False


def test_phase_detector_idle_can_enter_team_selected_directly(region_config) -> None:
    detector = PhaseDetector(region_config)
    transition = detector.detect_transition(_load_asset("team_selection.png"))
    assert transition.previous == BattlePhase.IDLE
    assert transition.current == BattlePhase.TEAM_SELECTED
    assert transition.entered_team_selected is True


def test_phase_detector_latches_action_selection_until_standby(region_config) -> None:
    """Strict rules: ACTION_SELECTION only exits via standby, not missing FIGHT."""
    detector = PhaseDetector(region_config)
    _advance_to_action_selection(detector)
    assert detector.phase == BattlePhase.ACTION_SELECTION

    transition = detector.detect_transition(_load_asset("battle_text.png"))
    assert transition.previous == BattlePhase.ACTION_SELECTION
    assert transition.current == BattlePhase.ACTION_SELECTION


def test_phase_detector_battle_end_returns_to_idle(region_config) -> None:
    detector = PhaseDetector(region_config)
    _advance_to_battle_animation(detector)
    assert detector.phase == BattlePhase.BATTLE_ANIMATION

    transition = detector.detect_transition(_load_asset("battle_end.png"))
    assert transition.previous == BattlePhase.BATTLE_ANIMATION
    assert transition.current == BattlePhase.IDLE


def test_has_battle_ended_on_battle_end_screenshot(region_config) -> None:
    image = _load_asset("battle_end.png")
    display_config = config_for_image(region_config, image)
    assert has_battle_ended(image, display_config) is True


def test_phase_detector_reset(region_config) -> None:
    detector = PhaseDetector(region_config)
    _advance_to_action_selection(detector)
    assert detector.phase == BattlePhase.ACTION_SELECTION

    detector.reset()
    assert detector.phase == BattlePhase.IDLE
