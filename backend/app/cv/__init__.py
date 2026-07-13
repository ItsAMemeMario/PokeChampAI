"""Computer vision package for BlueStacks screen capture and region processing."""

from app.cv.adb_capture import capture_screenshot, is_adb_connected
from app.cv.phase_detector import (
    PhaseDetector,
    PhaseTransition,
    detect_phase,
    has_battle_ui,
    is_fight_button_visible,
    is_standby_screen_visible,
    is_team_preview,
    has_battle_ended,
)
from app.cv.regions import RegionConfig, crop_region, default_config_path, load_regions, config_for_image, save_regions

__all__ = [
    "PhaseDetector",
    "PhaseTransition",
    "RegionConfig",
    "capture_screenshot",
    "config_for_image",
    "crop_region",
    "default_config_path",
    "detect_phase",
    "has_battle_ui",
    "is_adb_connected",
    "is_fight_button_visible",
    "is_standby_screen_visible",
    "is_team_preview",
    "has_battle_ended",
    "load_regions",
    "save_regions",
]
