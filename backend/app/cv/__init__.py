"""Computer vision package for BlueStacks screen capture and region processing."""

from app.cv.adb_capture import capture_screenshot, is_adb_connected
from app.cv.regions import RegionConfig, crop_region, default_config_path, load_regions, config_for_image, save_regions

__all__ = [
    "RegionConfig",
    "capture_screenshot",
    "config_for_image",
    "crop_region",
    "default_config_path",
    "is_adb_connected",
    "load_regions",
    "save_regions",
]
