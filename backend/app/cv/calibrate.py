"""
Interactive region calibration for BlueStacks 1600x900 captures.

Usage (from backend/):
    python -m app.cv.calibrate
    python -m app.cv.calibrate --image ../assets/cv/action_selection.png
    python -m app.cv.calibrate --live
    python -m app.cv.calibrate --export-all

Controls:
    n / p       Next / previous region
    Arrow keys  Move selected region (hold Shift for 10px steps)
    +/-         Grow / shrink region (hold Shift for 10px steps)
    r           Reset selected region to last saved value
    d           Draw a new rectangle with the mouse (click-drag)
    s           Save config to disk
    l           Refresh live ADB capture (--live mode only)
    q / Esc     Quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from app.cv.adb_capture import capture_screenshot, get_adb_device, is_adb_connected
from app.cv.regions import (
    RegionConfig,
    RegionRect,
    config_for_image,
    crop_region,
    default_assets_dir,
    default_config_path,
    load_regions,
    save_regions,
)

REGION_COLORS: dict[str, tuple[int, int, int]] = {
    "team_preview_prompt": (0, 200, 255),
    "team_selection_standby": (0, 200, 255),
    "player_team_selection": (0, 200, 255),
    "opponent_team_preview": (0, 128, 255),
    "fight_button": (255, 0, 255),
    "player_slot_1_card": (0, 255, 0),
    "player_slot_2_card": (0, 200, 0),
    "opponent_slot_1_card": (0, 0, 255),
    "opponent_slot_2_card": (80, 80, 255),
    "player_slot_1_banner": (0, 255, 128),
    "player_slot_2_banner": (0, 200, 128),
    "opponent_slot_1_banner": (128, 128, 255),
    "opponent_slot_2_banner": (80, 80, 255),
    "battle_text": (255, 255, 0),
    "action_selection_standby": (255, 180, 0),
}

HELP_LINES = [
    "n/p: next/prev region",
    "arrows: move | +/-: resize",
    "d: mouse draw | s: save",
    "l: refresh live | q: quit",
]


def _require_opencv_gui() -> None:
    try:
        cv2.namedWindow("__opencv_gui_probe__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__opencv_gui_probe__")
    except cv2.error as exc:
        raise SystemExit(
            "OpenCV GUI is unavailable. The calibration tool requires opencv-python "
            "(not opencv-python-headless).\n\n"
            "  pip uninstall opencv-python-headless\n"
            "  pip install opencv-python"
        ) from exc


def _rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _load_reference_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Reference image not found: {path}")
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return _rgb_to_bgr(rgb)


def _region_color(name: str) -> tuple[int, int, int]:
    return REGION_COLORS.get(name, (200, 200, 200))


def _draw_overlay(
    frame: np.ndarray,
    config: RegionConfig,
    selected: str,
    *,
    show_labels: bool = True,
) -> np.ndarray:
    canvas = frame.copy()
    for name in config.names():
        x, y, w, h = config.get(name)
        color = _region_color(name)
        thickness = 1
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)
        if show_labels:
            label = name
            cv2.putText(
                canvas,
                label,
                (x + 4, max(y - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    cv2.rectangle(canvas, (8, 8), (520, 8 + 18 * len(HELP_LINES) + 12), (20, 20, 20), -1)
    for idx, line in enumerate(HELP_LINES):
        cv2.putText(
            canvas,
            line,
            (14, 28 + idx * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    sel_rect = config.get(selected)
    status = (
        f"{selected}  [{sel_rect[0]}, {sel_rect[1]}, {sel_rect[2]}, {sel_rect[3]}]  "
        f"target {config.resolution[0]}x{config.resolution[1]}"
    )
    cv2.putText(
        canvas,
        status,
        (14, canvas.shape[0] - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


class Calibrator:
    def __init__(
        self,
        frame: np.ndarray,
        config: RegionConfig,
        *,
        live: bool = False,
    ) -> None:
        self.base_frame = frame
        self.config = config
        self.saved_regions = dict(config.regions)
        self.live = live
        self.selected_idx = 0
        self.window = "PokeChamp CV Calibrate"
        self._drawing = False
        self._drag_start: tuple[int, int] | None = None
        self._drag_preview: RegionRect | None = None

    @property
    def selected_name(self) -> str:
        return self.config.names()[self.selected_idx]

    def _set_rect(self, name: str, rect: RegionRect) -> None:
        regions = dict(self.config.regions)
        regions[name] = rect
        self.config = RegionConfig(resolution=self.config.resolution, regions=regions)

    def _nudge(self, dx: int, dy: int, dw: int, dh: int) -> None:
        name = self.selected_name
        x, y, w, h = self.config.get(name)
        self._set_rect(name, (max(0, x + dx), max(0, y + dy), max(1, w + dw), max(1, h + dh)))

    def _next_region(self, step: int) -> None:
        names = self.config.names()
        self.selected_idx = (self.selected_idx + step) % len(names)

    def _reset_region(self) -> None:
        name = self.selected_name
        self._set_rect(name, self.saved_regions[name])

    def _apply_drag_rect(self, rect: RegionRect) -> None:
        self._set_rect(self.selected_name, rect)

    def _mouse_callback(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._drag_start = (x, y)
            self._drag_preview = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing and self._drag_start is not None:
            x0, y0 = self._drag_start
            rx = min(x0, x)
            ry = min(y0, y)
            self._drag_preview = (rx, ry, abs(x - x0), abs(y - y0))
        elif event == cv2.EVENT_LBUTTONUP and self._drawing and self._drag_start is not None:
            x0, y0 = self._drag_start
            rect = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            if rect[2] >= 4 and rect[3] >= 4:
                self._apply_drag_rect(rect)
            self._drawing = False
            self._drag_start = None
            self._drag_preview = None

    def _render(self) -> np.ndarray:
        canvas = _draw_overlay(self.base_frame, self.config, self.selected_name)
        if self._drag_preview is not None:
            x, y, w, h = self._drag_preview
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 2)
        return canvas

    def refresh_live_frame(self) -> None:
        if not self.live:
            return
        rgb = capture_screenshot()
        self.base_frame = _rgb_to_bgr(rgb)

    def run(self) -> RegionConfig:
        _require_opencv_gui()
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 720)
        cv2.setMouseCallback(self.window, self._mouse_callback)

        while True:
            cv2.imshow(self.window, self._render())
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break

            step = 10 if key in (ord("N"), ord("P")) else 1
            if key in (ord("n"), ord("N")):
                self._next_region(1)
            elif key in (ord("p"), ord("P")):
                self._next_region(-1)
            elif key in (81, 2424832):  # left arrow
                self._nudge(-step, 0, 0, 0)
            elif key in (83, 2555904):  # right arrow
                self._nudge(step, 0, 0, 0)
            elif key in (82, 2490368):  # up arrow
                self._nudge(0, -step, 0, 0)
            elif key in (84, 2621440):  # down arrow
                self._nudge(0, step, 0, 0)
            elif key in (ord("+"), ord("=")):
                self._nudge(0, 0, step, step)
            elif key == ord("-"):
                self._nudge(0, 0, -step, -step)
            elif key == ord("r"):
                self._reset_region()
            elif key == ord("l") and self.live:
                try:
                    self.refresh_live_frame()
                except RuntimeError as exc:
                    print(f"Live capture failed: {exc}", file=sys.stderr)
            elif key == ord("s"):
                path = save_regions(self.config)
                self.saved_regions = dict(self.config.regions)
                print(f"Saved regions to {path}")

        cv2.destroyAllWindows()
        return self.config


def _export_overlays(config: RegionConfig, assets_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(assets_dir.glob("*.png")):
        frame = _load_reference_image(image_path)
        display_config = config_for_image(config, frame)
        overlay = _draw_overlay(frame, display_config, selected=display_config.names()[0], show_labels=True)
        out_path = output_dir / f"{image_path.stem}_regions.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"Wrote {out_path}")


def _preview_crops(config: RegionConfig, image_path: Path) -> None:
    frame_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    display_config = config_for_image(config, frame_rgb)
    print(f"\nCrops from {image_path.name} ({frame_rgb.shape[1]}x{frame_rgb.shape[0]}):")
    for name in display_config.names():
        try:
            crop = crop_region(frame_rgb, display_config.get(name))
            print(f"  {name}: {crop.shape[1]}x{crop.shape[0]}")
        except ValueError as exc:
            print(f"  {name}: SKIP ({exc})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate BlueStacks CV screen regions.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Region config JSON (default: config/bluestacks_1600x900.json)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=default_assets_dir() / "action_selection.png",
        help="Reference screenshot for static calibration",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Calibrate against a live ADB screencap (requires connected BlueStacks)",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export overlay PNGs for every reference screenshot, then exit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_assets_dir() / "calibration",
        help="Output directory for --export-all",
    )
    parser.add_argument(
        "--list-crops",
        action="store_true",
        help="Print crop sizes for each region on --image, then exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    config_path = args.config or default_config_path()
    config = load_regions(config_path)

    if args.export_all:
        _export_overlays(config, default_assets_dir(), args.output_dir)
        return 0

    if args.list_crops:
        _preview_crops(config, args.image)
        return 0

    if args.live:
        device = get_adb_device()
        if not is_adb_connected(device):
            print(
                f"ADB device '{device}' is not connected. "
                "Enable ADB in BlueStacks and run: adb connect 127.0.0.1:5555",
                file=sys.stderr,
            )
            return 1
        try:
            frame = _rgb_to_bgr(capture_screenshot(device))
        except RuntimeError as exc:
            print(f"Capture failed: {exc}", file=sys.stderr)
            return 1
        print(f"Live capture from {device} ({frame.shape[1]}x{frame.shape[0]})")
    else:
        frame = _load_reference_image(args.image)
        print(f"Calibrating on {args.image} ({frame.shape[1]}x{frame.shape[0]})")

    expected_w, expected_h = config.resolution
    if frame.shape[1] != expected_w or frame.shape[0] != expected_h:
        print(
            f"Note: image is {frame.shape[1]}x{frame.shape[0]} but config targets "
            f"{expected_w}x{expected_h}. Overlays are scaled for display; saved coords stay "
            "in config resolution. Use --live on a 1600x900 emulator for final calibration.",
            file=sys.stderr,
        )
        display_config = config_for_image(config, frame)
    else:
        display_config = config

    calibrator = Calibrator(frame, display_config, live=args.live)
    final_display = calibrator.run()

    if frame.shape[1] != expected_w or frame.shape[0] != expected_h:
        scale_x = expected_w / frame.shape[1]
        scale_y = expected_h / frame.shape[0]
        final_regions = {
            name: (
                int(round(rect[0] * scale_x)),
                int(round(rect[1] * scale_y)),
                max(1, int(round(rect[2] * scale_x))),
                max(1, int(round(rect[3] * scale_y))),
            )
            for name, rect in final_display.regions.items()
        }
        final_config = RegionConfig(resolution=config.resolution, regions=final_regions)
    else:
        final_config = final_display

    save_path = save_regions(final_config, config_path)
    print(f"Saved regions to {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
