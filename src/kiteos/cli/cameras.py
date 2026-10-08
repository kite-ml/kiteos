# Copyright 2026 Kite ML.
"""``kiteos cameras list|check``: find the station cameras and focus them.

``list`` only enumerates devices (it never opens a camera). ``check`` opens the three station
cameras in a live window with a sharpness meter; it needs a desktop session and, on macOS, a
terminal with camera permission. ``missing_station_cameras`` is the start gate of the setups that
record cameras (collect, rollout).
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import typer
from rich.table import Table

from kiteos import brand
from kiteos.cli._common import StationOption, console, die, load_station, note, run_capture
from kiteos.config import CAMERA_ROLES, StationConfig, is_mock_camera, is_placeholder, platform_key

app = typer.Typer(help="Find and focus the station cameras.", no_args_is_help=True)

WIDTH, HEIGHT, FPS = 640, 480, 30
ROI = (WIDTH // 4, HEIGHT // 4, 3 * WIDTH // 4, 3 * HEIGHT // 4)  # centre box: x0, y0, x1, y1
# macOS cameras that are never station cameras: the built-in one, an iPhone or iPad (Continuity,
# Desk View), a display's camera and virtual (software) cameras.
NON_STATION_CAMERA = re.compile(
    r"(?i)facetime|macbook|built-?in|iphone|ipad|continuity|desk view|studio display|virtual"
)


@dataclass(frozen=True)
class CameraDevice:
    selector: str
    label: str


def enumerate_linux(globber: Callable[[str], list[str]] = glob.glob) -> list[CameraDevice]:
    """Stable V4L2 names (by-id / by-path) with the /dev/videoN node they point to."""
    devices: list[CameraDevice] = []
    for pattern in ("/dev/v4l/by-id/*", "/dev/v4l/by-path/*"):
        for path in sorted(globber(pattern)):
            devices.append(CameraDevice(path, os.path.realpath(path)))
    listed = {d.label for d in devices}
    for node in sorted(globber("/dev/video*")):
        if node not in listed:
            devices.append(CameraDevice(node, "no stable name"))
    return devices


def darwin_camera_entries(runner: Callable[..., Any] = run_capture) -> list[dict[str, Any]] | None:
    """system_profiler's camera entries (no camera is opened); None when it cannot be read."""
    result = runner(["system_profiler", "SPCameraDataType", "-json"], timeout=30.0)
    if not result.ok:
        return None
    try:
        return [e for e in json.loads(result.stdout).get("SPCameraDataType", []) if isinstance(e, dict)]
    except (ValueError, AttributeError):
        return None


def enumerate_darwin(runner: Callable[..., Any] = run_capture) -> list[CameraDevice]:
    """Cameras from system_profiler (never opened), with their PROBABLE OpenCV index.

    OpenCV's AVFoundation backend orders devices by unique ID, not in system_profiler's order, so
    the index shown is that sort (``?`` when the unique ID is missing). It is a hint: confirm the
    mapping with the cover test (`kiteos cameras check`).
    """
    entries = darwin_camera_entries(runner) or []
    ids = [str(e.get("spcamera_unique-id", "")) for e in entries]
    order = {uid: i for i, uid in enumerate(sorted(u for u in ids if u))} if all(ids) else {}
    devices = []
    for entry, uid in zip(entries, ids, strict=True):
        index = str(order[uid]) if uid in order else "?"
        label = f"{entry.get('_name', 'camera')} ({entry.get('spcamera_model-id', 'unknown model')})"
        if uid:
            label += f" id {uid}"
        devices.append(CameraDevice(index, label))
    return sorted(devices, key=lambda d: (d.selector == "?", d.selector))


def enumerate_cameras() -> list[CameraDevice]:
    return enumerate_darwin() if platform_key() == "darwin" else enumerate_linux()


def station_cameras(station: StationConfig) -> list[tuple[str, int | str | None]]:
    return [(role, station.camera(role)) for role in CAMERA_ROLES]


def missing_station_cameras(
    station: StationConfig,
    runner: Callable[..., Any] = run_capture,
    exists: Callable[[str], bool] = os.path.exists,
) -> str | None:
    """Why the station's cameras are not all connected, or None when they are (none is opened).

    ``mock`` roles need no camera (placeholders are refused before this). On Linux every role's
    device must exist. On macOS the roles are OpenCV indices into one list of all the cameras, so
    with a USB camera missing the numbers shift and a role would record another camera, or the
    built-in or iPhone one: at least one USB camera per role must be connected. Which camera
    each role gets is still the cover test's to confirm (`kiteos cameras check`).
    """
    roles = {
        role: selector
        for role, selector in station_cameras(station)
        if selector is not None and not is_placeholder(selector) and not is_mock_camera(selector)
    }
    if not roles:
        return None
    if platform_key() == "linux":
        paths = {
            role: f"/dev/video{selector}" if isinstance(selector, int) or str(selector).isdigit() else str(selector)
            for role, selector in roles.items()
        }
        missing = [f"{role} ({path})" for role, path in paths.items() if not exists(path)]
        if not missing:
            return None
        return (
            f"station camera {', '.join(missing)} not connected: the device does not exist. by-path names "
            "follow the USB port, so plug the camera back into the same port, or find its name with "
            "`kiteos cameras list`."
        )
    entries = darwin_camera_entries(runner)
    if entries is None:
        return "could not list this Mac's cameras (system_profiler failed); check them with `kiteos cameras list`"
    names = [str(entry.get("_name", "camera")) for entry in entries]
    usb = [name for name in names if not NON_STATION_CAMERA.search(name)]
    if len(usb) >= len(roles):
        return None
    others = [name for name in names if NON_STATION_CAMERA.search(name)]
    seen = ", ".join(usb) or "no USB camera"
    if others:
        seen += f" (not counted: {', '.join(others)})"
    return (
        f"{len(usb)} of the station's {len(roles)} USB cameras are connected ({', '.join(roles)}); this Mac sees "
        f"{seen}. Cameras are picked by number from one list, so with one missing the numbers shift and a role "
        "would record the wrong camera, or the built-in or iPhone one. Connect them, then confirm each role with "
        "`kiteos cameras list` and the cover test (`kiteos cameras check`)."
    )


@app.command("list")
def list_cameras(station: StationOption = None) -> None:
    """Show the station camera mapping and the cameras this host can see (never opens them)."""
    config = load_station(station)
    key = platform_key()
    table = Table(title=f"Station cameras ({key})", title_justify="left", show_edge=False)
    table.add_column("role")
    table.add_column("selector")
    table.add_column("status")
    for role, selector in station_cameras(config):
        if selector is None or is_placeholder(selector):
            status = "[fail]not configured[/fail]"
        elif isinstance(selector, str) and selector.startswith("/dev/"):
            status = "[ok]present[/ok]" if os.path.exists(selector) else "[fail]missing[/fail]"
        else:
            status = "[muted]index[/muted]"
        table.add_row(role, str(selector), status)
    console.print(table)

    devices = enumerate_cameras()
    found = Table(title="Cameras on this host", title_justify="left", show_edge=False)
    found.add_column("probable index" if key == "darwin" else "selector")
    found.add_column("device")
    for device in devices:
        found.add_row(device.selector, device.label)
    console.print(found if devices else "[warn]no cameras found[/warn]")
    if key == "darwin":
        note(
            "macOS: the index is OpenCV's probable order (sorted by unique ID) and includes the built-in "
            "and iPhone cameras; confirm every role with the cover test (`kiteos cameras check`)"
        )
    else:
        note("use the *-video-index0 by-id/by-path entry of each camera in the station file")


def sharpness(frame: Any) -> float:
    """Variance of the Laplacian over the centre box (higher = sharper)."""
    import cv2

    x0, y0, x1, y1 = ROI
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class _Reader(threading.Thread):
    def __init__(self, selector: int | str) -> None:
        import cv2

        super().__init__(daemon=True)
        backend = cv2.CAP_ANY
        if platform_key() == "linux" and isinstance(selector, str) and selector.startswith("/dev/"):
            backend = cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(selector, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        self.frame: Any = None
        self.lock = threading.Lock()
        self.running = True

    def run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def latest(self) -> Any:
        with self.lock:
            return None if self.frame is None else self.frame.copy()


def _label(img: Any, lines: list[str], color: tuple[int, int, int] = (255, 255, 255)) -> None:
    import cv2

    cv2.rectangle(img, (0, 0), (img.shape[1], 30 * len(lines) + 10), (0, 0, 0), -1)
    for n, text in enumerate(lines):
        cv2.putText(img, text, (10, 30 + 30 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def focus_view(cams: list[tuple[str, int | str]]) -> None:
    """Live 2x2 grid (three cameras + a 2x zoom of one) with a smoothed sharpness meter."""
    import cv2
    import numpy as np

    window = f"{brand.BRAND_NAME} camera focus - 1/2/3 zoom, r reset, q quit"
    readers = {role: _Reader(selector) for role, selector in cams}
    for reader in readers.values():
        reader.start()
    smooth = {role: 0.0 for role, _ in cams}
    best = {role: 0.0 for role, _ in cams}
    zoom_role = cams[0][0]
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 960)
    last_print = 0.0
    try:
        while True:
            panels: dict[str, tuple[Any, Any]] = {}
            for role, selector in cams:
                frame = readers[role].latest()
                if frame is None:
                    view = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
                    _label(view, [f"{role} ({selector})", "no frames"], (0, 0, 255))
                    panels[role] = (view, None)
                    continue
                frame = cv2.resize(frame, (WIDTH, HEIGHT)) if frame.shape[:2] != (HEIGHT, WIDTH) else frame
                s = sharpness(frame)
                smooth[role] = s if smooth[role] == 0 else 0.7 * smooth[role] + 0.3 * s
                best[role] = max(best[role], smooth[role])
                view = frame.copy()
                x0, y0, x1, y1 = ROI
                cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 255), 1)
                ratio = smooth[role] / best[role] if best[role] else 0.0
                cv2.rectangle(view, (0, HEIGHT - 16), (int(WIDTH * ratio), HEIGHT), (0, 200, 0), -1)
                _label(view, [f"{role} ({selector})", f"sharpness {smooth[role]:.0f}   best {best[role]:.0f}"])
                panels[role] = (view, frame)

            zoom_raw = panels[zoom_role][1]
            if zoom_raw is not None:
                x0, y0, x1, y1 = ROI
                zoom = cv2.resize(zoom_raw[y0:y1, x0:x1], (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
                _label(zoom, [f"ZOOM 2x: {zoom_role}", f"sharpness {smooth[zoom_role]:.0f}"], (0, 255, 255))
            else:
                zoom = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
            views = [panels[role][0] for role, _ in cams] + [zoom]
            while len(views) < 4:
                views.append(np.zeros((HEIGHT, WIDTH, 3), np.uint8))
            cv2.imshow(window, np.vstack([np.hstack(views[:2]), np.hstack(views[2:4])]))

            now = time.time()
            if now - last_print > 2:
                console.print("sharpness  " + "  ".join(f"{r}={smooth[r]:.0f} (best {best[r]:.0f})" for r, _ in cams))
                last_print = now

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                best = {role: 0.0 for role, _ in cams}
            if ord("1") <= key < ord("1") + len(cams):
                zoom_role = cams[key - ord("1")][0]
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        for reader in readers.values():
            reader.running = False
            reader.join(timeout=1)
            reader.cap.release()
        cv2.destroyAllWindows()
    console.print("live view closed")


@app.command("check")
def check(station: StationOption = None) -> None:
    """Live view of the three cameras with a sharpness meter for focusing (opens the cameras)."""
    config = load_station(station)
    cams: list[tuple[str, int | str]] = []
    for role, selector in station_cameras(config):
        if selector is None or is_placeholder(selector):
            die(f"camera {role} is not configured for {platform_key()}; run `kiteos cameras list`", code=2)
        cams.append((role, selector))
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        die(f"OpenCV is not available ({exc}); run kiteos from the runtime virtualenv", code=2)
    note("turn each lens to maximise its sharpness; 1/2/3 pick the zoomed camera, r resets, q quits")
    focus_view(cams)
