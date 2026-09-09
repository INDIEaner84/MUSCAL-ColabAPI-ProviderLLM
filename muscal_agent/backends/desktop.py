"""Desktop control: mouse, keyboard, app launch, screenshots.

This is the dangerous part of the agent. Three independent brakes:

1. it only acts when ``MUSCAL_DESKTOP=1`` is set (otherwise dry-run),
2. the executor's policy gates every tool by risk class,
3. ``MUSCAL_SCREEN`` can restrict clicks to a region if you want a sandbox.

Platform notes -- be honest about the limits:
* Windows / macOS: pyautogui works out of the box.
* Linux X11: works.
* Linux Wayland: pyautogui does NOT work; use the portal-based tooling of your
  desktop (ydotool/wtype) or run an X11 session.

For anything beyond toy use, prefer accessibility-tree automation over blind
coordinates (Windows: uiautomation, Linux: pyatspi, macOS: AX via PyObjC).
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

_ENABLED = os.environ.get("MUSCAL_DESKTOP", "0").lower() in {"1", "true", "yes", "on"}


def _require():
    try:
        import pyautogui  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "desktop tools need pyautogui: pip install pyautogui (Linux also needs xdg deps)"
        ) from exc
    return pyautogui


def _region() -> tuple[int, int, int, int] | None:
    """Optional clickable region: MUSCAL_SCREEN=x,y,w,h"""
    raw = os.environ.get("MUSCAL_SCREEN")
    if not raw:
        return None
    try:
        x, y, w, h = (int(v) for v in raw.split(","))
    except ValueError:
        return None
    return x, y, w, h


def launch(app: str) -> dict[str, Any]:
    if not _ENABLED:
        return {"dry_run": True, "would_launch": app}
    if sys.platform.startswith("win"):
        os.startfile(app)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-a", app])
    else:
        subprocess.Popen([app], start_new_session=True)
    return {"launched": app}


def type_text(text: str) -> dict[str, Any]:
    if not _ENABLED:
        return {"dry_run": True, "would_type": text}
    pyautogui = _require()
    pyautogui.write(text, interval=0.01)
    return {"typed_chars": len(text)}


def hotkey(keys: str) -> dict[str, Any]:
    parts = [k.strip().lower() for k in keys.split("+") if k.strip()]
    if not _ENABLED:
        return {"dry_run": True, "would_press": parts}
    pyautogui = _require()
    pyautogui.hotkey(*parts)
    return {"pressed": parts}


def click(x: int, y: int, clicks: int = 1) -> dict[str, Any]:
    region = _region()
    if region:
        rx, ry, rw, rh = region
        if not (rx <= x <= rx + rw and ry <= y <= ry + rh):
            return {"error": f"({x},{y}) outside allowed region {region}", "clicked": False}
    if not _ENABLED:
        return {"dry_run": True, "would_click": [x, y, clicks]}
    pyautogui = _require()
    pyautogui.click(x, y, clicks=clicks)
    return {"clicked": [x, y, clicks]}


def screenshot(path: str = "screenshot.png") -> dict[str, Any]:
    try:
        import mss  # type: ignore

        with mss.mss() as sct:
            sct.shot(output=path)
        return {"path": path}
    except ImportError:
        pass
    if not _ENABLED:
        return {"dry_run": True, "would_capture": path}
    pyautogui = _require()
    pyautogui.screenshot(path)
    return {"path": path}
