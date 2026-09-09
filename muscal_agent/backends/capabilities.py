"""Platform probe: what can this machine actually run?

Run it before blaming the model:

    python -m muscal_agent --doctor
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import sys
from typing import Any


def _has(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:  # noqa: BLE001 - any import failure counts as missing
        return False


def session_type() -> str:
    """x11 | wayland | unknown -- decides whether pyautogui can work at all."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return "native"
    if os.environ.get("WAYLAND_DISPLAY"):
        return "wayland"
    if os.environ.get("DISPLAY"):
        return "x11"
    if shutil.which("loginctl"):
        try:
            import subprocess

            out = subprocess.run(
                ["loginctl", "show-session", "self", "-p", "Type"],
                capture_output=True, text=True, timeout=5,
            ).stdout.lower()
            if "wayland" in out:
                return "wayland"
            if "x11" in out:
                return "x11"
        except Exception:  # noqa: BLE001
            pass
    return "unknown"


def report() -> dict[str, Any]:
    session = session_type()
    desktop_ok = _has("pyautogui") and session in {"native", "x11"}
    return {
        "platform": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "session": session,
        "torch": _has("torch"),
        "cuda": _cuda(),
        "liquid_audio": _has("liquid_audio"),
        "playwright": _has("playwright"),
        "pyautogui": _has("pyautogui"),
        "mss": _has("mss"),
        "trafilatura": _has("trafilatura"),
        "sounddevice": _has("sounddevice"),
        "soundfile": _has("soundfile"),
        "desktop_tools_usable": desktop_ok,
        "browser_tools_usable": _has("playwright"),
    }


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


HINTS = {
    "wayland": (
        "Wayland erkannt: pyautogui funktioniert hier nicht. "
        "Entweder eine X11-Session nutzen oder ydotool/wtype einsetzen."
    ),
    "unknown": (
        "Sitzungstyp unbekannt -- desktop_* Werkzeuge erst nach --doctor freigeben."
    ),
}


def print_report() -> None:
    info = report()
    width = max(len(k) for k in info)
    for key, value in info.items():
        print(f"  {key:<{width}}  {value}")
    hint = HINTS.get(info["session"])
    if hint:
        print(f"\n  ! {hint}")
    if not info["playwright"]:
        print("  ! browser_* braucht: pip install playwright && playwright install chromium")
    if not info["pyautogui"] and info["session"] in {"native", "x11"}:
        print("  ! desktop_* braucht: pip install pyautogui mss")
    if not info["trafilatura"]:
        print("  ! web_read extrahiert ohne trafilatura nur mittelmaessig: pip install trafilatura")
