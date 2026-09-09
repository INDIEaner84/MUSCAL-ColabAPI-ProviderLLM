"""Playwright-backed browser control.

Deliberately text-first: `snapshot()` returns the accessibility tree, not a
screenshot. A 1.5B model cannot read a screenshot, and an a11y tree is both
cheaper in tokens and far more reliable to act on.

Install:
    pip install playwright && playwright install chromium
"""

from __future__ import annotations

import os
from typing import Any

_PAGE = None
_BROWSER = None


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "browser tools need Playwright: pip install playwright && playwright install chromium"
        ) from exc
    return sync_playwright


def _ensure_browser():
    global _BROWSER, _PAGE
    if _PAGE is not None:
        return _PAGE
    sync_playwright = _require_playwright()
    pw = sync_playwright().start()
    headless = os.environ.get("MUSCAL_BROWSER_HEADLESS", "1").lower() not in {"0", "false"}
    _BROWSER = pw.chromium.launch(headless=headless)
    _PAGE = _BROWSER.new_page()
    return _PAGE


def open(url: str) -> dict[str, Any]:  # noqa: A001 - tool name is browser_open
    page = _ensure_browser()
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    return {"url": page.url, "title": page.title()}


def click(selector: str) -> dict[str, Any]:
    page = _ensure_browser()
    page.click(selector, timeout=10_000)
    return {"clicked": selector, "url": page.url}


def type_text(selector: str, text: str, submit: bool = False) -> dict[str, Any]:
    page = _ensure_browser()
    page.fill(selector, text, timeout=10_000)
    if submit:
        page.keyboard.press("Enter")
    return {"typed": selector, "submit": submit, "url": page.url}


def snapshot(max_chars: int = 4000) -> str:
    """Accessibility tree as text -- the cheap, reliable way to 'see' a page."""
    page = _ensure_browser()
    try:
        snap = page.accessibility.snapshot()
        text = _flatten(snap) if snap else ""
    except Exception:  # noqa: BLE001 - fall back to inner text
        text = page.inner_text("body")
    if not text.strip():
        text = page.inner_text("body")
    return text[:max_chars]


def _flatten(node: dict[str, Any], depth: int = 0) -> str:
    lines = []
    name = (node.get("name") or "").strip()
    role = node.get("role") or ""
    if name or role:
        lines.append(f"{'  ' * depth}{role}: {name}".rstrip())
    for child in node.get("children", []) or []:
        lines.append(_flatten(child, depth + 1))
    return "\n".join(line for line in lines if line.strip())


def close() -> None:
    global _BROWSER, _PAGE
    if _BROWSER is not None:
        _BROWSER.close()
    _BROWSER = None
    _PAGE = None
