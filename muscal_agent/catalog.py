"""Tool catalog for the audio agent: web research, browser, desktop control.

Every tool is described once and used three times:

1. as a JSON schema for the LFM2.5 text/VL tool-calling prompt,
2. as the target vocabulary when generating training data,
3. as the runtime implementation the executor dispatches to.

Keep the vocabulary small. A 1.5B model trained on 41 functions (Liquid's own
recipe) already shows a long tail of rare functions; every extra function is
another way to be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Param:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    name: str
    description: str
    params: list[Param] = field(default_factory=list)
    impl: Callable[..., Any] | None = None
    # Risk class drives the executor's policy: read-only actions are allowed by
    # default, anything that can change or destroy state needs an explicit opt-in.
    risk: str = "read"  # read | write | destructive

    def schema(self) -> dict[str, Any]:
        properties = {
            p.name: {"type": p.type, "description": p.description} for p in self.params
        }
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [p.name for p in self.params if p.required],
            },
        }

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        known = {p.name: p for p in self.params}
        unknown = set(args) - set(known)
        if unknown:
            raise KeyError(f"{self.name}: unknown argument(s) {sorted(unknown)}")
        missing = [p.name for p in self.params if p.required and p.name not in args]
        if missing:
            raise KeyError(f"{self.name}: missing required argument(s) {missing}")
        resolved = {p.name: p.default for p in self.params if p.default is not None}
        resolved.update(args)
        return resolved


# --------------------------------------------------------------------------- #
# implementations
# --------------------------------------------------------------------------- #


def _impl_web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    from .backends import web

    return {"results": web.search(query, max_results=max_results)}


def _impl_web_read(url: str, max_chars: int = 4000) -> dict[str, Any]:
    from .backends import web

    return {"text": web.read(url, max_chars=max_chars)}


def _impl_browser_open(url: str) -> dict[str, Any]:
    from .backends import browser

    return browser.open(url)


def _impl_browser_click(selector: str) -> dict[str, Any]:
    from .backends import browser

    return browser.click(selector)


def _impl_browser_type(selector: str, text: str, submit: bool = False) -> dict[str, Any]:
    from .backends import browser

    return browser.type_text(selector, text, submit=submit)


def _impl_browser_snapshot(max_chars: int = 4000) -> dict[str, Any]:
    from .backends import browser

    return {"text": browser.snapshot(max_chars=max_chars)}


def _impl_desktop_launch(app: str) -> dict[str, Any]:
    from .backends import desktop

    return desktop.launch(app)


def _impl_desktop_type(text: str) -> dict[str, Any]:
    from .backends import desktop

    return desktop.type_text(text)


def _impl_desktop_hotkey(keys: str) -> dict[str, Any]:
    from .backends import desktop

    return desktop.hotkey(keys)


def _impl_desktop_click(x: int, y: int, clicks: int = 1) -> dict[str, Any]:
    from .backends import desktop

    return desktop.click(x, y, clicks=clicks)


def _impl_desktop_screenshot() -> dict[str, Any]:
    from .backends import desktop

    return desktop.screenshot()


TOOLS: list[Tool] = [
    Tool(
        name="web_search",
        description="Search the web and return titles, urls and snippets.",
        params=[
            Param("query", description="Search query"),
            Param("max_results", "integer", "How many results to return", required=False, default=5),
        ],
        impl=_impl_web_search,
    ),
    Tool(
        name="web_read",
        description="Fetch a URL and return its main text content.",
        params=[
            Param("url", description="Absolute http(s) URL"),
            Param("max_chars", "integer", "Truncate output to N characters", required=False, default=4000),
        ],
        impl=_impl_web_read,
    ),
    Tool(
        name="browser_open",
        description="Open a URL in the controlled browser and wait for load.",
        params=[Param("url", description="Absolute http(s) URL")],
        impl=_impl_browser_open,
        risk="write",
    ),
    Tool(
        name="browser_click",
        description="Click an element identified by a CSS selector.",
        params=[Param("selector", description="CSS selector, e.g. #login or button.submit")],
        impl=_impl_browser_click,
        risk="write",
    ),
    Tool(
        name="browser_type",
        description="Type text into an input field, optionally pressing Enter.",
        params=[
            Param("selector", description="CSS selector of the input"),
            Param("text", description="Text to type"),
            Param("submit", "boolean", "Press Enter afterwards", required=False, default=False),
        ],
        impl=_impl_browser_type,
        risk="write",
    ),
    Tool(
        name="browser_snapshot",
        description="Return the current page as text (accessibility tree or readable text).",
        params=[Param("max_chars", "integer", "Truncate output to N characters", required=False, default=4000)],
        impl=_impl_browser_snapshot,
    ),
    Tool(
        name="desktop_launch",
        description="Start an application by name.",
        params=[Param("app", description="Application name or path")],
        impl=_impl_desktop_launch,
        risk="write",
    ),
    Tool(
        name="desktop_type",
        description="Type text with the keyboard into the focused window.",
        params=[Param("text", description="Text to type")],
        impl=_impl_desktop_type,
        risk="write",
    ),
    Tool(
        name="desktop_hotkey",
        description="Press a keyboard shortcut, e.g. ctrl+c or alt+tab.",
        params=[Param("keys", description="Keys joined by +")],
        impl=_impl_desktop_hotkey,
        risk="write",
    ),
    Tool(
        name="desktop_click",
        description="Click at absolute screen coordinates.",
        params=[
            Param("x", "integer", "Horizontal pixel position"),
            Param("y", "integer", "Vertical pixel position"),
            Param("clicks", "integer", "1 single, 2 double", required=False, default=1),
        ],
        impl=_impl_desktop_click,
        risk="destructive",
    ),
    Tool(
        name="desktop_screenshot",
        description="Capture the screen and return the file path.",
        params=[],
        impl=_impl_desktop_screenshot,
    ),
]

TOOLS_BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}


def schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in TOOLS]


def system_prompt_block(fmt: str = "pipe") -> str:
    """Short tool description for the text/VL prompt. Keep it token-cheap."""
    lines = []
    for tool in TOOLS:
        args = ", ".join(p.name for p in tool.params)
        if fmt == "pipe":
            example = "|".join([tool.name] + [f"{p.name}=<{p.type}>" for p in tool.params])
            lines.append(f"{tool.name}({args}) -> {example}")
        else:
            lines.append(f"{tool.name}({args})")
    return "\n".join(lines)
