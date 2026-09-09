"""Policy-enforcing dispatcher: tool call in, result out.

The model is untrusted input. Everything here exists so that a misheard word
cannot become an unwanted click, keystroke or app launch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from .catalog import TOOLS_BY_NAME, Tool
from .toolcall import ToolCall


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> set[str] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


@dataclass
class ToolResult:
    name: str
    ok: bool
    output: Any = None
    error: str | None = None
    skipped: bool = False

    def as_message(self, max_chars: int = 2000) -> str:
        """What the model sees in the `tool` role message."""
        if self.skipped:
            return json.dumps({"error": self.error or "skipped"}, ensure_ascii=False)
        if not self.ok:
            return json.dumps({"error": self.error}, ensure_ascii=False)
        payload = json.dumps({"result": self.output}, ensure_ascii=False)
        if len(payload) > max_chars:
            payload = payload[: max_chars - 3] + "..."
        return payload


@dataclass
class Policy:
    """What the agent is allowed to do. Defaults are deliberately boring."""

    allowed_tools: set[str] | None = None  # None = everything in the catalog
    allow_write: bool = False
    allow_destructive: bool = False
    dry_run: bool = True
    max_output_chars: int = 2000
    confirm: Callable[[ToolCall, Tool], bool] | None = None

    @classmethod
    def from_env(cls) -> "Policy":
        return cls(
            allowed_tools=_env_list("MUSCAL_AGENT_TOOLS"),
            allow_write=_env_flag("MUSCAL_AGENT_WRITE"),
            allow_destructive=_env_flag("MUSCAL_AGENT_DESTRUCTIVE"),
            dry_run=_env_flag("MUSCAL_AGENT_DRY_RUN", default=True),
            max_output_chars=int(os.environ.get("MUSCAL_AGENT_MAX_CHARS", "2000")),
        )


@dataclass
class Executor:
    policy: Policy = field(default_factory=Policy)

    def check(self, call: ToolCall) -> tuple[bool, str | None]:
        """Return (allowed, reason_if_denied)."""
        tool = TOOLS_BY_NAME.get(call.name)
        if tool is None:
            return False, f"unknown tool {call.name!r}"
        if self.policy.allowed_tools and call.name not in self.policy.allowed_tools:
            return False, f"tool {call.name!r} is not on the allow-list"
        if tool.risk == "write" and not self.policy.allow_write:
            return False, f"{call.name} can change state; set MUSCAL_AGENT_WRITE=1 to allow"
        if tool.risk == "destructive" and not self.policy.allow_destructive:
            return False, f"{call.name} is destructive; set MUSCAL_AGENT_DESTRUCTIVE=1 to allow"
        return True, None

    def execute(self, call: ToolCall) -> ToolResult:
        allowed, reason = self.check(call)
        if not allowed:
            return ToolResult(call.name, ok=False, error=reason, skipped=True)

        tool = TOOLS_BY_NAME[call.name]
        try:
            args = tool.validate(call.args)
        except KeyError as exc:
            return ToolResult(call.name, ok=False, error=str(exc), skipped=True)

        if self.policy.confirm and not self.policy.confirm(call, tool):
            return ToolResult(call.name, ok=False, error="rejected by confirmation", skipped=True)

        if self.policy.dry_run:
            return ToolResult(
                call.name, ok=True, output={"dry_run": True, "would_call": call.name, "args": args}
            )

        try:
            output = tool.impl(**args) if tool.impl else None
        except Exception as exc:  # noqa: BLE001 - a tool must never kill the loop
            return ToolResult(call.name, ok=False, error=f"{type(exc).__name__}: {exc}")
        return ToolResult(call.name, ok=True, output=output)

    def run(self, calls: list[ToolCall]) -> list[ToolResult]:
        return [self.execute(call) for call in calls]
