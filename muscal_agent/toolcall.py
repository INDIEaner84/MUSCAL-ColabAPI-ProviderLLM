"""Parse and serialise tool calls for the speech -> function-call track.

Two wire formats are supported, controled by ``configs/audio_agent_ft.yaml``:

``pipe`` (default, recommended for the audio model)
    ``web_search|query=liquid ai lfm|max_results=5``

    Compact, no special tokens, few tokens per call. This is the shape Liquid's
    own voice-assistant recipe uses (``HassStartTimer|minutes=5|name=oven``).

    Constraint: a value must not contain ``|`` (it separates arguments). ``=``
    is allowed -- parsing splits on the first ``=`` only, so URLs with query
    strings and CSS attribute selectors survive.

``pythonic``
    ``<|tool_call_start|>[web_search(query="liquid ai lfm", max_results=5)]<|tool_call_end|>``

    The native LFM2.5 tool format, consistent with the text/VL models. Costs
    more tokens per call, which matters for a 1.5B model.

Why this choice is not cosmetic: the training target *is* the interface. Every
prediction has to be parsed by ``muscal_agent.executor``, so one format must be
picked before generating training data and never changed afterwards.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"


class ToolCallError(ValueError):
    """Raised when a model output cannot be parsed into a valid tool call."""


@dataclass
class ToolCall:
    name: str
    args: dict[str, object] = field(default_factory=dict)

    def to_pipe(self) -> str:
        parts = [self.name]
        for key, value in self.args.items():
            parts.append(f"{key}={_scalar(value)}")
        return "|".join(parts)

    def to_pythonic(self) -> str:
        args = ", ".join(f"{k}={_py_repr(v)}" for k, v in self.args.items())
        return f"{TOOL_CALL_START}[{self.name}({args})]{TOOL_CALL_END}"

    def serialise(self, fmt: str = "pipe") -> str:
        if fmt == "pipe":
            return self.to_pipe()
        if fmt == "pythonic":
            return self.to_pythonic()
        raise ValueError(f"unknown tool-call format: {fmt!r}")

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.to_pipe()


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    # "|" separates arguments and cannot be escaped -- a value containing it
    # would be ambiguous. "=" is fine: parsing splits on the FIRST one only,
    # so URLs (...?a=1) and CSS selectors (input[name=q]) survive round-trips.
    if "|" in text or "\n" in text:
        raise ToolCallError(f"argument value may not contain '|' or newlines: {text!r}")
    return text


def _py_repr(value: object) -> str:
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

_PIPE_RE = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<rest>\|.*)?$", re.DOTALL)


def parse_pipe(text: str) -> ToolCall:
    """``web_search|query=lfm|max_results=5`` -> ToolCall."""
    match = _PIPE_RE.match(text.strip())
    if not match:
        raise ToolCallError(f"not a pipe-style tool call: {text!r}")
    name = match.group("name")
    args: dict[str, object] = {}
    rest = match.group("rest")
    if rest:
        for chunk in rest.split("|"):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ToolCallError(f"argument without '=' in {text!r}: {chunk!r}")
            key, _, raw = chunk.partition("=")
            args[key.strip()] = _coerce(raw.strip())
    return ToolCall(name=name, args=args)


def parse_pythonic(text: str) -> ToolCall:
    """Extract ``<tool_call_start>[fn(a="b")]<tool_call_end|>`` from model output."""
    body = text
    if TOOL_CALL_START in text:
        body = text.split(TOOL_CALL_START, 1)[1]
        if TOOL_CALL_END in body:
            body = body.split(TOOL_CALL_END, 1)[0]
    body = body.strip()
    if body.startswith("[") and body.endswith("]"):
        body = body[1:-1]

    try:
        tree = ast.parse(body, mode="eval")
    except SyntaxError as exc:
        raise ToolCallError(f"cannot parse tool call {text!r}: {exc}") from exc
    node = tree.body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ToolCallError(f"expected a single function call, got {body!r}")

    args: dict[str, object] = {}
    for kw in node.keywords:
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except ValueError:
            args[kw.arg] = _unparse_str(kw.value)
    return ToolCall(name=node.func.id, args=args)


def _unparse_str(node: ast.expr) -> str:
    """Best-effort fallback for string literals built from non-literal parts."""
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.JoinedStr):
        out = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                out.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                out.append(_unparse_str(value.value))
        return "".join(out)
    return ast.unparse(node)


def parse_tool_call(text: str, fmt: str = "pipe") -> ToolCall:
    if fmt == "pipe":
        return parse_pipe(text)
    if fmt == "pythonic":
        return parse_pythonic(text)
    raise ValueError(f"unknown tool-call format: {fmt!r}")


def extract_tool_calls(text: str, fmt: str = "pipe") -> list[ToolCall]:
    """Pull every tool call out of a model response."""
    if fmt == "pythonic":
        return [
            parse_pythonic(chunk)
            for chunk in re.findall(
                rf"{re.escape(TOOL_CALL_START)}(.*?){re.escape(TOOL_CALL_END)}", text, re.DOTALL
            )
        ]
    calls = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            calls.append(parse_pipe(line))
        except ToolCallError:
            continue
    return calls


def _coerce(raw: str) -> object:
    """Turn pipe-format strings back into bools/ints/floats where obvious."""
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


# --------------------------------------------------------------------------- #
# comparison (used by scripts/eval_toolcalls.py)
# --------------------------------------------------------------------------- #


def normalise(value: object) -> object:
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    return normalise(str(value))
