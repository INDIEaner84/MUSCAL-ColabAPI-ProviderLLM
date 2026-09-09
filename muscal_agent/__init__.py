"""MUSCAL audio agent: speech -> tool call -> web / browser / desktop.

Runtime half of the audio track. The training half lives in
``notebooks/05_audio_agent_finetune.ipynb`` and ``scripts/``.
"""

from .catalog import TOOLS, TOOLS_BY_NAME
from .executor import Executor, Policy, ToolResult
from .toolcall import ToolCall, parse_tool_call

__all__ = [
    "TOOLS",
    "TOOLS_BY_NAME",
    "Executor",
    "Policy",
    "ToolResult",
    "ToolCall",
    "parse_tool_call",
]
