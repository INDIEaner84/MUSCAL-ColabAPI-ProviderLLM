"""Named capability profiles: how much of the agent is switched on.

The catalog is deliberately bigger than what should ever run at once. A profile
bundles two things that belong together -- which tools exist for the model and
which risk classes the executor accepts. Handing the model 11 tools when 3 are
enough does not make it more capable, it makes it more wrong: every extra
function is another way to mishear.

Start with `web`. Add `browser` when the web tier hits its numbers. Only then
`desktop` -- and preferably in a VM.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    tools: list[str]
    allow_write: bool
    allow_destructive: bool
    description: str


WEB_TOOLS = ["web_search", "web_read"]
BROWSER_TOOLS = WEB_TOOLS + ["browser_open", "browser_click", "browser_type", "browser_snapshot"]
DESKTOP_TOOLS = BROWSER_TOOLS + [
    "desktop_launch",
    "desktop_type",
    "desktop_hotkey",
    "desktop_click",
    "desktop_screenshot",
]

PROFILES: dict[str, Profile] = {
    "web": Profile(
        "web",
        WEB_TOOLS,
        allow_write=False,
        allow_destructive=False,
        description="Nur Recherche und Seiten lesen. Nichts veraendert den Rechner.",
    ),
    "browser": Profile(
        "browser",
        BROWSER_TOOLS,
        allow_write=True,
        allow_destructive=False,
        description="Zusaetzlich Browser-Steuerung. Kann Klicks und Eingaben ausloesen.",
    ),
    "desktop": Profile(
        "desktop",
        DESKTOP_TOOLS,
        allow_write=True,
        allow_destructive=True,
        description="Alles inklusive Desktop-Steuerung. Erste Laeufe in eine VM.",
    ),
    "all": Profile(
        "all",
        DESKTOP_TOOLS,
        allow_write=True,
        allow_destructive=True,
        description="Identisch zu 'desktop'; Alias fuer Klarheit im Config-File.",
    ),
}

DEFAULT_PROFILE = "web"


def get(name: str) -> Profile:
    key = (name or DEFAULT_PROFILE).strip().lower()
    if key not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose one of {sorted(PROFILES)}")
    return PROFILES[key]


def describe() -> str:
    lines = []
    for name, profile in PROFILES.items():
        lines.append(f"  {name:<9} {len(profile.tools):>2} tools  write={str(profile.allow_write):<5} "
                     f"destructive={str(profile.allow_destructive):<5} {profile.description}")
    return "\n".join(lines)
