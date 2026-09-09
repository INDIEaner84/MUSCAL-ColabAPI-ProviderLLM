"""Load a corpus from disk.

Supported out of the box: ``.md``, ``.txt``, ``.html``/``.htm`` (tags stripped
with the stdlib). ``.pdf`` needs ``pypdf`` and is skipped with a warning if the
package is missing -- installing it is a one-liner, not a design decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt", ".markdown", ".rst"}
HTML_SUFFIXES = {".html", ".htm"}
PDF_SUFFIXES = {".pdf"}


@dataclass
class Document:
    id: str
    path: Path
    text: str


class _HTMLText(HTMLParser):
    SKIP = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "section", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\n\s*\n\s*", "\n\n", "".join(self.parts)).strip()


def load_documents(root: str | Path) -> list[Document]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"docs_dir not found: {root}")

    files = sorted(p for p in root.rglob("*") if p.is_file())
    docs: list[Document] = []
    skipped_pdf = False

    for path in files:
        suffix = path.suffix.lower()
        try:
            if suffix in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8")
            elif suffix in HTML_SUFFIXES:
                parser = _HTMLText()
                parser.feed(path.read_text(encoding="utf-8", errors="replace"))
                text = parser.text()
            elif suffix in PDF_SUFFIXES:
                text = _read_pdf(path)
                if text is None:
                    skipped_pdf = True
                    continue
            else:
                continue
        except (OSError, UnicodeDecodeError) as exc:
            print(f"[warn] skipped {path.name}: {exc}")
            continue

        if text.strip():
            docs.append(Document(id=path.stem, path=path, text=text))

    if skipped_pdf:
        print("[warn] PDFs uebersprungen -- pip install pypdf")
    return docs


def _read_pdf(path: Path) -> str | None:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return None
    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)
