"""Web research backend: search + page text extraction.

Search order:
1. ``MUSCAL_SEARX_URL`` -- self-hosted SearxNG, no rate limits, best option.
2. DuckDuckGo's HTML endpoint -- no API key, but brittle and rate-limited.
3. ``MUSCAL_SEARCH_API`` (Tavily/Brave style) if you set one.

For extraction, `trafilatura` is used when installed; otherwise a small
stdlib reader does a respectable job on article-shaped pages.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

UA = "muscal-agent/0.1 (+https://github.com/INDIEaner84/MUSCAL-ColabAPI-ProviderLLM)"


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in {"p", "br", "div", "li", "h1", "h2", "h3", "section", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\n\s*\n\s*", "\n\n", raw)
        return raw.strip()


def _get(url: str, *, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user-provided url
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def html_to_text(html: str) -> str:
    try:
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(html)
        if extracted:
            return extracted
    except Exception:  # noqa: BLE001 - optional dependency
        pass
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def read(url: str, max_chars: int = 4000) -> str:
    html = _get(url)
    text = html_to_text(html)
    return text[:max_chars]


def search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    searx = os.environ.get("MUSCAL_SEARX_URL")
    if searx:
        return _search_searx(searx, query, max_results)
    return _search_ddg(query, max_results)


def _search_searx(base: str, query: str, max_results: int) -> list[dict[str, str]]:
    url = f"{base.rstrip('/')}/search?{urllib.parse.urlencode({'q': query, 'format': 'json'})}"
    payload = json.loads(_get(url))
    out = []
    for item in payload.get("results", [])[:max_results]:
        out.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": (item.get("content") or "")[:300],
            }
        )
    return out


def _search_ddg(query: str, max_results: int) -> list[dict[str, str]]:
    """DuckDuckGo HTML endpoint. No key, but rate-limited and brittle."""
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    html = _get(url)
    results: list[dict[str, str]] = []
    pattern = re.compile(
        r'class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
        re.DOTALL,
    )
    for match in pattern.finditer(html):
        raw_href = match.group("href")
        if "uddg=" in raw_href:
            raw_href = urllib.parse.unquote(raw_href.split("uddg=")[1].split("&")[0])
        results.append(
            {
                "title": re.sub(r"<[^>]+>", "", match.group("title")).strip(),
                "url": raw_href,
                "snippet": re.sub(r"<[^>]+>", "", match.group("snippet")).strip(),
            }
        )
        if len(results) >= max_results:
            break
    return results
