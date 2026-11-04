"""Fetching and main-content extraction.

Fixes for PROBLEMS 1-3 in naive.py:
  * a Session with timeouts, retry with backoff on 429/5xx, a browser-like User-Agent, a
    content-type check, and a hard byte limit (streamed, so a 2 GB file is never downloaded);
  * boilerplate removal (scripts, styles, nav, footers, cookie banners, share widgets), then the
    main block is chosen: <article>/<main> if present, otherwise the element with the most
    paragraph text and the lowest link density;
  * if what is left is suspiciously short and the HTML looks like a JavaScript app shell, the page
    is rendered with headless Chromium (optional `[js]` extra); without Playwright you get a clear
    error instead of a summary of an empty page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "Mozilla/5.0 (compatible; aac-summarizer/0.1; +https://github.com/weilun9)"
MAX_BYTES = 5_000_000
MIN_MAIN_CHARS = 400

_DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "form",
    "button",
    "nav",
    "footer",
    "aside",
)
# Matched against whole tokens of id/class/role ("cookie-banner" -> cookie, banner), so an article
# with class "has-comments" is kept while a div with class "comments" is dropped.
_BOILERPLATE = frozenset(
    "cookie cookies consent banner newsletter subscribe promo advert ad ads sponsor sponsored share sharing social "
    "related recommended sidebar breadcrumb breadcrumbs comments popup modal menu navbar navigation "
    # Wikipedia/MediaWiki furniture
    "navbox reflist editsection catlinks noprint".split()
)
_MARKER_SPLIT = re.compile(r"[\s_\-]+")
_BLOCKS = ("h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre", "td", "figcaption")
_SPA_HINT = re.compile(r'id=["\'](?:root|app|__next|__nuxt)["\']|data-reactroot|ng-version', re.IGNORECASE)


class FetchError(RuntimeError):
    pass


@dataclass
class Page:
    url: str
    title: str
    text: str
    raw_chars: int  # what naive get_text() would have sent to the model
    rendered_js: bool = False
    notes: list[str] = field(default_factory=list)


def make_session(retries: int = 3) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=retries, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET", "HEAD")
    )
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"})
    return s


def fetch_html(url: str, session: requests.Session | None = None, timeout=(5, 20), max_bytes: int = MAX_BYTES) -> str:
    if not url.startswith(("http://", "https://")):
        path = Path(url)
        if path.exists():  # local file, used by the tests and the offline demo
            return path.read_text(encoding="utf-8", errors="replace")
        raise FetchError(f"not a URL or an existing file: {url}")
    session = session or make_session()
    try:
        with session.get(url, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if ctype and "html" not in ctype and "xml" not in ctype:
                raise FetchError(f"{url} is {ctype!r}, not an HTML page")
            body = bytearray()
            for chunk in r.iter_content(64 * 1024):
                body += chunk
                if len(body) > max_bytes:
                    raise FetchError(f"{url} is larger than {max_bytes:,} bytes")
            encoding = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else r.apparent_encoding
            return body.decode(encoding or "utf-8", errors="replace")
    except requests.RequestException as exc:
        raise FetchError(f"could not fetch {url}: {exc}") from exc


def _is_boilerplate(tag: Tag) -> bool:
    if getattr(tag, "decomposed", False) or not tag.attrs:
        return False
    marker = " ".join([tag.get("id") or "", *(tag.get("class") or []), tag.get("role") or ""]).lower()
    return not _BOILERPLATE.isdisjoint(_MARKER_SPLIT.split(marker))


def _link_density(tag: Tag) -> float:
    text = len(tag.get_text(" ", strip=True)) or 1
    links = sum(len(a.get_text(" ", strip=True)) for a in tag.find_all("a"))
    return links / text


def _pick_main(soup: BeautifulSoup) -> Tag:
    for selector in ("article", "main", "[role=main]"):
        candidates = soup.select(selector)
        if candidates:
            best = max(candidates, key=lambda t: len(t.get_text(" ", strip=True)))
            if len(best.get_text(" ", strip=True)) >= MIN_MAIN_CHARS:
                return best
    # No semantic container: score every block-holding element by its paragraph text.
    best, best_score = soup.body or soup, 0.0
    for tag in soup.find_all(("div", "section", "td")):
        p_text = sum(len(p.get_text(" ", strip=True)) for p in tag.find_all("p", recursive=False))
        score = p_text * (1 - _link_density(tag))
        if score > best_score:
            best, best_score = tag, score
    return best


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, main text) with one block per line and duplicate lines removed."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        if tag.name not in ("html", "body") and _is_boilerplate(tag):
            tag.decompose()
    main = _pick_main(soup)
    lines, seen = [], set()
    for block in main.find_all(_BLOCKS):
        if block.find(_BLOCKS):  # nested block (li containing p): the inner one is emitted instead
            continue
        line = re.sub(r"\s+", " ", block.get_text(" ", strip=True))
        if len(line) < 3 or line in seen:
            continue
        seen.add(line)
        lines.append(f"## {line}" if block.name in ("h1", "h2", "h3", "h4") else line)
    if not lines:  # plain-text-ish page with no block tags
        lines = [re.sub(r"\s+", " ", main.get_text(" ", strip=True))]
    return title, "\n".join(lines).strip()


def _render_with_browser(url: str, timeout_ms: int = 20_000) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise FetchError(
            "this page is rendered by JavaScript; install the extra with "
            "`pip install -e .[js]` and `playwright install chromium` to summarise it"
        ) from exc
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, timeout=timeout_ms, wait_until="networkidle")
            return page.content()
        finally:
            browser.close()


def get_page(url: str, session: requests.Session | None = None) -> Page:
    html = fetch_html(url, session)
    raw_chars = len(BeautifulSoup(html, "html.parser").get_text())
    title, text = html_to_text(html)
    page = Page(url=url, title=title, text=text, raw_chars=raw_chars)
    if len(text) < MIN_MAIN_CHARS and _SPA_HINT.search(html) and url.startswith("http"):
        page.notes.append(f"only {len(text)} characters of static text; rendering JavaScript")
        title, text = html_to_text(_render_with_browser(url))
        page.title, page.text, page.rendered_js = title or page.title, text, True
    if not page.text:
        raise FetchError(f"no readable text found at {url}")
    return page
