from pathlib import Path

import pytest
import requests

from aac.scraper.fetch import FetchError, Page, fetch_html, get_page, html_to_text
from aac.scraper.summarize import Summarizer, check_concise, chunk_text, enforce_concise

from .conftest import scripted

FIXTURE = Path(__file__).parent / "fixtures" / "complex_page.html"


def test_extraction_keeps_the_article_and_drops_boilerplate():
    title, text = html_to_text(FIXTURE.read_text(encoding="utf-8"))
    assert title == "How Coral Reefs Recover After Bleaching"
    assert "fewer than 3 bleaching events" in text and "## What helps recovery" in text
    for junk in ("cookies", "LONG_JSON_BLOB", "Trending", "Share on", "newsletter", "cheap watches", "Copyright"):
        assert junk not in text, junk


def test_extraction_is_much_smaller_than_naive_get_text():
    page = get_page(str(FIXTURE))
    assert len(page.text) < page.raw_chars * 0.8


def test_has_comments_class_does_not_delete_the_page():
    _, text = html_to_text('<body class="has-comments"><article><p>' + "Real content. " * 50 + "</p></article></body>")
    assert "Real content." in text


def test_page_without_article_uses_densest_block():
    html = (
        "<body><div><a href='#'>Home</a> <a href='#'>About</a></div><div><p>"
        + "Useful paragraph text. " * 40
        + "</p></div></body>"
    )
    _, text = html_to_text(html)
    assert text.startswith("Useful paragraph text.") and "Home" not in text


class _FakeResponse:
    def __init__(self, body=b"<html><body><p>hi</p></body></html>", ctype="text/html; charset=utf-8", status=200):
        self.body, self.headers, self.status_code, self.encoding, self.apparent_encoding = (
            body,
            {"Content-Type": ctype},
            status,
            "utf-8",
            "utf-8",
        )

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i : i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.kwargs = response, exc, None

    def get(self, url, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.response


def test_fetch_uses_timeout_and_streams():
    s = _FakeSession(_FakeResponse())
    assert "hi" in fetch_html("https://example.com", s)
    assert s.kwargs["timeout"] and s.kwargs["stream"] is True


def test_fetch_rejects_non_html_and_oversized_and_network_errors():
    with pytest.raises(FetchError, match="not an HTML page"):
        fetch_html("https://x", _FakeSession(_FakeResponse(ctype="application/pdf")))
    with pytest.raises(FetchError, match="larger than"):
        fetch_html("https://x", _FakeSession(_FakeResponse(body=b"x" * 2000)), max_bytes=1000)
    with pytest.raises(FetchError, match="could not fetch"):
        fetch_html("https://x", _FakeSession(exc=requests.ConnectTimeout("slow")))


def test_js_shell_without_playwright_gives_a_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_playwright(name, *a, **k):
        if name.startswith("playwright"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_playwright)
    shell = _FakeResponse(body=b'<html><body><div id="root"></div><script src="app.js"></script></body></html>')
    with pytest.raises(FetchError, match="rendered by JavaScript"):
        get_page("https://spa.example", _FakeSession(shell))


def test_chunks_respect_the_budget_and_keep_all_text():
    text = "\n".join(f"Paragraph {i}. " + "word " * 100 for i in range(40))
    chunks = chunk_text(text, max_tokens=300)
    assert len(chunks) > 1 and all(len(c) <= 300 * 4 for c in chunks)
    assert "".join(chunks).count("Paragraph") == 40


def test_one_giant_line_is_split():
    chunks = chunk_text("x" * 10_000, max_tokens=500)
    assert all(len(c) <= 2000 for c in chunks) and sum(map(len, chunks)) == 10_000


def test_concise_checker_and_enforcer():
    long = "Here is a summary of the page:\n" + "\n".join(f"- point {i} " + "word " * 20 for i in range(8))
    v = check_concise(long, max_words=60, max_bullets=5)
    assert any("words" in x for x in v) and any("bullets" in x for x in v) and any("introduction" in x for x in v)
    cut = enforce_concise(long, max_words=60, max_bullets=5)
    assert check_concise(cut, 60, 5) == []


def _page(paragraphs: int) -> Page:
    text = "\n".join(f"Fact number {i} is important. " + "detail " * 120 for i in range(paragraphs))
    return Page(url="u", title="T", text=text, raw_chars=len(text) * 3)


def _by_role(final):
    """Map calls get notes, reduce calls get the final summary."""
    return lambda messages: "- note" if "condensing one part" in messages[0].content else final


def test_short_page_is_one_call():
    llm = scripted("- one\n- two")
    result = Summarizer(llm, chunk_tokens=5000).summarize(_page(2))
    assert result.llm_calls == 1 and result.summary == "- one\n- two" and result.guardrail == []


def test_long_page_is_map_reduced_and_every_call_stays_small():
    llm = scripted(*[_by_role("- final point")] * 30)
    s = Summarizer(llm, chunk_tokens=400, workers=3)
    result = s.summarize(_page(20))
    assert result.chunks_total > 1 and result.summary == "- final point"
    assert max(sum(len(m.content) for m in call) for call in llm.calls) < 400 * 4 + 1000


def test_guardrail_asks_once_then_truncates():
    verbose = "Sure! Here is the summary:\n" + "\n".join(f"- point {i} " + "word " * 30 for i in range(9))
    llm = scripted(verbose, verbose)  # the model ignores the rewrite request too
    result = Summarizer(llm, max_words=50, max_bullets=3, chunk_tokens=5000).summarize(_page(1))
    assert result.words <= 50 and result.bullets <= 3
    assert result.guardrail[-1] == "truncated deterministically" and result.llm_calls == 2


def test_guardrail_accepts_a_good_rewrite():
    verbose = "\n".join(f"- point {i}" for i in range(9))
    llm = scripted(verbose, "- a\n- b")
    result = Summarizer(llm, max_bullets=3, chunk_tokens=5000).summarize(_page(1))
    assert result.summary == "- a\n- b" and result.guardrail[-1] == "rewritten by the model"


def test_a_failing_chunk_is_skipped_not_fatal():
    from aac.llm import LLMError

    llm = scripted(LLMError("timeout"), *[_by_role("- final")] * 30)
    result = Summarizer(llm, chunk_tokens=400, workers=1, retries=0).summarize(_page(10))
    assert result.summary == "- final" and any("failed and were skipped" in n for n in result.notes)
