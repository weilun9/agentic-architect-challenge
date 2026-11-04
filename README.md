# Agentic Architect Challenge

Three small agent systems that share one LLM layer and one tracer:

| Part | What it does | Where |
|---|---|---|
| 1. Support triage | Classifies support emails (multi-label), **escalates data loss / outage / security breach / 4+ contacts in 7 days before any draft is written**, drafts KB-grounded replies, and blocks refund-policy hallucination with a rule checker plus an LLM judge | [`src/aac/support/`](src/aac/support) |
| 2. Scrape and summarise | The broken original script ([`naive.py`](src/aac/scraper/naive.py)), the bottleneck diagnosis with evidence, and the fix: robust fetching, main-content extraction, map-reduce summarisation, a conciseness guardrail | [`src/aac/scraper/`](src/aac/scraper) |
| 3. Document agent | Answers questions about one sample document, remembers the user's name and earlier context, and calls a calculator only when the question needs arithmetic | [`src/aac/docagent/`](src/aac/docagent) |

Stack: Python 3.12, LangGraph, LangChain chat models (Ollama locally, or OpenAI / Gemini by env var),
BM25, BeautifulSoup. **85 offline tests** (every LLM call scripted) run in CI on Linux and Windows.
All the results below come from a local `qwen3.5:9b` (RTX 4060 laptop, 8 GB VRAM).

The 1-page architecture summary is [`docs/architecture.pdf`](docs/architecture.pdf).

---

## Run it

```bash
git clone https://github.com/weilun9/agentic-architect-challenge && cd agentic-architect-challenge
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                            # 85 tests, no model or network needed
```

**Model.** The default is a free local model, so you don't need an API key:

```bash
ollama pull qwen3.5:9b          # https://ollama.com ; any tool-calling model works
```

To use a hosted model, copy `.env.example` to `.env` and set `AAC_LLM_PROVIDER=openai` with
`OPENAI_API_KEY`, or `AAC_LLM_PROVIDER=gemini` with `GEMINI_API_KEY`. Keys are only read from the environment and never logged.

```bash
python -m aac triage -v                                   # Part 1 on data/emails/sample_inbox.jsonl
python -m aac scrape https://peps.python.org/pep-0008/    # Part 2, fixed pipeline
python -m aac scrape https://peps.python.org/pep-0008/ --naive   # Part 2, the original script
python -m aac chat                                        # Part 3, interactive
python -m aac eval-agent                                  # Part 3, tool-decision + memory checks
python -m aac trace-summary                               # per-step latency, tokens, errors, guardrails
```

Optional: `pip install -e ".[js]" && playwright install chromium` lets Part 2 render pages built with JavaScript.

---

## Part 1: support-email triage

```
screen ─► classify ─┬─(any flag)─────────────────────────────► escalate ─► END   (hand-off note, no draft)
 rules,   LLM JSON,  │
 history  validated  └─► retrieve ─► draft ─► verify ─┬─(ok, mentions refunds)─► judge ─┐
                                               ▲      ├─(ok)───────────────────────────────┴─► finalize ─► END
                                               └──────┤ (problems, 1 revision allowed)
                                                      └─(still failing)───────────────────► fallback ─► END
```

* **Flag before drafting.** `screen` runs without an LLM. It stores the contact in SQLite, counts
  that customer's contacts in the 7 days up to this email, and runs keyword rules for data loss,
  outage and breach. `classify` adds categories and the model's own `risk_signals`, which catch
  paraphrases the rules miss. Either source is enough. The gate comes before `retrieve` and `draft`,
  so a flagged email never gets a drafted reply. It gets a hand-off note for the right queue
  (`security-incident`, `tier2-technical`, `customer-care-lead`).
* **Fail closed.** If the classifier is down, or still returns invalid JSON after one repair prompt,
  the email goes to a human. An unexpected exception in one email is contained. That email is
  escalated and the rest of the batch continues.
* **No refund hallucination: three layers.**
  1. The refund-policy sections are always retrieved for billing/refund emails. The prompt requires a `[n]` citation per claim and forbids promising refunds.
  2. [`verify.py`](src/aac/support/verify.py) is deterministic. Every number has to appear, with its unit, in the cited passage. Refund terms have to cite the refund policy. Commitments like "we will refund you" or "you are eligible" are blocked unless the sentence is conditional or negated.
  3. An **LLM judge** re-reads each refund sentence next to the full policy, which catches meaning errors the rules can't see. The judge fails closed: no verdict counts as "unsupported".

  A draft that still fails after one revision is replaced by a **fixed template that quotes the policy word for word**. That template cannot hallucinate.

**Live run** (`python -m aac triage --fresh`, 10 emails in [`data/emails/`](data/emails)):

| Email | Outcome | Why |
|---|---|---|
| e04 "SkyVault is down... 503" | escalated → tier2-technical | rule: service_outage |
| e05 "folder is completely empty, not in Trash" | escalated → tier2-technical | **LLM** risk signal (no keyword matched) |
| e06 "someone signed in... hacked" | escalated → security-incident | rule: security_breach |
| e08 "fourth time I am writing this week" | escalated → customer-care-lead | 4 contacts in 7 days (3 seeded + this one) |
| e01 refund, monthly plan | **needs_review**, policy-quote template | model wrote "you are eligible for a full refund" twice |
| e02 double charge, e09 annual refund | drafted, passed the judge | |
| e03 sync, e07 feedback, e10 email change | drafted | |

What the live runs taught me (and the tests now pin): the first checker flagged "agents *cannot*
guarantee approval" as a promise, and one draft said "once approved, credits are applied to your
next invoice". That draft had a valid citation and no numbers, but it confused service credits
with refunds. The rules couldn't see that, which is why the judge exists. Both cases are now tests.

---

## Part 2: scrape and summarise

The brief didn't include the script, so [`naive.py`](src/aac/scraper/naive.py) is my version of how
it is usually written: `requests.get(url).text` → `soup.get_text()` → the whole text in one prompt.

**The bottleneck is the single prompt holding the whole page.** Its size grows with the page, and
past the context window the model silently stops seeing it. Evidence from running the naive script
against the local model ([`docs/evidence/`](docs/evidence)):

| Page | Naive script | Fixed pipeline |
|---|---|---|
| PEP 8 (~11.7k tokens) | Ollama log: `truncating input prompt limit=2050 prompt=10903`: the model saw **19%** of the page, then wrote a ~570-word summary mostly from memory, with no length limit | 6 chunks, 7 calls, all of the page read, 110 words, ~36 s wall clock |
| Wikipedia "Coral bleaching" (~28k tokens) | No User-Agent and no status check: Wikipedia returned its **403 robot-policy page**, and the script summarised that page as if it were the article | 20 chunks, 25 calls. The first draft was 178 words; the guardrail had it rewritten to 94 |

The fix, in order of the failure modes:

1. **Fetching** ([`fetch.py`](src/aac/scraper/fetch.py)): session with connect/read timeouts, retry
   with backoff on 429/5xx, a real User-Agent, `raise_for_status`, content-type check, and a streamed
   5 MB cap.
2. **Extraction**: drop scripts, styles, nav, footers, cookie banners and share widgets (matched on
   whole class/id tokens, so `discomment` is not "comment"; an element holding over half the page text is never dropped, so `<article class="has-comments">` survives). Then take `<article>`/`<main>`, or
   else the block with the most paragraph text and the lowest link density.
3. **JavaScript pages**: if little text is left and the HTML looks like an app shell, render it with
   headless Chromium (optional extra). Without Playwright you get a clear error, not a summary of an empty page.
4. **Map-reduce** ([`summarize.py`](src/aac/scraper/summarize.py)): about 1,500-token chunks are condensed
   in parallel (bounded pool), then merged, as a tree if needed. Every call is a fixed small size.
   Very long pages are sampled evenly, not cut off at the top, and the output says so. One failed chunk is skipped and reported.
5. **Conciseness guardrail**: the budget is in the prompt (≤ 5 bullets, ≤ 120 words) and output
   tokens are capped. `check_concise` validates the result. On a violation the model gets **one**
   rewrite with the exact violations, and if that still fails `enforce_concise` cuts it at a sentence
   boundary. A summary that leaves the module always meets the budget.

---

## Part 3: document agent

```
START ─► remember ─► agent ──(tool call, budget left)──► tools ─┐
                       ▲                                        │
                       ├────────────────────────────────────────┘
                       └──(plain answer)──► check ──(numbers grounded)──► END
                       ▲                      │
                       └──(1 retry, feedback)─┘
```

* **Document**: [`data/docagent/skyvault_guide.md`](data/docagent/skyvault_guide.md), a fictional
  plans-and-pricing guide (about 600 tokens). It is small, so it goes in the system prompt instead of
  through retrieval: no retrieval step that can miss.
* **Tool**: a `calculator` that parses with `ast` (no `eval`, no names, exponent and size limits).
  The model decides when to call it. The prompt says to use it for arithmetic and never for looking things up.
* **Memory**: the LangGraph checkpointer keeps the whole conversation per `thread_id`, and the last
  20 messages go to the model (the window never starts on an orphan tool result). The user's name
  is pulled into state by `remember` and written into every system prompt, so it survives the
  window. Threads are isolated.
* **Loop safety**: a per-turn tool budget (4). After that the model is called without tools and has to answer.
* **Answer check** (the same idea as Part 1's verifier): every number in the final answer must come
  from the document, the user's messages or a calculator result. If one doesn't, the answer is removed
  from memory and the model retries once with the offending numbers named. If it is still
  ungrounded, the answer is returned with a `warnings` entry and a trace event.

**Live eval** (`python -m aac eval-agent`, 11 cases: 5 need the calculator, 6 don't, 2 test memory):

| | Before the answer check | With the answer check |
|---|---|---|
| Answer accuracy (scored on the **first sentence**) | 10 / 11 | **11 / 11** |
| Tool precision / recall | 1.0 / 1.0 | 1.0 / 1.0 |
| Unneeded tool calls | 0 | 0 |
| Memory cases (name, "team of 8" → USD 144) | 2 / 2 | 2 / 2 |
| Check retries that fired | — | 2 (both then correct) |

The one miss, and how I found it: on *"How much would I save per year on Personal by paying
annually?"* the model called the calculator for `4*12*0.8 = 38.4`, then did the subtraction in its
head and answered **"save USD 6.40"**. My first scorer looked for the expected value anywhere in the
answer, found "9.60" in the working underneath, and reported 11/11. Scoring the headline sentence
showed the real 10/11. Tightening the prompt didn't fix it (the next run said "7.60"), so the agent
got the deterministic `check` step. The raw outputs of each stage are in `docs/evidence/eval_agent_*`.

---

## Failure points and how they're handled

| Failure | Handling |
|---|---|
| LLM timeout / connection error | retry with exponential backoff (`llm.call`), every attempt traced; then fail closed (Part 1 → human; Part 2 → skip the chunk, or error if over half fail) |
| Model returns invalid JSON | Pydantic validation, one repair prompt quoting the error, then `LLMError` |
| Small model says "Billing" / "Service Outage" | validators normalise case and spacing and drop unknown labels |
| Classifier misses a keyword | rules first, LLM signals second; either one escalates |
| Same email processed twice / batch order | contact history is idempotent on email id; the batch records all contacts in time order before the parallel workers start |
| Refund hallucination | rules → LLM judge → verbatim template (see Part 1) |
| Page blocks bots / 4xx / huge / not HTML | `raise_for_status`, User-Agent, content-type check, 5 MB streamed cap |
| Page longer than the context | chunked map-reduce; even sampling beyond 24 chunks, reported |
| Wordy or chatty summary | validator → one rewrite → deterministic truncation |
| Agent states a number it didn't compute | answer `check`: removed from memory, one retry with feedback, then flagged |
| Agent loops on tools / bad expression | per-turn tool budget; tool errors go back to the model as text, not exceptions |
| Unsafe calculator input | AST whitelist; `__import__`, names, `9**9**9` are rejected (tests) |

## AgentOps

Every step writes one JSON line to `runs/trace.jsonl`, with `run_id`, the email id, step, status, latency,
input and output tokens, and which guardrail fired. `python -m aac trace-summary` aggregates it into
calls, errors, p50/p95 latency, tokens and guardrail counts. After the live Part 1 run, for example, it showed
`classify` p50 7.7 s / p95 9.5 s, `judge` p50 5.1 s, end-to-end p50 12.9 s / p95 26.1 s per email (4
workers sharing one laptop GPU), 2 grounding violations, 1 template fallback and the 4 escalations
by reason. A sample trace is in [`docs/evidence/trace_sample.jsonl`](docs/evidence/trace_sample.jsonl). Those numbers would feed a dashboard or an alert (e.g. "template_fallback rate > 20%").

## Trade-offs and known limits

* **BM25, not embeddings** for the KB: no model download, deterministic tests, exact matches on
  "annual" or "503". My [grounded-rag](https://github.com/weilun9/grounded-rag) repo shows the
  hybrid + reranker version; switching only changes `KnowledgeBase.search`.
* **Rules favour recall.** "No data loss, just a question" is escalated. A false escalation costs a
  human a minute; a missed breach costs much more.
* **Drafts go to a human for one-click approval**, not straight to the customer. `drafted` means
  "passed every check", not "sent".
* **Non-refund claims are only checked for numbers.** In e07 the model wrote "setting a default expiry
  is not yet available". That's plausible, but it isn't in the KB, and no check covers it. Widening
  the judge to every sentence would catch it, at one more LLM call per email.
* **The judge is the same small model.** It catches mix-ups the rules miss, but it isn't infallible.
  That is why it fails closed and why the template is the last resort.
* **Part 2 token counts are estimates** (4 characters ≈ 1 token). That is good enough for budgeting
  chunks, with a wide safety margin.
* **Memory is in-process** (`MemorySaver`). For production, swap in the SQLite/Postgres checkpointer;
  the graph code doesn't change.

## Layout

```
src/aac/
  config.py  llm.py  obs.py  kb.py  __main__.py      shared: settings, model factory + retry/JSON repair, tracer, BM25 KB, CLI
  support/   models.py rules.py history.py verify.py graph.py
  scraper/   naive.py fetch.py summarize.py
  docagent/  tools.py agent.py evaluate.py
data/kb/            refund policy + billing/technical/account FAQs (fictional "SkyVault")
data/emails/        sample inbox + seeded contact history
data/docagent/      the Part 3 sample document
tests/              85 offline tests + an HTML fixture with every kind of page clutter
docs/evidence/      raw outputs and the Ollama truncation log behind the tables above
```
