"""Command line entry point.

python -m aac triage [inbox.jsonl]           Part 1: process a support inbox
python -m aac scrape URL_OR_FILE [--naive]   Part 2: scrape and summarise a page
python -m aac chat [--thread ID]             Part 3: talk to the document agent
python -m aac eval-agent                     Part 3: tool-decision and memory checks
python -m aac trace-summary                  AgentOps: latency, tokens, errors, guardrails
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .config import ROOT, Settings
from .obs import Tracer, setup_logging, summarize_trace


def _llm(settings: Settings, max_tokens: int | None = None):
    from .llm import get_chat_model

    return get_chat_model(settings, max_tokens=max_tokens)


def cmd_triage(args, s: Settings) -> int:
    from .kb import KnowledgeBase
    from .support.graph import SupportTriage
    from .support.history import ContactHistory
    from .support.models import Email

    history = ContactHistory(":memory:" if args.fresh else s.history_db)
    if args.history and Path(args.history).exists():  # seed earlier contacts (demo data)
        for line in Path(args.history).read_text(encoding="utf-8").splitlines():
            h = json.loads(line)
            history.record(h["id"], h["sender"], datetime.fromisoformat(h["received_at"]))
    emails = [
        Email.model_validate_json(line)
        for line in Path(args.inbox).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tracer = Tracer(s.trace_path, part="triage")
    triage = SupportTriage(s, _llm(s, max_tokens=600), KnowledgeBase.from_dir(s.kb_dir), history, tracer)
    results = triage.process_batch(emails)

    out = ROOT / "runs" / f"triage_{tracer.run_id}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(r.model_dump_json() for r in results) + "\n", encoding="utf-8")
    for e, r in zip(emails, results, strict=True):
        flags = ",".join(f"{f.reason}({f.source})" for f in r.flags) or "-"
        print(
            f"{r.email_id}  {r.status:<12} {r.queue:<20} cats={','.join(r.categories) or '-':<22} flags={flags}  {r.latency_ms} ms"
        )
        if args.verbose:
            print(f"    subject: {e.subject}")
            print("    " + (r.draft or r.handoff_note).replace("\n", "\n    "))
            if r.problems:
                print(f"    problems: {r.problems}")
    print(f"\nrun_id={tracer.run_id}  results -> {out}")
    print(json.dumps(summarize_trace(s.trace_path, tracer.run_id), indent=2))
    return 0


def cmd_scrape(args, s: Settings) -> int:
    from .scraper.fetch import FetchError, get_page
    from .scraper.summarize import Summarizer

    llm = _llm(s, max_tokens=None if args.naive else 400)
    if args.naive:
        from .scraper.naive import scrape_and_summarize

        print(scrape_and_summarize(args.url, llm))
        return 0
    tracer = Tracer(s.trace_path, part="scrape")
    try:
        page = get_page(args.url)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summarizer = Summarizer(
        llm,
        max_words=args.max_words or s.summary_max_words,
        max_bullets=s.summary_max_bullets,
        chunk_tokens=s.chunk_tokens,
        max_chunks=s.max_chunks,
        workers=s.map_workers,
        retries=s.llm_retries,
        tracer=tracer,
    )
    result = summarizer.summarize(page)
    print(f"# {page.title}\n\n{result.summary}\n")
    meta = {k: v for k, v in vars(result).items() if k != "summary"}
    print(
        json.dumps({"url": page.url, "clean_chars": len(page.text), "rendered_js": page.rendered_js, **meta}, indent=2)
    )
    return 0


def cmd_chat(args, s: Settings) -> int:
    from .docagent.agent import DocAgent

    agent = DocAgent.from_file(
        _llm(s),
        s.doc_path,
        history_messages=s.history_messages,
        max_tool_calls=s.max_tool_calls,
        tracer=Tracer(s.trace_path, part="docagent"),
    )
    print(f"Document: {s.doc_path.name}. Ask a question (empty line to quit).")
    while True:
        try:
            q = input("you> ").strip()
        except EOFError:
            break
        if not q:
            break
        out = agent.ask(q, args.thread)
        for tc in out["tool_calls"]:
            print(f"  [tool] {tc['name']}({tc['args']})")
        print(f"agent> {out['answer']}")
    return 0


def cmd_eval_agent(args, s: Settings) -> int:
    from .docagent.agent import DocAgent
    from .docagent.evaluate import print_report, run

    tracer = Tracer(s.trace_path, part="docagent-eval")
    agent = DocAgent.from_file(
        _llm(s), s.doc_path, history_messages=s.history_messages, max_tool_calls=s.max_tool_calls, tracer=tracer
    )
    report = run(agent)
    print_report(report)
    out = ROOT / "runs" / f"eval_agent_{s.model_name.replace(':', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": s.model_name, **report}, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


def cmd_trace_summary(args, s: Settings) -> int:
    print(json.dumps(summarize_trace(s.trace_path, args.run_id), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aac", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="WARNING")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("triage", help="Part 1: process a support inbox")
    t.add_argument("inbox", nargs="?", default=str(ROOT / "data" / "emails" / "sample_inbox.jsonl"))
    t.add_argument("--history", default=str(ROOT / "data" / "emails" / "contact_history.jsonl"))
    t.add_argument("--fresh", action="store_true", help="use an in-memory contact history")
    t.add_argument("-v", "--verbose", action="store_true", help="print drafts and hand-off notes")

    sc = sub.add_parser("scrape", help="Part 2: scrape and summarise a page")
    sc.add_argument("url")
    sc.add_argument("--naive", action="store_true", help="run the original broken script instead")
    sc.add_argument("--max-words", type=int)

    c = sub.add_parser("chat", help="Part 3: chat with the document agent")
    c.add_argument("--thread", default="cli")

    sub.add_parser("eval-agent", help="Part 3: tool-decision and memory checks")
    ts = sub.add_parser("trace-summary", help="latency/tokens/errors per step from runs/trace.jsonl")
    ts.add_argument("--run-id")

    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8")
    setup_logging(args.log_level)
    handlers = {
        "triage": cmd_triage,
        "scrape": cmd_scrape,
        "chat": cmd_chat,
        "eval-agent": cmd_eval_agent,
        "trace-summary": cmd_trace_summary,
    }
    return handlers[args.cmd](args, Settings())


if __name__ == "__main__":
    sys.exit(main())
