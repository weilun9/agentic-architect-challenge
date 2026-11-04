"""Behaviour checks for the document agent against a live model.

Measures the two things the brief asks for:
  * tool decisions: did it call the calculator exactly when the question needs arithmetic?
    (precision = calls that were needed / all calls; recall = needed calls that were made)
  * memory: does it recall the user's name and earlier context in a later turn?
Answers are scored by "expected substring appears in the answer's first sentence" (numbers compared
without commas).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass

from .agent import DocAgent


@dataclass
class Case:
    question: str
    expect: list[str]  # any one of these must appear in the answer
    needs_tool: bool
    setup: tuple[str, ...] = ()  # earlier turns in the same conversation


CASES = [
    Case("How much does the Business plan cost per user per month?", ["18"], False),
    Case("What is the largest file I can upload through the web app?", ["5 GB", "5GB"], False),
    Case("Which storage regions can I choose from?", ["Frankfurt"], False),
    Case("Is phone support available on weekends?", ["weekday", "no", "not"], False),
    Case(
        "How much does the Enterprise plan cost?",
        ["don't know", "do not know", "not mention", "no enterprise", "doesn't", "does not"],
        False,
    ),
    Case(
        "How much would a team of 12 pay per year on the Business plan if they pay annually?",
        ["2073.6", "2,073.6", "2073.60"],
        True,
    ),
    Case("We have 3 Pro users and want one 10 TB storage pack. What is our monthly total?", ["80"], True),
    Case("If uptime is exactly 99.9% in a 30-day month, how many minutes of downtime is that?", ["43.2"], True),
    Case(
        "How much would I save in a year on the Personal plan by paying annually instead of monthly?",
        ["9.6", "9.60"],
        True,
    ),
    # memory
    Case("What is my name?", ["Lawrance"], False, setup=("Hi, my name is Lawrance.",)),
    Case("What would we pay per month in total?", ["144"], True, setup=("We are a team of 8 on the Business plan.",)),
]


def _norm(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3})", "", s).lower()


def headline(answer: str) -> str:
    return re.split(r"(?<=[.!?])\s+|\n", answer.strip(), maxsplit=1)[0]


def run(agent: DocAgent) -> dict:
    rows = []
    for case in CASES:
        thread = uuid.uuid4().hex
        for turn in case.setup:
            agent.ask(turn, thread)
        out = agent.ask(case.question, thread)
        called = any(tc["name"] == "calculator" for tc in out["tool_calls"])
        # Score the first sentence only: a wrong headline followed by a right working is still wrong
        # (qwen3.5:9b answered "save USD 6.40" and then showed 9.60 in its working).
        correct = any(_norm(e) in _norm(headline(out["answer"])) for e in case.expect)
        rows.append({**asdict(case), "answer": out["answer"], "called_tool": called, "correct": correct})

    needed = [r for r in rows if r["needs_tool"]]
    called = [r for r in rows if r["called_tool"]]
    tp = sum(r["needs_tool"] for r in called)
    return {
        "cases": len(rows),
        "answer_accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
        "tool_precision": round(tp / len(called), 3) if called else None,
        "tool_recall": round(tp / len(needed), 3) if needed else None,
        "unneeded_tool_calls": len(called) - tp,
        "memory_cases_correct": sum(r["correct"] for r in rows if r["setup"]),
        "rows": rows,
    }


def print_report(report: dict) -> None:
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    for r in report["rows"]:
        mark = "ok " if r["correct"] and r["called_tool"] == r["needs_tool"] else "XX "
        print(
            f"{mark} tool={'Y' if r['called_tool'] else 'n'} (need {'Y' if r['needs_tool'] else 'n'}) | {r['question']}"
        )
        print(f"     -> {r['answer'][:160]!r}")
