"""Observability: every run gets a run_id, and every step writes one JSON line.

The trace file (runs/trace.jsonl by default) is the AgentOps record: which node ran, how long it
took, how many tokens it used, which guardrail fired and why. `summarize_trace` turns it into the
numbers you would put on a dashboard (calls, errors, p50/p95 latency, tokens, guardrail hits).
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("aac")

_write_lock = threading.Lock()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class Tracer:
    """Collects events for one run in memory and appends them to a JSONL file (thread-safe)."""

    def __init__(self, path: Path | None = None, run_id: str | None = None, **context):
        self.path = path
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.context = context
        self.events: list[dict] = []

    def child(self, **context) -> Tracer:
        """Same file and run_id, extra context (e.g. the email id) on every event."""
        return Tracer(self.path, self.run_id, **{**self.context, **context})

    def event(self, step: str, **fields) -> dict:
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            **self.context,
            "step": step,
            **fields,
        }
        self.events.append(record)
        log.debug("%s", record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with _write_lock, self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def timed(self, step: str):
        return _Timer(self, step)


class _Timer:
    def __init__(self, tracer: Tracer, step: str):
        self.tracer, self.step, self.fields = tracer, step, {}

    def __enter__(self):
        self.start = time.perf_counter()
        return self.fields

    def __exit__(self, exc_type, exc, tb):
        ms = round((time.perf_counter() - self.start) * 1000)
        if exc:
            self.tracer.event(self.step, status="error", ms=ms, error=f"{exc_type.__name__}: {exc}"[:300])
        else:
            self.tracer.event(self.step, status="ok", ms=ms, **self.fields)
        return False


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, round(p * (len(values) - 1)))]


def summarize_trace(path: Path, run_id: str | None = None) -> dict:
    """Per-step call counts, errors, latency percentiles and token totals for a trace file."""
    steps: dict[str, dict] = defaultdict(lambda: {"calls": 0, "errors": 0, "ms": [], "in_tokens": 0, "out_tokens": 0})
    guardrails: dict[str, int] = defaultdict(int)
    for line in path.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if run_id and e.get("run_id") != run_id:
            continue
        s = steps[e["step"]]
        s["calls"] += 1
        s["errors"] += e.get("status") in ("error", "invalid_json")
        if isinstance(e.get("ms"), int | float):
            s["ms"].append(e["ms"])
        s["in_tokens"] += e.get("in_tokens") or 0
        s["out_tokens"] += e.get("out_tokens") or 0
        for g in e.get("guardrails", []) or []:
            guardrails[g] += 1
    out = {}
    for name, s in steps.items():
        out[name] = {
            "calls": s["calls"],
            "errors": s["errors"],
            "p50_ms": _pct(s["ms"], 0.5),
            "p95_ms": _pct(s["ms"], 0.95),
            "in_tokens": s["in_tokens"],
            "out_tokens": s["out_tokens"],
        }
    return {"steps": out, "guardrails": dict(guardrails)}
