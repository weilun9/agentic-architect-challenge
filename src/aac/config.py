"""Settings read from the environment (or a .env file). Nothing secret is ever logged."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODELS = {
    "ollama": "qwen3.5:9b",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    llm_provider: str = field(default_factory=lambda: _env("AAC_LLM_PROVIDER", "ollama"))
    llm_model: str = field(default_factory=lambda: _env("AAC_LLM_MODEL", ""))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("AAC_LLM_TIMEOUT_S", "120")))
    llm_retries: int = field(default_factory=lambda: int(_env("AAC_LLM_RETRIES", "2")))
    trace_path: Path = field(default_factory=lambda: Path(_env("AAC_TRACE_PATH", str(ROOT / "runs" / "trace.jsonl"))))

    # Part 1: support triage
    kb_dir: Path = field(default_factory=lambda: Path(_env("AAC_KB_DIR", str(ROOT / "data" / "kb"))))
    history_db: Path = field(default_factory=lambda: Path(_env("AAC_HISTORY_DB", str(ROOT / "runs" / "history.db"))))
    repeat_contact_limit: int = 3  # escalate when contacts in the window are MORE than this
    repeat_contact_days: int = 7
    max_revisions: int = 1
    triage_workers: int = field(default_factory=lambda: int(_env("AAC_TRIAGE_WORKERS", "4")))

    # Part 2: scrape and summarize
    summary_max_words: int = 120
    summary_max_bullets: int = 5
    chunk_tokens: int = 1500
    max_chunks: int = 24
    map_workers: int = 4

    # Part 3: document agent
    doc_path: Path = field(
        default_factory=lambda: Path(_env("AAC_DOC_PATH", str(ROOT / "data" / "docagent" / "skyvault_guide.md")))
    )
    history_messages: int = 20
    max_tool_calls: int = 4

    @property
    def model_name(self) -> str:
        return self.llm_model or DEFAULT_MODELS.get(self.llm_provider.lower(), "")
