"""Contact history per customer, in SQLite, for the "more than 3 contacts in 7 days" rule.

`record` is idempotent on the email id, so re-processing an inbox never inflates the count.
Counts are taken over the window that ends at the email's own timestamp, which keeps results
the same whether emails are processed one by one, in a batch, or replayed later.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path


class ContactHistory:
    def __init__(self, path: Path | str = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS contacts (email_id TEXT PRIMARY KEY, sender TEXT NOT NULL, ts TEXT NOT NULL)"
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_sender_ts ON contacts(sender, ts)")
            self._db.commit()

    def record(self, email_id: str, sender: str, ts: datetime) -> None:
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO contacts VALUES (?, ?, ?)", (email_id, sender.lower(), _utc(ts)))
            self._db.commit()

    def count(self, sender: str, until: datetime, days: int) -> int:
        """Contacts from `sender` in the `days` before `until`, inclusive of `until` itself."""
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) FROM contacts WHERE sender = ? AND ts > ? AND ts <= ?",
                (sender.lower(), _utc(until - timedelta(days=days)), _utc(until)),
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self._db.close()


def _utc(ts: datetime) -> str:
    # ISO strings compare correctly only in one timezone; naive timestamps are treated as UTC.
    if ts.tzinfo is not None:
        ts = ts.astimezone(UTC).replace(tzinfo=None)
    return ts.isoformat(timespec="seconds")
