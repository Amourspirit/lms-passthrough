"""SQLite-backed response-chain persistence."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StoredResponse:
    """Stored response chain record."""

    response_id: str
    provider_name: str
    model: str
    created_at: datetime
    payload: dict[str, Any]
    previous_response_id: str | None = None


class StateStore:
    """Small SQLite store for response chains."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            current = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            ).fetchone()[0]
            if current < 1:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS response_chains (
                        response_id TEXT PRIMARY KEY,
                        provider_name TEXT NOT NULL,
                        model TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        previous_response_id TEXT,
                        payload TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                    (1, datetime.now(tz=UTC).isoformat()),
                )

    def save_response(self, record: StoredResponse) -> None:
        """Persist a response chain."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO response_chains(
                    response_id, provider_name, model, created_at, previous_response_id, payload
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.response_id,
                    record.provider_name,
                    record.model,
                    record.created_at.isoformat(),
                    record.previous_response_id,
                    json.dumps(record.payload),
                ),
            )

    def get_response(self, response_id: str) -> StoredResponse | None:
        """Load a response chain by id."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT response_id, provider_name, model, created_at, previous_response_id, payload
                FROM response_chains
                WHERE response_id = ?
                """,
                (response_id,),
            ).fetchone()
        if row is None:
            return None
        return StoredResponse(
            response_id=row["response_id"],
            provider_name=row["provider_name"],
            model=row["model"],
            created_at=datetime.fromisoformat(row["created_at"]),
            previous_response_id=row["previous_response_id"],
            payload=json.loads(row["payload"]),
        )

    def delete_expired(self, retention_days: int) -> int:
        """Delete chains older than retention."""
        cutoff = datetime.now(tz=UTC) - timedelta(days=retention_days)
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM response_chains WHERE created_at < ?",
                (cutoff.isoformat(),),
            )
            return cursor.rowcount
