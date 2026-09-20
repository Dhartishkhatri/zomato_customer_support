"""
===============================================================================
DATABASE LAYER                                                       [CORE]
===============================================================================

SQLite, chosen because it needs no server and ships with Python. Everything
the agent reads or writes at runtime lives here: orders, live delivery
tracking, conversation history, ratings, executed actions, per-turn telemetry
and evaluation scores.

To point this at real systems, replace the backend adapters in
`zomato_support/backends/` - they are the only things that touch this module.

    from zomato_support.db import get_db, reset_db
    reset_db()                  # drop, recreate and re-seed
    with get_db() as conn: ...  # row_factory is sqlite3.Row

THE PATH IS OVERRIDABLE, and that matters for tests. Because the database now
holds mutable state (refunds, cancellations, chat history), tests that shared
one file would leak state into each other. `set_db_path()` lets the test suite
give every test its own freshly seeded file - see tests/conftest.py.
Precedence: set_db_path() > $ZOMATO_DB_PATH > the default file next to this
module.
===============================================================================
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "zomato.db"
SCHEMA = Path(__file__).resolve().parent / "schema.sql"

_override: Path | None = None


def set_db_path(path: Path | str | None) -> None:
    """Point every subsequent connection at `path`. None restores the default."""
    global _override
    _override = Path(path) if path is not None else None


def current_db_path() -> Path:
    if _override is not None:
        return _override
    env = os.environ.get("ZOMATO_DB_PATH")
    return Path(env) if env else DEFAULT_DB_PATH


@contextmanager
def get_db(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Connection with dict-like rows and foreign keys on. Commits on exit."""
    conn = sqlite3.connect(path or current_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | None = None) -> None:
    """Create any missing tables. Safe to call repeatedly."""
    with get_db(path) as conn:
        conn.executescript(SCHEMA.read_text(encoding="utf-8"))


def ensure_db() -> None:
    """Create and seed the database if it does not exist yet.

    Lets a fresh checkout run the demo, the UI or the tests with no setup step.
    """
    if not current_db_path().exists():
        init_db()
        from .seed import seed_all

        seed_all()


def reset_db(path: Path | None = None) -> None:
    """Delete the database file and rebuild it from scratch with seed data."""
    target = path or current_db_path()
    if target.exists():
        target.unlink()
    init_db(target)
    from .seed import seed_all

    seed_all(target)


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


__all__ = [
    "DEFAULT_DB_PATH", "current_db_path", "set_db_path",
    "get_db", "init_db", "ensure_db", "reset_db", "rows_to_dicts",
]
