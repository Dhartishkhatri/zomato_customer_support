"""Shared test fixtures.

THE IMPORTANT ONE is `isolated_db`. The system now keeps mutable state in
SQLite - refunds, cancellations, delivery instructions, chat history - so
tests sharing a single database file would leak state into each other (a
refund issued by one test makes the next test's order look already-refunded).
Each test therefore gets its own freshly seeded database file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zomato_support import db as _db
from zomato_support.db.seed import seed_all


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Give every test a private, freshly seeded database."""
    path = tmp_path / "test.db"
    _db.set_db_path(path)
    _db.init_db(path)
    seed_all(path)
    yield path
    _db.set_db_path(None)
