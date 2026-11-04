"""DB layer: TranscriptDatabase schema, writes and reads."""

from __future__ import annotations

import sqlite3

import pytest

from tests.conftest import db_rows
from transcript_database import TranscriptDatabase


def _journal_mode(path: str) -> str:
    with sqlite3.connect(path) as c:
        return c.execute("PRAGMA journal_mode").fetchone()[0]


def test_reopen_fresh_db_does_not_crash(tmp_path):
    """Regression: 2nd open of a fresh DB raised 'cannot change into wal mode from within a transaction'."""
    p = str(tmp_path / "data" / "t.db")
    TranscriptDatabase(p)
    TranscriptDatabase(p)  # used to raise sqlite3.OperationalError
    assert _journal_mode(p) == "wal"


def test_open_existing_rollback_journal_db(tmp_path):
    """A DB created elsewhere in rollback-journal mode (backup copy, merge script) must open."""
    p = str(tmp_path / "data" / "t.db")
    (tmp_path / "data").mkdir()
    with sqlite3.connect(p) as c:
        c.execute("PRAGMA journal_mode=DELETE")
        c.execute("CREATE TABLE videos (video_id TEXT PRIMARY KEY, title TEXT, channel TEXT, url TEXT)")
    TranscriptDatabase(p)
    assert _journal_mode(p) == "wal"


def test_schema_has_core_tables(corpus):
    names = {r[0] for r in db_rows(corpus["db_path"], "SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("videos", "transcripts", "entities", "synthetic_questions", "processing_queue",
              "facts", "segments", "events", "segment_index", "video_signatures"):
        assert t in names, t
