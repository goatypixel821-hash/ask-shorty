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


def test_add_video_again_keeps_metadata_and_watch_date(corpus_copy):
    """Regression: INSERT OR REPLACE wiped json_metadata / watch_date on re-grab."""
    db = TranscriptDatabase(corpus_copy["db_path"])
    vid = "KSTRL42rel1"
    before = db_rows(db.db_path, "SELECT json_metadata, watch_date, created_at FROM videos WHERE video_id=?", (vid,))[0]
    assert before[0] and before[1]
    db.add_video(vid, "Kestrel DB 4.2 Release Deep Dive (re-grab)", "Byte Harbor", "https://youtu.be/KSTRL42rel1")
    after = db_rows(db.db_path, "SELECT json_metadata, watch_date, created_at, title FROM videos WHERE video_id=?", (vid,))[0]
    assert after[:3] == before
    assert after[3].endswith("(re-grab)")


def test_latest_transcript_wins_within_same_second(tmp_path):
    """
    Regression: 'latest transcript' was ORDER BY created_at DESC only. created_at has
    1-second resolution, so a re-save in the same second could attach the Shorty to the
    old row and read back the old text.
    """
    db = TranscriptDatabase(str(tmp_path / "data" / "t.db"))
    db.add_video("SAMESECOND1", "t", "c", "u")
    with sqlite3.connect(db.db_path) as c:
        for text in ("old text", "new text"):
            c.execute(
                "INSERT INTO transcripts (video_id, text, created_at) VALUES (?, ?, '2024-01-01 00:00:00')",
                ("SAMESECOND1", text),
            )
    assert db.get_transcript("SAMESECOND1") == "new text"
    assert db.save_shorty("SAMESECOND1", "SHORTY")
    assert db.get_transcript_and_shorty("SAMESECOND1") == {
        "text": "new text", "shorty": "SHORTY", "shorty_generated_at": pytest.approx(
            db_rows(db.db_path, "SELECT shorty_generated_at FROM transcripts WHERE text='new text'")[0][0])
    }
    assert db_rows(db.db_path, "SELECT shorty FROM transcripts WHERE text='old text'")[0][0] is None


def test_synthetic_question_embedding_ids_match_positions(corpus):
    """Regression: same-second created_at ties reversed the question -> embedding_id mapping."""
    rows = db_rows(corpus["db_path"], "SELECT question, embedding_id FROM synthetic_questions WHERE video_id='VGRrec00004' ORDER BY id")
    assert [r[1] for r in rows] == [f"VGRrec00004:sq:{i}" for i in range(len(rows))]


def test_bare_filename_db_path(tmp_path, monkeypatch):
    """os.makedirs('') raised FileNotFoundError for a db_path with no directory part."""
    monkeypatch.chdir(tmp_path)
    TranscriptDatabase("plain.db")
    assert (tmp_path / "plain.db").exists()


def test_replace_facts_and_counts(corpus_copy):
    db = TranscriptDatabase(corpus_copy["db_path"])
    n = db.replace_facts_for_video("SRDGH780003", [
        {"subject": "a", "relation": "b", "object": "c"},
        {"subject": "", "relation": "skipped", "object": "x"},
    ])
    assert n == 1 and db.count_facts_for_video("SRDGH780003") == 1
    assert db_rows(db.db_path, "SELECT value FROM (SELECT stale AS value FROM global_graph_meta)")[0][0] == 1


def test_enqueue_processing_tasks_default_set(corpus_copy):
    db = TranscriptDatabase(corpus_copy["db_path"])
    db.enqueue_processing_tasks("WGGLdnc0007")
    tasks = {r[0] for r in db_rows(db.db_path, "SELECT task FROM processing_queue WHERE video_id='WGGLdnc0007' AND status='pending'")}
    assert tasks == {"shorty", "synthetic_questions", "entities", "segments", "events", "triples"}


def test_search_transcripts_like(corpus):
    db = TranscriptDatabase(corpus["db_path"])
    hits = db.search_transcripts("Tidewater")
    assert {h["video_id"] for h in hits} == {"KSTRL42rel1", "KSTRLpm0002"}
