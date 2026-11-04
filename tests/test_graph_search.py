"""Per-video graph search over the facts (triples) table."""

from __future__ import annotations

import sqlite3

import pytest

from graph_search import GraphSearch
from transcript_database import TranscriptDatabase
from tests.conftest import db_rows


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Who fixed the Tidewater page-split race?", "KSTRLpm0002"),
        ("Who chaired the Golden Record committee?", "VGRrec00004"),
        ("What caused sparks from telegraph keys?", "CRRNGTN1859"),
        ("who decoded the waggle dance", "WGGLdnc0007"),
    ],
)
def test_graph_top_hit(corpus, query, expected):
    hits = GraphSearch(corpus["db_path"]).search(query)
    assert hits[0]["video_id"] == expected
    assert " — " in hits[0]["text_preview"]


def test_cross_video_one_hop(corpus):
    """'Tidewater' links the release video and the postmortem via shared entities."""
    vids = [h["video_id"] for h in GraphSearch(corpus["db_path"]).search("Tidewater")]
    assert set(vids[:2]) == {"KSTRL42rel1", "KSTRLpm0002"}


def test_find_connections_and_video_facts(corpus):
    gs = GraphSearch(corpus["db_path"])
    paths = gs.find_connections("Marcus Oyelaran", "Tidewater")
    assert ["Marcus Oyelaran", "fixed", "Tidewater page-split race"] in paths
    assert len(gs.get_video_facts("VGRrec00004")) == 3


def test_missing_facts_table_returns_empty(tmp_path):
    p = tmp_path / "empty.db"
    sqlite3.connect(p).close()
    assert GraphSearch(str(p)).search("anything") == []


def test_stopwords_do_not_crowd_out_the_real_match(corpus_copy):
    """
    Regression: tokens like 'who'/'the'/'is' matched nearly every fact and the first 200
    rows (no ORDER BY) were taken, so on a big facts table the real match was cut off.
    """
    with sqlite3.connect(corpus_copy["db_path"]) as c:
        c.execute("DELETE FROM facts WHERE video_id='KSTRLpm0002'")
        c.executemany(
            "INSERT INTO facts (video_id, subject, relation, object) VALUES (?,?,?,?)",
            [("WGGLdnc0007", f"bee {i}", "is the member of", "the hive") for i in range(400)],
        )
        c.execute("INSERT INTO facts (video_id, subject, relation, object) VALUES "
                  "('KSTRLpm0002', 'Marcus Oyelaran', 'fixed', 'Tidewater page-split race')")
    hits = GraphSearch(corpus_copy["db_path"]).search("who is the one that fixed the Tidewater race?")
    assert hits[0]["video_id"] == "KSTRLpm0002"


def test_zero_confidence_is_kept(corpus_copy):
    """Regression: `confidence or 1.0` turned an explicit 0.0 into 1.0 (storage and scoring)."""
    db = TranscriptDatabase(corpus_copy["db_path"])
    db.replace_facts_for_video("SRDGH780003", [{"subject": "x", "relation": "y", "object": "z", "confidence": 0.0}])
    assert db_rows(db.db_path, "SELECT confidence FROM facts WHERE video_id='SRDGH780003'") == [(0.0,)]
    db.replace_facts_for_video("SRDGH780003", [{"subject": "x", "relation": "y", "object": "z"}])
    assert db_rows(db.db_path, "SELECT confidence FROM facts WHERE video_id='SRDGH780003'") == [(1.0,)]


def test_like_wildcards_in_entities_are_literal(corpus_copy):
    """An entity named '%' in a matched fact must not one-hop-expand to every fact."""
    with sqlite3.connect(corpus_copy["db_path"]) as c:
        c.execute("INSERT INTO facts (video_id, subject, relation, object) VALUES "
                  "('EMBSSYrs005', '%', 'zorblat', 'quux')")
    vids = {h["video_id"] for h in GraphSearch(corpus_copy["db_path"]).search("zorblat", top_k=20)}
    assert vids == {"EMBSSYrs005"}
