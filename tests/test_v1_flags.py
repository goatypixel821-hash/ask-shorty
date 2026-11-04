"""V1 answer_question with the optional layers switched on (HSC routing, graph, BM25)."""

from __future__ import annotations

import pytest


@pytest.fixture()
def v1(corpus, monkeypatch):
    import ask_shorty

    monkeypatch.setenv("ASK_SHORTY_DB_PATH", corpus["db_path"])
    monkeypatch.setattr(ask_shorty, "ENABLE_BM25", True)
    monkeypatch.setattr(ask_shorty, "ENABLE_GRAPH", True)
    monkeypatch.setattr(ask_shorty, "ENABLE_HSC", True)
    eng = ask_shorty.AskShorty()
    eng._rag = corpus["rag"]
    return eng


@pytest.mark.parametrize(
    "q,vid",
    [
        ("Who fixed the Tidewater page-split corruption bug?", "KSTRLpm0002"),
        ("What caused the telegraph sparks during the Carrington Event?", "CRRNGTN1859"),
        ("Why does 78% hydration sourdough get an open crumb?", "SRDGH780003"),
    ],
)
def test_all_layers_on(v1, fake_llm, q, vid):
    events = []
    res = v1.answer_question(q, emit=events.append)
    assert res["sources"][0]["video_id"] == vid
    fusion = next(e for e in events if e["type"] == "fusion")
    assert fusion["bm25_hits"] > 0 and fusion["hsc_rank_len"] > 0
    assert fusion["rrf_top"][0]["video_id"] == vid


def test_video_id_filter(v1, fake_llm):
    res = v1.answer_question("Tidewater", video_ids=["KSTRL42rel1"])
    assert {s["video_id"] for s in res["sources"]} == {"KSTRL42rel1"}


def test_empty_question_rejected(v1):
    with pytest.raises(ValueError):
        v1.answer_question("   ")
