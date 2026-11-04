"""V2 hierarchical RAG (ask_shorty_v2) end to end with the fake LLM."""

from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

import pytest

import ask_shorty_v2 as v2
import v2_bm25
from tests.fakes import FakeLLM, install_fake_llm
from tests.fixtures.corpus import EXPECTED_SOURCES
from v2_memory_maps import V2MemoryMaps


@pytest.fixture()
def engine(corpus):
    return v2.AskShortyV2(db_path=corpus["db_path"])


@pytest.mark.parametrize("question,expected", sorted(EXPECTED_SOURCES.items()))
def test_answer_cites_right_video(engine, fake_llm, question, expected):
    res = engine.answer(question)
    assert res["sources"][0]["video_id"] == expected[0]
    assert f"video_id={expected[0]}" in res["answer"]
    assert res["grounding_audit"] == []
    # Context blocks are built from the DB (Shorty + segments), headed by the video id.
    assert f"VIDEO_ID: {expected[0]}" in res["used_context"][0]
    assert res["sources"][0]["timestamp_range"]


def test_cross_video_fact_brings_both_videos(engine, fake_llm):
    """'Who fixed the bug in the engine introduced in 4.2?' needs the release video AND the postmortem."""
    res = engine.answer("Who fixed the bug in the Tidewater storage engine introduced in Kestrel 4.2?")
    vids = [s["video_id"] for s in res["sources"]]
    assert {"KSTRL42rel1", "KSTRLpm0002"} <= set(vids[:3])
    assert "Marcus Oyelaran" in res["answer"]


def test_prompt_contains_grounding_rules_and_summary(engine, fake_llm):
    engine.answer("How many images are on the Voyager Golden Record?")
    call = [c for c in fake_llm.calls if "Ask Shorty (V2)" in c["system"]][-1]
    assert "GROUNDING (mandatory)" in call["system"]
    assert "RETRIEVAL SUMMARY" in call["user"] and "Indexed library: 7 videos total; 7 have a Shorty" in call["user"]


def test_restrict_videos(engine, fake_llm):
    vids, dbg = engine.retrieve_videos("Tidewater", restrict_videos=["KSTRL42rel1"])
    assert vids == ["KSTRL42rel1"]


def test_cause_effect_route_uses_events(engine, fake_llm):
    res = engine.answer("What caused the telegraph sparks during the Carrington Event?")
    assert res["sources"][0]["route_type"] == "cause_effect"
    assert "Events:" in res["used_context"][0]


def test_no_candidates_skips_llm(engine, fake_llm):
    res = engine.answer("zzqx blorptang vrillium")
    assert res["sources"] == [] and "No indexed videos matched" in res["answer"]
    assert not any("Ask Shorty (V2)" in c["system"] for c in fake_llm.calls)


def test_cancel_before_llm(engine, fake_llm):
    res = engine.answer("How many images are on the Voyager Golden Record?", should_cancel=lambda: True)
    assert res["answer"] == "" and res["sources"] == []
    assert not any("Ask Shorty (V2)" in c["system"] for c in fake_llm.calls)


@pytest.mark.parametrize("mode", ["garbage", "empty", "timeout", "error"])
def test_query_expansion_failures_fall_back_to_question(engine, monkeypatch, mode):
    install_fake_llm(monkeypatch, FakeLLM(mode=mode))
    assert v2._expand_query("Tidewater latch") == ["Tidewater latch"]


@pytest.mark.parametrize("mode", ["timeout", "error"])
def test_answer_llm_failure_raises(engine, monkeypatch, mode):
    """The answer call must raise so the job runner marks the job 'error', not 'completed'."""
    install_fake_llm(monkeypatch, FakeLLM(mode=mode))
    with pytest.raises(Exception):
        engine.answer("How many images are on the Voyager Golden Record?")


def test_fast_search(engine):
    out = engine.search_fast("Embassy RP2040") if hasattr(engine, "search_fast") else None
    assert out is not None


def test_missing_v2_indexes_are_reported(corpus_copy, fake_llm):
    """Regression: with the BM25 pickles missing V2 silently ran with empty layers."""
    shutil.rmtree(Path(corpus_copy["db_path"]).parent / "v2_indexes")
    res = v2.AskShortyV2(db_path=corpus_copy["db_path"]).answer("Who fixed the Tidewater page-split corruption bug?")
    warnings = [e for e in res["debug_events"] if e["type"] == "warning"]
    assert warnings and "video_bm25_index.pkl" in warnings[0]["label"]


def test_rebuilt_index_is_picked_up_without_restart(corpus_copy):
    """Regression: load_shared_bm25_index cached by path forever; rebuilds were invisible."""
    eng = v2.AskShortyV2(db_path=corpus_copy["db_path"])
    assert eng.retrieve_videos("zanzibarite")[0] == []
    with sqlite3.connect(corpus_copy["db_path"]) as c:
        c.execute("UPDATE video_signatures SET routing_text = routing_text || ' zanzibarite' WHERE video_id='WGGLdnc0007'")
    time.sleep(0.01)
    v2_bm25.build_video_bm25(corpus_copy["db_path"], eng.index_dir / "video_bm25_index.pkl")
    assert eng.retrieve_videos("zanzibarite")[0][:1] == ["WGGLdnc0007"]


def test_instant_candidates_handle_punctuation_and_multiword_aliases(corpus):
    """Regression: str.split() kept '?', so the last word of a question never matched;
    multi-word aliases ('marcus oyelaran') were unreachable; order depended on hash seed."""
    mm = V2MemoryMaps.load(corpus["db_path"])
    assert mm.instant_candidates("what is tidewater?") == ["KSTRL42rel1", "KSTRLpm0002"]
    assert "KSTRLpm0002" in mm.instant_candidates("who is marcus oyelaran?")
    assert mm.instant_candidates("tidewater kestrel") == sorted(mm.instant_candidates("tidewater kestrel"))


def test_select_llm_context_prefers_topic_tokens(corpus):
    ranked = ["WGGLdnc0007", "VGRrec00004", "SRDGH780003"]
    sel, toks, n = v2._select_llm_context_video_ids(corpus["db_path"], ranked, "tell me about the golden record")
    assert sel == ["VGRrec00004"] and toks == ["golden", "record"] and n == 1


def test_request_log_goes_to_patched_path(engine, fake_llm, tmp_path):
    engine.answer("How many images are on the Voyager Golden Record?", job_id=7)
    assert (tmp_path / "v2_request_log.html").is_file()


@pytest.mark.parametrize(
    "q,route",
    [
        ("What caused the telegraph sparks?", "cause_effect"),
        ("What causes an open crumb?", "cause_effect"),
        ("What were the effects of the storm?", "cause_effect"),
        ("Why did Kestrel batch fsync?", "cause_effect"),
        ("compare Tidewater versus the old pager", "cross_video_compare"),
    ],
)
def test_router_cause_effect_inflections(q, route):
    """Regression: 'caused'/'causes'/'effects' fell through to the general route."""
    from query_router import QueryRouter

    assert QueryRouter().classify(q).route_type == route
