"""RRF fusion in AskShorty._apply_rrf_fusion and the claimed per-layer weights."""

from __future__ import annotations

import pytest


def _hit(vid, layer, dist, idx=0, text="..."):
    meta = {"video_id": vid, "type": layer}
    if layer == "chunk":
        meta["chunk_index"] = idx
    rid = f"{vid}:{'chunk:%d' % idx if layer == 'chunk' else layer}"
    return {"id": rid, "text": text, "score": dist, "metadata": meta, "query": "q"}


@pytest.fixture()
def v1(corpus, monkeypatch):
    import ask_shorty

    monkeypatch.setenv("ASK_SHORTY_DB_PATH", corpus["db_path"])
    monkeypatch.setattr(ask_shorty, "ENABLE_BM25", True)
    monkeypatch.setattr(ask_shorty, "ENABLE_GRAPH", False)
    eng = ask_shorty.AskShorty()
    eng._rag = corpus["rag"]
    return eng


def test_layer_weights_claim_is_not_implemented(v1):
    """
    ROADMAP Phase 2 lists 'weighted retrieval (Shorty 2.0x, questions 1.5x, chunks 1.0x)'
    as unchecked, and the code agrees: _vector_video_rank folds all three Chroma layers
    into one list keyed by best cosine distance, then RRF weights every list equally.
    This test pins that behaviour so a future weighting change is a conscious one.
    """
    chunk = [_hit("AAAAAAAAAAA", "chunk", 0.30)]
    shorty = [_hit("BBBBBBBBBBB", "shorty", 0.30)]
    synq = [_hit("CCCCCCCCCCC", "synthetic_question", 0.30)]
    rank = v1._vector_video_rank(chunk, shorty, synq)
    assert set(rank) == {"AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC"}
    # A Shorty hit is not preferred over a chunk hit at equal distance: order is input order.
    assert rank == ["AAAAAAAAAAA", "BBBBBBBBBBB", "CCCCCCCCCCC"]


def test_fused_scores_stay_in_distance_domain(v1):
    """Regression: fused score was -rrf + 0.01*dist (negative), but the reranker treats score as cosine distance."""
    chunks = [_hit("SRDGH780003", "chunk", 0.2), _hit("KSTRLpm0002", "chunk", 0.9)]
    out, shorty, synq, dbg = v1._apply_rrf_fusion("Tidewater latch Oyelaran page-split", chunks, [], [])
    for r in out:
        assert 0.0 <= r["score"] <= 2.0, r
        assert "vector_distance" in r["metadata"] or r["id"].endswith(":synth")


def test_fused_order_is_applied_to_lists(v1):
    """
    Regression: _apply_rrf_fusion rewrote scores but never re-sorted, so the default
    (no-rerank) path sliced [:top_k] from the raw cosine order and fusion had no effect
    beyond appending BM25 'synth' chunks.
    """
    # Vector layer likes the sourdough video best; BM25 (real fixture index) and
    # vector both contain the postmortem, so RRF puts it first.
    chunks = [
        _hit("SRDGH780003", "chunk", 0.10, 0),
        _hit("WGGLdnc0007", "chunk", 0.15, 0),
        _hit("KSTRLpm0002", "chunk", 0.20, 1),
    ]
    out, _, _, dbg = v1._apply_rrf_fusion("Tidewater latch Oyelaran page-split race", chunks, [], [])
    assert dbg["rrf_top"][0]["video_id"] == "KSTRLpm0002"
    assert out[0]["metadata"]["video_id"] == "KSTRLpm0002"
    assert "bm25" in out[0]["metadata"]["fusion_source"]


def test_bm25_only_video_gets_synth_chunk(v1):
    chunks = [_hit("SRDGH780003", "chunk", 0.1)]
    out, _, _, dbg = v1._apply_rrf_fusion("Embassy RP2040 firmware", chunks, [], [])
    synth = [r for r in out if r["id"].endswith(":bm25:synth")]
    assert any(r["metadata"]["video_id"] == "EMBSSYrs005" for r in synth)
    assert dbg["bm25_hits"] >= 1


def test_answer_question_context_follows_fusion(v1, fake_llm, monkeypatch):
    """End to end (no rerank): the first chunk block in the LLM context is the fused winner."""
    def fake_layer(variants, type_filter=None, top_k=4):
        if type_filter == "chunk":
            return [_hit("SRDGH780003", "chunk", 0.10, 0, "sourdough text"),
                    _hit("WGGLdnc0007", "chunk", 0.12, 0, "bee text"),
                    _hit("VGRrec00004", "chunk", 0.13, 0, "voyager text"),
                    _hit("CRRNGTN1859", "chunk", 0.14, 0, "carrington text"),
                    _hit("KSTRLpm0002", "chunk", 0.30, 1, "Marcus Oyelaran added a latch around the split path")]
        return []

    monkeypatch.setattr(v1, "_search_layer", fake_layer)
    res = v1.answer_question("Tidewater latch Oyelaran page-split race")
    first_chunk = next(b for b in res["used_context"] if b.startswith("[chunk]"))
    assert "video_id=KSTRLpm0002" in first_chunk.split("\n")[0]
    assert "KSTRLpm0002" in res["answer"]
