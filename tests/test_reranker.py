"""Second-stage reranker: normalisation, grouping, neighbour expansion, blending."""

from __future__ import annotations

import pytest

import reranker as rr
from reranker import Reranker, RetrievalHit


def H(vid, stype="chunk", idx=None, dist=0.5, sid=None, text="t", q="q"):
    sid = sid or (f"{vid}:chunk:{idx}" if stype == "chunk" else f"{vid}:{stype}")
    return RetrievalHit(sid, vid, vid, stype, idx, dist, q, text)


def test_normalize_chroma_result():
    res = {"ids": [["v:chunk:2"]], "documents": [["doc"]], "metadatas": [[{"video_id": "v", "chunk_index": "2"}]],
           "distances": [[0.25]]}
    (h,) = Reranker().normalize_chroma_result(res, "chunk", "q", {"v": "Title"})
    assert (h.video_title, h.chunk_index, h.retrieval_score, h.text) == ("Title", 2, 0.25, "doc")


def test_normalize_flat_results_keeps_type_and_query():
    flat = [{"id": "v:shorty", "text": "s", "score": 0.4, "metadata": {"video_id": "v", "type": "shorty"}, "query": "q2"}]
    (h,) = Reranker().normalize_flat_results(flat, {})
    assert (h.source_type, h.query_variant, h.video_title) == ("shorty", "q2", "v")


def test_grouping_rules():
    hits = [
        H("v", idx=0, dist=0.4), H("v", idx=2, dist=0.3), H("v", idx=0, dist=0.2, q="variant2"),
        H("v", idx=7), H("v", "shorty"), H("v", "synthetic_question"), H("w", idx=0),
    ]
    groups = {g.group_key: g for g in Reranker().group_hits(hits, expand_neighbors=False)}
    assert set(groups) == {"v|chunk|0", "v|chunk|2", "v|shorty", "v|synq", "w|chunk|0"}
    g0 = groups["v|chunk|0"]
    assert g0.support_count == 3 and g0.best_retrieval_score == 0.2
    assert g0.chunk_ids == ["v:chunk:0", "v:chunk:2"]  # duplicate id across variants collapses


def test_bucket_boundary_is_floor_division_not_radius():
    """
    Docstring says chunks 'within CHUNK_NEIGHBORHOOD_RADIUS of each other' share a group,
    but the key is chunk_index // 3: adjacent chunks 2 and 3 split, 0 and 2 merge.
    Pinned so the mismatch is visible (see HEALTH.md).
    """
    keys = [Reranker()._group_key(H("v", idx=i)) for i in (0, 2, 3)]
    assert keys == ["v|chunk|0", "v|chunk|0", "v|chunk|1"]


def test_best_score_and_text_when_all_distances_above_one():
    """Regression: best_retrieval_score defaulted to 1.0, so distances >1 never registered."""
    hits = [H("v", idx=0, dist=1.6, text="worse"), H("v", idx=1, dist=1.3, text="better")]
    (g,) = Reranker().group_hits(hits, expand_neighbors=False)
    assert g.best_retrieval_score == pytest.approx(1.3)
    assert g.compact_text == "better"


def test_expand_chunk_neighbors_against_real_collection(corpus):
    col = corpus["rag"].collection
    hit = H("KSTRL42rel1", idx=1, sid="KSTRL42rel1:chunk:1",
            text=col.get(ids=["KSTRL42rel1:chunk:1"])["documents"][0])
    text = Reranker().expand_chunk_neighbors(hit, col)
    docs = col.get(ids=[f"KSTRL42rel1:chunk:{i}" for i in (0, 1)])["documents"]
    assert all(d in text for d in docs) and len(text) > len(hit.text)


def test_expand_handles_old_id_format_and_failures():
    class Boom:
        def get(self, ids):
            assert ids == ["v_chunk_0", "v_chunk_1", "v_chunk_2"]
            raise RuntimeError("boom")

    hit = H("v", idx=1, sid="v_chunk_1", text="orig")
    assert Reranker().expand_chunk_neighbors(hit, Boom()) == "orig"


def test_rerank_and_blend_orders_by_relevance():
    hits = [
        H("SRDGH780003", idx=0, dist=0.2, text="levain hydration crumb sourdough"),
        H("KSTRLpm0002", idx=1, dist=0.6, text="Marcus Oyelaran added a latch around the split path"),
    ]
    groups = Reranker().group_hits(hits, expand_neighbors=False)
    ranked = Reranker().rerank_and_blend("who added the latch around the split path", groups)
    assert ranked[0].video_id == "KSTRLpm0002"
    assert all(0.0 <= g.rerank_score <= 1.0 for g in ranked)
    assert ranked[0].final_score >= ranked[1].final_score


def test_blend_weights_sum_to_one_and_bounded():
    assert rr.WEIGHT_RERANK + rr.WEIGHT_RETRIEVAL + rr.WEIGHT_SUPPORT + rr.WEIGHT_DIVERSITY == pytest.approx(1.0)
    g = rr.EvidenceGroup("k", "v", "v", 0, support_count=8, support_types=["chunk", "shorty", "synthetic_question"],
                         best_retrieval_score=0.0, rerank_score=1.0)
    assert Reranker()._blend(g) == pytest.approx(1.0)


def test_groups_cap_and_context_blocks():
    hits = [H(f"v{i:010d}", idx=0, dist=0.1 + i / 100, text=f"text {i}") for i in range(rr.MAX_GROUPS_TO_RERANK + 5)]
    groups = Reranker().group_hits(hits, expand_neighbors=False)
    ranked = Reranker().rerank_and_blend("text", groups)
    assert len(ranked) == len(hits)
    blocks = Reranker().groups_to_context_blocks(ranked, top_n=3)
    assert len(blocks) == 3 and blocks[0].startswith("[EVIDENCE] video_id=")


def test_answer_question_rerank_path_end_to_end(corpus, fake_llm, monkeypatch):
    import ask_shorty

    monkeypatch.setenv("ASK_SHORTY_DB_PATH", corpus["db_path"])
    monkeypatch.setattr(ask_shorty, "ENABLE_RERANKING", True)
    eng = ask_shorty.AskShorty()
    eng._rag = corpus["rag"]
    res = eng.answer_question("Who fixed the Tidewater page-split corruption bug?")
    assert res["sources"][0]["video_id"] == "KSTRLpm0002"
    assert "Marcus Oyelaran" in res["answer"]
