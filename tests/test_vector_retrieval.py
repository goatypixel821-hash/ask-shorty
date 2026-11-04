"""Chroma vector retrieval: indexing ids/metadata, chunk + Shorty + synthetic-question layers."""

from __future__ import annotations

import pytest

from tests.conftest import db_rows
from tests.fixtures.corpus import VIDEOS, video_by_id

QUESTIONS = {
    "p99 write latency fsync batching Kestrel": "KSTRL42rel1",
    "Marcus Oyelaran latch page-split race fix": "KSTRLpm0002",
    "levain temperature hydration open crumb sourdough": "SRDGH780003",
    "Golden Record images languages Sagan": "VGRrec00004",
    "telegraph sparks coronal mass ejection 1859": "CRRNGTN1859",
    "Embassy RP2040 async fn in traits firmware bytes": "EMBSSYrs005",
    "waggle dance von Frisch distance": "WGGLdnc0007",
}


@pytest.fixture()
def v1(corpus, monkeypatch):
    import ask_shorty

    monkeypatch.setenv("ASK_SHORTY_DB_PATH", corpus["db_path"])
    eng = ask_shorty.AskShorty()
    eng._rag = corpus["rag"]
    return eng


def test_index_ids_and_metadata(corpus):
    col = corpus["rag"].collection
    got = col.get(ids=["KSTRL42rel1:shorty", "KSTRL42rel1:chunk:0", "KSTRL42rel1:sq:0"])
    metas = dict(zip(got["ids"], got["metadatas"]))
    assert metas["KSTRL42rel1:shorty"] == {"video_id": "KSTRL42rel1", "type": "shorty"}
    assert metas["KSTRL42rel1:chunk:0"]["chunk_index"] == 0
    assert metas["KSTRL42rel1:sq:0"]["type"] == "synthetic_question"


def test_reindex_is_idempotent(corpus):
    """index_single_transcript uses upsert, so re-indexing must not duplicate rows."""
    rag = corpus["rag"]
    v = video_by_id("WGGLdnc0007")
    before = rag.collection.count()
    rag.index_single_transcript(v["video_id"], v["transcript"], shorty=v["shorty"], synthetic_questions=v["questions"])
    assert rag.collection.count() == before


def test_empty_transcript_is_skipped(corpus):
    before = corpus["rag"].collection.count()
    corpus["rag"].index_single_transcript("EMPTYvid001", "   ")
    assert corpus["rag"].collection.count() == before


@pytest.mark.parametrize("query,expected", sorted(QUESTIONS.items()))
def test_chunk_layer_top_hit(v1, query, expected):
    hits = v1._search_layer([query], type_filter="chunk", top_k=3)
    assert hits[0]["metadata"]["video_id"] == expected


@pytest.mark.parametrize("query,expected", sorted(QUESTIONS.items()))
def test_shorty_layer_top_hit(v1, query, expected):
    hits = v1._search_layer([query], type_filter="shorty", top_k=3)
    assert hits[0]["metadata"]["type"] == "shorty"
    assert hits[0]["metadata"]["video_id"] == expected


def test_synthetic_question_layer(v1):
    hits = v1._search_layer(["Who chaired the Golden Record committee?"], type_filter="synthetic_question", top_k=2)
    assert hits[0]["metadata"]["video_id"] == "VGRrec00004"


def test_query_uses_indexing_model_not_chroma_default(corpus):
    """
    Regression: V1 queried with collection.query(query_texts=...), which embeds with
    Chroma's default model. Documents here are embedded by a different model (as with
    the fine-tuned model in production), so neighbours were near-random. rag.query()
    must embed with the same model that indexed.
    """
    res = corpus["rag"].query(["Marcus Oyelaran latch page-split race fix"], n_results=1, where={"type": "chunk"})
    assert res["metadatas"][0][0]["video_id"] == "KSTRLpm0002"


def test_hsc_vector_rank_uses_embed_fn(corpus):
    from hsc.hsc_search import _vector_rank

    rag = corpus["rag"]
    ranked = _vector_rank(rag.collection, "waggle dance von Frisch distance", embed_fn=rag._embed_texts)
    assert ranked[0] == "WGGLdnc0007"


def test_hsc_vector_rank_failure_is_logged_not_silent(caplog):
    from hsc.hsc_search import _vector_rank

    class Broken:
        def query(self, **_):
            raise RuntimeError("chroma exploded")

    assert _vector_rank(Broken(), "anything") == []
    assert "chroma exploded" in caplog.text
