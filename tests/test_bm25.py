"""BM25: the V1 keyword index (bm25_index) and the V2 per-level indexes (v2_bm25)."""

from __future__ import annotations

import pytest

import v2_bm25
from bm25_index import BM25Search, _tokenize, reciprocal_rank_fusion


@pytest.fixture()
def bm25(corpus):
    return BM25Search(corpus["bm25_path"])


def test_tokenize():
    assert _tokenize("Rust 1.75 stabilised async-fn!") == ["rust", "1", "75", "stabilised", "async", "fn"]
    assert _tokenize("") == []


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Tidewater latch Oyelaran", "KSTRLpm0002"),
        ("fsync batching p99", "KSTRL42rel1"),
        ("levain hydration", "SRDGH780003"),
        ("uranium-238 phonograph", "VGRrec00004"),
        ("telegraph auroral current", "CRRNGTN1859"),
        ("Embassy RP2040", "EMBSSYrs005"),
        ("waggle Frisch", "WGGLdnc0007"),
    ],
)
def test_bm25_top_hit(bm25, query, expected):
    hits = bm25.search(query, top_k=5)
    assert hits[0]["video_id"] == expected


def test_bm25_one_hit_per_video_and_doc_types(bm25, corpus):
    hits = bm25.search("Kestrel Tidewater", top_k=20)
    vids = [h["video_id"] for h in hits]
    assert len(vids) == len(set(vids))
    assert set(vids) == {"KSTRL42rel1", "KSTRLpm0002"}
    types = set(BM25Search(corpus["bm25_path"]).payload["doc_types"])
    assert types == {"chunk", "shorty", "entity"}


def test_bm25_no_match_and_empty(bm25):
    assert bm25.search("zzzqqq nonexistentterm") == []
    assert bm25.search("   ") == []


def test_bm25_missing_index_file_is_empty(tmp_path):
    assert BM25Search(str(tmp_path / "nope.pkl")).search("anything") == []


def test_rrf_math():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)
    assert reciprocal_rank_fusion([[], ["x"]]) == [("x", pytest.approx(1 / 61))]


def test_v2_tokenizer_is_stemmed_and_lemmatised():
    assert v2_bm25.tokenize("Bitten running dances") == ["bite", "run", "danc"]


def test_v2_video_and_segment_indexes(corpus):
    vix = v2_bm25.GenericBM25Index(corpus["v2_paths"]["video"])
    assert vix.search("Tidewater", 5)[0][0] in {"KSTRL42rel1", "KSTRLpm0002"}
    six = v2_bm25.GenericBM25Index(corpus["v2_paths"]["segment"])
    hits = six.search_in_video_subset("latch split", {"KSTRLpm0002"}, 5)
    assert hits and all(six.payload["video_ids"][six.payload["doc_ids"].index(d)] == "KSTRLpm0002" for d, _ in hits)
