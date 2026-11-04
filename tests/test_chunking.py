"""Chunking. There are (at least) four copies of the same 800/200 char-window chunker."""

from __future__ import annotations

import pytest

import ask_shorty_v2
import bm25_index
from tests.fixtures.corpus import VIDEOS

LONG = " ".join(f"word{i:04d}" for i in range(600))  # ~5,400 chars


@pytest.fixture()
def rag_chunker(corpus):
    return corpus["rag"]._chunk_transcript


def test_short_text_is_single_chunk(rag_chunker):
    assert rag_chunker("  hello world  ") == ["hello world"]
    assert bm25_index._chunk_text("x" * 800) == ["x" * 800]


def test_window_and_overlap(rag_chunker):
    chunks = rag_chunker(LONG)
    assert all(len(c) <= 800 for c in chunks)
    # Consecutive windows overlap by 200 chars (before strip()).
    for a, b in zip(chunks, chunks[1:]):
        assert a[-150:].strip()[:50] in b
    # Every word of the input survives chunking.
    joined = " ".join(chunks)
    assert all(f"word{i:04d}" in joined for i in range(600))


def test_terminates_and_last_chunk_reaches_end(rag_chunker):
    chunks = rag_chunker(LONG)
    assert chunks[-1].endswith("word0599")
    assert len(chunks) == 9


@pytest.mark.parametrize("text", [LONG] + [v["transcript"] for v in VIDEOS])
def test_all_chunker_copies_agree(rag_chunker, text):
    """
    Chroma chunks, BM25 chunks and V2 verification chunks must line up, otherwise a
    BM25 'chunk N' and a Chroma 'chunk N' are different text. Guard the duplication.
    """
    assert rag_chunker(text) == bm25_index._chunk_text(text) == ask_shorty_v2._chunk_transcript_text(text)


def test_empty_inputs():
    assert bm25_index._chunk_text("") == []
    assert ask_shorty_v2._chunk_transcript_text(None) == []


def test_fixture_transcripts_produce_multiple_chunks(corpus):
    got = corpus["rag"].collection.get(where={"type": "chunk"})
    per_video = {}
    for m in got["metadatas"]:
        per_video[m["video_id"]] = per_video.get(m["video_id"], 0) + 1
    assert all(n >= 2 for n in per_video.values()), per_video
