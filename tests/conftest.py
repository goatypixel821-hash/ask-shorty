"""
Shared fixtures: build the synthetic corpus into a temporary DB / Chroma / BM25 /
V2-index directory using the project's own code paths, and keep every test away
from the network and from the repo's real ``data/`` directory.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Make sure nothing below can pick up a real key from the developer's shell.
for _k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(_k, None)
# Feature flags the tests rely on being off unless a test turns them on.
for _k in ("ASK_SHORTY_RERANK", "ASK_SHORTY_GRAPH", "ASK_SHORTY_HSC", "ASK_SHORTY_AGENT", "ASK_SHORTY_V2_VERIFY"):
    os.environ.pop(_k, None)

from tests.fixtures.corpus import VIDEOS  # noqa: E402

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class StandinEmbedder:
    """
    Tiny deterministic stand-in for SentenceTransformer: hashed bag-of-words into
    384 dims, L2-normalised. Used when the real model can't be downloaded and in
    tests that need an embedding space *different* from Chroma's default one.
    """

    dim = 384

    def encode(self, texts, show_progress_bar: bool = False, **_: Any):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in "".join(c.lower() if c.isalnum() else " " for c in t).split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0 if (h >> 9) & 1 else -1.0
            n = float(np.linalg.norm(v)) or 1.0
            out.append(v / n)
        return np.stack(out)


_REAL_MODEL_CACHE: Dict[str, Any] = {}


USE_REAL_MODELS = os.environ.get("ASK_SHORTY_TEST_REAL_MODELS") == "1"


class StandinCrossEncoder:
    """Stand-in for sentence_transformers.CrossEncoder: query-token overlap as a logit."""

    def predict(self, pairs, **_: Any):
        out = []
        for q, doc in pairs:
            qt = {t for t in "".join(c.lower() if c.isalnum() else " " for c in q).split() if len(t) > 2}
            dt = set("".join(c.lower() if c.isalnum() else " " for c in doc).split())
            out.append(float(len(qt & dt)) - 2.0)
        return np.array(out, dtype=np.float32)


def load_cross_encoder():
    if USE_REAL_MODELS:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(RERANK_MODEL), RERANK_MODEL
    return StandinCrossEncoder(), "standin"


def load_embedder():
    """
    Stand-in by default: model weights come from us.aws.cdn.hf.co, which the
    sandbox proxy blocks (huggingface.co itself is reachable). Set
    ASK_SHORTY_TEST_REAL_MODELS=1 to use the real all-MiniLM-L6-v2.
    """
    if not USE_REAL_MODELS:
        return StandinEmbedder(), "standin"
    if "m" in _REAL_MODEL_CACHE:
        return _REAL_MODEL_CACHE["m"], EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(EMBED_MODEL)
        _REAL_MODEL_CACHE["m"] = m
        return m, EMBED_MODEL
    except Exception as exc:  # pragma: no cover - depends on network policy
        print(f"[tests] could not load {EMBED_MODEL} from huggingface.co ({exc!r}); using stand-in")
        return StandinEmbedder(), "standin"


def make_rag(db_path: str, chroma_dir: str, embedder=None, monkeypatch=None):
    """Construct the project's EnhancedTranscriptRAG with an injected embedder."""
    import transcript_rag_enhanced as tre

    emb = embedder or load_embedder()[0]
    orig = tre.SentenceTransformer
    tre.SentenceTransformer = lambda *_a, **_k: emb  # type: ignore[assignment]
    saved_shared = tre._shared_model
    tre._shared_model = None
    try:
        rag = tre.EnhancedTranscriptRAG(
            transcript_db=db_path, chroma_dir=chroma_dir, embedding_model_name="test-injected-model"
        )
    finally:
        tre.SentenceTransformer = orig  # type: ignore[assignment]
        tre._shared_model = saved_shared
    return rag


def build_corpus(base: Path, embedder=None) -> Dict[str, Any]:
    """Index every fixture video through the same functions the real pipeline uses."""
    from transcript_database import TranscriptDatabase
    from entity_extractor import store_entities
    from bm25_index import BM25Search, default_bm25_index_path
    import v2_bm25

    data = base / "data"
    data.mkdir(parents=True, exist_ok=True)
    db_path = str(data / "transcripts.db")
    chroma_dir = str(data / "transcript_chroma")
    db = TranscriptDatabase(db_path)

    for v in VIDEOS:
        vid = v["video_id"]
        db.add_video(vid, v["title"], v["channel"], v["url"])
        db.save_transcript(vid, v["transcript"])
        db.save_metadata(vid, {"upload_date": v["upload_date"], "duration": 180})
        db.set_watch_date(vid, datetime.strptime(v["watch_date"], "%Y-%m-%d %H:%M:%S"))
        assert db.save_shorty(vid, v["shorty"])
        with sqlite3.connect(db_path) as conn:  # same INSERT batch_processor uses
            for q in v["questions"]:
                conn.execute(
                    "INSERT INTO synthetic_questions (video_id, question, embedding_id) VALUES (?, ?, NULL)",
                    (vid, q),
                )
        store_entities(vid, v["entities"], db=db)
        db.replace_facts_for_video(vid, v["facts"])
        db.replace_segments_for_video(
            vid, [{"start_time": s, "end_time": e, "summary": t} for s, e, t in v["segments"]]
        )
        db.replace_events_for_video(vid, v["events"])

    rag = make_rag(db_path, chroma_dir, embedder)
    for v in VIDEOS:
        rag.index_single_transcript(v["video_id"], v["transcript"], shorty=v["shorty"], synthetic_questions=v["questions"])

    bm25_path = default_bm25_index_path(db_path)
    BM25Search(str(bm25_path)).rebuild(db_path)

    # V2: populate tables then build the three BM25 pickles (same as the CLI scripts).
    import populate_v2_indexes

    argv = sys.argv
    sys.argv = ["populate_v2_indexes.py", "--db-path", db_path]
    try:
        populate_v2_indexes.main()
    finally:
        sys.argv = argv
    v2_paths = v2_bm25.build_all_v2_bm25(db_path)
    return {
        "base": base,
        "db_path": db_path,
        "chroma_dir": chroma_dir,
        "bm25_path": str(bm25_path),
        "v2_paths": v2_paths,
        "rag": rag,
        "db": db,
    }


@pytest.fixture(scope="session")
def embedder():
    return load_embedder()


@pytest.fixture(scope="session")
def corpus(tmp_path_factory, embedder):
    """Read-only corpus shared by the session. Tests that mutate must use ``corpus_copy``."""
    base = tmp_path_factory.mktemp("corpus")
    env = build_corpus(base, embedder[0])
    env["embed_name"] = embedder[1]
    return env


@pytest.fixture()
def corpus_copy(corpus, tmp_path):
    """A private, writable copy of the corpus SQLite DB (Chroma stays shared, read-only)."""
    data = tmp_path / "data"
    data.mkdir()
    src = Path(corpus["db_path"])
    with sqlite3.connect(str(src)) as s, sqlite3.connect(str(data / "transcripts.db")) as d:
        s.backup(d)
    shutil.copytree(Path(corpus["base"]) / "data" / "v2_indexes", data / "v2_indexes")
    shutil.copy(corpus["bm25_path"], data / "bm25_index.pkl")
    return {**corpus, "db_path": str(data / "transcripts.db"), "base": tmp_path}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Keep tests off the network and out of the repo's data/ directory."""
    import ask_shorty_v2
    import v2_bm25

    monkeypatch.setattr(ask_shorty_v2, "_v2_log_path", lambda: tmp_path / "v2_request_log.html")
    # Each test sees fresh process-wide caches.
    monkeypatch.setattr(ask_shorty_v2, "_v2_maps_singleton", {})
    monkeypatch.setattr(v2_bm25, "_bm25_loaded", {})
    yield


@pytest.fixture(autouse=True)
def _cross_encoder(monkeypatch):
    """Inject the (stand-in or real) cross-encoder into both reranker paths."""
    import reranker
    import ask_shorty_v2

    ce = _CE_CACHE.setdefault("ce", load_cross_encoder()[0])
    monkeypatch.setattr(reranker, "_cross_encoder", ce)
    monkeypatch.setattr(ask_shorty_v2.AskShortyV2, "_get_cross_encoder", lambda self: ce)
    return ce


_CE_CACHE: Dict[str, Any] = {}


@pytest.fixture()
def fake_llm(monkeypatch):
    from tests.fakes import install_fake_llm

    return install_fake_llm(monkeypatch)


def db_rows(db_path: str, sql: str, params=()) -> List[tuple]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(sql, params).fetchall()
