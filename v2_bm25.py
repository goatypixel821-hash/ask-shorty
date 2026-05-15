#!/usr/bin/env python3
"""
V2 BM25 indexes: separate pickles for video routing text, segments, and events.
Compatible payload shape with rank_bm25.BM25Okapi (see bm25_index.py).

Tokenization uses a tiny irregular-verb map (Porter misses e.g. bitten→bite) plus
NLTK PorterStemmer for all other tokens (fast). After changing :func:`tokenize`, you must
rebuild pickles so corpus and query use the same vocabulary, e.g.::

    python v2_bm25.py --db-path data/transcripts.db

Old index files are incompatible with a new tokenizer (silent score skew if mixed).
"""

from __future__ import annotations

import pickle
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

# Irregular verb surface forms → lemma for search (WordNet-quality on these, no WordNet I/O).
_IRREGULAR_VERB_LEMMA: Dict[str, str] = {
    "am": "be",
    "are": "be",
    "been": "be",
    "being": "be",
    "is": "be",
    "was": "be",
    "were": "be",
    "bit": "bite",
    "bites": "bite",
    "bitten": "bite",
    "biting": "bite",
    "ran": "run",
    "running": "run",
    "runs": "run",
}

# Porter is fast; shared by index build and queries.
_stemmer = PorterStemmer()
# Same alphanumeric word pattern as bm25_index._tokenize.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """
    Shared tokenizer for V2 BM25: lowercase → word tokens → irregular map or Porter stem.
    Used for both index construction and query scoring — must not diverge.
    """
    if not text or not text.strip():
        return []
    out: List[str] = []
    for raw in _WORD_RE.findall(text):
        w = raw.lower()
        if not w:
            continue
        if w in _IRREGULAR_VERB_LEMMA:
            out.append(_IRREGULAR_VERB_LEMMA[w])
        else:
            out.append(_stemmer.stem(w))
    return out

# One loaded index per resolved path — avoids re-reading multi‑MB pickles and
# rebuilding BM25Okapi on every new AskShortyV2 / eval query.
_bm25_singleton_lock = threading.Lock()
_bm25_loaded: Dict[str, "GenericBM25Index"] = {}


def load_shared_bm25_index(index_path: Path | str) -> Optional["GenericBM25Index"]:
    """
    Return a process-wide cached GenericBM25Index for this file path.

    Building BM25Okapi from ~10k–30k docs after each pickle.load dominates latency
    (~seconds–tens of seconds); caching keeps warm indices in RAM across queries.
    """
    resolved = Path(index_path).resolve()
    key = str(resolved)
    with _bm25_singleton_lock:
        hit = _bm25_loaded.get(key)
        if hit is not None:
            return hit
        if not resolved.is_file():
            return None
        inst = GenericBM25Index.__new__(GenericBM25Index)
        inst.index_path = resolved
        inst._bm25 = None
        inst._payload = {}
        inst._load()
        _bm25_loaded[key] = inst
        return inst


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_v2_index_dir(db_path: str) -> Path:
    return Path(db_path).resolve().parent / "v2_indexes"


def _save_pkl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _build_tokenized_corpus(texts: Sequence[str]) -> Tuple[List[str], List[List[str]]]:
    corpus: List[str] = []
    tokenized: List[List[str]] = []
    for t in texts:
        s = (t or "").strip()
        corpus.append(s)
        tok = tokenize(s)
        tokenized.append(tok if tok else ["_"])
    return corpus, tokenized


class GenericBM25Index:
    """BM25 over arbitrary doc ids (video_id, segment_id, event_id, ...)."""

    def __init__(self, index_path: Path | str):
        """Load from disk immediately (prefer :func:`load_shared_bm25_index` for reuse)."""
        self.index_path = Path(index_path).resolve()
        self._bm25: Optional[BM25Okapi] = None
        self._payload: Dict[str, Any] = {}
        if self.index_path.is_file():
            self._load()

    def _load(self) -> None:
        with open(self.index_path, "rb") as f:
            self._payload = pickle.load(f)
        tok = self._payload.get("tokenized") or []
        self._bm25 = BM25Okapi(tok) if tok else None

    @property
    def payload(self) -> Dict[str, Any]:
        return self._payload

    def search(
        self,
        query: str,
        top_k: int,
        *,
        restrict_ids: Optional[set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return (doc_id, score) best-first. Optionally filter doc_ids."""
        if not self._bm25 or not (query or "").strip():
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        doc_ids: List[str] = list(self._payload.get("doc_ids") or [])
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: List[Tuple[str, float]] = []
        for i in ranked_idx:
            if scores[i] <= 0:
                continue
            did = doc_ids[i] if i < len(doc_ids) else ""
            if not did:
                continue
            if restrict_ids is not None and did not in restrict_ids:
                continue
            out.append((did, float(scores[i])))
            if len(out) >= top_k:
                break
        return out

    def search_in_video_subset(
        self,
        query: str,
        allowed_videos: Set[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """For segment/event indexes: doc_ids + parallel video_ids in payload."""
        if not self._bm25 or not (query or "").strip():
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        doc_ids: List[str] = list(self._payload.get("doc_ids") or [])
        vids: List[str] = list(self._payload.get("video_ids") or [])
        if len(vids) != len(doc_ids):
            vids = [""] * len(doc_ids)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: List[Tuple[str, float]] = []
        for i in ranked_idx:
            if scores[i] <= 0:
                continue
            if i >= len(doc_ids):
                break
            if vids[i] not in allowed_videos:
                continue
            out.append((doc_ids[i], float(scores[i])))
            if len(out) >= top_k:
                break
        return out


def build_video_bm25(db_path: str, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or (default_v2_index_dir(db_path) / "video_bm25_index.pkl")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT video_id, routing_text FROM video_signatures ORDER BY video_id"
        )
        rows = cur.fetchall()
    texts = [(r["routing_text"] or "") for r in rows]
    doc_ids = [str(r["video_id"]) for r in rows]
    corpus, tokenized = _build_tokenized_corpus(texts)
    payload = {
        "kind": "v2_video",
        "corpus": corpus,
        "tokenized": tokenized,
        "doc_ids": doc_ids,
        "built_at": _utc_now(),
        "doc_count": len(corpus),
    }
    _save_pkl(out_path, payload)
    return out_path


def build_segment_bm25(db_path: str, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or (default_v2_index_dir(db_path) / "segment_bm25_index.pkl")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT segment_id, video_id, summary, clean_text
            FROM segment_index ORDER BY segment_id
            """
        )
        rows = cur.fetchall()
    texts: List[str] = []
    doc_ids: List[str] = []
    meta_vid: List[str] = []
    for r in rows:
        s = ((r["summary"] or "") + " " + (r["clean_text"] or "")).strip()
        texts.append(s)
        doc_ids.append(str(int(r["segment_id"])))
        meta_vid.append(str(r["video_id"]))
    corpus, tokenized = _build_tokenized_corpus(texts)
    payload = {
        "kind": "v2_segment",
        "corpus": corpus,
        "tokenized": tokenized,
        "doc_ids": doc_ids,
        "video_ids": meta_vid,
        "built_at": _utc_now(),
        "doc_count": len(corpus),
    }
    _save_pkl(out_path, payload)
    return out_path


def build_event_bm25(db_path: str, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or (default_v2_index_dir(db_path) / "event_bm25_index.pkl")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT event_id, video_id, title, cause, effect, systems
            FROM event_index ORDER BY event_id
            """
        )
        rows = cur.fetchall()
    texts: List[str] = []
    doc_ids: List[str] = []
    meta_vid: List[str] = []
    for r in rows:
        parts = [r["title"], r["cause"], r["effect"], r["systems"]]
        s = " ".join((p or "") for p in parts if p).strip()
        texts.append(s)
        doc_ids.append(str(int(r["event_id"])))
        meta_vid.append(str(r["video_id"]))
    corpus, tokenized = _build_tokenized_corpus(texts)
    payload = {
        "kind": "v2_event",
        "corpus": corpus,
        "tokenized": tokenized,
        "doc_ids": doc_ids,
        "video_ids": meta_vid,
        "built_at": _utc_now(),
        "doc_count": len(corpus),
    }
    _save_pkl(out_path, payload)
    return out_path


def build_all_v2_bm25(db_path: str) -> Dict[str, str]:
    return {
        "video": str(build_video_bm25(db_path)),
        "segment": str(build_segment_bm25(db_path)),
        "event": str(build_event_bm25(db_path)),
    }


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Build V2 BM25 pickle indexes")
    ap.add_argument("--db-path", default=None)
    args = ap.parse_args()
    db = args.db_path or os.environ.get("ASK_SHORTY_DB_PATH") or "data/transcripts.db"
    paths = build_all_v2_bm25(db)
    for k, p in paths.items():
        sz = Path(p).stat().st_size
        print(f"{k}: {p} ({sz:,} bytes)")
