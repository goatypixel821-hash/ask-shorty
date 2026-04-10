#!/usr/bin/env python3
"""
BM25 keyword index for Ask Shorty.
Indexes transcripts, Shorties, and entities for exact-match retrieval.
Complements vector search — never replaces it.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Tokenization (simple word-based, matches keyword search expectations)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", text) if len(t) >= 1]


def _chunk_text(text: str, max_chars: int = 800, overlap: int = 200) -> List[str]:
    """Match transcript_rag_enhanced chunking for BM25 chunks."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == length:
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    RRF score = sum(1 / (k + rank)) across all result lists (rank is 1-based).
    Standard k=60 from literature.
    Returns (video_id, rrf_score) sorted by score descending.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        if not ranked:
            continue
        for rank, vid in enumerate(ranked, start=1):
            if not vid:
                continue
            scores[vid] = scores.get(vid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Build / load index
# ---------------------------------------------------------------------------


def build_bm25_payload(db_path: str) -> Dict[str, Any]:
    """
    Scan SQLite and build serializable index payload (corpus + metadata).
    """
    corpus: List[str] = []
    doc_ids: List[str] = []
    doc_types: List[str] = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute(
            """
            SELECT t.video_id, t.text, t.shorty
            FROM transcripts t
            WHERE t.text IS NOT NULL AND trim(t.text) != ''
            """
        )
        for row in cur.fetchall():
            vid = row["video_id"]
            raw_text = row["text"] or ""
            shorty = (row["shorty"] or "").strip()

            for ch in _chunk_text(raw_text):
                corpus.append(ch)
                doc_ids.append(vid)
                doc_types.append("chunk")

            if shorty:
                corpus.append(shorty)
                doc_ids.append(vid)
                doc_types.append("shorty")

        cur.execute(
            """
            SELECT video_id, name FROM entities
            WHERE name IS NOT NULL AND trim(name) != ''
            """
        )
        for row in cur.fetchall():
            name = (row["name"] or "").strip()
            if not name:
                continue
            corpus.append(name)
            doc_ids.append(row["video_id"])
            doc_types.append("entity")

    tokenized = [_tokenize(c) for c in corpus]
    # rank_bm25 skips empty docs — keep alignment by using single placeholder token
    tokenized = [t if t else ["_"] for t in tokenized]

    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "corpus": corpus,
        "tokenized": tokenized,
        "doc_ids": doc_ids,
        "doc_types": doc_types,
        "built_at": built_at,
        "doc_count": len(corpus),
    }


class BM25Search:
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
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

    def rebuild(self, db_path: str) -> None:
        """Rebuild index from SQLite and persist."""
        payload = build_bm25_payload(db_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "corpus": payload["corpus"],
            "tokenized": payload["tokenized"],
            "doc_ids": payload["doc_ids"],
            "doc_types": payload["doc_types"],
            "built_at": payload["built_at"],
            "doc_count": payload["doc_count"],
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._payload = out
        self._bm25 = BM25Okapi(out["tokenized"]) if out["tokenized"] else None

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Returns [{video_id, score, doc_type, text_preview}, ...] best-first.
        Scores are BM25 relevance (higher is better). One entry per video (best-scoring doc).
        """
        if not self._bm25 or not query or not query.strip():
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        corpus = self._payload.get("corpus") or []
        doc_ids = self._payload.get("doc_ids") or []
        doc_types = self._payload.get("doc_types") or []

        best_by_vid: Dict[str, Tuple[float, int]] = {}
        for idx, s in enumerate(scores):
            if s <= 0:
                continue
            if idx >= len(doc_ids):
                break
            vid = doc_ids[idx]
            if not vid:
                continue
            sf = float(s)
            prev = best_by_vid.get(vid)
            if prev is None or sf > prev[0]:
                best_by_vid[vid] = (sf, idx)

        ranked = sorted(best_by_vid.items(), key=lambda x: -x[1][0])[:top_k]
        out: List[Dict[str, Any]] = []
        for vid, (sc, idx) in ranked:
            text = corpus[idx] if idx < len(corpus) else ""
            preview = (text[:200] + "…") if len(text) > 200 else text
            dt = doc_types[idx] if idx < len(doc_types) else "chunk"
            out.append(
                {
                    "video_id": vid,
                    "score": sc,
                    "doc_type": dt,
                    "text_preview": preview.replace("\n", " "),
                }
            )
        return out


def default_bm25_index_path(db_path: Optional[str] = None) -> Path:
    """Prefer bm25_index.pkl beside the DB; else project data/bm25_index.pkl."""
    env = os.getenv("ASK_SHORTY_BM25_INDEX", "").strip()
    if env:
        return Path(env)
    if db_path:
        p = Path(db_path).resolve()
        return p.parent / "bm25_index.pkl"
    base = Path(__file__).resolve().parent
    return base / "data" / "bm25_index.pkl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query BM25 index for Ask Shorty.")
    parser.add_argument("--db-path", type=str, required=True, help="Path to transcripts.db")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from SQLite")
    parser.add_argument("--query", type=str, default=None, help="Test query (BM25 search)")
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Override output path (default: beside DB or data/bm25_index.pkl)",
    )
    args = parser.parse_args()

    index_path = args.index_path or str(default_bm25_index_path(args.db_path))
    search = BM25Search(index_path)

    if args.rebuild:
        print(f"Rebuilding BM25 index -> {index_path}")
        search.rebuild(args.db_path)
        print(f"Done. doc_count={search.payload.get('doc_count', 0)} built_at={search.payload.get('built_at')}")

    if args.query:
        if not search._bm25:
            print("Index missing or empty; run with --rebuild first.")
            return
        results = search.search(args.query, top_k=20)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
