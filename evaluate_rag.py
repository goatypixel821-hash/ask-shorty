#!/usr/bin/env python3
"""
Rigorous evaluation framework for Ask Shorty RAG system.

Three pipeline modes are compared:

  baseline        - current pipeline: chunk+shorty+synq, video-level dedup, no reranking
  rerank_isolated - retrieve same hits, rerank each hit independently with CrossEncoder
  rerank_grouped  - group hits by video+neighbourhood, expand neighbours, rerank groups
                    (full second-stage pipeline from reranker.py)

Metrics per mode:
  Recall@k  - is a relevant video in the top-k results?
  MRR       - mean reciprocal rank of first relevant video
  NDCG@10   - normalised discounted cumulative gain at depth 10

Outputs:
  eval_results/<timestamp>/eval_results.json   - full per-query data
  eval_results/<timestamp>/eval_summary.csv    - aggregate metrics
  eval_results/<timestamp>/queries/            - one JSON artifact per query
                                                 showing top hits before/after reranking
  eval_results/eval_summary.csv                - latest run summary (overwritten each run)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIGS    = [
    "chunk_only",
    "chunk_shorty",
    "chunk_synq",
    "full_system",
    "chunk_bm25",
    "full_bm25",
]
CATEGORIES = ["specific_fact", "thematic", "cross_video", "causal_chain"]
EVAL_MODES = ["baseline", "rerank_isolated", "rerank_grouped", "rerank_grouped_expanded"]
# rerank_grouped          = group + rerank, no neighbour expansion (just matched chunk)
# rerank_grouped_expanded = group + rerank + expand prev/next chunks for context

TOP_K_RETRIEVAL   = 20   # retrieve this many hits per layer before any reranking
TOP_K_FOR_METRICS = 10   # evaluate metrics at this depth

ANSWER_SYSTEM = (
    "You are an assistant that answers questions using ONLY the provided context. "
    "Be concise and cite the source."
)


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def load_queries(path: str) -> List[Dict[str, Any]]:
    """
    Load eval queries from a file.  Supports:
    - .jsonl  (one JSON object per line — format used by the local dataset builder)
    - .json   (legacy format: array or {queries:[...]} )
    """
    if not os.path.isfile(path):
        return []

    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("queries", [])

    # Normalise field names from the newer local-dataset format
    for r in records:
        # query_id → id
        if "id" not in r and "query_id" in r:
            r["id"] = r["query_id"]
        # expected_video_ids → relevant_video_ids
        if "relevant_video_ids" not in r and "expected_video_ids" in r:
            r["relevant_video_ids"] = r["expected_video_ids"]
        # query_type → category  (using the mapping from build_eval_dataset.py)
        if "category" not in r and "query_type" in r:
            r["category"] = {
                "fact_lookup":          "specific_fact",
                "entity_topic":         "thematic",
                "summary_comparison":   "cross_video",
                "paraphrase":           "specific_fact",
                "tricky":               "specific_fact",
            }.get(r["query_type"], r["query_type"])

    return records


# ---------------------------------------------------------------------------
# RAG / Chroma helpers
# ---------------------------------------------------------------------------

def get_rag(db_path: str, chroma_path: Optional[str] = None):
    from transcript_rag import TranscriptRAG
    kwargs: Dict[str, Any] = {"transcript_db": db_path}
    if chroma_path:
        kwargs["chroma_dir"] = chroma_path
    return TranscriptRAG(**kwargs)


def detect_chroma_format(rag) -> bool:
    """
    Return True if the Chroma collection has 'type' metadata on its documents.
    Old indices (pre-shorty project) lack the 'type' field; new ones have it.
    """
    try:
        sample = rag.collection.get(limit=20, include=["metadatas"])
        metas = sample.get("metadatas") or []
        if not metas:
            return False
        has_type = sum(1 for m in metas if (m or {}).get("type"))
        return has_type > 0
    except Exception:
        return False


def _query_chroma(
    collection,
    query: str,
    source_type: str,
    n: int,
    chroma_has_type: bool = True,
) -> List[Tuple[str, float, Dict, str]]:
    """
    Returns list of (source_id, score, metadata, text) for one type filter.

    When chroma_has_type=False (old index format without 'type' metadata) the
    filter is omitted and all document types are returned together for 'chunk'
    queries; non-chunk types return empty so we don't double-count.
    """
    if not chroma_has_type:
        # Old Chroma: no type field — only query once (for 'chunk') and treat
        # all docs as chunks.  Skip shorty / synthetic_question passes.
        if source_type != "chunk":
            return []
        where_filter = None
    else:
        where_filter = {"type": source_type}

    try:
        res = collection.query(
            query_texts=[query],
            n_results=n,
            where=where_filter,
        )
    except Exception:
        return []
    ids    = (res.get("ids")       or [[]])[0]
    metas  = (res.get("metadatas") or [[]])[0]
    scores = (res.get("distances") or [[]])[0]
    docs   = (res.get("documents") or [[]])[0]
    out = []
    for i in range(len(ids)):
        m = metas[i] if i < len(metas) else {}
        # For old-format docs that lack 'type', inject it
        if m and "type" not in m:
            m = dict(m)
            m["type"] = "chunk"
        out.append((
            ids[i]   if i < len(ids)    else "",
            float(scores[i]) if i < len(scores) else 1.0,
            m,
            docs[i]  if i < len(docs)   else "",
        ))
    return out


def _types_for_config(config: str) -> List[str]:
    if config in ("chunk_bm25", "chunk_only"):
        return ["chunk"]
    if config == "chunk_shorty":
        return ["chunk", "shorty"]
    if config == "chunk_synq":
        return ["chunk", "synthetic_question"]
    if config in ("full_bm25", "full_system"):
        return ["chunk", "shorty", "synthetic_question"]
    return ["chunk", "shorty", "synthetic_question"]


def _config_uses_bm25(config: str) -> bool:
    return config in ("chunk_bm25", "full_bm25")


def _strip_bm25_config(config: str) -> str:
    """Map hybrid eval configs to underlying Chroma-only config for rerank paths."""
    if config == "chunk_bm25":
        return "chunk_only"
    if config == "full_bm25":
        return "full_system"
    return config


def _load_title_map(db_path: str) -> Dict[str, str]:
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.cursor()
        cur.execute("SELECT video_id, title FROM videos")
        result = {r[0]: (r[1] or r[0]) for r in cur.fetchall()}
        conn.close()
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> bool:
    """True if any relevant video_id appears in the top-k retrieved list."""
    return any(r in set(retrieved[:k]) for r in relevant)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """Mean Reciprocal Rank — returns 1/rank of first relevant hit, or 0."""
    rel_set = set(relevant)
    for rank, vid in enumerate(retrieved, start=1):
        if vid in rel_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Binary-relevance NDCG@k.
    IDCG assumes all relevant docs are at the top.
    """
    rel_set = set(relevant)
    dcg  = sum(
        1.0 / math.log2(rank + 1)
        for rank, vid in enumerate(retrieved[:k], start=1)
        if vid in rel_set
    )
    # Ideal: min(len(relevant), k) relevant docs at positions 1..k
    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Retrieval pipelines
# ---------------------------------------------------------------------------

def _flatten_to_video_list(
    hits: List[Tuple[str, float, Dict, str]]
) -> List[Tuple[str, float]]:
    """Deduplicate by video_id, keep best score.  Returns sorted list."""
    by_video: Dict[str, float] = {}
    for _, score, meta, _ in hits:
        vid = (meta or {}).get("video_id")
        if vid and (vid not in by_video or score < by_video[vid]):
            by_video[vid] = score
    return sorted(by_video.items(), key=lambda x: x[1])


def run_baseline_hybrid(
    rag,
    bm25_search,
    query: str,
    config: str,
    n: int = TOP_K_RETRIEVAL,
    chroma_has_type: bool = True,
) -> Tuple[List[str], List[Dict]]:
    """
    Vector layers per config + BM25 keyword search, merged with reciprocal rank fusion.
    """
    from bm25_index import reciprocal_rank_fusion

    all_hits: List[Tuple[str, float, Dict, str]] = []
    for stype in _types_for_config(config):
        all_hits.extend(_query_chroma(rag.collection, query, stype, n, chroma_has_type))

    ranked = _flatten_to_video_list(all_hits)
    vector_rank = [v for v, _ in ranked]

    bm25_hits = bm25_search.search(query, top_k=n) if bm25_search is not None else []
    bm25_rank = [h["video_id"] for h in bm25_hits]

    fused = reciprocal_rank_fusion([vector_rank, bm25_rank], k=60)
    video_ids = [v for v, _ in fused]

    artifact_hits = [
        {
            "source_id": hits[0],
            "score": round(hits[1], 4),
            "video_id": (hits[2] or {}).get("video_id"),
            "source_type": (hits[2] or {}).get("type"),
            "text_snippet": (hits[3] or "")[:120],
        }
        for hits in sorted(all_hits, key=lambda h: h[1])[:15]
    ]
    return video_ids, artifact_hits


def run_baseline(
    rag,
    query: str,
    config: str,
    n: int = TOP_K_RETRIEVAL,
    chroma_has_type: bool = True,
) -> Tuple[List[str], List[Dict]]:
    """
    Current pipeline: retrieve from each layer, deduplicate by video_id,
    rank by best cosine distance.

    Returns (video_ids_ranked, top_hits_for_artifact).
    """
    all_hits: List[Tuple[str, float, Dict, str]] = []
    for stype in _types_for_config(config):
        all_hits.extend(_query_chroma(rag.collection, query, stype, n, chroma_has_type))

    ranked = _flatten_to_video_list(all_hits)
    video_ids = [v for v, _ in ranked]

    artifact_hits = [
        {
            "source_id": hits[0],
            "score": round(hits[1], 4),
            "video_id": (hits[2] or {}).get("video_id"),
            "source_type": (hits[2] or {}).get("type"),
            "text_snippet": (hits[3] or "")[:120],
        }
        for hits in sorted(all_hits, key=lambda h: h[1])[:15]
    ]
    return video_ids, artifact_hits


def run_rerank_isolated(
    rag,
    reranker,
    query: str,
    config: str,
    title_map: Dict[str, str],
    n: int = TOP_K_RETRIEVAL,
    chroma_has_type: bool = True,
) -> Tuple[List[str], List[Dict], List[Dict]]:
    """
    Rerank each individual hit independently with the CrossEncoder.
    Returns (video_ids_ranked, before_hits, after_hits).
    """
    from sentence_transformers import CrossEncoder

    all_hits: List[Tuple[str, float, Dict, str]] = []
    for stype in _types_for_config(config):
        all_hits.extend(_query_chroma(rag.collection, query, stype, n, chroma_has_type))

    if not all_hits:
        return [], [], []

    before_hits = [
        {
            "source_id": h[0],
            "score": round(h[1], 4),
            "video_id": (h[2] or {}).get("video_id"),
            "source_type": (h[2] or {}).get("type"),
            "text_snippet": (h[3] or "")[:120],
        }
        for h in sorted(all_hits, key=lambda h: h[1])[:15]
    ]

    ce = reranker._get_cross_encoder() if hasattr(reranker, "_get_cross_encoder") else None
    if ce is None:
        # fall back — just return same order
        return [v for v, _ in _flatten_to_video_list(all_hits)], before_hits, before_hits

    pairs = [(query, (h[3] or "")) for h in all_hits]
    raw_scores = ce.predict(pairs)

    scored = sorted(
        zip(all_hits, raw_scores),
        key=lambda x: x[1],
        reverse=True,  # higher = better for CrossEncoder
    )

    # Deduplicate by video_id, keep first (best rerank score)
    seen: Dict[str, bool] = {}
    video_ids: List[str] = []
    after_hits: List[Dict] = []
    for (src_id, ret_score, meta, text), ce_score in scored:
        vid = (meta or {}).get("video_id")
        if vid and vid not in seen:
            seen[vid] = True
            video_ids.append(vid)
        after_hits.append({
            "source_id": src_id,
            "retrieval_score": round(ret_score, 4),
            "rerank_score": round(float(ce_score), 4),
            "video_id": vid,
            "source_type": (meta or {}).get("type"),
            "text_snippet": (text or "")[:120],
        })
    return video_ids, before_hits, after_hits[:15]


def run_rerank_grouped(
    rag,
    reranker,
    query: str,
    config: str,
    title_map: Dict[str, str],
    n: int = TOP_K_RETRIEVAL,
    chroma_has_type: bool = True,
    expand_neighbors: bool = False,
) -> Tuple[List[str], List[Dict], List[Dict]]:
    """
    Full grouped reranking pipeline from reranker.py.

    expand_neighbors=False  rerank_grouped          — just the matched chunk text
    expand_neighbors=True   rerank_grouped_expanded — prev+chunk+next window

    Returns (video_ids_ranked, before_hits, after_groups).
    """
    all_hits_raw: List[Tuple[str, float, Dict, str]] = []
    for stype in _types_for_config(config):
        all_hits_raw.extend(_query_chroma(rag.collection, query, stype, n, chroma_has_type))

    if not all_hits_raw:
        return [], [], []

    before_hits = [
        {
            "source_id": h[0],
            "score": round(h[1], 4),
            "video_id": (h[2] or {}).get("video_id"),
            "source_type": (h[2] or {}).get("type", "chunk"),
            "text_snippet": (h[3] or "")[:120],
        }
        for h in sorted(all_hits_raw, key=lambda h: h[1])[:15]
    ]

    # Convert to RetrievalHit objects
    from reranker import RetrievalHit
    hits = []
    for src_id, score, meta, text in all_hits_raw:
        meta = meta or {}
        vid  = meta.get("video_id", "unknown")
        cidx = meta.get("chunk_index")
        if cidx is not None:
            try:
                cidx = int(cidx)
            except (ValueError, TypeError):
                cidx = None
        hits.append(RetrievalHit(
            source_id=src_id,
            video_id=vid,
            video_title=title_map.get(vid, vid),
            source_type=meta.get("type", "chunk"),
            chunk_index=cidx,
            retrieval_score=score,
            query_variant=query,
            text=text or "",
        ))

    groups = reranker.group_hits(
        hits,
        collection=rag.collection if expand_neighbors else None,
        expand_neighbors=expand_neighbors,
    )
    ranked  = reranker.rerank_and_blend(query, groups, verbose=False)

    # Deduplicate video_ids preserving rerank order
    seen: Dict[str, bool] = {}
    video_ids: List[str] = []
    for g in ranked:
        if g.video_id not in seen:
            seen[g.video_id] = True
            video_ids.append(g.video_id)

    after_groups = reranker.groups_to_debug_dict(ranked[:15])
    return video_ids, before_hits, after_groups


# ---------------------------------------------------------------------------
# Answer generation (specific_fact queries only)
# ---------------------------------------------------------------------------

def answer_with_context(context_blocks: List[str], query: str) -> str:
    from anthropic_client import get_client
    merged = "\n---\n".join(context_blocks)
    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Context:\n{merged}\n\n"
        "Answer using ONLY the context above."
    )
    client = get_client()
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.3,
        system=ANSWER_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


def run_answer_baseline(rag, query: str, config: str, top_k_docs: int = 10) -> str:
    blocks: List[str] = []
    for stype in _types_for_config(config):
        res = rag.collection.query(
            query_texts=[query], n_results=top_k_docs, where={"type": stype}
        )
        docs  = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        for i, doc in enumerate(docs):
            meta = metas[i] or {}
            blocks.append(f"[{stype}] video_id={meta.get('video_id','?')}\n{doc}")
    return answer_with_context(blocks, query) if blocks else ""


# ---------------------------------------------------------------------------
# Per-query artifact writer
# ---------------------------------------------------------------------------

def _classify_failure(
    video_ids: List[str],
    relevant_ids: List[str],
    before_hits: List[Dict],
    after_hits: List[Dict],
    mode: str,
) -> str:
    """
    Return a short failure reason string for inspection.
    Returns empty string if the query succeeded (relevant in top-10).
    """
    rel_set = set(relevant_ids)
    if not rel_set:
        return "no_relevant_ids_defined"

    in_before = any((h.get("video_id") or "") in rel_set for h in before_hits)
    in_after  = any(
        (h.get("video_id") or h.get("group_key", "").split("|")[0]) in rel_set
        for h in (after_hits or before_hits)
    )
    in_top10 = any(v in rel_set for v in video_ids[:10])

    if in_top10:
        return ""  # success

    if not in_before:
        return "not_retrieved_at_all"   # Chroma never returned the expected video
    if in_before and not in_after:
        return "reranked_out"           # was retrieved but reranking pushed it out
    return "outside_top10"             # retrieved but ranked too low


def save_query_artifact(
    out_dir: str,
    query_id: str,
    query_text: str,
    relevant_ids: List[str],
    expected_chunk_ids: List[str],
    mode_results: Dict[str, Any],
    query_obj: Dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    safe_id = query_id.replace("/", "_").replace("\\", "_")
    path = os.path.join(out_dir, f"{safe_id}.json")

    # Per-mode failure classification
    failure_summary: Dict[str, str] = {}
    for key, res in mode_results.items():
        vids        = res.get("top_video_ids", [])
        before_hits = res.get("before_hits", [])
        after_hits  = res.get("after_hits", [])
        mode_name   = key.split("__")[-1] if "__" in key else key
        reason = _classify_failure(vids, relevant_ids, before_hits, after_hits, mode_name)
        if reason:
            failure_summary[key] = reason

    # Check which expected chunk IDs actually appeared in retrieval results
    found_chunk_ids: List[str] = []
    if expected_chunk_ids:
        for res in mode_results.values():
            for h in res.get("before_hits", []):
                if h.get("source_id") in expected_chunk_ids:
                    found_chunk_ids.append(h["source_id"])
    found_chunk_ids = list(set(found_chunk_ids))

    artifact = {
        "query_id": query_id,
        "query": query_text,
        "query_type": query_obj.get("query_type", "?"),
        "source_video_title": query_obj.get("source_video_title", ""),
        "relevant_video_ids": relevant_ids,
        "expected_chunk_ids": expected_chunk_ids,
        "expected_chunk_texts": query_obj.get("expected_chunk_texts", []),
        "neighbor_chunk_ids": query_obj.get("neighbor_chunk_ids", []),
        "gold_answer_note": query_obj.get("gold_answer_note", ""),
        "support_count": query_obj.get("support_count"),
        "support_types": query_obj.get("support_types"),
        "retrieval_feasible": query_obj.get("retrieval_feasible"),
        "expected_rank_global": query_obj.get("expected_rank_global"),
        # --- what actually happened ---
        "found_chunk_ids": found_chunk_ids,
        "expected_chunks_found": len(found_chunk_ids),
        "expected_chunks_total": len(expected_chunk_ids),
        # --- failure diagnosis ---
        "failure_summary": failure_summary,
        "overall_success": not bool(failure_summary),
        # --- full per-mode results ---
        "modes": mode_results,
        # --- inspection guide ---
        "_how_to_inspect": {
            "not_retrieved_at_all": (
                "Expected video was never returned by Chroma. "
                "Check: (1) is this video indexed? "
                "(2) is the query too different from the transcript vocabulary? "
                "(3) try --mode baseline to see raw cosine scores."
            ),
            "reranked_out": (
                "Video was retrieved but CrossEncoder pushed it out of top-10. "
                "Check: rerank score of expected group in after_hits. "
                "The compact_text sent to the reranker may be the wrong chunk."
            ),
            "outside_top10": (
                "Video was retrieved but ranked too low. "
                "Check: best_retrieval_score of expected video in before_hits. "
                "Consider adding a Shorty or synthetic questions for this video."
            ),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Ask Shorty RAG retrieval and answers."
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help=(
            "Path to transcripts.db.  "
            "If omitted, auto-detects the largest known local corpus DB."
        ),
    )
    parser.add_argument(
        "--chroma-path",
        "--chroma-dir",
        dest="chroma_path",
        default=None,
        help="Path to Chroma vector store directory.  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--queries-file",
        default=str(Path(__file__).parent / "eval_data" / "final" / "golden.jsonl"),
        help=(
            "Path to eval queries file (.jsonl or .json).  "
            "Default: eval_data/final/golden.jsonl"
        ),
    )
    parser.add_argument(
        "--config",
        default="all",
        choices=["all"] + CONFIGS,
        help="Which retrieval config to test",
    )
    parser.add_argument(
        "--category",
        default="all",
        choices=["all"] + CATEGORIES,
        help="Which query category to test",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["all"] + EVAL_MODES,
        help=(
            "Pipeline mode: "
            "baseline=current pipeline, "
            "rerank_isolated=hit-level reranking, "
            "rerank_grouped=grouped reranking with support tracking"
        ),
    )
    parser.add_argument(
        "--output-dir", default="eval_results", help="Where to save results"
    )
    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Skip answer generation (faster; only measures retrieval metrics)",
    )
    args = parser.parse_args()

    queries = load_queries(args.queries_file)
    if not queries:
        print(f"No queries found in {args.queries_file}")
        print(
            "\nTo build the eval dataset, run:\n"
            "  python build_eval_dataset.py\n"
            "  python review_eval_dataset.py\n"
            "  python review_eval_dataset.py --finalize\n"
            "\nOr pass a custom file with --queries-file path/to/queries.jsonl"
        )
        return

    if args.category != "all":
        queries = [q for q in queries if q.get("category") == args.category]
    if not queries:
        print("No queries left after category filter.")
        return

    configs = CONFIGS if args.config == "all" else [args.config]
    modes   = EVAL_MODES if args.mode == "all" else [args.mode]

    # Create timestamped output directory for this run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir      = os.path.join(args.output_dir, ts)
    artifact_dir = os.path.join(run_dir, "queries")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    # Resolve DB and Chroma paths (use build_eval_dataset's auto-detect logic)
    from build_eval_dataset import find_best_db, find_companion_chroma
    db_path     = find_best_db(args.db_path)
    chroma_path = find_companion_chroma(db_path, args.chroma_path)

    print(f"Using DB       : {db_path}  ({round(db_path.stat().st_size / 1e6, 1)} MB)")
    if chroma_path:
        print(f"Using Chroma   : {chroma_path}")
    else:
        print("Chroma         : using project default (from TranscriptRAG)")

    print(f"Loading RAG …")
    rag = get_rag(str(db_path), str(chroma_path) if chroma_path else None)
    title_map = _load_title_map(str(db_path))
    print(f"  {len(title_map)} videos in title map.")

    # Detect whether this Chroma has the 'type' metadata field
    chroma_has_type = detect_chroma_format(rag)
    if not chroma_has_type:
        print(
            "  [WARN] Chroma index lacks 'type' metadata (old format).\n"
            "         Falling back: all vectors treated as 'chunk'.\n"
            "         Shorty and synthetic-question layers will be skipped.\n"
            "         Run reindex_all.py to rebuild with type metadata for full multi-layer eval."
        )

    # Lazy-load reranker only if needed
    reranker = None
    needs_reranker = any(
        m in modes for m in ["rerank_isolated", "rerank_grouped", "rerank_grouped_expanded"]
    )
    if needs_reranker:
        from reranker import Reranker
        reranker = Reranker()
        print("  CrossEncoder reranker loaded.")

    bm25_search = None
    if any(_config_uses_bm25(c) for c in configs):
        from bm25_index import BM25Search, default_bm25_index_path

        bm25_search = BM25Search(str(default_bm25_index_path(str(db_path))))
        print(f"  BM25 index: {default_bm25_index_path(str(db_path))}")

    # Accumulators: mode -> config -> category -> list of metric dicts
    AccType = Dict[str, Dict[str, Dict[str, List[Dict[str, float]]]]]
    acc: AccType = {
        m: {c: {cat: [] for cat in CATEGORIES} for c in configs} for m in modes
    }

    all_query_results: List[Dict[str, Any]] = []

    for q_obj in queries:
        qid          = q_obj.get("id", "?")
        category     = q_obj.get("category", "?")
        query_text   = q_obj.get("query", "")
        relevant_ids = q_obj.get("relevant_video_ids") or []
        ground_truth = (q_obj.get("ground_truth") or "").strip()
        is_sf        = category == "specific_fact"

        print(f"\n--- [{qid}] {category} ---")
        print(f"    Q: {query_text[:80].encode('ascii', 'replace').decode()}{'...' if len(query_text) > 80 else ''}")
        if relevant_ids:
            print(f"    Relevant: {relevant_ids}")

        query_row: Dict[str, Any] = {
            "id": qid,
            "category": category,
            "query": query_text,
            "relevant_video_ids": relevant_ids,
            "ground_truth": ground_truth,
            "configs": {},
        }
        artifact_mode_results: Dict[str, Any] = {}

        for config in configs:
            query_row["configs"][config] = {}

            for mode in modes:
                before_hits: List[Dict] = []
                after_hits:  List[Dict] = []
                video_ids:   List[str]  = []

                eff_config = _strip_bm25_config(config) if mode != "baseline" else config

                if mode == "baseline":
                    if _config_uses_bm25(config):
                        video_ids, before_hits = run_baseline_hybrid(
                            rag,
                            bm25_search,
                            query_text,
                            config,
                            n=TOP_K_RETRIEVAL,
                            chroma_has_type=chroma_has_type,
                        )
                    else:
                        video_ids, before_hits = run_baseline(
                            rag, query_text, config, n=TOP_K_RETRIEVAL,
                            chroma_has_type=chroma_has_type,
                        )
                    after_hits = before_hits  # no reranking

                elif mode == "rerank_isolated":
                    video_ids, before_hits, after_hits = run_rerank_isolated(
                        rag, reranker, query_text, eff_config, title_map,
                        n=TOP_K_RETRIEVAL, chroma_has_type=chroma_has_type,
                    )

                elif mode == "rerank_grouped":
                    video_ids, before_hits, after_hits = run_rerank_grouped(
                        rag, reranker, query_text, eff_config, title_map,
                        n=TOP_K_RETRIEVAL, chroma_has_type=chroma_has_type,
                        expand_neighbors=False,
                    )

                elif mode == "rerank_grouped_expanded":
                    video_ids, before_hits, after_hits = run_rerank_grouped(
                        rag, reranker, query_text, eff_config, title_map,
                        n=TOP_K_RETRIEVAL, chroma_has_type=chroma_has_type,
                        expand_neighbors=True,
                    )

                r5   = recall_at_k(video_ids, relevant_ids, 5)
                r10  = recall_at_k(video_ids, relevant_ids, 10)
                mrr_score  = mrr(video_ids, relevant_ids)
                ndcg_score = ndcg_at_k(video_ids, relevant_ids, TOP_K_FOR_METRICS)

                answer_correct = False
                if is_sf and ground_truth and not args.no_answer:
                    try:
                        ans = run_answer_baseline(
                            rag, query_text, _strip_bm25_config(config)
                        )
                        answer_correct = ground_truth.lower() in (ans or "").lower()
                    except Exception as exc:
                        print(f"    [{config}/{mode}] answer error: {exc}")

                metrics = {
                    "recall_at_5":       r5,
                    "recall_at_10":      r10,
                    "mrr":               round(mrr_score, 4),
                    "ndcg_at_10":        round(ndcg_score, 4),
                    "answer_correctness": answer_correct,
                    "top_video_ids":     video_ids[:10],
                }
                acc[mode][config][category].append({
                    "r5": float(r5),
                    "r10": float(r10),
                    "mrr": mrr_score,
                    "ndcg": ndcg_score,
                    "ac": float(answer_correct),
                })

                label = f"{config}/{mode}"
                print(
                    f"    {label}: R@5={int(r5)} R@10={int(r10)} "
                    f"MRR={mrr_score:.3f} NDCG={ndcg_score:.3f}"
                    + (f" ans_ok={answer_correct}" if is_sf and not args.no_answer else "")
                )

                key = f"{config}__{mode}"
                query_row["configs"][config][mode] = metrics
                artifact_mode_results[key] = {
                    "top_video_ids": video_ids[:10],
                    "before_hits": before_hits,
                    "after_hits": after_hits,
                    "metrics": metrics,
                }

        all_query_results.append(query_row)
        save_query_artifact(
            artifact_dir,
            qid,
            query_text,
            relevant_ids,
            q_obj.get("expected_chunk_ids") or [],
            artifact_mode_results,
            q_obj,
        )

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY  (recall@5, recall@10, MRR, NDCG@10, answer_correctness)")
    print("=" * 70)

    for mode in modes:
        print(f"\n  === Mode: {mode} ===")
        for config in configs:
            print(f"  [{config}]")
            for cat in CATEGORIES:
                L = acc[mode][config][cat]
                if not L:
                    continue
                n   = len(L)
                r5  = sum(x["r5"]   for x in L) / n
                r10 = sum(x["r10"]  for x in L) / n
                mv  = sum(x["mrr"]  for x in L) / n
                ng  = sum(x["ndcg"] for x in L) / n
                if cat == "specific_fact":
                    ac = sum(x["ac"] for x in L) / n
                    print(
                        f"    {cat}: R@5={r5:.2f} R@10={r10:.2f} "
                        f"MRR={mv:.3f} NDCG={ng:.3f} ans_correct={ac:.2f} (n={n})"
                    )
                else:
                    print(
                        f"    {cat}: R@5={r5:.2f} R@10={r10:.2f} "
                        f"MRR={mv:.3f} NDCG={ng:.3f} (n={n})"
                    )

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    out_json = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "db_path": str(db_path),
        "queries_file": args.queries_file,
        "configs": configs,
        "modes": modes,
        "results_per_query": all_query_results,
    }
    json_path = os.path.join(run_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nSaved full results  -> {json_path}")

    # -----------------------------------------------------------------------
    # Save summary CSV (in run dir AND overwrite latest)
    # -----------------------------------------------------------------------
    csv_rows: List[List] = []
    header = ["mode", "config", "category", "recall_at_5", "recall_at_10",
              "mrr", "ndcg_at_10", "answer_correctness", "n_queries"]
    csv_rows.append(header)

    for mode in modes:
        for config in configs:
            for cat in CATEGORIES:
                L = acc[mode][config][cat]
                if not L:
                    continue
                n   = len(L)
                r5  = sum(x["r5"]   for x in L) / n
                r10 = sum(x["r10"]  for x in L) / n
                mv  = sum(x["mrr"]  for x in L) / n
                ng  = sum(x["ndcg"] for x in L) / n
                ac  = sum(x["ac"]   for x in L) / n if cat == "specific_fact" else 0.0
                csv_rows.append([
                    mode, config, cat,
                    f"{r5:.4f}", f"{r10:.4f}",
                    f"{mv:.4f}", f"{ng:.4f}",
                    f"{ac:.4f}", n,
                ])

    run_csv_path    = os.path.join(run_dir, "eval_summary.csv")
    latest_csv_path = os.path.join(args.output_dir, "eval_summary.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    for csv_path in (run_csv_path, latest_csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(csv_rows)

    print(f"Saved summary CSV   -> {run_csv_path}")
    print(f"Updated latest CSV  -> {latest_csv_path}")
    print(f"\nPer-query artifacts -> {artifact_dir}/")


if __name__ == "__main__":
    main()
