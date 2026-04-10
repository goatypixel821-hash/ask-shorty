#!/usr/bin/env python3
"""
Enrich eval candidates with corpus-aware evidence labels.

For each candidate in candidates.jsonl, this script:

  1. expected_chunk_ids   — queries Chroma restricted to the source video,
                            records the top-k chunk IDs that most match the
                            query.  Falls back gracefully if Chroma is missing.

  2. expected_chunk_texts — short snippet from each expected chunk.

  3. support_count        — how many chunks from the expected video scored
                            below the "relevant" cosine-distance threshold.

  4. support_types        — which retrieval layers carried evidence
                            (chunk / shorty / synthetic_question — what Chroma
                            has for that video).

  5. neighbor_chunk_ids   — immediate neighbours of the best expected chunk,
                            so labellers can see the surrounding context window.

  6. retrieval_feasible   — boolean: was the expected video retrievable at all
                            with an unfiltered query?  Useful for flagging
                            un-answerable queries before human review.

Supports both Chroma index formats:
  old project  {video_id}_chunk_{N}   (no 'type' metadata)
  new project  {video_id}:chunk:{N}   (has 'type' metadata)

Usage:
  python enrich_candidates.py
  python enrich_candidates.py --chroma-path PATH --top-k 5
  python enrich_candidates.py --input eval_data/candidates/candidates.jsonl
                               --output eval_data/candidates/candidates_enriched.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths (all git-ignored)
# ---------------------------------------------------------------------------
EVAL_DATA = Path(__file__).parent / "eval_data"
DEFAULT_INPUT  = EVAL_DATA / "candidates" / "candidates.jsonl"
DEFAULT_OUTPUT = EVAL_DATA / "candidates" / "candidates_enriched.jsonl"

# Cosine distance below which a chunk counts as "supporting evidence"
SUPPORT_THRESHOLD = 0.60

# Number of top chunks to record per query
TOP_K_CHUNKS = 3


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Chroma helpers
# ---------------------------------------------------------------------------

def _detect_id_format(collection) -> str:
    """
    Return 'old' if chunk IDs use {video_id}_chunk_{N},
           'new' if they use {video_id}:chunk:{N}.
    """
    sample = collection.get(limit=5, include=[])
    ids = sample.get("ids", [])
    for id_ in ids:
        if re.search(r"_chunk_\d+$", id_):
            return "old"
        if re.search(r":chunk:\d+$", id_):
            return "new"
    return "old"  # default


def _chunk_id(video_id: str, n: int, fmt: str) -> str:
    if fmt == "old":
        return f"{video_id}_chunk_{n}"
    return f"{video_id}:chunk:{n}"


def _extract_chunk_index_from_id(id_: str) -> Optional[int]:
    m = re.search(r"[_:]chunk[_:](\d+)$", id_)
    return int(m.group(1)) if m else None


def _query_video_chunks(
    collection,
    query_text: str,
    video_id: str,
    n: int = TOP_K_CHUNKS,
) -> List[Tuple[str, float, str]]:
    """
    Query Chroma restricted to a specific video_id.
    Returns list of (chunk_id, score, text_snippet).
    """
    try:
        res = collection.query(
            query_texts=[query_text],
            n_results=n,
            where={"video_id": video_id},
        )
        ids    = (res.get("ids")       or [[]])[0]
        scores = (res.get("distances") or [[]])[0]
        docs   = (res.get("documents") or [[]])[0]
        out = []
        for i in range(len(ids)):
            snippet = (docs[i] or "")[:100]
            out.append((ids[i], float(scores[i]), snippet))
        return out
    except Exception:
        return []


def _query_global(
    collection,
    query_text: str,
    n: int = 10,
) -> List[Tuple[str, str, float]]:
    """
    Unfiltered query.  Returns list of (video_id, chunk_id, score).
    """
    try:
        res = collection.query(
            query_texts=[query_text],
            n_results=n,
        )
        ids    = (res.get("ids")       or [[]])[0]
        metas  = (res.get("metadatas") or [[]])[0]
        scores = (res.get("distances") or [[]])[0]
        out = []
        for i in range(len(ids)):
            m = metas[i] or {}
            vid = m.get("video_id", "unknown")
            out.append((vid, ids[i], float(scores[i])))
        return out
    except Exception:
        return []


def _get_neighbor_ids(
    chunk_id: str,
    video_id: str,
    fmt: str,
    n_before: int = 1,
    n_after: int = 1,
) -> List[str]:
    """Return IDs for neighboring chunks."""
    cidx = _extract_chunk_index_from_id(chunk_id)
    if cidx is None:
        return [chunk_id]
    return [
        _chunk_id(video_id, cidx + offset, fmt)
        for offset in range(-n_before, n_after + 1)
        if cidx + offset >= 0
    ]


# ---------------------------------------------------------------------------
# Per-candidate enrichment
# ---------------------------------------------------------------------------

def _enrich_one(
    q: Dict[str, Any],
    collection,
    id_fmt: str,
) -> Dict[str, Any]:
    """Add evidence labels to a single candidate dict (mutates and returns it)."""
    query_text   = q.get("query", "")
    source_vid   = q.get("source_video_id", "")
    expected_ids = q.get("expected_video_ids") or []
    query_type   = q.get("query_type", "")

    # Skip if no Chroma / no source video
    if not query_text or not source_vid:
        q.setdefault("expected_chunk_ids", [])
        q.setdefault("expected_chunk_texts", [])
        q.setdefault("neighbor_chunk_ids", [])
        q.setdefault("support_count", 0)
        q.setdefault("support_types", [])
        q.setdefault("retrieval_feasible", None)
        return q

    # --- Step 1: find best chunks from the source video ---
    video_hits = _query_video_chunks(collection, query_text, source_vid, n=TOP_K_CHUNKS)

    expected_chunk_ids: List[str] = []
    expected_chunk_texts: List[str] = []
    support_count = 0
    support_types = ["chunk"]  # old Chroma has only chunks

    for cid, score, snippet in video_hits:
        expected_chunk_ids.append(cid)
        expected_chunk_texts.append(snippet)
        if score < SUPPORT_THRESHOLD:
            support_count += 1

    # Expand to neighbors of best chunk
    neighbor_chunk_ids: List[str] = []
    if expected_chunk_ids:
        neighbor_chunk_ids = _get_neighbor_ids(
            expected_chunk_ids[0], source_vid, id_fmt
        )

    # --- Step 2: check global retrieval feasibility ---
    # Is the expected video retrievable without any filter?
    global_hits = _query_global(collection, query_text, n=10)
    top_global_vids = [vid for vid, _, _ in global_hits]
    retrieval_feasible = any(v in top_global_vids for v in (expected_ids or [source_vid]))

    # Rank position of expected video in global results
    expected_rank: Optional[int] = None
    for rank, (vid, _, _) in enumerate(global_hits, 1):
        if vid in (expected_ids or [source_vid]):
            expected_rank = rank
            break

    q["expected_chunk_ids"]   = expected_chunk_ids
    q["expected_chunk_texts"] = expected_chunk_texts
    q["neighbor_chunk_ids"]   = neighbor_chunk_ids
    q["support_count"]        = support_count
    q["support_types"]        = support_types
    q["retrieval_feasible"]   = retrieval_feasible
    q["expected_rank_global"] = expected_rank   # None if not in top-10

    # Enrich gold_answer_note from chunk text if it was empty
    if not q.get("gold_answer_note") and expected_chunk_texts:
        q["gold_answer_note"] = f"[auto] best chunk snippet: {expected_chunk_texts[0][:80]}"

    return q


def _enrich_summary_comparison(
    q: Dict[str, Any],
    collection,
) -> Dict[str, Any]:
    """For summary_comparison, record best chunk per expected video."""
    query_text   = q.get("query", "")
    expected_ids = q.get("expected_video_ids") or []

    all_chunk_ids: List[str] = []
    all_chunk_texts: List[str] = []
    retrieved_vids: List[str] = []

    for vid in expected_ids[:5]:  # cap at 5 to keep it fast
        hits = _query_video_chunks(collection, query_text, vid, n=1)
        if hits:
            cid, score, snippet = hits[0]
            all_chunk_ids.append(cid)
            all_chunk_texts.append(snippet)
            retrieved_vids.append(vid)

    feasible_count = len(retrieved_vids)
    q["expected_chunk_ids"]   = all_chunk_ids
    q["expected_chunk_texts"] = all_chunk_texts
    q["neighbor_chunk_ids"]   = []
    q["support_count"]        = feasible_count
    q["support_types"]        = ["chunk"]
    q["retrieval_feasible"]   = feasible_count > 0
    q["expected_rank_global"] = None
    return q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich eval candidates with expected_chunk_ids and support labels."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input JSONL candidates file",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output enriched JSONL file",
    )
    parser.add_argument(
        "--chroma-path",
        default=None,
        help=(
            "Path to Chroma directory.  "
            "Auto-detected if omitted (prefers full corpus Chroma)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_CHUNKS,
        help="Number of top chunks to store per query",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N candidates (0 = all).  Useful for quick tests.",
    )
    args = parser.parse_args()

    # Resolve Chroma path
    if args.chroma_path:
        chroma_path = Path(args.chroma_path)
    else:
        from build_eval_dataset import find_best_db, find_companion_chroma
        db_path = find_best_db()
        chroma_path = find_companion_chroma(db_path) or Path(__file__).parent / "data" / "transcript_chroma_new"

    print(f"Input    : {args.input}")
    print(f"Output   : {args.output}")
    print(f"Chroma   : {chroma_path}")

    # Load candidates
    candidates = _load_jsonl(Path(args.input))
    if not candidates:
        print(f"No candidates found in {args.input}")
        print("Run: python build_eval_dataset.py")
        return

    print(f"Candidates loaded: {len(candidates)}")

    if args.limit:
        candidates = candidates[: args.limit]
        print(f"Limiting to first {args.limit} candidates")

    # Load Chroma
    collection = None
    id_fmt = "old"
    if chroma_path.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_path))
            collection = client.get_collection("transcripts")
            id_fmt = _detect_id_format(collection)
            print(f"Chroma loaded: {collection.count()} vectors  id_fmt={id_fmt}")
        except Exception as e:
            print(f"[WARN] Could not load Chroma: {e}")
            print("  expected_chunk_ids will be empty for all candidates.")
    else:
        print(f"[WARN] Chroma not found at {chroma_path}")
        print("  expected_chunk_ids will be empty — run reindex_all.py to build it.")

    # Enrich each candidate
    enriched: List[Dict[str, Any]] = []
    t0 = time.time()
    n_feasible = 0
    n_skipped  = 0

    for i, q in enumerate(candidates):
        qt = q.get("query_type", "")

        if collection is None:
            # No Chroma — just pass through with empty labels
            q.setdefault("expected_chunk_ids", [])
            q.setdefault("expected_chunk_texts", [])
            q.setdefault("neighbor_chunk_ids", [])
            q.setdefault("support_count", 0)
            q.setdefault("support_types", [])
            q.setdefault("retrieval_feasible", None)
            q.setdefault("expected_rank_global", None)
            n_skipped += 1
        elif qt == "summary_comparison":
            q = _enrich_summary_comparison(q, collection)
            if q.get("retrieval_feasible"):
                n_feasible += 1
        else:
            q = _enrich_one(q, collection, id_fmt)
            if q.get("retrieval_feasible"):
                n_feasible += 1

        enriched.append(q)

        if (i + 1) % 25 == 0 or (i + 1) == len(candidates):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta  = (len(candidates) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i + 1}/{len(candidates)}]  "
                f"feasible={n_feasible}  elapsed={elapsed:.0f}s  "
                f"ETA={eta:.0f}s"
            )

    _write_jsonl(Path(args.output), enriched)

    # Summary stats
    feasible_pct = 100 * n_feasible / max(len(enriched) - n_skipped, 1)
    with_chunks = sum(1 for q in enriched if q.get("expected_chunk_ids"))
    print(f"\nEnrichment complete:")
    print(f"  Total candidates   : {len(enriched)}")
    print(f"  With chunk labels  : {with_chunks}")
    print(f"  Retrieval feasible : {n_feasible} ({feasible_pct:.0f}%)")
    print(f"  Skipped (no Chroma): {n_skipped}")
    print(f"  Output             : {args.output}")

    infeasible = [q for q in enriched if q.get("retrieval_feasible") is False]
    if infeasible:
        print(f"\n  [WARN] {len(infeasible)} queries where expected video NOT in top-10")
        print("  These are hard negatives — useful for eval but may not pass review.")
        print("  Inspect with: python review_eval_dataset.py --stats")


if __name__ == "__main__":
    main()
