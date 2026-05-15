#!/usr/bin/env python3
"""
Generate cross-video (sc_*) eval replacements from topic clusters + DB excerpts.

Samples semantic clusters (data/clusters.json), picks 2–3 videos per cluster,
calls Claude via OpenRouter/Anthropic, and writes eval JSON files to
queries/replacements/ (same layout as generate_tr_replacements.py).

Usage:
  python generate_sc_replacements.py eval_results/20260514_065902/queries
  python generate_sc_replacements.py --db-path data/transcripts.db --limit 5
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from generate_tr_replacements import (
    _load_dotenv,
    _make_llm_caller,
    _valid_eval_query,
)

SYSTEM_PROMPT = """You write cross-video evaluation questions for a personal transcript search system.

Given a shared topic and transcript excerpts from 2–3 different videos, produce:
1. A specific question that requires retrieving those particular videos (not a vague channel summary)
2. A short ground-truth answer with concrete facts from the excerpts
3. The video IDs needed to answer it (2–3 ids from the input list)

Rules:
- Name a specific person, product, event, technique, policy, number, or claim from the excerpts
- Do NOT ask what a channel "focuses on across all videos" or similar broad summaries
- Do NOT use phrases like "across all indexed videos", "all their videos", or "summarize the channel"
- Question: at least 12 words, must end with ?
- ground_truth: 1–3 sentences, factual, states what the videos agree on, contrast, or each contribute
- relevant_video_ids: exactly the 2–3 video ids from the input that the question requires

Return ONLY valid JSON:
{"query": "...", "ground_truth": "...", "relevant_video_ids": ["id1", "id2"]}"""

_BAD_QUERY_RE = re.compile(
    r"across (?:all |their )?(?:indexed )?videos|"
    r"what does .+ focus on|"
    r"summar(?:y|ise|ize) (?:the )?(?:main )?points|"
    r"common themes appear|"
    r"compare what .+ discussed across",
    re.I,
)

_WORD_RE = re.compile(r"[a-z0-9]+", re.I)


def _slugify(text: str, max_len: int = 24) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return (s[:max_len] if s else "cluster")


def _resolve_queries_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir() and path.name != "queries" and (path / "queries").is_dir():
        return path / "queries"
    return path


def _load_clusters(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clusters = data.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError(f"No 'clusters' list in {path}")
    return clusters


def _videos_with_shorty(conn: sqlite3.Connection) -> Set[str]:
    rows = conn.execute(
        """
        SELECT v.video_id
        FROM videos v
        JOIN transcripts t ON t.video_id = v.video_id
        WHERE t.shorty IS NOT NULL AND trim(t.shorty) != ''
        """
    ).fetchall()
    return {str(r[0]) for r in rows}


def _select_clusters(
    clusters: Sequence[Dict[str, Any]],
    shorty_ids: Set[str],
    *,
    n: int,
    seed: int,
) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Pick n diverse clusters that have >=2 videos with Shorties in the DB."""
    rng = random.Random(seed)
    candidates: List[Tuple[Dict[str, Any], List[Dict[str, Any]], Set[str]]] = []

    for c in clusters:
        label = (c.get("label") or "").strip()
        if not label or label.lower() in {"noise", "uncategorized", "other"}:
            continue
        vids = [
            v
            for v in (c.get("videos") or [])
            if isinstance(v, dict) and v.get("video_id") in shorty_ids
        ]
        if len(vids) < 2:
            continue
        tokens = set(_WORD_RE.findall(label.lower()))
        candidates.append((c, vids, tokens))

    rng.shuffle(candidates)
    chosen: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    used_tokens: Set[str] = set()

    def _score(item: Tuple[Dict[str, Any], List[Dict[str, Any]], Set[str]]) -> float:
        _, vids, tokens = item
        novelty = len(tokens - used_tokens)
        channels = len({v.get("channel") or "" for v in vids})
        size_penalty = 0 if 3 <= len(vids) <= 12 else -1
        return novelty * 2 + channels + size_penalty

    pool = list(candidates)
    while pool and len(chosen) < n:
        pool.sort(key=_score, reverse=True)
        c, vids, tokens = pool.pop(0)
        chosen.append((c, vids))
        used_tokens |= tokens

    return chosen


def _pick_videos(
    videos: Sequence[Dict[str, Any]],
    *,
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Prefer diverse channels; return 2–3 videos."""
    k = max(2, min(k, 3, len(videos)))
    by_ch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for v in videos:
        by_ch[(v.get("channel") or "unknown").strip() or "unknown"].append(v)

    picked: List[Dict[str, Any]] = []
    channels = sorted(by_ch.keys(), key=lambda ch: -len(by_ch[ch]))
    for ch in channels:
        if len(picked) >= k:
            break
        picked.append(rng.choice(by_ch[ch]))

    remaining = [v for v in videos if v not in picked]
    rng.shuffle(remaining)
    for v in remaining:
        if len(picked) >= k:
            break
        picked.append(v)
    return picked[:k]


def _fetch_video_excerpt(
    conn: sqlite3.Connection,
    video_id: str,
    *,
    label_tokens: Sequence[str],
) -> Tuple[str, str, Optional[str], str]:
    """Return title, channel, best segment chunk_id, excerpt text."""
    row = conn.execute(
        "SELECT title, channel FROM videos WHERE video_id = ?",
        (video_id,),
    ).fetchone()
    title = (row[0] if row else video_id) or video_id
    channel = (row[1] if row else "") or ""

    shorty_row = conn.execute(
        """
        SELECT shorty FROM transcripts
        WHERE video_id = ? AND shorty IS NOT NULL AND trim(shorty) != ''
        ORDER BY id DESC LIMIT 1
        """,
        (video_id,),
    ).fetchone()
    shorty = (shorty_row[0] if shorty_row else "") or ""

    seg_rows = conn.execute(
        """
        SELECT id, summary FROM segments
        WHERE video_id = ? AND summary IS NOT NULL AND trim(summary) != ''
        ORDER BY start_time
        """,
        (video_id,),
    ).fetchall()

    tokens = [t for t in label_tokens if len(t) >= 3]
    best_idx = 0
    best_score = -1
    summaries: List[str] = []
    for i, (seg_id, summ) in enumerate(seg_rows):
        s = (summ or "").strip()
        summaries.append(s)
        score = sum(1 for t in tokens if t in s.lower())
        if score > best_score:
            best_score = score
            best_idx = i

    if summaries:
        summ = summaries[best_idx]
        chunk_id = f"{video_id}_chunk_{best_idx}"
        excerpt = summ[:600]
    else:
        chunk_id = None
        excerpt = shorty[:600]

    if shorty and len(shorty) > 80:
        excerpt = (shorty[:400] + "\n...\n" + excerpt)[:900]

    return title, channel, chunk_id, excerpt.strip()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    if start < 0:
        return None
    blob = text[start:]
    for end in range(len(blob), start, -1):
        if blob[end - 1] != "}":
            continue
        try:
            obj = json.loads(blob[:end])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    # Truncated output: close open string + object
    if '"query"' in blob:
        repaired = blob.rstrip()
        if repaired.count('"') % 2 == 1:
            repaired += '"'
        if not repaired.endswith("}"):
            repaired += '"}'
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def _parse_cross_video_json(
    raw: str,
    allowed_ids: Set[str],
) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    obj = _extract_json_object(text)
    if not obj:
        return None

    query = (obj.get("query") or "").strip()
    ground_truth = (obj.get("ground_truth") or obj.get("gold_answer_note") or "").strip()
    vids = obj.get("relevant_video_ids") or obj.get("expected_video_ids") or []
    if not isinstance(vids, list):
        vids = []
    vids = [str(v).strip() for v in vids if str(v).strip() in allowed_ids]
    if len(vids) < 2:
        vids = sorted(allowed_ids)[: min(3, len(allowed_ids))]

    if not _valid_eval_query(query):
        return None
    if len(query) < 30:
        return None
    if _BAD_QUERY_RE.search(query):
        return None
    if not ground_truth or len(ground_truth) < 12:
        return None
    if len(vids) < 2 or len(vids) > 3:
        return None
    return {
        "query": query,
        "ground_truth": ground_truth,
        "relevant_video_ids": vids,
    }


def _generate_cross_video(
    llm: Callable[..., str],
    *,
    cluster_label: str,
    videos: Sequence[Dict[str, Any]],
    excerpts: Sequence[Tuple[str, str, str, str]],
) -> Dict[str, Any]:
    allowed = {v["video_id"] for v in videos}
    label_tokens = list(_WORD_RE.findall(cluster_label.lower()))

    blocks: List[str] = []
    for v, (title, channel, vid, excerpt) in zip(videos, excerpts):
        blocks.append(
            f"VIDEO_ID: {vid}\n"
            f"TITLE (do not quote in question): {title}\n"
            f"CHANNEL (do not ask broad channel summaries): {channel}\n"
            f"EXCERPT:\n{excerpt[:2500]}"
        )

    user_prompt = (
        f"SHARED TOPIC / CLUSTER LABEL: {cluster_label}\n\n"
        f"VIDEOS ({len(videos)}):\n\n"
        + "\n\n---\n\n".join(blocks)
        + "\n\nWrite one cross-video question requiring 2–3 of these video IDs."
    )

    raw = llm(SYSTEM_PROMPT, user_prompt, max_tokens=512, temperature=0.35)
    parsed = _parse_cross_video_json(raw, allowed)
    if parsed:
        return parsed

    raw2 = llm(
        SYSTEM_PROMPT,
        user_prompt
        + '\n\nReply with ONLY compact JSON (one line): {"query":"...","ground_truth":"...","relevant_video_ids":["id1","id2"]}',
        max_tokens=512,
        temperature=0.15,
    )
    parsed = _parse_cross_video_json(raw2, allowed)
    if parsed:
        return parsed

    raise ValueError(f"Could not parse cross-video JSON: {raw[:220]!r}")


def _build_artifact(
    *,
    query_id: str,
    cluster_id: int,
    cluster_label: str,
    parsed: Dict[str, Any],
    source_title: str,
    chunk_ids: List[str],
    chunk_texts: List[str],
) -> Dict[str, Any]:
    vids = parsed["relevant_video_ids"]
    gt = parsed["ground_truth"]
    return {
        "query_id": query_id,
        "id": query_id,
        "query": parsed["query"],
        "query_type": "summary_comparison",
        "category": "cross_video",
        "source_video_title": source_title,
        "relevant_video_ids": vids,
        "expected_video_ids": vids,
        "expected_chunk_ids": chunk_ids,
        "expected_chunk_texts": chunk_texts,
        "neighbor_chunk_ids": [],
        "gold_answer_note": gt,
        "ground_truth": gt,
        "support_count": len(vids),
        "support_types": ["chunk"],
        "retrieval_feasible": True,
        "expected_rank_global": None,
        "found_chunk_ids": [],
        "expected_chunks_found": 0,
        "expected_chunks_total": len(chunk_ids),
        "failure_summary": {},
        "overall_success": None,
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "notes": (
            f"generated cross-video replacement; cluster_id={cluster_id}; "
            f"videos={len(vids)}"
        ),
    }


def main() -> None:
    _load_dotenv()
    root = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        description="Generate cross-video sc_* eval replacements from topic clusters."
    )
    ap.add_argument(
        "queries_dir",
        type=Path,
        nargs="?",
        default=root / "eval_results" / "20260514_065902" / "queries",
        help="Eval queries directory (writes sc_*.json to queries/replacements/)",
    )
    ap.add_argument(
        "--db-path",
        default=None,
        help="transcripts.db (default: data/transcripts.db)",
    )
    ap.add_argument(
        "--clusters-path",
        type=Path,
        default=root / "data" / "clusters.json",
        help="Topic clusters JSON (default: data/clusters.json)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of cross-video questions to generate (default: 25)",
    )
    ap.add_argument(
        "--videos-per-cluster",
        type=int,
        default=3,
        help="Max videos to show the model per cluster (2–3 used in output)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for cluster/video sampling",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sc_*.json files that already exist in replacements/",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0.6,
        help="Seconds between API calls",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Override LLM model",
    )
    args = ap.parse_args()

    db_path = Path(args.db_path or root / "data" / "transcripts.db")
    if not db_path.is_file():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    if not args.clusters_path.is_file():
        print(f"Clusters file not found: {args.clusters_path}", file=sys.stderr)
        sys.exit(1)

    queries_dir = _resolve_queries_dir(args.queries_dir)
    out_dir = queries_dir / "replacements"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        llm, backend = _make_llm_caller(args.model)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    clusters = _load_clusters(args.clusters_path)
    conn = sqlite3.connect(str(db_path))
    shorty_ids = _videos_with_shorty(conn)
    selected = _select_clusters(
        clusters, shorty_ids, n=args.limit, seed=args.seed
    )

    if len(selected) < args.limit:
        print(
            f"[WARN] Only {len(selected)} clusters available (requested {args.limit})",
            file=sys.stderr,
        )

    rng = random.Random(args.seed)
    used_slugs: Set[str] = set()
    ok = skipped = failed = 0

    print(f"DB       : {db_path}")
    print(f"Clusters : {args.clusters_path}")
    print(f"Output   : {out_dir}")
    print(f"Backend  : {backend}")
    print(f"Targets  : {len(selected)}\n")

    for i, (cluster, pool_videos) in enumerate(selected, 1):
        cid = int(cluster.get("id", i))
        label = (cluster.get("label") or f"cluster_{cid}").strip()
        slug = _slugify(label)
        if slug in used_slugs:
            slug = f"{slug}_{cid}"
        used_slugs.add(slug)
        qid = f"sc_{slug}"
        out_path = out_dir / f"{qid}.json"

        if args.skip_existing and out_path.is_file():
            print(f"[{i}/{len(selected)}] skip (exists) {out_path.name}")
            skipped += 1
            continue

        picked = _pick_videos(
            pool_videos, k=min(args.videos_per_cluster, 3), rng=rng
        )
        label_tokens = list(_WORD_RE.findall(label.lower()))

        try:
            excerpt_data: List[Tuple[str, str, str, str]] = []
            chunk_ids: List[str] = []
            chunk_texts: List[str] = []
            for v in picked:
                vid = v["video_id"]
                title, channel, chunk_id, excerpt = _fetch_video_excerpt(
                    conn, vid, label_tokens=label_tokens
                )
                excerpt_data.append((title, channel, vid, excerpt))
                if chunk_id:
                    chunk_ids.append(chunk_id)
                    chunk_texts.append(excerpt[:200])

            parsed = _generate_cross_video(
                llm,
                cluster_label=label,
                videos=picked,
                excerpts=excerpt_data,
            )

            # Keep only videos the model selected; align chunks.
            sel_set = set(parsed["relevant_video_ids"])
            final_chunks: List[str] = []
            final_texts: List[str] = []
            for v, (_, _, vid, ex) in zip(picked, excerpt_data):
                if vid in sel_set:
                    idx = len(final_chunks)
                    final_chunks.append(f"{vid}_chunk_{idx}")
                    final_texts.append(ex[:200])

            artifact = _build_artifact(
                query_id=qid,
                cluster_id=cid,
                cluster_label=label,
                parsed=parsed,
                source_title=excerpt_data[0][0],
                chunk_ids=final_chunks,
                chunk_texts=final_texts,
            )
        except Exception as exc:
            print(f"[{i}/{len(selected)}] FAIL {qid}: {exc}")
            failed += 1
            continue

        out_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[{i}/{len(selected)}] OK {out_path.name}")
        print(f"    topic: {label[:70]}")
        print(f"    Q: {parsed['query'][:90]}{'...' if len(parsed['query']) > 90 else ''}")
        print(f"    videos: {', '.join(parsed['relevant_video_ids'])}")
        ok += 1

        if args.delay > 0 and i < len(selected):
            time.sleep(args.delay)

    conn.close()
    print(f"\nDone: {ok} written, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
