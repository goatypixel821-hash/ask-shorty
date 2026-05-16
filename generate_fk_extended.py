#!/usr/bin/env python3
"""
Stratified specific_fact (fk_) questions from cluster labels + Shorties.

Samples 3 videos each from 10 diverse topic clusters (videos with Shorties,
not already used in eval_results), generates one concrete question per video,
writes eval_results/fk_extended/.

Usage:
  python generate_fk_extended.py
  python generate_fk_extended.py --clusters 10 --per-cluster 3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from generate_tr_replacements import (
    _load_dotenv,
    _make_llm_caller,
    _title_leaks,
    _valid_eval_query,
    _WORD_RE,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "transcripts.db"
DEFAULT_CLUSTERS = ROOT / "data" / "clusters.json"
DEFAULT_OUT = ROOT / "eval_results" / "fk_extended"
SKIP_LABELS = {"noise", "uncategorized", "other", ""}

# One cluster per bucket for stratified variety (first match wins).
TOPIC_BUCKETS: List[Tuple[str, Tuple[str, ...]]] = [
    ("3d_printing", ("3d print", "filament", "bambu", "fdm")),
    ("woodworking_diy", ("woodwork", "wood ", "carpentry", "furniture")),
    ("space_science", ("space", "astronom", "planet", "solar system", "cosmos")),
    ("engines_aviation", ("engine", "aviation", "aircraft", "motor", "mechanical")),
    ("retro_computing", ("retro", "embedded", "vintage computer", "apple ii", "commodore")),
    ("diy_electronics", ("diy electronic", "electronics", "solder", "pcb", "raspberry pi")),
    ("ai_ml", (" ai", "artificial intelligence", "llm", "machine learning", "chatgpt")),
    ("economics_policy", ("economic", "finance", "market", "inflation", "gdp")),
    ("privacy_security", ("privacy", "surveillance", "security", "encryption", "hack")),
    ("skepticism_science", ("skeptic", "debunk", "myth", "pseudoscience", "fact check")),
]

FK_SYSTEM = """You write specific_fact evaluation questions for a personal video transcript search system.

Given a Shorty summary from ONE video, produce:
1. A single concrete factual question answerable from that Shorty alone
2. A short ground_truth answer (1-2 sentences) with the specific fact(s)

Rules:
- Ask for a precise fact: number, date, version, name, mechanism, comparison, outcome, etc.
- At least 10 words, must end with ?
- Do NOT mention the video title, channel name, creator, or "this video"
- Do NOT use vague templates like "what does the video discuss"
- ground_truth must be factual and drawn only from the Shorty

Return ONLY valid JSON:
{"query": "...", "ground_truth": "..."}"""


def _collect_reserved_video_ids(eval_root: Path) -> Set[str]:
    reserved: Set[str] = set()
    if not eval_root.is_dir():
        return reserved
    for jf in eval_root.rglob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("relevant_video_ids", "expected_video_ids"):
            for vid in data.get(key) or []:
                v = str(vid).strip()
                if v:
                    reserved.add(v)
    return reserved


def _shorty_video_ids(db_path: Path) -> Set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT video_id FROM transcripts
            WHERE shorty IS NOT NULL AND trim(shorty) != ''
            """
        ).fetchall()
        return {str(r[0]) for r in rows}
    finally:
        conn.close()


def _load_clusters(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    clusters = data.get("clusters")
    if not isinstance(clusters, list):
        raise ValueError(f"No clusters list in {path}")
    return clusters


def _bucket_for_label(label: str) -> Optional[str]:
    low = f" {label.lower()} "
    for bucket, patterns in TOPIC_BUCKETS:
        if any(p in low for p in patterns):
            return bucket
    return None


def _select_diverse_clusters(
    clusters: Sequence[Dict[str, Any]],
    shorty_ids: Set[str],
    reserved: Set[str],
    *,
    n_clusters: int,
    per_cluster: int,
    seed: int,
) -> List[Tuple[Dict[str, Any], List[str]]]:
    rng = random.Random(seed)
    by_bucket: Dict[str, List[Tuple[Dict[str, Any], List[str], int]]] = {}
    unbucketed: List[Tuple[Dict[str, Any], List[str], int]] = []

    for c in clusters:
        label = (c.get("label") or "").strip()
        if not label or label.lower() in SKIP_LABELS:
            continue
        eligible: List[str] = []
        for v in c.get("videos") or []:
            if not isinstance(v, dict):
                continue
            vid = str(v.get("video_id") or "").strip()
            if vid and vid in shorty_ids and vid not in reserved:
                eligible.append(vid)
        if len(eligible) < per_cluster:
            continue
        item = (c, eligible, len(eligible))
        bucket = _bucket_for_label(label)
        if bucket:
            by_bucket.setdefault(bucket, []).append(item)
        else:
            unbucketed.append(item)

    chosen: List[Tuple[Dict[str, Any], List[str]]] = []

    # Best cluster per topic bucket (most eligible videos).
    for bucket in [b[0] for b in TOPIC_BUCKETS]:
        pool = by_bucket.get(bucket, [])
        if not pool:
            continue
        pool.sort(key=lambda x: x[2], reverse=True)
        cluster, eligible, _ = pool[0]
        rng.shuffle(eligible)
        chosen.append((cluster, eligible[:per_cluster]))

    # Fill remaining slots from large unbucketed clusters.
    if len(chosen) < n_clusters:
        unbucketed.sort(key=lambda x: x[2], reverse=True)
        used_ids = {int(c[0].get("id", -1)) for c, _ in chosen}
        for cluster, eligible, _ in unbucketed:
            if len(chosen) >= n_clusters:
                break
            cid = int(cluster.get("id", -1))
            if cid in used_ids:
                continue
            rng.shuffle(eligible)
            chosen.append((cluster, eligible[:per_cluster]))
            used_ids.add(cid)

    return chosen[:n_clusters]


def _fetch_video_row(db_path: Path, video_id: str) -> Dict[str, str]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT v.video_id, v.title, v.channel, t.shorty
            FROM videos v
            JOIN transcripts t ON t.video_id = v.video_id
            WHERE v.video_id = ?
              AND t.shorty IS NOT NULL AND trim(t.shorty) != ''
            ORDER BY t.id DESC LIMIT 1
            """,
            (video_id,),
        ).fetchone()
        if not row:
            return {"video_id": video_id, "title": video_id, "channel": "", "shorty": ""}
        return {
            "video_id": row["video_id"],
            "title": row["title"] or video_id,
            "channel": row["channel"] or "",
            "shorty": row["shorty"] or "",
        }
    finally:
        conn.close()


def _segment_excerpts(db_path: Path, video_id: str, *, max_parts: int = 3) -> Tuple[List[str], List[str]]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT id, summary FROM segments
            WHERE video_id = ? AND summary IS NOT NULL AND trim(summary) != ''
            ORDER BY start_time
            LIMIT ?
            """,
            (video_id, max_parts),
        ).fetchall()
    finally:
        conn.close()
    chunk_ids: List[str] = []
    texts: List[str] = []
    for i, (seg_id, summ) in enumerate(rows):
        chunk_ids.append(f"{video_id}_chunk_{i}")
        texts.append((summ or "").strip()[:300])
    return chunk_ids, texts


def _extract_json_obj(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_fk_response(
    raw: str,
    *,
    title: str,
    channel: str,
    shorty: str,
) -> Optional[Dict[str, str]]:
    obj = _extract_json_obj(raw)
    if not obj:
        return None
    query = (obj.get("query") or "").strip()
    gt = (obj.get("ground_truth") or "").strip()
    if not _valid_eval_query(query):
        return None
    if not gt or len(gt) < 8:
        return None
    ch = (channel or "").strip()
    if ch and len(ch) >= 4 and ch.lower() in query.lower():
        return None
    if _title_leaks(query, title, [shorty]):
        return None
    return {"query": query, "ground_truth": gt}


def _generate_fk(
    llm,
    *,
    title: str,
    channel: str,
    shorty: str,
    video_id: str,
) -> Dict[str, str]:
    excerpt = shorty[:3500]
    user = (
        f"VIDEO_ID (do not put in question): {video_id}\n"
        f"TITLE (do not mention): {title}\n"
        f"CHANNEL (do not mention): {channel}\n\n"
        f"SHORTY:\n{excerpt}\n\n"
        "Write one specific_fact question and ground_truth."
    )
    raw = llm(FK_SYSTEM, user, max_tokens=400, temperature=0.3)
    parsed = _parse_fk_response(raw, title=title, channel=channel, shorty=shorty)
    if parsed:
        return parsed
    raw2 = llm(
        FK_SYSTEM,
        user + '\n\nReply with ONLY: {"query":"...?","ground_truth":"..."}',
        max_tokens=400,
        temperature=0.15,
    )
    parsed = _parse_fk_response(raw2, title=title, channel=channel, shorty=shorty)
    if parsed:
        return parsed
    raise ValueError(f"Could not parse FK JSON: {raw[:180]!r}")


def _build_fk_artifact(
    *,
    query_id: str,
    query: str,
    ground_truth: str,
    video_id: str,
    title: str,
    chunk_ids: List[str],
    chunk_texts: List[str],
    cluster_id: int,
    cluster_label: str,
) -> Dict[str, Any]:
    return {
        "query_id": query_id,
        "id": query_id,
        "query": query,
        "query_type": "fact_lookup",
        "category": "specific_fact",
        "source_video_title": title,
        "relevant_video_ids": [video_id],
        "expected_video_ids": [video_id],
        "expected_chunk_ids": chunk_ids,
        "expected_chunk_texts": chunk_texts,
        "neighbor_chunk_ids": [],
        "gold_answer_note": ground_truth,
        "ground_truth": ground_truth,
        "support_count": len(chunk_ids),
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
        "notes": f"fk_extended; cluster_id={cluster_id}",
    }


def main() -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(description="Generate fk_extended specific_fact eval queries.")
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    ap.add_argument("--clusters-path", type=Path, default=DEFAULT_CLUSTERS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--eval-root", type=Path, default=ROOT / "eval_results")
    ap.add_argument("--clusters", type=int, default=10, help="Number of clusters")
    ap.add_argument("--per-cluster", type=int, default=3, help="Videos per cluster")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true", help="Sample only; no LLM / no writes")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    if not args.db_path.is_file():
        print(f"DB not found: {args.db_path}", file=sys.stderr)
        sys.exit(1)
    if not args.clusters_path.is_file():
        print(f"Clusters not found: {args.clusters_path}", file=sys.stderr)
        sys.exit(1)

    reserved = _collect_reserved_video_ids(args.eval_root)
    shorty_ids = _shorty_video_ids(args.db_path)
    clusters = _load_clusters(args.clusters_path)
    selected = _select_diverse_clusters(
        clusters,
        shorty_ids,
        reserved,
        n_clusters=args.clusters,
        per_cluster=args.per_cluster,
        seed=args.seed,
    )

    if len(selected) < args.clusters:
        print(
            f"[WARN] Only {len(selected)} clusters with >={args.per_cluster} eligible videos "
            f"(requested {args.clusters})",
            file=sys.stderr,
        )

    total_videos = sum(len(vids) for _, vids in selected)
    print(f"Reserved video IDs in eval_results: {len(reserved)}")
    print(f"Videos with Shorties: {len(shorty_ids)}")
    print(f"Selected {len(selected)} clusters, {total_videos} videos\n")

    summary_lines = ["CLUSTER SAMPLE SUMMARY", "-" * 72]
    for cluster, vids in selected:
        cid = int(cluster.get("id", -1))
        label = (cluster.get("label") or "").strip()
        bucket = _bucket_for_label(label) or "other"
        summary_lines.append(f"  cluster {cid:4d}  [{bucket}]  ({len(vids)} videos)  {label}")
        for vid in vids:
            meta = _fetch_video_row(args.db_path, vid)
            title = (meta["title"] or "")[:55].encode("ascii", "replace").decode("ascii")
            summary_lines.append(f"      {vid}  {title}")
    summary_lines.append("")
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "_cluster_sample_summary.txt").write_text(
        summary_text, encoding="utf-8"
    )

    if args.dry_run:
        print("Dry run — no files written.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        llm, backend = _make_llm_caller()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(f"LLM backend: {backend}")
    print(f"Output: {args.output_dir}\n")

    ok = failed = skipped = 0
    seq = 0
    for cluster, vids in selected:
        cid = int(cluster.get("id", -1))
        label = (cluster.get("label") or "").strip()
        for vid in vids:
            seq += 1
            qid = f"fk_ext_{vid[:8]}_{seq}"
            out_path = args.output_dir / f"{qid}.json"
            if args.skip_existing and out_path.is_file():
                print(f"[{seq}/{total_videos}] skip {qid}")
                skipped += 1
                continue

            meta = _fetch_video_row(args.db_path, vid)
            title = meta["title"]
            channel = meta["channel"]
            shorty = meta["shorty"]
            if not shorty.strip():
                print(f"[{seq}/{total_videos}] FAIL {vid}: no shorty")
                failed += 1
                continue

            chunk_ids, chunk_texts = _segment_excerpts(args.db_path, vid)
            try:
                parsed = _generate_fk(
                    llm,
                    title=title,
                    channel=channel,
                    shorty=shorty,
                    video_id=vid,
                )
            except Exception as exc:
                print(f"[{seq}/{total_videos}] FAIL {qid}: {exc}")
                failed += 1
                continue

            artifact = _build_fk_artifact(
                query_id=qid,
                query=parsed["query"],
                ground_truth=parsed["ground_truth"],
                video_id=vid,
                title=title,
                chunk_ids=chunk_ids,
                chunk_texts=chunk_texts,
                cluster_id=cid,
                cluster_label=label,
            )
            out_path.write_text(
                json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"[{seq}/{total_videos}] OK {qid}")
            print(f"    cluster {cid}: {label[:60]}")
            print(f"    Q: {parsed['query'][:85]}{'...' if len(parsed['query']) > 85 else ''}")
            ok += 1
            if args.delay > 0 and seq < total_videos:
                time.sleep(args.delay)

    print(f"\nDone: {ok} written, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
