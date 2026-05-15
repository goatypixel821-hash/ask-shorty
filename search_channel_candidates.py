#!/usr/bin/env python3
"""
Search for a likely channel match across transcripts/shorties/questions/metadata.

Writes a human-readable report to data/channel_search_results.txt.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TEXT_TERMS = [
    "fell through",
    "through the floor",
    "rotting floor",
    "soft floor",
    "floor gave",
    "getting back on their feet",
    "back on your feet",
    "buyer's agent",
    "buyers agent",
    "consulting",
]

SHORTY_TERMS = list(TEXT_TERMS)

SYNQ_TERMS = [
    "property condition",
    "walkthrough",
    "walk through",
    "buyer's agent",
    "buyers agent",
    "real estate consultant",
    "consulting",
    "floor",
    "rotting",
]

REAL_ESTATE_HINTS = [
    "real estate",
    "property",
    "home buying",
    "buyers agent",
    "buyer's agent",
    "walkthrough",
    "walk through",
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _find_snippet(text: str, terms: Sequence[str], context: int = 50) -> Tuple[Optional[str], Optional[str]]:
    """Return (matched_term, snippet) with 50 chars around first hit."""
    t = text or ""
    low = t.lower()
    first_idx = None
    first_term = None
    for term in terms:
        i = low.find(term.lower())
        if i != -1 and (first_idx is None or i < first_idx):
            first_idx = i
            first_term = term
    if first_idx is None:
        return None, None
    start = max(0, first_idx - context)
    end = min(len(t), first_idx + len(first_term or "") + context)
    snippet = _norm(t[start:end])
    return first_term, snippet


def _has_walkthrough_client(text: str) -> bool:
    low = (text or "").lower()
    walk = any(k in low for k in ("walk through", "walkthrough", "walk-thru", "walk thru"))
    client = any(k in low for k in ("client", "buyer", "buyers", "buyer's"))
    return walk and client


def _extract_description(json_metadata: str) -> str:
    raw = (json_metadata or "").strip()
    if not raw:
        return ""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for key in ("description", "shortDescription", "videoDescription", "desc"):
                v = obj.get(key)
                if isinstance(v, str) and v.strip():
                    return v
    except Exception:
        pass
    # Fallback: search as plain text blob.
    return raw


@dataclass
class Hit:
    approach: str
    video_id: str
    title: str
    channel: str
    watch_date: str
    matched: str
    snippet: str


def _base_rows(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
          v.video_id,
          v.title,
          v.channel,
          v.watch_date,
          v.json_metadata,
          COALESCE(t.text, '')   AS text,
          COALESCE(t.shorty, '') AS shorty
        FROM videos v
        LEFT JOIN transcripts t ON t.video_id = v.video_id
        """
    )
    return cur.fetchall()


def run_search(db_path: Path, out_path: Path) -> Dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    hits_by_approach: Dict[str, List[Hit]] = defaultdict(list)
    channels_to_videos: Dict[str, set] = defaultdict(set)
    channel_topic_counts: List[Tuple[str, int]] = []

    rows = list(_base_rows(conn))

    # 1 + 2 + 6 on shared row scan
    for r in rows:
        video_id = r["video_id"] or ""
        title = r["title"] or ""
        channel = r["channel"] or ""
        watch_date = (r["watch_date"] or "")[:10]
        text = r["text"] or ""
        shorty = r["shorty"] or ""
        meta_desc = _extract_description(r["json_metadata"] or "")

        # 1) transcripts.text
        term, snip = _find_snippet(text, TEXT_TERMS)
        if term and snip:
            hits_by_approach["1_transcripts_text"].append(
                Hit("1_transcripts_text", video_id, title, channel, watch_date, term, snip)
            )
            channels_to_videos[channel].add(video_id)
        elif _has_walkthrough_client(text):
            _, snip2 = _find_snippet(text, ["walk through", "walkthrough", "client", "buyer"])
            hits_by_approach["1_transcripts_text"].append(
                Hit(
                    "1_transcripts_text",
                    video_id,
                    title,
                    channel,
                    watch_date,
                    "walk through + client",
                    snip2 or _norm(text[:140]),
                )
            )
            channels_to_videos[channel].add(video_id)

        # 2) transcripts.shorty
        term, snip = _find_snippet(shorty, SHORTY_TERMS)
        if term and snip:
            hits_by_approach["2_transcripts_shorty"].append(
                Hit("2_transcripts_shorty", video_id, title, channel, watch_date, term, snip)
            )
            channels_to_videos[channel].add(video_id)

        # 6) videos.json_metadata description
        term, snip = _find_snippet(meta_desc, TEXT_TERMS + ["real estate", "property", "home buying"])
        if term and snip:
            hits_by_approach["6_json_metadata_description"].append(
                Hit("6_json_metadata_description", video_id, title, channel, watch_date, term, snip)
            )
            channels_to_videos[channel].add(video_id)

    # 3) synthetic questions
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
          sq.video_id,
          sq.question,
          v.title,
          v.channel,
          v.watch_date
        FROM synthetic_questions sq
        JOIN videos v ON v.video_id = sq.video_id
        """
    )
    for r in cur.fetchall():
        q = r["question"] or ""
        term, snip = _find_snippet(q, SYNQ_TERMS)
        if term and snip:
            hit = Hit(
                "3_synthetic_questions",
                r["video_id"] or "",
                r["title"] or "",
                r["channel"] or "",
                (r["watch_date"] or "")[:10],
                term,
                snip,
            )
            hits_by_approach["3_synthetic_questions"].append(hit)
            channels_to_videos[hit.channel].add(hit.video_id)

    # 4) channels with 3+ topic videos
    cur.execute(
        """
        SELECT
          v.channel AS channel,
          COUNT(DISTINCT v.video_id) AS n
        FROM videos v
        LEFT JOIN transcripts t ON t.video_id = v.video_id
        WHERE
          lower(COALESCE(v.title,'')) LIKE '%real estate%' OR
          lower(COALESCE(v.title,'')) LIKE '%property%' OR
          lower(COALESCE(v.title,'')) LIKE '%home buying%' OR
          lower(COALESCE(t.text,'')) LIKE '%real estate%' OR
          lower(COALESCE(t.text,'')) LIKE '%property%' OR
          lower(COALESCE(t.text,'')) LIKE '%home buying%' OR
          lower(COALESCE(t.shorty,'')) LIKE '%real estate%' OR
          lower(COALESCE(t.shorty,'')) LIKE '%property%' OR
          lower(COALESCE(t.shorty,'')) LIKE '%home buying%'
        GROUP BY v.channel
        HAVING COUNT(DISTINCT v.video_id) >= 3
        ORDER BY n DESC, v.channel
        """
    )
    channel_topic_counts = [(r["channel"] or "", int(r["n"] or 0)) for r in cur.fetchall()]

    # enrich approach 4 with sample videos/snippets
    for ch, _n in channel_topic_counts:
        cur.execute(
            """
            SELECT v.video_id, v.title, v.channel, v.watch_date,
                   COALESCE(t.text,'') AS text, COALESCE(t.shorty,'') AS shorty
            FROM videos v
            LEFT JOIN transcripts t ON t.video_id = v.video_id
            WHERE v.channel = ?
            """,
            (ch,),
        )
        for r in cur.fetchall():
            blob = (r["text"] or "") + "\n" + (r["shorty"] or "")
            term, snip = _find_snippet(blob, REAL_ESTATE_HINTS)
            if term and snip:
                hit = Hit(
                    "4_channel_3plus_topic_videos",
                    r["video_id"] or "",
                    r["title"] or "",
                    r["channel"] or "",
                    (r["watch_date"] or "")[:10],
                    term,
                    snip,
                )
                hits_by_approach["4_channel_3plus_topic_videos"].append(hit)
                channels_to_videos[hit.channel].add(hit.video_id)

    # Prepare final report.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("CHANNEL SEARCH RESULTS")
    lines.append(f"DB: {db_path}")
    lines.append("")
    lines.append("Counts by approach:")
    ordered_approaches = [
        "1_transcripts_text",
        "2_transcripts_shorty",
        "3_synthetic_questions",
        "4_channel_3plus_topic_videos",
        "6_json_metadata_description",
    ]
    counts = {k: len(hits_by_approach.get(k, [])) for k in ordered_approaches}
    for k in ordered_approaches:
        lines.append(f"- {k}: {counts[k]}")
    lines.append("")

    lines.append("Top channels by unique matched videos (all approaches 1/2/3/4/6):")
    channel_rank = sorted(
        ((ch, len(vs)) for ch, vs in channels_to_videos.items() if ch),
        key=lambda x: (-x[1], x[0].lower()),
    )
    for ch, n in channel_rank[:60]:
        lines.append(f"- {ch}: {n} matched videos")
    lines.append("")

    lines.append("Approach 4 channel counts (>=3 topic videos):")
    for ch, n in channel_topic_counts[:120]:
        lines.append(f"- {ch}: {n}")
    lines.append("")

    for key in ordered_approaches:
        hits = hits_by_approach.get(key, [])
        lines.append("=" * 95)
        lines.append(f"{key} — {len(hits)} matches")
        lines.append("=" * 95)
        for h in hits:
            lines.append(
                f"{h.video_id} | {h.title} | {h.channel} | {h.watch_date} | matched={h.matched}"
            )
            lines.append(f"  snippet: {h.snippet}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    conn.close()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Search likely channel from transcripts.db")
    ap.add_argument(
        "--db-path",
        default=r"C:\Users\number2\Desktop\shorty\data\transcripts.db",
        help="Path to SQLite DB",
    )
    ap.add_argument(
        "--out-path",
        default=r"data\channel_search_results.txt",
        help="Output report path",
    )
    args = ap.parse_args()

    db_path = Path(args.db_path)
    out_path = Path(args.out_path)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path

    counts = run_search(db_path, out_path)
    print("Wrote:", out_path)
    for k, v in counts.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
