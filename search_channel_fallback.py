#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple


DATE_MIN = "2024-06-01"
DATE_MAX = "2025-06-01"

HIGH_SIGNAL = [
    "fell through",
    "almost fell through",
    "just about fell through",
    "nearly fell through",
    "through the floor",
    "floor gave",
    "soft spot",
    "rotting",
    "back on their feet",
    "back on your feet",
    "getting back",
    "buyer's agent",
    "buyers agent",
    "consulting",
    "first time buyer",
    "property condition",
    "camera missed it",
    "didn't get that on camera",
    "missed that on camera",
    "you missed it",
    "wasn't on camera",
]

WALK_CLIENT_TERMS = ["walk through", "walkthrough", "walk-thru", "walk thru"]
CAMERA_TERMS = ["camera", "filming"]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _snippet(text: str, idx: int, span: int = 150) -> str:
    start = max(0, idx - span)
    end = min(len(text), idx + span)
    return _norm(text[start:end])


def _find_matches(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    low = (text or "").lower()
    raw = text or ""
    for phrase in HIGH_SIGNAL:
        i = low.find(phrase.lower())
        if i != -1:
            out.append((phrase, _snippet(raw, i, 150)))
    # walk through near client/buyer
    for w in WALK_CLIENT_TERMS:
        wi = low.find(w)
        if wi == -1:
            continue
        for c in ("client", "buyer", "buyers", "buyer's"):
            ci = low.find(c)
            if ci != -1 and abs(ci - wi) <= 200:
                out.append((f"{w} near {c}", _snippet(raw, min(wi, ci), 150)))
                break
    # floor within 200 chars of camera/filming
    floor_positions = [m.start() for m in re.finditer(r"floor", low)]
    cam_positions = [m.start() for m in re.finditer(r"camera|filming", low)]
    for fp in floor_positions:
        near = [cp for cp in cam_positions if abs(cp - fp) <= 200]
        if near:
            out.append(("floor near camera/filming", _snippet(raw, fp, 150)))
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--db-path",
        default=r"C:\Users\number2\Desktop\shorty\data\transcripts.db",
    )
    ap.add_argument(
        "--out-path",
        default=r"data\channel_search_fallback.txt",
    )
    args = ap.parse_args()

    db_path = Path(args.db_path)
    out_path = Path(args.out_path)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            v.video_id,
            v.title,
            v.channel,
            COALESCE(v.watch_date, '') AS watch_date,
            COALESCE(t.text, '') AS text,
            COALESCE(t.shorty, '') AS shorty
        FROM videos v
        JOIN transcripts t ON t.video_id = v.video_id
        WHERE COALESCE(v.watch_date, '') >= ?
          AND COALESCE(v.watch_date, '') <= ?
        ORDER BY v.watch_date DESC
        """,
        (DATE_MIN, DATE_MAX),
    )
    rows = cur.fetchall()

    per_video_hits = []
    channel_to_vids = defaultdict(set)

    for r in rows:
        blob = (r["text"] or "") + "\n" + (r["shorty"] or "")
        matches = _find_matches(blob)
        if not matches:
            continue
        vid = r["video_id"] or ""
        ch = r["channel"] or ""
        channel_to_vids[ch].add(vid)
        per_video_hits.append(
            {
                "video_id": vid,
                "title": r["title"] or "",
                "channel": ch,
                "watch_date": (r["watch_date"] or "")[:10],
                "matches": matches,
            }
        )

    # keep channels with 2-10 matched videos
    keep_channels = {ch for ch, vids in channel_to_vids.items() if 2 <= len(vids) <= 10}
    filtered = [x for x in per_video_hits if x["channel"] in keep_channels]

    # Direct SQL-style search requested
    cur.execute(
        """
        SELECT
            v.video_id,
            v.title,
            v.channel,
            v.watch_date,
            COALESCE(t.text, '') AS text
        FROM videos v
        JOIN transcripts t ON t.video_id = v.video_id
        WHERE (
            lower(COALESCE(t.text,'')) LIKE '%fell through%'
            OR lower(COALESCE(t.text,'')) LIKE '%through the floor%'
            OR lower(COALESCE(t.text,'')) LIKE '%floor gave%'
            OR lower(COALESCE(t.text,'')) LIKE '%soft spot%floor%'
            OR lower(COALESCE(t.text,'')) LIKE '%almost fell through%'
            OR lower(COALESCE(t.text,'')) LIKE '%just about fell through%'
            OR lower(COALESCE(t.text,'')) LIKE '%nearly fell through%'
        )
        AND COALESCE(v.watch_date,'') >= ?
        AND COALESCE(v.watch_date,'') <= ?
        ORDER BY v.watch_date DESC
        """,
        (DATE_MIN, DATE_MAX),
    )
    direct_rows = cur.fetchall()

    lines: List[str] = []
    lines.append("Channel Search Fallback")
    lines.append(f"DB: {db_path}")
    lines.append(f"Date range: {DATE_MIN} .. {DATE_MAX}")
    lines.append("")
    lines.append(f"Matched videos (all channels): {len(per_video_hits)}")
    lines.append(f"Channels with 2-10 matched videos: {len(keep_channels)}")
    lines.append(f"Filtered videos kept: {len(filtered)}")
    lines.append("")

    lines.append("=== FILTERED CHANNEL RESULTS (2-10 videos/channel) ===")
    if not filtered:
        lines.append("No channels met the 2-10 rule with these high-signal patterns.")
    else:
        by_channel = defaultdict(list)
        for hit in filtered:
            by_channel[hit["channel"]].append(hit)
        for ch, vids in sorted(by_channel.items(), key=lambda kv: (-len(kv[1]), kv[0].lower())):
            uniq_vids = {v["video_id"] for v in vids}
            lines.append("")
            lines.append(f"- CHANNEL: {ch} ({len(uniq_vids)} matched videos)")
            for v in vids:
                lines.append(f"  {v['watch_date']} | {v['video_id']} | {v['title']}")
                for phrase, snip in v["matches"][:4]:
                    lines.append(f"    [{phrase}] {snip}")

    lines.append("")
    lines.append("=== DIRECT SEARCH (floor/fell-through phrases) ===")
    lines.append(f"Direct rows: {len(direct_rows)}")
    for r in direct_rows:
        txt = r["text"] or ""
        low = txt.lower()
        idx = -1
        for p in (
            "almost fell through",
            "just about fell through",
            "nearly fell through",
            "fell through",
            "through the floor",
            "floor gave",
            "soft spot",
        ):
            idx = low.find(p)
            if idx != -1:
                break
        if idx == -1:
            idx = 0
        snip = _snippet(txt, idx, 150)
        lines.append(
            f"{(r['watch_date'] or '')[:10]} | {r['video_id']} | {r['channel']} | {r['title']}"
        )
        lines.append(f"  {snip}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    conn.close()
    print(f"Wrote: {out_path}")
    print(f"Matched videos (all): {len(per_video_hits)}")
    print(f"Channels kept (2-10): {len(keep_channels)}")
    print(f"Direct rows: {len(direct_rows)}")


if __name__ == "__main__":
    main()
