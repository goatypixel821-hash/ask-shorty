#!/usr/bin/env python3
"""
Review and optionally skip likely ad/commercial videos in processing_queue.

Behavior:
1) Loads all DISTINCT videos with pending queue tasks.
2) Flags videos that look like ads based on channel/title heuristics.
3) Prints flagged videos for review.
4) Always writes full pending list to data/pending_review.csv.
5) With --execute, marks pending queue rows for flagged videos as permanently_failed.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


DEFAULT_DB_PATH = r"C:\Users\number2\Desktop\shorty\data\transcripts.db"
DEFAULT_CSV_PATH = Path("data") / "pending_review.csv"

BRAND_CHANNEL_KEYWORDS = [
    "pizza hut",
    "domino",
    "mcdonald",
    "facebook",
    "fanduel",
    "spectrum",
    "harbor freight",
    "walmart",
    "target",
    "best buy",
    "verizon",
    "at&t",
    "att",
    "t-mobile",
    "xfinity",
    "comcast",
    "geico",
    "progressive",
    "state farm",
    "draftkings",
    "temu",
]

TITLE_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("contains 'NAM Acq'", re.compile(r"\bnam\s*acq\b", re.IGNORECASE)),
    ("contains WEB_15 style token", re.compile(r"\bweb[_\-\s]?\d{1,3}\b", re.IGNORECASE)),
    ("contains 16x9 token", re.compile(r"\b16x9\b", re.IGNORECASE)),
    ("contains OTT token", re.compile(r"\bott\b", re.IGNORECASE)),
    ("contains ad-tech token", re.compile(r"\b(pre[-\s]?roll|mid[-\s]?roll|bumper|cta)\b", re.IGNORECASE)),
]


@dataclass
class PendingVideo:
    video_id: str
    title: str
    channel: str


def resolve_db_path(cli_db_path: str | None) -> Path:
    if cli_db_path:
        return Path(cli_db_path)
    env_db = (os.environ.get("ASK_SHORTY_DB_PATH") or "").strip()
    if env_db:
        return Path(env_db)
    return Path(DEFAULT_DB_PATH)


def fetch_pending_videos(db_path: Path) -> List[PendingVideo]:
    sql = """
    SELECT DISTINCT
        pq.video_id,
        COALESCE(v.title, '') AS title,
        COALESCE(v.channel, '') AS channel
    FROM processing_queue pq
    LEFT JOIN videos v ON v.video_id = pq.video_id
    WHERE pq.status = 'pending'
    ORDER BY pq.video_id
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(sql).fetchall()
        return [PendingVideo(video_id=r[0], title=r[1], channel=r[2]) for r in rows]
    finally:
        conn.close()


def is_brand_channel(channel: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    c = (channel or "").strip().lower()
    if not c:
        return False, reasons
    for kw in BRAND_CHANNEL_KEYWORDS:
        if kw in c:
            reasons.append(f"channel contains brand keyword '{kw}'")
    return bool(reasons), reasons


def title_looks_like_ad(title: str, is_brand: bool) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    t = (title or "").strip()
    if not t:
        return False, reasons

    for label, pattern in TITLE_PATTERNS:
        if pattern.search(t):
            reasons.append(label)

    # Common ad slug style: lots of uppercase tokens + digits/underscores
    tokens = re.findall(r"[A-Za-z0-9_#\-]+", t)
    caps_tokens = [x for x in tokens if len(x) >= 3 and x.upper() == x and re.search(r"[A-Z]", x)]
    if len(caps_tokens) >= 3 and re.search(r"\d", t):
        reasons.append("all-caps+numeric slug pattern")

    if "#shorts" in t.lower() and is_brand:
        reasons.append("brand channel #shorts pattern")

    return bool(reasons), reasons


def looks_like_ad(video: PendingVideo) -> Tuple[bool, List[str]]:
    is_brand, channel_reasons = is_brand_channel(video.channel)
    looks_ad_title, title_reasons = title_looks_like_ad(video.title, is_brand)

    # Conservative rule: brand signal OR strong title signal
    flagged = is_brand or looks_ad_title
    reasons = channel_reasons + title_reasons
    return flagged, reasons


def write_pending_csv(rows: List[PendingVideo], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "title", "channel"])
        for r in rows:
            w.writerow([r.video_id, r.title, r.channel])


def mark_flagged_permanently_failed(db_path: Path, flagged_ids: List[str]) -> int:
    if not flagged_ids:
        return 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reason = "Skipped by filter_ad_videos.py: likely ad/commercial"
    placeholders = ",".join("?" for _ in flagged_ids)
    sql = f"""
    UPDATE processing_queue
    SET status = 'permanently_failed',
        completed_at = ?,
        error = ?
    WHERE status = 'pending'
      AND video_id IN ({placeholders})
    """
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(sql, [now, reason, *flagged_ids])
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Review pending queue videos, flag likely ads, optionally skip them permanently."
    )
    p.add_argument("--db-path", default=None, help="Path to transcripts.db")
    p.add_argument("--execute", action="store_true", help="Apply changes: mark flagged pending rows as permanently_failed")
    args = p.parse_args()

    db_path = resolve_db_path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    pending = fetch_pending_videos(db_path)
    write_pending_csv(pending, DEFAULT_CSV_PATH)

    flagged: List[Tuple[PendingVideo, List[str]]] = []
    for row in pending:
        is_ad, reasons = looks_like_ad(row)
        if is_ad:
            flagged.append((row, reasons))

    print(f"DB: {db_path}")
    print(f"Pending videos total: {len(pending)}")
    print(f"Flagged as likely ads/commercials: {len(flagged)}")
    print(f"Full pending review CSV: {DEFAULT_CSV_PATH}")

    if flagged:
        print("\nFlagged videos:")
        for i, (v, reasons) in enumerate(flagged, start=1):
            reason_text = "; ".join(reasons) if reasons else "heuristic match"
            print(f"{i:>3}. {v.video_id} | {v.channel} | {v.title}")
            print(f"     reasons: {reason_text}")

    if not args.execute:
        print("\nDry run only. Re-run with --execute to mark flagged pending rows as permanently_failed.")
        return

    changed = mark_flagged_permanently_failed(db_path, [v.video_id for v, _ in flagged])
    print(f"\nUpdated queue rows: {changed}")
    print("Done.")


if __name__ == "__main__":
    main()
