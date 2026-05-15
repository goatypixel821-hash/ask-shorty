#!/usr/bin/env python3
"""
Import YouTube watch dates from a Google Takeout (watch-history.json or .html)
into youtube-history-viewer-copy's transcripts.db videos table.

- For rows that already exist: only sets watch_date when it is NULL or empty
  (never overwrites an existing watch_date).
- For video IDs not yet in the table: INSERT a new row with video_id,
  watch_date (latest from history), and url only; title/channel left NULL.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_DOWNLOADS = Path.home() / "Downloads"
DEFAULT_DB = Path(
    r"C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db"
)
HISTORY_REL = Path("Takeout") / "YouTube and YouTube Music" / "history"

VID_RE = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]{11})",
    re.I,
)
HTML_BLOCK_MARK = '<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">'
HTML_DATE_RE = re.compile(
    r"([A-Z][a-z]{2} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2}\s+[AP]M)\s+[A-Z]{2,5}"
)


def _fmt_db_ts(dt: datetime) -> str:
    """Match existing DB style: YYYY-MM-DDTHH:MM:SS (no timezone suffix)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _parse_iso_time(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _parse_html_date_fragment(s: str) -> Optional[datetime]:
    s = s.strip()
    try:
        return datetime.strptime(s, "%b %d, %Y, %I:%M:%S %p")
    except ValueError:
        return None


def find_latest_takeout_zip(downloads: Path) -> Path:
    zips = sorted(downloads.glob("takeout-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No takeout-*.zip under {downloads}")
    return zips[0]


def locate_history_file(takeout_root: Path) -> Path:
    hist_dir = takeout_root / HISTORY_REL
    for name in ("watch-history.json", "watch-history.html"):
        p = hist_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"No watch-history.json or watch-history.html under {hist_dir}"
    )


def extract_takeout_zip(zip_path: Path) -> Tuple[Path, bool]:
    """
    Returns (root_folder_containing_Takeout, owns_temp_cleanup).
    """
    tmp = Path(tempfile.mkdtemp(prefix="takeout_watch_import_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)
    # Zip may contain Takeout/ at top level or a single folder wrapping it
    takeout = tmp / "Takeout"
    if takeout.is_dir():
        return tmp, True
    # search for Takeout
    for p in tmp.rglob("Takeout"):
        if p.is_dir():
            return p.parent, True
    shutil.rmtree(tmp, ignore_errors=True)
    raise FileNotFoundError(f"No Takeout/ folder found after extracting {zip_path}")


def parse_watch_history_json(path: Path) -> List[Tuple[str, datetime]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    data = json.loads(raw)
    if not isinstance(data, list):
        data = [data]
    out: List[Tuple[str, datetime]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        url = (
            item.get("titleUrl")
            or item.get("titleURL")
            or item.get("url")
            or ""
        )
        m = VID_RE.search(url.replace("&amp;", "&"))
        if not m:
            continue
        vid = m.group(1)
        ts_raw = item.get("time") or item.get("timestamp") or ""
        dt = _parse_iso_time(str(ts_raw))
        if dt is None:
            continue
        out.append((vid, dt))
    return out


def parse_watch_history_html(path: Path) -> List[Tuple[str, datetime]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = text.split(HTML_BLOCK_MARK)[1:]
    out: List[Tuple[str, datetime]] = []
    for block in parts:
        m = VID_RE.search(block.replace("&amp;", "&"))
        if not m:
            continue
        vid = m.group(1)
        dm = HTML_DATE_RE.search(block)
        if not dm:
            continue
        dt = _parse_html_date_fragment(dm.group(1))
        if dt is None:
            continue
        out.append((vid, dt))
    return out


def parse_history(path: Path) -> Tuple[str, List[Tuple[str, datetime]]]:
    suf = path.suffix.lower()
    if suf == ".json":
        rows = parse_watch_history_json(path)
        return "JSON", rows
    if suf == ".html":
        rows = parse_watch_history_html(path)
        return "HTML", rows
    raise ValueError(f"Unsupported watch history file type: {path}")


def latest_watch_per_video(rows: Iterable[Tuple[str, datetime]]) -> Dict[str, datetime]:
    """Most recent watch time per video_id."""
    best: Dict[str, datetime] = {}
    for vid, ts in rows:
        cur = best.get(vid)
        if cur is None or ts > cur:
            best[vid] = ts
    return best


def _watch_url(video_id: str) -> str:
    return f"https://youtube.com/watch?v={video_id}"


def run_import(
    db_path: Path,
    best_by_vid: Dict[str, datetime],
    dry_run: bool,
) -> Tuple[int, int, int]:
    """
    Returns (updated, skipped_already_set, inserted_new_rows).
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT video_id, watch_date FROM videos")
    db_rows = cur.fetchall()
    db_map = {r[0]: r[1] for r in db_rows}

    updated = 0
    skipped = 0
    inserted = 0

    for vid, ts in best_by_vid.items():
        val = _fmt_db_ts(ts)
        url = _watch_url(vid)
        if vid not in db_map:
            if dry_run:
                inserted += 1
            else:
                cur.execute(
                    "INSERT INTO videos (video_id, watch_date, url) VALUES (?, ?, ?)",
                    (vid, val, url),
                )
                inserted += cur.rowcount
            continue
        wd = db_map[vid]
        if wd is not None and str(wd).strip() != "":
            skipped += 1
            continue
        if dry_run:
            updated += 1
        else:
            cur.execute(
                "UPDATE videos SET watch_date = ? WHERE video_id = ? "
                "AND (watch_date IS NULL OR trim(watch_date) = '')",
                (val, vid),
            )
            if cur.rowcount:
                updated += 1
    if not dry_run:
        conn.commit()
    conn.close()
    return updated, skipped, inserted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--zip",
        type=Path,
        help="Path to takeout-*.zip (default: newest takeout-*.zip in Downloads)",
    )
    ap.add_argument(
        "--takeout-root",
        type=Path,
        help="Folder that contains Takeout/ (skip --zip extraction if set)",
    )
    ap.add_argument(
        "--history-file",
        type=Path,
        help="Direct path to watch-history.json or .html (skips zip and takeout-root)",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"SQLite database path (default: {DEFAULT_DB})",
    )
    ap.add_argument(
        "--downloads",
        type=Path,
        default=DEFAULT_DOWNLOADS,
        help=f"Folder to search for takeout-*.zip (default: {DEFAULT_DOWNLOADS})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and report counts only; do not write to the database",
    )
    args = ap.parse_args()

    tmp_extract: Optional[Path] = None
    try:
        if args.history_file:
            hist = args.history_file.expanduser().resolve()
            if not hist.is_file():
                print(f"ERROR: history file not found: {hist}", file=sys.stderr)
                return 1
        elif args.takeout_root:
            root = args.takeout_root.expanduser().resolve()
            hist = locate_history_file(root)
        else:
            zip_path = (
                args.zip.expanduser().resolve()
                if args.zip
                else find_latest_takeout_zip(args.downloads)
            )
            print(f"Using Takeout zip: {zip_path}")
            takeout_parent, owns = extract_takeout_zip(zip_path)
            tmp_extract = takeout_parent if owns else None
            hist = locate_history_file(takeout_parent)

        fmt, rows = parse_history(hist)
        print(f"History file: {hist}")
        print(f"Format: {fmt}")
        print(f"Parsed watch entries (video + timestamp): {len(rows)}")

        best = latest_watch_per_video(rows)
        print(f"Unique video IDs in history: {len(best)}")

        if not args.db.is_file():
            print(f"ERROR: database not found: {args.db}", file=sys.stderr)
            return 1

        updated, skipped, inserted = run_import(args.db, best, args.dry_run)
        mode = "DRY-RUN (no DB writes)" if args.dry_run else "IMPORT"
        print(f"\n{mode} summary (per unique video_id vs videos table):")
        print(f"  Would update / updated rows (NULL watch_date): {updated}")
        print(f"  Would insert / inserted rows (new video_id):   {inserted}")
        print(f"  Skipped (watch_date already set):             {skipped}")
        return 0
    finally:
        if tmp_extract is not None and tmp_extract.exists():
            shutil.rmtree(tmp_extract, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
