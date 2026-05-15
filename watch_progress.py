#!/usr/bin/env python3
"""
Watch batch_processor progress in real time.
Run this in any terminal — reads the local DB and pings the Ollama tunnel.

  python watch_progress.py          # refresh every 30 s
  python watch_progress.py --every 10  # faster
"""
import argparse
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path

DB = Path(r"C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db")
OLLAMA_URL = "http://127.0.0.1:11434/v1/models"
TASKS_PER_VIDEO = 4  # shorty + synthetic_questions + entities + triples


def safe(text: str, width: int = 0) -> str:
    """Strip characters that Windows cp1252 can't print, then optionally truncate."""
    cleaned = text.encode("ascii", errors="replace").decode("ascii")
    if width and len(cleaned) > width:
        cleaned = cleaned[:width] + "..."
    return cleaned


def check_tunnel() -> str:
    try:
        with urllib.request.urlopen(OLLAMA_URL, timeout=4) as r:
            if r.status == 200:
                return "OK"
            return f"HTTP {r.status}"
    except urllib.error.URLError as e:
        return f"DOWN ({e.reason})"
    except Exception as e:
        return f"DOWN ({e})"


def query_db() -> dict:
    if not DB.exists():
        return {}
    try:
        conn = sqlite3.connect(str(DB), timeout=5)
        c = conn.cursor()

        # Queue totals
        c.execute("SELECT status, COUNT(*) FROM processing_queue GROUP BY status")
        by_status = dict(c.fetchall())

        total   = sum(by_status.values())
        done    = by_status.get("completed", 0)
        pending = by_status.get("pending", 0)
        started = by_status.get("started", 0)
        failed  = by_status.get("failed", 0)

        # Current in-progress video
        c.execute("""
            SELECT pq.video_id, pq.task, pq.started_at,
                   COALESCE(v.title, pq.video_id) AS title
            FROM processing_queue pq
            LEFT JOIN videos v ON v.video_id = pq.video_id
            WHERE pq.status = 'started'
            ORDER BY pq.started_at DESC
            LIMIT 1
        """)
        row = c.fetchone()
        current = row if row else None

        # Completion rate: tasks completed in last 10 minutes
        ten_min_ago = (datetime.now() - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
        c.execute("""
            SELECT COUNT(*) FROM processing_queue
            WHERE status = 'completed'
            AND completed_at IS NOT NULL
            AND completed_at >= ?
        """, (ten_min_ago,))
        recent_done = c.fetchone()[0]

        # Completed videos (all 4 tasks done)
        c.execute("""
            SELECT COUNT(*) FROM (
                SELECT video_id
                FROM processing_queue
                GROUP BY video_id
                HAVING SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) = COUNT(*)
            )
        """)
        videos_done = c.fetchone()[0]

        c.execute("SELECT COUNT(DISTINCT video_id) FROM processing_queue")
        videos_total = c.fetchone()[0]

        # Recently completed (last 5)
        c.execute("""
            SELECT COALESCE(v.title, pq.video_id), pq.task, pq.completed_at
            FROM processing_queue pq
            LEFT JOIN videos v ON v.video_id = pq.video_id
            WHERE pq.status = 'completed' AND pq.completed_at IS NOT NULL
            ORDER BY pq.completed_at DESC LIMIT 5
        """)
        recent = c.fetchall()

        # Recent failures
        c.execute("""
            SELECT pq.video_id, pq.task, pq.error
            FROM processing_queue pq
            WHERE pq.status = 'failed' AND pq.error IS NOT NULL
            ORDER BY pq.id DESC LIMIT 3
        """)
        recent_fails = c.fetchall()

        conn.close()
        return {
            "total": total, "done": done, "pending": pending,
            "started": started, "failed": failed,
            "current": current,
            "recent_done_10m": recent_done,
            "videos_done": videos_done,
            "videos_total": videos_total,
            "recent": recent,
            "recent_fails": recent_fails,
        }
    except Exception as e:
        return {"error": str(e)}


def eta_str(remaining_tasks: int, rate_per_min: float) -> str:
    if rate_per_min <= 0:
        return "unknown"
    mins = remaining_tasks / rate_per_min
    if mins < 60:
        return f"{mins:.0f} min"
    hours = mins / 60
    if hours < 24:
        return f"{hours:.1f} hr"
    return f"{hours/24:.1f} days"


def render(d: dict, tunnel: str, refresh_s: int) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Clear screen (works on Windows and Linux)
    print("\033[2J\033[H", end="")

    print("=" * 64)
    print(f"  Ask Shorty — batch progress monitor    {now}")
    print("=" * 64)

    if "error" in d:
        print(f"\n  [DB ERROR] {d['error']}\n")
        return

    # Tunnel health
    tunnel_icon = "[OK]" if tunnel == "OK" else "[!!]"
    print(f"  Tunnel to Qwen:  {tunnel_icon} {tunnel}")
    print()

    # Task counts
    total  = d["total"]
    done   = d["done"]
    failed = d["failed"]
    pct    = (done / total * 100) if total else 0
    bar_w  = 40
    filled = int(bar_w * done / total) if total else 0
    bar    = "#" * filled + "-" * (bar_w - filled)

    print(f"  Tasks  [{bar}] {pct:.1f}%")
    print(f"         {done:,} done / {total:,} total   "
          f"({d['pending']:,} pending, {d['started']} active, {failed:,} failed)")
    print()

    # Video-level progress
    vd = d["videos_done"]
    vt = d["videos_total"]
    vpct = (vd / vt * 100) if vt else 0
    print(f"  Videos  {vd:,} / {vt:,} fully done  ({vpct:.1f}%)")
    print()

    # ETA (based on last 10 min rate)
    recent10 = d["recent_done_10m"]
    rate_per_min = recent10 / 10.0
    remaining = total - done
    print(f"  Rate (last 10 min): {rate_per_min:.1f} tasks/min  "
          f"({rate_per_min * 60:.0f}/hr)")
    print(f"  ETA (tasks left):   {eta_str(remaining, rate_per_min)}")
    print()

    # Currently in progress
    cur = d.get("current")
    if cur:
        vid_id, task, started_at, title = cur
        short_title = safe(title, 50)
        print(f"  Now:  [{task}]  {short_title}")
        print(f"        video_id: {vid_id}   started: {started_at or 'unknown'}")
    else:
        print("  Now:  (nothing active — may be between batches)")
    print()

    # Recent completed
    if d["recent"]:
        print("  Recent completions:")
        for title, task, ts in d["recent"]:
            short = safe(title or "?", 45)
            print(f"    {ts or '':>19}  [{task:<22}]  {short}")
    print()

    # Failures
    if d["recent_fails"]:
        print("  Recent failures:")
        for vid, task, err in d["recent_fails"]:
            short_err = safe((err or ""), 60)
            print(f"    [{task}] {vid}  -> {short_err}")
        print()

    print(f"  Refreshing every {refresh_s}s — Ctrl+C to stop")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch batch_processor progress")
    ap.add_argument("--every", type=int, default=30, help="Refresh interval in seconds (default 30)")
    args = ap.parse_args()

    print("Starting monitor — first refresh in 2 s...")
    time.sleep(2)

    while True:
        tunnel = check_tunnel()
        data   = query_db()
        render(data, tunnel, args.every)
        try:
            time.sleep(args.every)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
