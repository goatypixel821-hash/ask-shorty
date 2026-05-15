#!/usr/bin/env python3
"""
Print progress stats for the transcript/Shorty pipeline:
- Videos with Shorty count
- Videos with triples (distinct videos that have rows in facts)
- Queue: pending, failed, completed (all tasks)
- Queue breakdown for triples only
- Optional: per-task queue summary (pending / failed / completed)
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, DefaultDict, Tuple


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    )
    return cur.fetchone() is not None


def _aggregate_queue(cur: sqlite3.Cursor) -> Tuple[Dict[str, int], DefaultDict[str, DefaultDict[str, int]]]:
    """Overall status counts and nested task -> status -> count."""
    cur.execute("SELECT status, COUNT(*) AS n FROM processing_queue GROUP BY status")
    overall: Dict[str, int] = {row[0]: row[1] for row in cur.fetchall()}
    by_task: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
    cur.execute(
        """
        SELECT task, status, COUNT(*) AS n
        FROM processing_queue
        GROUP BY task, status
        """
    )
    for row in cur.fetchall():
        task, status, n = row[0], row[1], row[2]
        by_task[task][status] = n
    return overall, by_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Shorty/queue/triples progress for a transcripts DB.")
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "transcripts.db"),
        help="Path to transcripts.db",
    )
    args = parser.parse_args()
    db_path = args.db_path

    if not Path(db_path).is_file():
        print(f"Error: DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Videos with Shorty (distinct video_id in transcripts with non-empty shorty)
    cur.execute(
        """
        SELECT COUNT(DISTINCT video_id) AS n
        FROM transcripts
        WHERE shorty IS NOT NULL AND TRIM(shorty) != ''
        """
    )
    videos_with_shorty = cur.fetchone()[0]

    print("Videos with Shorty count:", videos_with_shorty)

    if _table_exists(cur, "facts"):
        cur.execute("SELECT COUNT(DISTINCT video_id) AS n FROM facts")
        videos_with_triples = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) AS n FROM facts")
        fact_rows = cur.fetchone()[0]
        print("Videos with triples (facts) count:", videos_with_triples)
        print("Total fact rows (all videos):", fact_rows)
    else:
        print("Videos with triples (facts) count: (no facts table)")

    if not _table_exists(cur, "processing_queue"):
        print("No processing_queue table in this DB.")
        conn.close()
        return

    overall, by_task = _aggregate_queue(cur)

    pending = overall.get("pending", 0)
    failed = overall.get("failed", 0)
    perm_failed = overall.get("permanently_failed", 0)
    completed = overall.get("completed", 0)
    started = overall.get("started", 0)  # rare if worker crashed mid-task

    print("Still pending in queue count:", pending)
    print("Failed in queue count (retryable):", failed)
    if perm_failed:
        print("Permanently failed in queue count:", perm_failed)
    print("Completed in queue count:", completed)
    if started:
        print("Started (in-flight / stale) count:", started)

    triples = by_task.get("triples", {})
    if triples:
        print("")
        print("--- Queue: triples only ---")
        print("  pending:", triples.get("pending", 0))
        print("  failed:", triples.get("failed", 0))
        if triples.get("permanently_failed"):
            print("  permanently_failed:", triples.get("permanently_failed", 0))
        print("  completed:", triples.get("completed", 0))
        if triples.get("started"):
            print("  started:", triples.get("started", 0))
    else:
        print("")
        print("--- Queue: triples only --- (no triples rows in queue)")

    print("")
    print("--- Queue: by task (pending / failed / perm_failed / completed) ---")
    task_names = sorted(by_task.keys())
    for task in task_names:
        t = by_task[task]
        p = t.get("pending", 0)
        f = t.get("failed", 0)
        pf = t.get("permanently_failed", 0)
        c = t.get("completed", 0)
        print(f"  {task}: pending={p} failed={f} perm_failed={pf} completed={c}")

    conn.close()


if __name__ == "__main__":
    main()
