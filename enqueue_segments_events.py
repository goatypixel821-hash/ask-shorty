#!/usr/bin/env python3
"""
Enqueue segments + events processing_queue rows for videos that have a Shorty
but do not already have a pending or completed queue row for that task.

Does not modify existing rows (only INSERTs new pending tasks).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent / "data" / "transcripts.db"


def resolve_db_path(cli: str | None) -> Path:
    if cli:
        return Path(cli)
    env = (os.environ.get("ASK_SHORTY_DB_PATH") or "").strip()
    if env:
        return Path(env)
    return DEFAULT_DB


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enqueue segments and events tasks for videos with Shorty but missing active queue rows."
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to transcripts.db (default: ASK_SHORTY_DB_PATH or shorty data/transcripts.db)",
    )
    args = parser.parse_args()

    db_path = resolve_db_path(args.db_path)
    if not db_path.is_file():
        raise SystemExit(f"Database not found: {db_path}")

    sql_segments = """
    INSERT INTO processing_queue (video_id, task, status)
    SELECT DISTINCT t.video_id, 'segments', 'pending'
    FROM transcripts t
    WHERE t.shorty IS NOT NULL
      AND TRIM(t.shorty) != ''
      AND NOT EXISTS (
        SELECT 1 FROM processing_queue pq
        WHERE pq.video_id = t.video_id
          AND pq.task = 'segments'
          AND pq.status IN ('pending', 'completed')
      )
    """

    sql_events = """
    INSERT INTO processing_queue (video_id, task, status)
    SELECT DISTINCT t.video_id, 'events', 'pending'
    FROM transcripts t
    WHERE t.shorty IS NOT NULL
      AND TRIM(t.shorty) != ''
      AND NOT EXISTS (
        SELECT 1 FROM processing_queue pq
        WHERE pq.video_id = t.video_id
          AND pq.task = 'events'
          AND pq.status IN ('pending', 'completed')
      )
    """

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(sql_segments)
    n_segments = cur.rowcount
    cur.execute(sql_events)
    n_events = cur.rowcount
    conn.commit()
    conn.close()

    print(f"DB: {db_path}")
    print(f"New pending rows inserted for task 'segments': {n_segments}")
    print(f"New pending rows inserted for task 'events':   {n_events}")


if __name__ == "__main__":
    main()
