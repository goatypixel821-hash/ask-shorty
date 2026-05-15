#!/usr/bin/env python3
"""
Reset processing_queue rows stuck in status 'started' back to 'pending'.

These are usually orphaned when a batch_processor worker is killed (Ctrl+C, crash,
SSH drop) after claiming a task but before writing completed/failed.
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset stale 'started' processing_queue rows to pending."
    )
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
    cur = conn.cursor()

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='processing_queue' LIMIT 1"
    )
    if cur.fetchone() is None:
        print("Error: no processing_queue table in this database.", file=sys.stderr)
        conn.close()
        sys.exit(1)

    cur.execute("PRAGMA table_info(processing_queue)")
    cols = {row[1] for row in cur.fetchall()}

    if "started_at" in cols:
        cur.execute(
            """
            UPDATE processing_queue
            SET status = 'pending', started_at = NULL
            WHERE status = 'started'
            """
        )
    else:
        cur.execute(
            "UPDATE processing_queue SET status = 'pending' WHERE status = 'started'"
        )

    reset_count = cur.rowcount
    conn.commit()
    conn.close()

    print(f"Reset {reset_count} stale started task(s) to pending.")


if __name__ == "__main__":
    main()
