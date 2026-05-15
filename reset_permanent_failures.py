#!/usr/bin/env python3
"""
Reset processing_queue rows in status permanently_failed back to pending
for one task type (e.g. segments) so batch_processor can retry them.
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset permanently_failed processing_queue rows to pending for a given task."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "transcripts.db"),
        help="Path to transcripts.db",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task column value to filter (e.g. segments, shorty, triples).",
    )
    args = parser.parse_args()
    db_path = Path(args.db_path)
    task = str(args.task).strip()
    if not task:
        print("Error: --task must be non-empty.", file=sys.stderr)
        sys.exit(1)

    if not db_path.is_file():
        print(f"Error: DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
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

    set_parts = ["status = 'pending'"]
    if "started_at" in cols:
        set_parts.append("started_at = NULL")
    if "completed_at" in cols:
        set_parts.append("completed_at = NULL")
    if "error" in cols:
        set_parts.append("error = NULL")
    if "attempts" in cols:
        set_parts.append("attempts = 0")

    sql = (
        "UPDATE processing_queue SET "
        + ", ".join(set_parts)
        + " WHERE status = 'permanently_failed' AND task = ?"
    )
    cur.execute(sql, (task,))
    reset_count = cur.rowcount
    conn.commit()
    conn.close()

    print(
        f"Reset {reset_count} permanently_failed row(s) with task={task!r} to pending."
    )


if __name__ == "__main__":
    main()
