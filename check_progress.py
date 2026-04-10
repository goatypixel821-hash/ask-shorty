#!/usr/bin/env python3
"""
Print progress stats for the transcript/Shorty pipeline:
- Videos with Shorty count
- Queue: pending, failed, completed
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Shorty/queue progress for a transcripts DB.")
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

    # Queue counts by status
    cur.execute(
        "SELECT status, COUNT(*) AS n FROM processing_queue GROUP BY status"
    )
    status_counts = {row["status"]: row["n"] for row in cur.fetchall()}
    pending = status_counts.get("pending", 0)
    failed = status_counts.get("failed", 0)
    completed = status_counts.get("completed", 0)

    conn.close()

    print("Videos with Shorty count:", videos_with_shorty)
    print("Still pending in queue count:", pending)
    print("Failed in queue count:", failed)
    print("Completed in queue count:", completed)


if __name__ == "__main__":
    main()
