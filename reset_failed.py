#!/usr/bin/env python3
"""
Reset all failed tasks in the processing_queue back to pending so they can be retried.
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset failed processing_queue tasks to pending.")
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "transcripts.db"),
        help="Path to transcripts.db",
    )
    parser.add_argument(
        "--show-stuck",
        action="store_true",
        help="Show video_id and error for all failed tasks instead of resetting them.",
    )
    args = parser.parse_args()
    db_path = args.db_path

    if not Path(db_path).is_file():
        print(f"Error: DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if args.show_stuck:
        cur.execute(
            "SELECT video_id, task, error, status FROM processing_queue WHERE status IN ('failed', 'permanently_failed') ORDER BY status, video_id, task"
        )
        rows = cur.fetchall()
        if not rows:
            print("No failed or permanently_failed tasks in queue.")
        else:
            for status_label in ("failed", "permanently_failed"):
                subset = [r for r in rows if r[3] == status_label]
                if not subset:
                    continue
                print(f"{status_label} tasks ({len(subset)}):")
                for video_id, task, error, _ in subset:
                    err = (error or "").strip() or "(no error message)"
                    print(f"  {video_id}  {task}: {err}")
        conn.close()
        return

    cur.execute(
        "UPDATE processing_queue SET status = 'pending' WHERE status = 'failed'"
    )
    reset_count = cur.rowcount
    conn.commit()
    conn.close()

    print(f"Reset {reset_count} task(s) from failed to pending.")


if __name__ == "__main__":
    main()
