#!/usr/bin/env python3
"""
Populate the processing queue for videos that have a transcript but no Shorty yet.

Finds videos with transcript text but no Shorty, skips those that already have
pending or completed queue tasks, and enqueues 3 tasks per video: shorty,
synthetic_questions, entities.

Usage:
  python enqueue_backfill.py [--db-path PATH] [--dry-run]
"""

import argparse
import sqlite3
import sys
from typing import List, Tuple

from transcript_database import TranscriptDatabase


def get_candidates(conn: sqlite3.Connection) -> List[Tuple[str]]:
    """Videos that have a transcript (with non-empty text) but no Shorty."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT DISTINCT v.video_id
        FROM videos v
        JOIN transcripts t ON t.video_id = v.video_id
        WHERE (t.shorty IS NULL OR trim(coalesce(t.shorty, '')) = '')
          AND t.text IS NOT NULL AND trim(t.text) != ''
        ORDER BY v.video_id
        """
    )
    return cursor.fetchall()


def has_any_queue_tasks(conn: sqlite3.Connection, video_id: str) -> bool:
    """True if this video has any pending or completed tasks in the queue."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 1 FROM processing_queue
        WHERE video_id = ? AND status IN ('pending', 'completed')
        LIMIT 1
        """,
        (video_id,),
    )
    return cursor.fetchone() is not None


def enqueue_video(db: TranscriptDatabase, video_id: str, dry_run: bool) -> bool:
    """Enqueue shorty, synthetic_questions, entities for one video. Returns True if enqueued."""
    if dry_run:
        return True
    db.enqueue_processing_tasks(video_id, tasks=["shorty", "synthetic_questions", "entities"])
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enqueue backfill tasks for videos that have transcript but no Shorty."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/transcripts.db",
        help="Path to transcripts SQLite database",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be enqueued without writing to the database",
    )
    args = parser.parse_args()

    db = TranscriptDatabase(db_path=args.db_path)

    with sqlite3.connect(db.db_path) as conn:
        candidates = get_candidates(conn)
        total_found = len(candidates)

    if total_found == 0:
        print("No videos found that have a transcript but no Shorty.")
        return

    enqueued = 0
    skipped = 0
    would_enqueue_ids: List[str] = []

    for (video_id,) in candidates:
        with sqlite3.connect(db.db_path) as conn:
            if has_any_queue_tasks(conn, video_id):
                skipped += 1
                continue
        if args.dry_run:
            would_enqueue_ids.append(video_id)
        enqueue_video(db, video_id, args.dry_run)
        enqueued += 1

    if args.dry_run:
        print("[DRY RUN] Would enqueue %d videos (3 tasks each = %d tasks)." % (enqueued, enqueued * 3))
        if would_enqueue_ids:
            preview = ", ".join(would_enqueue_ids[:20])
            if len(would_enqueue_ids) > 20:
                preview += ", ..."
            print("Video IDs: %s" % preview)
        print("Skipped %d (already have pending/completed queue tasks)." % skipped)
    else:
        print("Total candidates (transcript, no Shorty): %d" % total_found)
        print("Enqueued: %d videos (%d tasks)." % (enqueued, enqueued * 3))
        print("Skipped: %d (already in queue)." % skipped)


if __name__ == "__main__":
    main()
    sys.exit(0)
