#!/usr/bin/env python3
"""
Populate the processing queue for videos that need LLM backfill.

Default mode: videos with transcript text but no Shorty — enqueues:
  shorty → synthetic_questions → entities → segments → events → triples

--task triples: Shorty present but no facts — triples only
--task segments: Shorty present but no segment rows — segments only
--task events: segments present but no events — events only

Usage:
  python enqueue_backfill.py [--db-path PATH] [--dry-run]
  python enqueue_backfill.py --task triples [--db-path PATH] [--dry-run]
  python enqueue_backfill.py --task segments [--db-path PATH] [--dry-run]
  python enqueue_backfill.py --task events [--db-path PATH] [--dry-run]
"""

import argparse
import sqlite3
import sys
from typing import List, Tuple

from transcript_database import TranscriptDatabase


def get_candidates_no_shorty(conn: sqlite3.Connection) -> List[Tuple[str]]:
    """Videos that have a transcript (non-empty text) but no Shorty."""
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


def get_candidates_triples_only(conn: sqlite3.Connection) -> List[Tuple[str]]:
    """Videos with Shorty but no rows in facts."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT DISTINCT t.video_id
            FROM transcripts t
            WHERE t.shorty IS NOT NULL AND trim(t.shorty) != ''
              AND NOT EXISTS (SELECT 1 FROM facts f WHERE f.video_id = t.video_id)
            ORDER BY t.video_id
            """
        )
        return cursor.fetchall()
    except sqlite3.OperationalError:
        return []


def get_candidates_segments_only(conn: sqlite3.Connection) -> List[Tuple[str]]:
    """Videos with Shorty but no HSC segments yet."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT DISTINCT t.video_id
            FROM transcripts t
            WHERE t.shorty IS NOT NULL AND trim(t.shorty) != ''
              AND NOT EXISTS (SELECT 1 FROM segments s WHERE s.video_id = t.video_id)
            ORDER BY t.video_id
            """
        )
        return cursor.fetchall()
    except sqlite3.OperationalError:
        return []


def get_candidates_events_only(conn: sqlite3.Connection) -> List[Tuple[str]]:
    """Videos with at least one segment but no events row."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT DISTINCT s.video_id
            FROM segments s
            WHERE NOT EXISTS (SELECT 1 FROM events e WHERE e.video_id = s.video_id)
            ORDER BY s.video_id
            """
        )
        return cursor.fetchall()
    except sqlite3.OperationalError:
        return []


def has_any_queue_tasks(conn: sqlite3.Connection, video_id: str) -> bool:
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


def has_pending_task(conn: sqlite3.Connection, video_id: str, task: str) -> bool:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 1 FROM processing_queue
        WHERE video_id = ? AND task = ? AND status = 'pending'
        LIMIT 1
        """,
        (video_id, task),
    )
    return cursor.fetchone() is not None


def enqueue_tasks(db: TranscriptDatabase, video_id: str, tasks: List[str], dry_run: bool) -> bool:
    if dry_run:
        return True
    db.enqueue_processing_tasks(video_id, tasks=tasks)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enqueue backfill tasks for videos missing Shorties, HSC layers, or triples."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/transcripts.db",
        help="Path to transcripts SQLite database",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["default", "triples", "segments", "events"],
        default="default",
        help="default=no Shorty full pipeline; triples/segments/events=targeted backfill",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be enqueued without writing to the database",
    )
    args = parser.parse_args()

    db = TranscriptDatabase(db_path=args.db_path)

    with sqlite3.connect(db.db_path) as conn:
        if args.task == "triples":
            candidates = get_candidates_triples_only(conn)
        elif args.task == "segments":
            candidates = get_candidates_segments_only(conn)
        elif args.task == "events":
            candidates = get_candidates_events_only(conn)
        else:
            candidates = get_candidates_no_shorty(conn)

    if not candidates:
        print("No matching candidates for this --task mode.")
        return

    enqueued = 0
    skipped = 0
    would_enqueue_ids: List[str] = []

    task_map = {
        "triples": ["triples"],
        "segments": ["segments"],
        "events": ["events"],
    }
    single = task_map.get(args.task)

    for (video_id,) in candidates:
        with sqlite3.connect(db.db_path) as conn:
            if args.task == "default" and has_any_queue_tasks(conn, video_id):
                skipped += 1
                continue
            if single and has_pending_task(conn, video_id, single[0]):
                skipped += 1
                continue

        if args.dry_run:
            would_enqueue_ids.append(video_id)
        if args.task == "default":
            enqueue_tasks(db, video_id, tasks=None, dry_run=args.dry_run)
        else:
            enqueue_tasks(db, video_id, tasks=single or [], dry_run=args.dry_run)
        enqueued += 1

    n_tasks = 6 if args.task == "default" else 1
    if args.dry_run:
        print(
            "[DRY RUN] Would enqueue %d videos (%d tasks each where applicable)."
            % (enqueued, n_tasks)
        )
        if would_enqueue_ids:
            preview = ", ".join(would_enqueue_ids[:20])
            if len(would_enqueue_ids) > 20:
                preview += ", ..."
            print("Video IDs: %s" % preview)
        print("Skipped %d." % skipped)
    else:
        print("Total candidates: %d" % len(candidates))
        print("Enqueued: %d videos." % enqueued)
        print("Skipped: %d." % skipped)


if __name__ == "__main__":
    main()
    sys.exit(0)
