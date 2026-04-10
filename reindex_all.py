#!/usr/bin/env python3
"""
Rebuild Chroma from SQLite for all videos that have a Shorty.

Chroma reindexing of Shorties disabled on Windows due to os._exit() crash - SQLite search used instead.

- Takes --db-path argument.
- Loops through all videos in SQLite that have a shorty.
- For each video: index transcript chunks, shorty text, synthetic questions (in-process).
- Progress: "video X of Y: title".

Usage:
  python reindex_all.py --db-path data/transcripts.db
  python reindex_all.py --db-path data/transcripts.db --video-id ABC123  # single-video only
"""

import argparse
import os
import sys
from pathlib import Path

# Project root = directory containing this script
SCRIPT_DIR = Path(__file__).resolve().parent


def _chroma_dir_for_db(db_path: str) -> Path:
    """Derive Chroma directory from DB path (e.g. data/transcripts.db -> data/transcript_chroma_new)."""
    return Path(db_path).resolve().parent / "transcript_chroma_new"


def _videos_to_index(db_path: str, all_transcripts: bool = False):
    """Yield (video_id, title) for videos to index.

    all_transcripts=False  → only videos that already have a Shorty (original behavior)
    all_transcripts=True   → all videos that have any transcript text (chunk-level search)
    """
    import sqlite3
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        if all_transcripts:
            cur.execute(
                """
                SELECT DISTINCT t.video_id, COALESCE(v.title, t.video_id) AS title
                FROM transcripts t
                LEFT JOIN videos v ON v.video_id = t.video_id
                WHERE t.text IS NOT NULL AND length(trim(t.text)) > 0
                ORDER BY t.video_id
                """
            )
        else:
            cur.execute(
                """
                SELECT DISTINCT t.video_id, COALESCE(v.title, t.video_id) AS title
                FROM transcripts t
                LEFT JOIN videos v ON v.video_id = t.video_id
                WHERE t.shorty IS NOT NULL AND t.shorty != ''
                ORDER BY t.video_id
                """
            )
        for row in cur.fetchall():
            yield row["video_id"], (row["title"] or row["video_id"])


# Keep old name as alias for backward compatibility
def _videos_with_shorty(db_path: str):
    yield from _videos_to_index(db_path, all_transcripts=False)


def _index_one_video_shared(db, rag, db_path: str, video_id: str) -> None:
    """Index one video using shared DB and RAG instances (fast path for bulk runs)."""
    import sqlite3

    info = db.get_transcript_and_shorty(video_id)
    if not info or not info.get("text"):
        raise ValueError(f"No transcript text for {video_id}")
    text = info["text"]
    shorty = (info.get("shorty") or "").strip() or None

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT question FROM synthetic_questions WHERE video_id = ? ORDER BY created_at ASC",
            (video_id,),
        )
        questions = [row[0] for row in cur.fetchall() if row[0]]

    rag.index_single_transcript(
        video_id,
        text,
        shorty=shorty,
        synthetic_questions=questions if questions else None,
    )


def _index_one_video(db_path: str, chroma_dir: Path, video_id: str) -> None:
    """Load RAG and DB, then index this one video (used for single-video mode only)."""
    from transcript_database import TranscriptDatabase
    from transcript_rag import TranscriptRAG

    db = TranscriptDatabase(db_path)
    rag = TranscriptRAG(transcript_db=str(db_path), chroma_dir=str(chroma_dir))
    _index_one_video_shared(db, rag, db_path, video_id)


def main() -> None:
    FULL_DB = "C:/Users/number2/Desktop/youtube-history-viewer-copy/data/transcripts.db"

    parser = argparse.ArgumentParser(description="Rebuild Chroma from SQLite for videos with Shorty (or all transcripts).")
    parser.add_argument(
        "--db-path",
        type=str,
        default=FULL_DB,
        help="Path to transcripts.db (default: full corpus DB)",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Index only this video.",
    )
    parser.add_argument(
        "--all-transcripts",
        action="store_true",
        default=False,
        help="Index ALL videos with transcript text, not just those with a Shorty.",
    )
    args = parser.parse_args()
    db_path = os.path.abspath(args.db_path)
    chroma_dir = _chroma_dir_for_db(db_path)

    if not os.path.isfile(db_path):
        print(f"[reindex_all] DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"DB:     {db_path}")
    print(f"Chroma: {chroma_dir}")

    # Single-video mode: run in this process and exit
    if args.video_id:
        _index_one_video(db_path, chroma_dir, args.video_id)
        return

    # Full run — create shared DB + RAG once, reuse across all videos
    from transcript_database import TranscriptDatabase
    from transcript_rag import TranscriptRAG

    print("Loading embedding model and Chroma (one-time)...")
    db = TranscriptDatabase(db_path)
    rag = TranscriptRAG(transcript_db=str(db_path), chroma_dir=str(chroma_dir))
    print("Ready.\n")

    videos = list(_videos_to_index(db_path, all_transcripts=args.all_transcripts))
    total = len(videos)

    if args.all_transcripts:
        print(f"Found {total} videos with any transcript text.")
    else:
        print(f"Found {total} videos with Shorty.")
    if total == 0:
        print("Nothing to do.")
        return

    num_processed = 0
    num_failed = 0

    for index, (video_id, title) in enumerate(videos, 1):
        short_title = (title[:55] + "...") if len(title) > 55 else title
        print(f"video {index} of {total}: {short_title}", flush=True)

        try:
            _index_one_video_shared(db, rag, db_path, video_id)
            num_processed += 1
        except Exception as e:
            print(f"  ! {e}", file=sys.stderr)
            num_failed += 1

    print("\nDone.")
    print(f"Summary: {num_processed} processed, {num_failed} failed.")
    chroma_count = rag.collection.count()
    print(f"Chroma collection now has {chroma_count} vectors.")


if __name__ == "__main__":
    main()
