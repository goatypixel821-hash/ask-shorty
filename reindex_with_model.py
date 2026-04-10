#!/usr/bin/env python3
"""
Rebuild Chroma using a specified SentenceTransformer embedding model.

Creates a NEW Chroma directory (does not overwrite the default minilm store).
Writes the model id to data/chroma_model.txt so transcript_rag_enhanced picks it up at runtime.

Usage:
  python reindex_with_model.py \\
    --db-path "C:/path/to/transcripts.db" \\
    --model BAAI/bge-large-en-v1.5 \\
    --chroma-dir data/transcript_chroma_bge
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _videos_with_shorty(db_path: str):
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
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


def _index_one_video_shared(db, rag, db_path: str, video_id: str) -> None:
    from reindex_all import _index_one_video_shared as _shared

    _shared(db, rag, db_path, video_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild Chroma with a chosen embedding model into a new directory."
    )
    parser.add_argument("--db-path", type=str, required=True, help="Path to transcripts.db")
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model id (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        required=True,
        help="New Chroma persistent directory (must not be the active store if you want to compare)",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db_path)
    chroma_dir = Path(args.chroma_dir)
    if not chroma_dir.is_absolute():
        chroma_dir = (SCRIPT_DIR / chroma_dir).resolve()

    if not os.path.isfile(db_path):
        print(f"[reindex_with_model] DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    model_name = args.model.strip()
    data_dir = SCRIPT_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_file = data_dir / "chroma_model.txt"
    model_file.write_text(model_name + "\n", encoding="utf-8")
    chroma_dir.mkdir(parents=True, exist_ok=True)
    sidecar = chroma_dir / "chroma_model.txt"
    sidecar.write_text(model_name + "\n", encoding="utf-8")

    print(f"DB:          {db_path}")
    print(f"Chroma:      {chroma_dir}")
    print(f"Model:       {model_name}")
    print(f"Wrote:       {model_file}")
    print(f"Sidecar:     {sidecar}")

    from transcript_database import TranscriptDatabase
    from transcript_rag import TranscriptRAG

    print("Loading embedding model and Chroma (one-time)...")
    db = TranscriptDatabase(db_path)
    rag = TranscriptRAG(
        transcript_db=str(db_path),
        chroma_dir=str(chroma_dir),
        embedding_model_name=model_name,
    )
    print("Ready.\n")

    videos = list(_videos_with_shorty(db_path))
    total = len(videos)
    if total == 0:
        print("No videos with Shorty — nothing to index.")
        return

    print(f"Indexing {total} videos (with Shorty).\n")
    num_ok = 0
    num_fail = 0
    for index, (video_id, title) in enumerate(videos, 1):
        short_title = (title[:55] + "...") if len(title) > 55 else title
        print(f"video {index} of {total}: {short_title}", flush=True)
        try:
            _index_one_video_shared(db, rag, db_path, video_id)
            num_ok += 1
        except Exception as e:
            print(f"  ! {e}", file=sys.stderr)
            num_fail += 1

    print(f"\nDone. OK={num_ok} failed={num_fail}")
    print(f"Chroma docs: {rag.collection.count()}")


if __name__ == "__main__":
    main()
