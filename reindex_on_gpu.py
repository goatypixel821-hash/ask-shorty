#!/usr/bin/env python3
"""
Standalone Chroma reindexer — runs on vast.ai (no project imports needed).

Reads transcripts + shorties + synthetic questions directly from SQLite,
embeds them with a local SentenceTransformer model, and writes to Chroma.

Usage (on vast.ai):
  python reindex_on_gpu.py \
      --db      /workspace/transcripts.db \
      --model   /workspace/shorty_embedding_model \
      --chroma  /workspace/transcript_chroma_finetuned

Then scp the chroma folder back to your local machine.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Chunking  (mirrors transcript_rag_enhanced.py)
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = 800, overlap: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Index one video
# ---------------------------------------------------------------------------

def index_video(
    collection,
    model,
    video_id: str,
    text: str,
    shorty: Optional[str],
    questions: List[str],
) -> int:
    """Embed and upsert all layers for one video. Returns total docs upserted."""
    total = 0

    # Chunks
    chunks = chunk_text(text) or [text[:800]]
    chunk_ids   = [f"{video_id}:chunk:{i}" for i in range(len(chunks))]
    chunk_metas = [{"video_id": video_id, "type": "chunk", "chunk_index": i}
                   for i in range(len(chunks))]
    chunk_embs  = model.encode(chunks, show_progress_bar=False).tolist()
    collection.upsert(ids=chunk_ids, embeddings=chunk_embs,
                      metadatas=chunk_metas, documents=chunks)
    total += len(chunks)

    # Shorty
    if shorty and shorty.strip():
        s = shorty.strip()
        emb = model.encode([s], show_progress_bar=False).tolist()
        collection.upsert(
            ids=[f"{video_id}:shorty"],
            embeddings=emb,
            metadatas=[{"video_id": video_id, "type": "shorty"}],
            documents=[s],
        )
        total += 1

    # Synthetic questions
    clean_qs = [q.strip() for q in questions if q and q.strip()]
    if clean_qs:
        q_ids   = [f"{video_id}:sq:{i}" for i in range(len(clean_qs))]
        q_metas = [{"video_id": video_id, "type": "synthetic_question", "index": i}
                   for i in range(len(clean_qs))]
        q_embs  = model.encode(clean_qs, show_progress_bar=False).tolist()
        collection.upsert(ids=q_ids, embeddings=q_embs,
                          metadatas=q_metas, documents=clean_qs)
        total += len(clean_qs)

    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone GPU reindexer for Ask Shorty.")
    parser.add_argument("--db",     required=True, help="Path to transcripts.db")
    parser.add_argument("--model",  required=True, help="SentenceTransformer model dir or HF id")
    parser.add_argument("--chroma", required=True, help="Output Chroma directory (created if needed)")
    parser.add_argument("--batch",  type=int, default=256,
                        help="Embed this many texts at once (larger = faster on GPU, default 256)")
    parser.add_argument("--limit",  type=int, default=None,
                        help="Only index first N videos (for testing)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos already indexed in Chroma (for resuming interrupted runs)")
    args = parser.parse_args()

    db_path    = args.db
    model_path = args.model
    chroma_dir = args.chroma

    if not Path(db_path).exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Ask Shorty — GPU Reindexer")
    print("=" * 60)
    print(f"DB     : {db_path}")
    print(f"Model  : {model_path}")
    print(f"Chroma : {chroma_dir}")
    print()

    # Load model
    print("Loading embedding model …")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path)
    print(f"  Device: {model.device}")

    # Open Chroma
    print("Opening Chroma collection …")
    import chromadb
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    col = client.get_or_create_collection(
        name="transcripts",
        metadata={"hnsw:space": "cosine"},
    )
    existing_count = col.count()
    print(f"  Existing vectors: {existing_count}")

    # Write model name sidecar so local reindex_with_model / rag_enhanced picks it up
    sidecar = Path(chroma_dir) / "chroma_model.txt"
    sidecar.write_text(str(Path(model_path).resolve()), encoding="utf-8")

    # Load videos from SQLite
    print("Loading videos from SQLite …")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT DISTINCT t.video_id, COALESCE(v.title, t.video_id) AS title,
               t.text, t.shorty
        FROM transcripts t
        LEFT JOIN videos v ON v.video_id = t.video_id
        WHERE t.text IS NOT NULL AND length(trim(t.text)) > 0
          AND t.video_id != 'test123'
        ORDER BY t.video_id
        """
    )
    videos = cur.fetchall()

    # Load all synthetic questions into a dict
    cur.execute("SELECT video_id, question FROM synthetic_questions WHERE question IS NOT NULL")
    synq_map: dict = {}
    for row in cur.fetchall():
        synq_map.setdefault(row["video_id"], []).append(row["question"])
    conn.close()

    if args.limit:
        videos = videos[:args.limit]

    total_videos = len(videos)
    print(f"  Videos to index: {total_videos}")
    print()

    total_docs = 0
    failed = 0
    skipped = 0

    for idx, row in enumerate(videos, 1):
        vid   = row["video_id"]
        title = row["title"] or vid
        text  = row["text"] or ""
        shorty = row["shorty"] or ""
        questions = synq_map.get(vid, [])

        short_title = (title[:55] + "...") if len(title) > 55 else title

        if args.skip_existing:
            # Check if this video already has at least one chunk indexed
            existing = col.get(ids=[f"{vid}:chunk:0"], include=[])
            if existing and existing.get("ids"):
                print(f"[{idx}/{total_videos}] SKIP {short_title}", flush=True)
                skipped += 1
                continue

        print(f"[{idx}/{total_videos}] {short_title}", flush=True)

        try:
            n = index_video(col, model, vid, text, shorty or None, questions)
            total_docs += n
        except Exception as e:
            print(f"  ! ERROR: {e}", file=sys.stderr)
            failed += 1

    print()
    print("=" * 60)
    print(f"Done.  Videos: {total_videos - failed - skipped} indexed, {skipped} skipped, {failed} failed.")
    print(f"Total vectors in Chroma: {col.count()}")
    print(f"Output: {chroma_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
