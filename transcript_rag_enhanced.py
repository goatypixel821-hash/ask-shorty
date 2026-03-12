#!/usr/bin/env python3
"""
Enhanced RAG System for YouTube Transcripts
- Chunking with overlap
- Hybrid search (semantic + keyword)
- Query expansion
- Re-indexing support
"""

import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import re
import sys
import threading
from typing import List, Dict, Tuple, Optional

# Global shared model instance to prevent multiple simultaneous initializations
_shared_model = None
_model_lock = threading.Lock()

# Global shared Chroma client to avoid race conditions on initialization
_shared_chroma_client = None
_chroma_lock = threading.Lock()

# Safe print function for Windows console encoding
def safe_print(*args, **kwargs):
    """Print that handles emoji encoding errors on Windows"""
    try:
        text = ' '.join(str(arg) for arg in args)
        if 'end' not in kwargs:
            text += '\n'
        elif kwargs.get('end') != '':
            text += kwargs.get('end', '\n')
        sys.stdout.write(text)
        sys.stdout.flush()
    except (UnicodeEncodeError, OSError):
        # Replace emojis with plain text alternatives
        text = ' '.join(str(arg) for arg in args)
        replacements = {
            '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARN]',
            '🔍': '[SEARCH]', '📥': '[QUEUE]', '🚀': '[START]',
            '💾': '[SAVE]', '📋': '[INFO]', '🎯': '[TARGET]',
            '📊': '[STATS]', '📹': '[VIDEO]', '📝': '[TEXT]',
            '📄': '[DOC]', '🎉': '[DONE]', 'ℹ️': '[INFO]',
            '🤔': '[QUESTION]', '🤖': '[AI]', '📚': '[SOURCES]'
        }
        for emoji, replacement in replacements.items():
            text = text.replace(emoji, replacement)
        text = text.encode('ascii', 'replace').decode('ascii')
        if 'end' not in kwargs:
            text += '\n'
        elif kwargs.get('end') != '':
            text += kwargs.get('end', '\n')
        sys.stdout.write(text)
        sys.stdout.flush()

class EnhancedTranscriptRAG:
    """Enhanced RAG system with chunking, hybrid search, and query expansion"""
    
    def __init__(self, transcript_db=None, chroma_dir=None):
        # Get base directory relative to this script
        base_dir = Path(__file__).parent
        
        if transcript_db is None:
            self.transcript_db = base_dir / 'data' / 'transcripts.db'
        else:
            self.transcript_db = Path(transcript_db)
            
        if chroma_dir is None:
            self.chroma_dir = base_dir / 'data' / 'transcript_chroma'
        else:
            self.chroma_dir = Path(chroma_dir)
        
        # Use shared model instance to prevent PyTorch tensor conflicts
        global _shared_model, _model_lock
        with _model_lock:
            if _shared_model is None:
                _shared_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_model = _shared_model

        # Use shared Chroma client with a lock to avoid race conditions
        global _shared_chroma_client, _chroma_lock
        with _chroma_lock:
            if _shared_chroma_client is None:
                _shared_chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self.chroma_client = _shared_chroma_client

        self.collection = self.chroma_client.get_or_create_collection(
            name="transcripts",
            metadata={"hnsw:space": "cosine"}
        )

        # SQLite connection string
        self._db_path = str(self.transcript_db)
        
        safe_print("✅ Enhanced Transcript RAG initialized")
        safe_print(f"📄 Transcript DB: {self.transcript_db}")
        safe_print(f"📚 Chroma directory: {self.chroma_dir}")

    # ---------- Embedding helpers ----------

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using the shared SentenceTransformer model."""
        if not texts:
            return []
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    # ---------- Chroma indexing ----------

    def index_single_transcript(
        self,
        video_id: str,
        text: str,
        shorty: Optional[str] = None,
        synthetic_questions: Optional[List[str]] = None,
    ) -> None:
        """
        Index a single video's transcript into Chroma, along with optional
        Shorty and synthetic questions.

        - Transcript is split into chunks and stored with type="chunk".
        - Shorty (if provided) stored with type="shorty".
        - Synthetic questions stored with type="synthetic_question".
        """
        if not text or not text.strip():
            safe_print(f"[WARN] Empty transcript for {video_id}, skipping index.")
            return

        # IMPORTANT: clear any existing items for this video_id first so we
        # don't hit duplicate ID errors and always get a clean upsert.
        try:
            self.collection.delete(where={"video_id": video_id})
        except Exception as e:
            safe_print(f"[WARN] Failed to delete existing Chroma items for {video_id}: {e}")

        # Basic paragraph / sentence-ish splitting with overlap
        chunks = self._chunk_transcript(text)
        if not chunks:
            chunks = [text.strip()]

        # Index transcript chunks
        chunk_ids = [f"{video_id}:chunk:{i}" for i in range(len(chunks))]
        chunk_metadatas = [
            {
                "video_id": video_id,
                "type": "chunk",
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        chunk_embeddings = self._embed_texts(chunks)

        self.collection.add(
            ids=chunk_ids,
            embeddings=chunk_embeddings,
            metadatas=chunk_metadatas,
            documents=chunks,
        )

        # Index Shorty if provided
        if shorty and shorty.strip():
            shorty_id = f"{video_id}:shorty"
            shorty_emb = self._embed_texts([shorty])[0]
            self.collection.add(
                ids=[shorty_id],
                embeddings=[shorty_emb],
                metadatas=[
                    {
                        "video_id": video_id,
                        "type": "shorty",
                    }
                ],
                documents=[shorty],
            )

        # Index synthetic questions if provided
        if synthetic_questions:
            clean_qs = [q.strip() for q in synthetic_questions if q and q.strip()]
            if clean_qs:
                q_ids = [f"{video_id}:sq:{i}" for i in range(len(clean_qs))]
                q_embs = self._embed_texts(clean_qs)
                q_metas = [
                    {
                        "video_id": video_id,
                        "type": "synthetic_question",
                        "index": i,
                    }
                    for i in range(len(clean_qs))
                ]
                self.collection.add(
                    ids=q_ids,
                    embeddings=q_embs,
                    metadatas=q_metas,
                    documents=clean_qs,
                )

                # Persist embedding IDs back to SQLite synthetic_questions table
                self._update_synthetic_question_ids(video_id, clean_qs, q_ids)

    def _chunk_transcript(self, text: str, max_chars: int = 800, overlap: int = 200) -> List[str]:
        """
        Naive chunking by characters with overlap.

        This avoids pulling in the original project's more complex logic while
        still giving reasonable chunks for retrieval.
        """
        text = text.strip()
        if len(text) <= max_chars:
            return [text]

        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + max_chars, length)
            chunk = text[start:end]
            chunks.append(chunk.strip())
            if end == length:
                break
            start = max(0, end - overlap)
        return [c for c in chunks if c]

    def _update_synthetic_question_ids(
        self,
        video_id: str,
        questions: List[str],
        embedding_ids: List[str],
    ) -> None:
        """
        Update the synthetic_questions table with embedding_ids for each question.

        Matching is done positionally for the latest inserted questions for this video.
        """
        if not questions or not embedding_ids:
            return
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            # Fetch the most recent N synthetic questions for this video (no embedding_id yet or any)
            cursor.execute(
                """
                SELECT id, question
                FROM synthetic_questions
                WHERE video_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (video_id, len(questions)),
            )
            rows = cursor.fetchall()
            # Reverse to align oldest->newest with our order
            rows = list(reversed(rows))
            for (row_id, q_text), emb_id in zip(rows, embedding_ids):
                cursor.execute(
                    """
                    UPDATE synthetic_questions
                    SET embedding_id = ?
                    WHERE id = ?
                    """,
                    (emb_id, row_id),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            safe_print(f"[WARN] Failed to update synthetic question embedding_ids for {video_id}: {e}")


