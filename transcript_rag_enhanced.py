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
import ollama
from pathlib import Path
import json
import re
import sys
import threading
from typing import List, Dict, Tuple, Optional

# Global shared model instance to prevent multiple simultaneous initializations
_shared_model = None
_model_lock = threading.Lock()

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
        
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.chroma_client.get_or_create_collection(
            name="transcripts",
            metadata={"hnsw:space": "cosine"}
        )
        
        safe_print("✅ Enhanced Transcript RAG initialized")
        safe_print(f"📄 Transcript DB: {self.transcript_db}")
        safe_print(f"📚 Chroma directory: {self.chroma_dir}")

    # ... original EnhancedTranscriptRAG implementation omitted for brevity ...

