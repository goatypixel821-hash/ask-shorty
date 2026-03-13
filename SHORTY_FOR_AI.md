# Ask Shorty — Shorty for Another AI

**Project:** Ask Shorty. Makes video/podcast transcripts queryable via a dense “Shorty” per video + RAG. URL → transcript → Shorty + synthetic Qs + entities → SQLite + Chroma → ask questions → multi-layer retrieval → Claude answer.

**Stack:** Python, Flask, SQLite (`data/transcripts.db`), Chroma (`data/transcript_chroma`), SentenceTransformer `all-MiniLM-L6-v2`, Anthropic Claude (and optional OpenAI-compatible API for batch).

**DB (SQLite):** `videos` (video_id, title, channel, has_transcript, …); `transcripts` (video_id, text, **shorty**, …); `entities` (video_id, name, type, aliases JSON); `synthetic_questions` (video_id, question); `processing_queue` (video_id, task: shorty|synthetic_questions|entities, status: pending|started|completed|failed). Shorty lives in **transcripts.shorty**.

**Chroma:** Collection `transcripts`, cosine. Metadata `type`: chunk | shorty | synthetic_question; `video_id`, chunk_index for chunks. **Queue-mode batch does NOT re-index Chroma** (avoids os._exit() kill); reindex separately; SQLite is source of truth.

**Key files:** `transcript_database.py` (schema, migrations, enqueue_processing_tasks); `shorty_generator.py` (Anthropic Shorty + synthetic Qs); `entity_extractor.py` (Anthropic tool-use **or** OpenAI-compatible JSON path: ENTITY_JSON_* prompts, parse_entities_from_json); `transcript_rag.py` → `transcript_rag_enhanced.py` (chunking, index_single_transcript, hybrid search); `ask_shorty.py` (rewrite query, metadata filter, 3-layer retrieval, Claude answer); `batch_processor.py` (--queue: process_queue_tasks from processing_queue; or legacy batch: videos needing shorties → Shorty+Qs+entities+Chroma); `enqueue_backfill.py` (find transcript+no shorty, skip if already in queue, enqueue 3 tasks; --dry-run); `video_grabber.py` (bookmarklet, port 5000); `ask_shorty_app.py` (UI port 5001, /ask, /api/ask); `library_app.py` (library port 5002).

**Env:** ANTHROPIC_API_KEY; OpenAI-compatible: OPENAI_API_KEY, OPENAI_BASE_URL (default http://localhost:8000/v1), OPENAI_MODEL. GRABBER_PORT=5000.

**Commands:** `python start_grabber.py`; `python ask_shorty_app.py` (5001); `python library_app.py` (5002); `python enqueue_backfill.py [--db-path] [--dry-run]`; `python batch_processor.py --queue [--limit N] [--provider anthropic|openai-compatible]`; `python entity_extractor.py --openai` (test entity JSON path).

**Gotchas:** Queue path skips Chroma re-index; entity extraction with openai-compatible uses JSON-only (no tool-use); reindex script (reindex_all / reindex_missing_videos) may need to be implemented to repopulate Chroma from SQLite.
