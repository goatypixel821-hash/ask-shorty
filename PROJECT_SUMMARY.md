# Ask Shorty — Project Summary

## What It Is

**Ask Shorty** makes video and podcast content queryable: you add transcripts, the system builds a dense “Shorty” per video and indexes everything; you ask questions and get answers with citations.

- **Core idea:** A “Shorty” is a dense, machine-oriented compression of a transcript (~90–97% token reduction) that keeps ~95% of answerable information (entities, numbers, causal chains, micro-details). It is used as an extra retrieval layer alongside normal RAG chunks.
- **Flow:** URL → transcript → Shorty + synthetic questions + entities → stored in SQLite and (optionally) vectorized in Chroma → query → multi-angle retrieval (chunks + Shorties + synthetic questions) → LLM answer.

---

## Repo Layout

| Path | Role |
|------|------|
| `transcript_database.py` | SQLite DB: videos, transcripts (with shorty column), entities, synthetic_questions, processing_queue. Creates tables and runs lightweight migrations. |
| `shorty_generator.py` | Anthropic: generates Shorty and synthetic questions (tool-use / JSON). Used by batch and library. |
| `entity_extractor.py` | **Dual path:** (1) Anthropic tool-use for Shorty; (2) **OpenAI-compatible:** JSON-only prompts + `parse_entities_from_json()`. Exports `extract_entities`, `store_entities`, `ENTITY_JSON_*`, `parse_entities_from_json`. |
| `transcript_rag.py` | Thin wrapper; imports `EnhancedTranscriptRAG` from `transcript_rag_enhanced.py` as `TranscriptRAG`. |
| `transcript_rag_enhanced.py` | Chroma + SentenceTransformer (`all-MiniLM-L6-v2`). Chunking, hybrid search. `index_single_transcript(video_id, transcript_text, shorty=..., synthetic_questions=...)` stores types: `chunk`, `shorty`, `synthetic_question`. DB path `data/transcripts.db`, Chroma under `data/transcript_chroma`. |
| `ask_shorty.py` | Query pipeline: query rewriting (Claude tool-use), metadata filter (channel/date → candidate video_ids), 3-layer retrieval (chunks, Shorties, synthetic questions), then Claude answer. Uses `TranscriptRAG` and `TranscriptDatabase`. |
| `ask_shorty_app.py` | Flask app (port **5001**): `GET /ask` (UI), `POST /api/ask` (JSON), `GET /debug/videos`, `GET /debug/video/<id>`. |
| `video_grabber.py` | Flask bookmarklet service (port **5000**, `GRABBER_PORT`): `POST /api/fetch-transcript`, `POST /api/save-pasted-transcript`. Saves transcript to DB, can trigger background vectorization. Uses `TranscriptDatabase`, `TranscriptRAG`, `SimpleTranscriptFetcher`, `VideoDownloader`. |
| `start_grabber.py` | Launches the video grabber (e.g. `python start_grabber.py`). |
| `batch_processor.py` | **Two modes:** (1) **Legacy batch:** `get_videos_needing_shorties()` → for each video generate Shorty, synthetic questions, entities, then index into Chroma. (2) **Queue mode (`--queue`):** `get_pending_queue_tasks()` → process tasks (shorty, synthetic_questions, entities) from `processing_queue`; **Chroma re-indexing is NOT run in queue path** (Chroma can call `os._exit()` and kill the process). Run a separate reindex script after queue processing; SQLite is source of truth. Supports `--provider anthropic` (default) and `--provider openai-compatible` (with `--base-url`, `--model`); openai path uses JSON prompts for entities and plain chat for Shorty/synthetic Qs. |
| `enqueue_backfill.py` | Backfill script: finds videos with transcript but no Shorty; skips videos that already have pending/completed rows in `processing_queue`; enqueues 3 tasks per video (shorty, synthetic_questions, entities). `--db-path`, `--dry-run`. |
| `library_app.py` | Flask app (port **5002**): library browser and admin (paginated videos, channel filter, has_shorty / missing_shorty, per-video Shorty/questions/entities, enqueue from UI). Uses Anthropic path for Shorty/questions/entities. |
| `anthropic_client.py` | Reads `ANTHROPIC_API_KEY`, exposes `get_client()` (Anthropic SDK). |
| `simple_transcript_fetcher.py` | Fetches YouTube transcript via youtube-transcript-api; uses `TranscriptDatabase`. |
| `video_downloader.py` | yt-dlp-based downloader. |

---

## Database (SQLite)

- **Default path:** `data/transcripts.db`
- **Tables:**
  - **videos:** `video_id` (PK), title, channel, url, has_transcript, transcript_fetched_at, watch_date, local_path, json_metadata, created_at.
  - **transcripts:** id, video_id, text, language, confidence, **shorty**, shorty_generated_at, created_at.
  - **entities:** id, video_id, name, type, aliases (JSON array text), created_at.
  - **synthetic_questions:** id, video_id, question, embedding_id, created_at.
  - **processing_queue:** id, video_id, task (`shorty` | `synthetic_questions` | `entities`), status (`pending` | `started` | `completed` | `failed`), created_at, started_at, completed_at, error.

Shorty lives in **transcripts.shorty**, not in the videos table. “No Shorty” = transcript row exists but `shorty` IS NULL or empty.

---

## Chroma (Vector Index)

- **Path:** `data/transcript_chroma`
- **Collection:** `transcripts` (cosine).
- **Stored types (metadata `type`):** `chunk`, `shorty`, `synthetic_question`.
- **Metadata:** includes `video_id`; chunks also have `chunk_index`.
- **Important:** In queue mode, batch_processor does **not** call Chroma (re-indexing was removed to avoid process kill). Rebuild Chroma separately (e.g. reindex script from SQLite) when needed.

---

## Environment / Config

- **Anthropic:** `ANTHROPIC_API_KEY` (required for Shorty, synthetic Qs, entities on Anthropic path, and for ask_shorty query/rewrite/answer).
- **OpenAI-compatible (batch + entity test):** `OPENAI_API_KEY`, `OPENAI_BASE_URL` or `OPENAI_API_BASE` (default `http://localhost:8000/v1`), `OPENAI_MODEL` (default `gpt-3.5-turbo`).
- **Grabber:** `GRABBER_PORT` (default 5000).
- **Ports:** Grabber 5000, Ask Shorty UI 5001, Library 5002.

---

## Commands (concise)

- **Grabber:** `python start_grabber.py` or `python video_grabber.py`
- **Ask Shorty UI:** `python ask_shorty_app.py` → http://localhost:5001/ask
- **Library:** `python library_app.py` → http://localhost:5002
- **Backfill queue:** `python enqueue_backfill.py [--db-path PATH] [--dry-run]`
- **Process queue:** `python batch_processor.py --queue [--limit N] [--provider anthropic|openai-compatible] [--base-url URL] [--model NAME]`
- **Legacy batch (no queue):** `python batch_processor.py [--limit N] [--retry-failed]` (prompts for confirmation; runs Shorty + synthetic Qs + entities + Chroma index).
- **Entity extraction test (OpenAI-compatible):** `python entity_extractor.py --openai` (hardcoded sample transcript, prints raw response and parsed entities).

---

## Entity Extraction (OpenAI-Compatible)

- Batch processor uses **ENTITY_JSON_SYSTEM_PROMPT** and **ENTITY_JSON_USER_TEMPLATE** and **parse_entities_from_json(raw)** so no tool-use is required.
- If entities are empty: debug logs in batch (raw API response, parse result, exceptions) and in `parse_entities_from_json` (JSONDecodeError snippet, “parsed N items but 0 had valid name/type”).

---

## Query Pipeline (ask_shorty.py)

1. Optional metadata filter: parse question for channel/date → SQLite → candidate video_ids.
2. Query rewriting: Claude tool-use → 3–4 alternate phrasings.
3. Retrieval: chunks (type=chunk), Shorties (type=shorty), synthetic questions (type=synthetic_question); optional filter by video_ids.
4. Context assembled and sent to Claude for final answer.

---

## Gotchas

- **Queue mode does not re-index Chroma.** Run a separate reindex step after queue processing if you need Chroma up to date.
- **Chroma/reindex:** Some Chroma code paths can call `os._exit()` and kill the process; re-indexing was removed from the queue loop for this reason.
- **Entity extraction with OpenAI-compatible:** Use JSON-only path (already wired in batch + entity_extractor); tool-use is Anthropic-only.
- **Reindex script:** README/comments mention “reindex_all.py” or “reindex_missing_videos.py”; if missing, you need to add a script that reads from SQLite and calls `rag.index_single_transcript()` for each video (or equivalent).

---

## Dependencies (from README)

- flask, anthropic, chromadb, sentence-transformers, youtube-transcript-api, yt-dlp (and openai for openai-compatible path).
