# Ask Shorty: Dense Transcript Compression for High‑Recall RAG

## 1. Motivation

Modern LLM applications rely heavily on Retrieval‑Augmented Generation (RAG) to answer questions over external data such as documents, chats, videos, and podcasts. YouTube and podcast transcripts are a particularly important source: they contain high‑value technical discussions, news, and educational material, but are long, noisy, and difficult to search.

Typical “RAG over transcripts” systems run into several problems:

- **Low signal per chunk** – transcripts include filler, repetition, digressions, and incomplete sentences, which dilute embedding quality.
- **Fragmented facts** – key facts and causal chains are split across multiple chunks, so a single retrieved chunk often misses crucial context.
- **Recall failures** – even with good embeddings, important details (numbers, names, versions, dates) are easy to miss when they appear only once in hours of speech.
- **Scaling limits** – storing and searching only raw text chunks becomes expensive and slow as libraries grow.

Ask Shorty was built from a practical need: searching a personal YouTube watch history and being able to ask, “Where did I hear X?” and “What exactly did they say about Y?” without re‑watching entire videos. What emerged is a more general, research‑grade component: a dense, machine‑oriented compression layer that sits alongside traditional RAG and dramatically improves recall.

## 2. Core Idea: The Shorty

At the center of Ask Shorty is the **Shorty**: a dense, structured compression of a single transcript optimized for LLM consumption rather than human readability.

A Shorty is:

- A 90–97% token reduction of the original transcript.
- Designed to retain ~95% of answerable information: entities, numbers, causal chains, “micro‑details”, and key claims.
- Stored as a separate field (`transcripts.shorty`) in SQLite and optionally vectorized into Chroma as its own document type.

Shorties are not conventional summaries. They aim to be lossy for humans but near‑lossless for LLMs:

- Preserve all entities (people, organizations, systems).
- Preserve all important numbers (dates, counts, sizes, versions).
- Preserve relationships and causal chains (who did what to what, and what happened).
- Separate facts from commentary/interpretation.
- Include a small **MICRO‑DETAILS** section for hard‑to‑recover but important details (software names, model names, specific legal terms, etc.).

In practice, a Shorty acts as a semantic index for the entire transcript: an LLM reading only the Shorty can answer most questions that could be answered from the original video.

## 3. System Overview

### 3.1 High‑Level Flow

Ask Shorty turns videos into queryable knowledge objects through the following pipeline:

**Ingest**

- A browser bookmarklet (grabber service) captures a YouTube URL and uses `simple_transcript_fetcher.py` plus `youtube-transcript-api` to fetch the transcript.
- Metadata and transcripts are stored in SQLite (`videos`, `transcripts`).

**Enqueue processing**

- `enqueue_backfill.py` scans for videos with transcripts but no Shorty and enqueues three tasks per video in `processing_queue`: `shorty`, `synthetic_questions`, `entities`.

**Generate Shorty, synthetic questions, and entities**

- `batch_processor.py` (queue mode) processes tasks using Anthropic or an OpenAI‑compatible API:
  - `shorty_generator.py` creates the Shorty and synthetic questions via LLM.
  - `entity_extractor.py` extracts entities via either Anthropic tool‑use or a JSON‑only OpenAI‑compatible path.
- Results are written back into SQLite (`transcripts.shorty`, `synthetic_questions`, `entities`).
- In legacy batch mode, the same script can also index vectors in Chroma, but in queue mode Chroma reindexing is decoupled to avoid `os._exit` issues.

**Index for retrieval**

- `transcript_rag_enhanced.py` handles chunking and indexing into Chroma (path `data/transcript_chroma`).
- Three main vector types share one collection (`transcripts`, cosine distance):
  - Transcript chunks (`type="chunk"`, with `video_id`, `chunk_index`).
  - Shorties (`type="shorty"`, one per video).
  - Synthetic questions (`type="synthetic_question"`).

**Query and answer**

- `ask_shorty.py` receives a user query (via Flask app `ask_shorty_app.py`).
- It optionally parses the question for channel/date filters, narrowing candidate `video_id`s via SQLite (`TranscriptDatabase`).
- A query rewriting step uses Claude to generate multiple alternate phrasings.
- It performs multi‑angle retrieval over Chroma: chunks, Shorties, synthetic questions, optionally filtered by `video_id`.
- Retrieved context is assembled and passed to Claude for the final answer, with citations back to videos/chunks.

**Browse and debug**

- `library_app.py` provides a simple UI for browsing videos, inspecting their Shorties, synthetic questions, and entities, and manually enqueueing processing tasks.
- SQLite (`data/transcripts.db`) is the source of truth; Chroma is a derived index that can be rebuilt from SQLite when needed.

## 4. Data Model

### 4.1 SQLite Schema

Key tables in `transcript_database.py`:

- **videos**
  - `video_id` (PK), `title`, `channel`, `url`, `has_transcript`, `transcript_fetched_at`, `watch_date`, `local_path`, `json_metadata`, `created_at`.
- **transcripts**
  - `id`, `video_id`, `text`, `language`, `confidence`, `shorty`, `shorty_generated_at`, `created_at`.
  - Shorty lives here, not in `videos`.
- **entities**
  - `id`, `video_id`, `name`, `type`, `aliases` (JSON string array), `created_at`.
- **synthetic_questions**
  - `id`, `video_id`, `question`, `embedding_id`, `created_at`.
- **processing_queue**
  - `id`, `video_id`, `task` (`shorty` \| `synthetic_questions` \| `entities`), `status` (`pending` \| `started` \| `completed` \| `failed`), timestamps, and `error`.

This schema supports both batch and interactive workflows and makes it easy to rebuild derived indexes.

### 4.2 Chroma Index

Chroma is used as a vector index under `data/transcript_chroma` with a single `transcripts` collection (cosine distance).

Each record includes:

- **embedding** – from SentenceTransformer `all-MiniLM-L6-v2`.
- **metadata** – at minimum:
  - `type` ∈ { `"chunk"`, `"shorty"`, `"synthetic_question"` }.
  - `video_id`.
  - `chunk_index` (for `chunk` only).

This design enables multi‑representation retrieval: the same video is represented as many chunks, one Shorty, and many synthetic questions.

## 5. Retrieval Pipeline

Ask Shorty’s query pipeline is designed to improve recall and robustness by combining several techniques.

### 5.1 Metadata Filtering

Before vector search, `ask_shorty.py` can parse a question for channel/date constraints and use SQLite to restrict the set of candidate `video_id`s. This avoids searching the entire corpus when the user implicitly references a specific channel or time window.

### 5.2 Query Rewriting

A single natural language question is often ambiguous or under‑specified. Ask Shorty therefore uses Claude with tool‑use to generate multiple alternate phrasings of the user query (typically 3–4). These rewrites capture:

- Different synonyms and formulations.
- Explicit mention of channels, people, or systems inferred from context.
- Variants that match how synthetic questions were phrased at index time.

Each rewrite is used as a separate query into Chroma.

### 5.3 Multi‑Representation Retrieval

For each rewritten query, Ask Shorty queries Chroma three times:

- **Shorty hits** (`type="shorty"`)
  - Identify which videos are globally relevant to the question.
  - Provide high‑density, whole‑video context.
- **Synthetic question hits** (`type="synthetic_question"`)
  - Match the user’s question against questions that were generated from the transcript at index time.
  - This is conceptually similar to “hypothetical question indexing”: matching question‑to‑question rather than question‑to‑raw text.
- **Chunk hits** (`type="chunk"`)
  - Retrieve local transcript segments that contain exact phrases, quotes, or detailed steps.

The results are merged and ranked by similarity (and potentially by type‑specific weights), yielding a small set of videos and chunks with strong evidence for answering the question.

### 5.4 Answer Synthesis

Ask Shorty then composes a context block including:

- Excerpts from the top Shorties (for global structure).
- The most relevant synthetic questions (to show why a video answers this query).
- Selected transcript chunks (for quotes and fine detail).

This aggregated context is sent to Claude, which produces the final answer with references/citations back to the underlying videos and segments.

By treating Shorties and synthetic questions as first‑class retrieval targets, Ask Shorty avoids many of the recall problems of pure chunk‑based RAG, especially for long, meandering transcripts.

## 6. Design Rationale

### 6.1 Why Dense Compression?

Storing only raw transcript chunks leads to several failure modes:

- **Concept dilution:** embeddings dominated by filler conversation rather than key facts.
- **Missing causal chains:** causes and consequences appear in different chunks.
- **Poor long‑range recall:** questions about “overall argument” or “big picture” are difficult when each chunk is narrow.

Shorties address this by packing entities + numbers + causal chains + micro‑details into a single dense representation per video. This makes it easy for both LLMs and the vector index to grasp “what this video is about” in a small token budget.

### 6.2 Why Multi‑Representation Retrieval?

Different representations support different query types:

| Layer              | Strength                                      |
|--------------------|-----------------------------------------------|
| Transcript chunks  | Exact phrasing, local detail                  |
| Shorty             | Global structure, cross‑chunk relationships   |
| Synthetic questions| Question‑to‑question matching, helps ambiguity|

Indexing all three and searching across them significantly improves recall compared to RAG that only uses chunks.

### 6.3 Why SQLite as Source of Truth?

Ask Shorty uses SQLite (`data/transcripts.db`) as the canonical store for all semantic artifacts (transcripts, Shorties, entities, synthetic questions), with Chroma as a derived index. This has several benefits:

- Easy to inspect and debug via standard SQL tools.
- Robust against vector index failures or format changes.
- Enables offline or batch reindexing scripts (for example, `reindex_all.py`) that rebuild Chroma from SQLite when needed.

### 6.4 Why a Queue‑Based Batch Processor?

Shorties and synthetic questions are relatively expensive to generate. Ask Shorty uses `processing_queue` and `batch_processor.py` in queue mode to process them asynchronously:

- New videos can be ingested quickly without blocking on LLM calls.
- Failed tasks can be retried and audited.
- Chroma reindexing is kept out of the queue loop because some Chroma code paths call `os._exit`, which can terminate the whole worker; instead, reindexing is handled in separate scripts.

## 7. Implementation Notes

### 7.1 Key Components

- `transcript_database.py` – SQLite schema, migrations, and enqueue helpers.
- `shorty_generator.py` – LLM client for generating Shorty and synthetic questions.
- `entity_extractor.py` – entity extraction via Anthropic tool‑use or OpenAI‑compatible JSON path.
- `transcript_rag.py` / `transcript_rag_enhanced.py` – chunking logic and Chroma integration.
- `batch_processor.py` – queue and legacy batch modes for generating Shorties, synthetic questions, and entities.
- `enqueue_backfill.py` – backfill script to enqueue processing tasks for existing transcripts.
- `video_grabber.py` / `start_grabber.py` – bookmarklet‑driven ingest service.
- `ask_shorty.py` – query pipeline (rewrite, retrieval, answer).
- `ask_shorty_app.py` – simple Flask UI for asking questions.
- `library_app.py` – admin/library UI to browse videos and their Shorties/entities/questions.

### 7.2 Environment & Dependencies

From the existing project summary:

- Python stack using Flask, SQLite, `sentence-transformers`, `chromadb`, `youtube-transcript-api`, `yt-dlp`, and Anthropic/OpenAI SDKs.
- Data locations:
  - SQLite: `data/transcripts.db`.
  - Chroma: `data/transcript_chroma` (collection `transcripts`).
- Environment variables:
  - `ANTHROPIC_API_KEY` – required for Shorty, synthetic questions, entities on Anthropic path, and for query rewriting/answers in `ask_shorty`.
  - `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` – for OpenAI‑compatible paths (entity extraction, batch runs).
  - `GRABBER_PORT` – default 5000 (bookmarklet ingest service).

## 8. Current Status and Future Work

Ask Shorty is currently a working prototype with:

- End‑to‑end ingest → Shorty synthesis → indexing → query answering.
- A personal YouTube history as an initial corpus and main use‑case.
- Ongoing improvements and bug fixes in both the pipeline and UI.

Planned and possible extensions:

- **Standardized Shorty template** – enforce sections like CONTEXT, INCIDENTS, ATTACK FLOW, IMPACT, MICRO‑DETAILS, TIMELINE to maximize recoverability.
- **Fact triples / knowledge graph layer** – an additional table of (subject, relation, object) facts per video for graph queries and higher‑level reasoning.
- **Confidence signals** – expose retrieval scores and video rankings to help users understand why particular sources were selected.
- **Public API / hosted service** – expose ingest and query endpoints so others can build on the Shorty + RAG model.

## 9. Conclusion

Ask Shorty demonstrates that adding a dense, machine‑oriented compression layer on top of conventional RAG substantially improves question‑answering over long, messy transcripts. By combining:

- Shorties (90–97% token reduction, ~95% information retention),
- Entities and synthetic questions as structured side‑channels,
- SQLite as a transparent source of truth and Chroma as a multi‑representation vector index,
- And a query pipeline that uses metadata filtering, query rewriting, and multi‑angle retrieval,

Ask Shorty moves beyond “just chunk the transcript” and toward a more robust, research‑grade approach to LLM knowledge retrieval over video and podcast content.

