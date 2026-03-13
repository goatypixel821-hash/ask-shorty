# Ask Shorty

> *Ask anything. Get the answer. No more scrubbing through videos.*

---

## The Problem

Video and podcast content is hard to search. You either watch everything or miss things. Standard search only hits titles and descriptions — not what was actually said. Plain RAG over raw transcripts helps a bit but often:

- Misses key details (names, numbers, versions)
- Splits causal chains across multiple chunks
- Returns noisy context filled with filler and digressions

Ask Shorty was built to fix this for a real use case: searching a personal YouTube history and asking *“Where did I hear X?”* without re‑watching hours of content.

---

## What Ask Shorty Does

Ask Shorty makes any video or podcast instantly queryable:

- Drop in a link (e.g. YouTube)
- Ask a question
- Get a precise answer with context and citations

Under the hood, Ask Shorty adds a **dense compression layer** on top of normal RAG. Each video gets a **Shorty**: a compact, machine‑oriented intelligence brief that sits alongside transcript chunks and synthetic questions in the index.

---

## The Shorty

The **Shorty** is the core innovation.

- Generated once per transcript at index time
- ~90–97 % token reduction vs. the original
- Designed to retain ~95 % of answerable information, including:
  - All named entities, systems, and people
  - Causal chains and relationships
  - Key numbers, dates, and technical details
  - Speaker framing and interpretation
  - Micro‑details that typical summarization drops

Shorties are stored in SQLite (`transcripts.shorty`) and also vectorized into Chroma as their own document type (`type="shorty"`).

They do **not** replace RAG over transcript chunks – they act as a **supplemental retrieval layer**. When a question is broad or the relevant fact is rare, the Shorty often catches it even when chunk‑only RAG would miss it.

---

## How It Works

```text
Video / podcast URL
↓
Transcript extracted and stored in SQLite
↓
LLM generates Shorty + synthetic questions + entities
↓
All representations vectorized into Chroma
  - chunks
  - Shorty
  - synthetic questions
↓
User query comes in
↓
Optional metadata filtering (channel / date)
↓
Query rewriting (multi-angle variants)
↓
Multi-representation retrieval in Chroma
  - type=shorty
  - type=syntheticquestion
  - type=chunk
↓
Best Shorty + questions + chunks assembled as context
↓
LLM answers with citations back to videos
```

This multi‑representation retrieval (Shorty + synthetic questions + chunks) gives much higher recall than naïve “chunk‑only” RAG.

---

## Architecture

### Storage

- **SQLite** (`data/transcripts.db`) – source of truth for:
  - `videos` – video metadata (id, title, channel, url, watch date, etc.)
  - `transcripts` – raw transcript text plus `shorty` and timestamps
  - `entities` – extracted entities with types and aliases
  - `syntheticquestions` – generated questions per video
  - `processingqueue` – background tasks (`shorty | syntheticquestions | entities`)

- **Chroma** (`data/transcript_chroma`) – vector index with a single collection `transcripts` using cosine similarity.  
  Each record has:
  - `embedding` (SentenceTransformer `all-MiniLM-L6-v2`)
  - `metadata.type` ∈ `{ "chunk", "shorty", "syntheticquestion" }`
  - `metadata.videoid`
  - `metadata.chunkindex` (for `type="chunk"`)

SQLite is the canonical store; Chroma is a derived index that can be rebuilt from SQLite if needed.

### Components

- `transcriptdatabase.py` – SQLite schema, migrations, and enqueue helpers
- `simpletranscriptfetcher.py` – fetches YouTube transcripts and persists them
- `videograbber.py` / `startgrabber.py` – bookmarklet‑based ingest service (Flask, default port 5000)
- `shortygenerator.py` – uses Anthropic to generate Shorty + synthetic questions
- `entityextractor.py` – extracts entities via Anthropic tool‑use or OpenAI‑compatible JSON prompts
- `transcriptrag.py` / `transcriptragenhanced.py` – chunking and Chroma integration (`index_single_transcript`, hybrid search)
- `enqueuebackfill.py` – backfill script to enqueue Shorty / questions / entities for existing transcripts
- `batchprocessor.py` – processes `processingqueue` tasks in queue or legacy batch mode
- `askshorty.py` – query pipeline (metadata filter → query rewrite → multi‑layer retrieval → answer)
- `askshortyapp.py` – Ask UI (Flask, default port 5001)
- `libraryapp.py` – library/admin UI for browsing videos, Shorties, entities, questions (Flask, default port 5002)

---

## Running Ask Shorty Locally

### 1. Install dependencies

Example (adjust as needed):

```bash
pip install flask anthropic chromadb sentence-transformers youtube-transcript-api yt-dlp
```

### 2. Set environment variables

Minimum for Anthropic‑based path:

```bash
# Windows PowerShell example
set ANTHROPIC_API_KEY=your_key_here
```

Optional OpenAI‑compatible batch/entity path:

```bash
set OPENAI_API_KEY=your_key_here
set OPENAI_BASE_URL=http://localhost:8000/v1   # or your provider
set OPENAI_MODEL=gpt-3.5-turbo                 # or another model
```

`GRABBER_PORT` defaults to `5000` if not set.

### 3. Start the grabber (bookmarklet backend)

```bash
python startgrabber.py
```

Endpoints:

- `POST /api/fetch-transcript` – grab transcript from URL
- `POST /api/save-pasted-transcript` – paste‑mode fallback

### 4. Generate Shorties, synthetic questions, entities

Backfill existing transcripts:

```bash
python enqueuebackfill.py --db-path data/transcripts.db --dry-run
# inspect, then:
python enqueuebackfill.py --db-path data/transcripts.db
```

Process the queue (Anthropic path):

```bash
python batchprocessor.py --queue --limit 50 --provider anthropic
```

This populates `transcripts.shorty`, `syntheticquestions`, `entities` and (in legacy batch mode) can also call into Chroma indexing.

> Note: In queue mode, Chroma reindexing is intentionally **not** run inside the queue loop because some Chroma code paths can terminate the worker. Run a separate reindex script from SQLite when needed.

### 5. Start the Ask Shorty UI

```bash
python askshortyapp.py
```

Open `http://localhost:5001/ask` to:

- Type a question
- Optionally restrict to specific `videoid`s
- Get answers with context drawn from transcript chunks, Shorties, and synthetic questions

### 6. Explore the library

```bash
python libraryapp.py
```

Open `http://localhost:5002` to browse:

- Videos and metadata
- Full transcripts
- Per‑video Shorty
- Synthetic questions
- Entities and processing queue status

---

## Scale Notes

Ask Shorty is designed to handle tens of thousands of videos on a single machine. With 30–40k videos, plus chunks, Shorties, and synthetic questions, the Chroma collection may reach 500k–1M vectors. For significantly larger libraries or multi‑user deployments, consider:

- Migrating from SQLite to PostgreSQL
- Migrating the vector layer from Chroma to a dedicated service (e.g. Qdrant or another hosted vector DB)

---

## Roadmap

- Standardized Shorty template (CONTEXT, INCIDENTS, ATTACK FLOW, IMPACT, MICRO‑DETAILS, TIMELINE)
- Automatic fact triple extraction for a lightweight knowledge‑graph layer
- Vast.ai / OpenAI‑compatible provider support for bulk processing
- PostgreSQL and external vector DB support at larger scales
- Multi‑user support and authentication
- Non‑YouTube sources (podcasts, local files, other platforms)
- Analytics on which layer (Shorty / chunk / synthetic question) answered each query
- Improved channel‑ and time‑based filters and timeline search

---

## Vision

Anyone should be able to ask questions across their entire video library — podcasts, lectures, meetings, research — and get precise answers instantly. Ask Shorty provides the **Shorty + RAG** architecture that makes that possible.

---

*Built in Burlington, VT*

This project is licensed under the MIT License (see `LICENSE`).

