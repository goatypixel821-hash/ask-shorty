# Ask Shorty

> *Ask anything. Get the answer. No more scrubbing through videos.*

---

## The Problem

Video and podcast content is unsearchable. You either watch everything or miss things. Standard search returns titles and descriptions — not what was actually said. RAG helps but misses details. Nothing works well enough.

## What Ask Shorty Does

Ask Shorty makes any video or podcast instantly queryable. Drop in a link, ask a question, get a precise answer with context.

Under the hood, Ask Shorty solves the retrieval problem that breaks most RAG systems — the **Shorty**.

## The Shorty

The core innovation is a dense compressed representation of each video generated at index time. Think of it as an intelligence brief — a ~90–97% token reduction that preserves ~95% of answerable information including:

- All named entities, systems, and people
- Causal chains and relationships
- Key numbers, dates, and technical details
- Speaker framing and interpretation
- Micro-details that standard summarization drops

The Shorty acts as a supplemental search layer alongside traditional RAG. When RAG misses something, the Shorty catches it.

## How It Works

```text
Video/podcast URL
↓
Transcript extraction
↓
LLM generates Shorty (dense compression)
↓
Both vectorized and stored
↓
Query comes in
↓
Multi-angle query rewriting
↓
RAG retrieves relevant chunks
↓
Shorty injected as context layer
↓
LLM answers with full picture
```

## Current Status

Early proof-of-concept. Working pipeline includes:
- Bookmarklet-based video capture into a local Flask service (`video_grabber.py`)
- Transcript extraction into SQLite (`transcript_database.py`)
- Background vectorization into a ChromaDB index (`transcript_rag_enhanced.py`)
- Shorty generation at index time via Anthropic (`shorty_generator.py`)
- Synthetic question generation and indexing
- Entity extraction and storage for alias-aware lookup
- Multi-angle query rewriting and hybrid retrieval in the Ask Shorty engine (`ask_shorty.py`)

### How to run it locally

1. **Install dependencies** (example, adjust to your environment):

```bash
pip install flask anthropic chromadb sentence-transformers youtube-transcript-api yt-dlp
```

2. **Set your Anthropic API key**:

```bash
set ANTHROPIC_API_KEY=your_key_here   # PowerShell / Windows
```

3. **Start the video grabber (for the bookmarklet)**:

```bash
python start_grabber.py
```

This exposes:
- `POST /api/fetch-transcript` – main bookmarklet endpoint
- `POST /api/save-pasted-transcript` – paste-mode fallback

4. **Start the Ask Shorty UI**:

```bash
python ask_shorty_app.py
```

Then open `http://localhost:5001/ask` in your browser to:
- Type a question
- Optionally restrict to specific `video_id`s
- Get an answer with context drawn from transcript chunks, Shorties, and synthetic questions

### Chroma scale considerations

Ask Shorty is designed to handle tens of thousands of videos. With 30–40k videos,
plus chunks, Shorties, and synthetic questions, the Chroma collection may reach
500k–1M vectors. Chroma can handle this on a decent machine, but if you push
significantly beyond that scale or need multi-user concurrency, consider
migrating the vector layer to a dedicated service such as Qdrant or another
hosted vector database.

## What's Coming

- Automatic Shorty generation at index time
- Synthetic question indexing
- Entity and alias mapping
- Knowledge graph layer for relationship queries
- Clean interface for non-technical users

## The Vision

Anyone should be able to ask questions across their entire video library — podcasts, lectures, meetings, research — and get precise answers instantly. Ask Shorty is the infrastructure that makes that possible.

---

*Built in Burlington, VT*

---

## TODO / Roadmap

- Vast.ai / OpenAI-compatible provider support for bulk processing
- Migrate from SQLite to PostgreSQL at scale
- Migrate from Chroma to Qdrant at 30k+ videos
- Multi-user support
- Non-YouTube sources (podcasts, local files)
- Analytics on which layer (Shorty/chunk/synthetic) answered each query
- Channel-based query filtering improvements
- Timeline-based search across library

