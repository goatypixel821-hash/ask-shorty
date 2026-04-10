# Ask Shorty

> *Ask anything. Get the answer. No more scrubbing through videos.*

---

## The Problem

Video and podcast content is hard to search. You either watch everything or miss things. Standard search only hits titles and descriptions — not what was actually said. Plain RAG over raw transcripts helps a bit but often:

- Misses key details (names, numbers, versions)
- Splits causal chains across multiple chunks
- Returns noisy context filled with filler and digressions
- Can't connect knowledge **across** videos

Ask Shorty was built to fix this for a real use case: searching a personal YouTube history and asking *"Where did I hear X?"* without re‑watching hours of content.

---

## What Ask Shorty Does

Ask Shorty makes any video or podcast instantly queryable:

- Drop in a link (e.g. YouTube)
- Ask a question
- Get a precise answer with context and citations

Under the hood, Ask Shorty adds a **dense compression layer** and **multi-layer retrieval** on top of normal RAG. Each video gets a **Shorty**: a compact, machine‑oriented intelligence brief. Questions are then answered through five retrieval layers fused together — not just vector search.

---

## The Shorty

The **Shorty** is the core innovation — a retrieval-optimized compression of each video's content.

- Generated once per transcript at index time
- ~90–97 % token reduction vs. the original
- Designed to retain ~95 % of answerable information, including:
  - All named entities, systems, and people
  - Causal chains and relationships
  - Key numbers, dates, and technical details
  - Speaker framing and interpretation
  - Micro‑details that typical summarization drops

Shorties are stored in SQLite and also vectorized into Chroma as their own document type. They do **not** replace RAG over transcript chunks — they act as a **supplemental retrieval layer** that catches things chunk-only RAG misses.

---

## Multi-Layer Retrieval

Ask Shorty doesn't rely on a single search method. It fuses **five retrieval layers** using Reciprocal Rank Fusion (RRF):

| Layer | What it does |
|-------|-------------|
| **Chroma vector search** | Semantic similarity across chunks, Shorties, and synthetic questions |
| **BM25 keyword search** | Exact keyword matching (catches names, acronyms, specific terms) |
| **Cross-encoder reranker** | Neural reranking of top candidates for precision |
| **Per-video graph reasoning** | Multi-hop traversal of subject→relation→object triples within a video |
| **Global knowledge graph** | Cross-video graph reasoning over normalized entities and facts |

The query router (HSC — Hierarchical Semantic Compression) classifies each question as a fact lookup, event query, or general question and selects the appropriate layers.

---

## Global Knowledge Graph

Every video gets entity extraction and triple extraction (subject → relation → object facts). These are normalized and merged into a **global knowledge graph** that spans all videos.

This enables:
- **Cross-video reasoning**: "How does X from video A connect to Y from video B?"
- **Multi-hop paths**: Following chains of relationships across sources
- **Agreement scoring**: Facts confirmed by multiple videos are boosted
- **Entity normalization**: Different spellings/aliases resolve to the same node

---

## Architecture

```text
Video / podcast URL
       ↓
Transcript extracted and stored in SQLite
       ↓
LLM generates:
  • Shorty (dense retrieval brief)
  • Synthetic questions (intent matching)
  • Entities (people, orgs, concepts, tech)
  • Triples (subject → relation → object facts)
       ↓
All representations indexed:
  • Chroma (vector embeddings)
  • BM25 (keyword index)
  • Knowledge graph (entities + triples)
       ↓
User asks a question
       ↓
HSC query router classifies question type
       ↓
Multi-layer retrieval:
  • Chroma semantic search
  • BM25 keyword search
  • Graph reasoning (per-video + global)
  • Cross-encoder reranking
       ↓
Reciprocal Rank Fusion merges all results
       ↓
LLM answers with citations back to source videos
```

### Storage

- **SQLite** (`data/transcripts.db`) — source of truth for videos, transcripts, Shorties, entities, synthetic questions, triples (facts), global normalized facts, processing queue
- **Chroma** (`data/transcript_chroma`) — vector index with cosine similarity (SentenceTransformer `all-MiniLM-L6-v2`)
- **BM25 index** — keyword search built from all document types

SQLite is the canonical store; Chroma and BM25 are derived indexes that can be rebuilt.

### Key Components

| File | Purpose |
|------|---------|
| `transcript_database.py` | SQLite schema, migrations, all table management |
| `shorty_generator.py` | LLM-powered Shorty + synthetic question generation |
| `entity_extractor.py` | Entity extraction (people, orgs, concepts, tech) |
| `triple_extractor.py` | Subject→relation→object fact extraction |
| `bm25_index.py` | Okapi BM25 keyword search index |
| `reranker.py` | Cross-encoder neural reranking |
| `transcript_rag_enhanced.py` | Chroma chunking and vector search |
| `graph_search.py` | Per-video graph reasoning over triples |
| `hsc/` | Hierarchical Semantic Compression — query routing, global graph, entity normalization |
| `ask_shorty.py` | Query pipeline — orchestrates all retrieval layers + RRF fusion |
| `ask_shorty_app.py` | Ask UI (Flask) with SSE streaming debug panel |
| `video_grabber.py` | Bookmarklet-based video ingest |
| `batch_processor.py` | Queue-based LLM task processing |
| `build_courses.py` | Auto-generates structured courses from video clusters |
| `build_clusters.py` | Topic clustering across the video library |
| `evaluate_rag.py` | Retrieval evaluation framework (Recall@K, MRR) |
| `reindex_all.py` | Rebuild all indexes from SQLite |

### Web UIs

- **Ask Shorty** (`/ask`) — question answering with real-time debug panel showing all retrieval layers
- **Knowledge Explorer** (`/knowledge`) — browse the knowledge graph, entities, facts, connections
- **Course Viewer** (`/courses`) — auto-generated courses from video clusters
- **Library Admin** — browse videos, Shorties, entities, queue status

---

## Running Locally

### 1. Install dependencies

```bash
pip install flask anthropic chromadb sentence-transformers youtube-transcript-api yt-dlp
```

### 2. Set environment variables

```bash
# Anthropic (primary LLM path)
set ANTHROPIC_API_KEY=your_key_here

# Optional: OpenAI-compatible provider for batch processing (e.g. local Ollama)
set OPENAI_API_KEY=your_key_here
set OPENAI_BASE_URL=http://localhost:8000/v1
set OPENAI_MODEL=qwen2.5:14b
```

### 3. Start the grabber and add videos

```bash
python start_grabber.py
```

Use YouTube's built-in transcript: click **···** under the video → **Show transcript** → select all → copy. Open the grab page, paste the transcript, click Save & Vectorize.

#### Bookmarklet

Paste this as a bookmark URL to grab videos with one click:

```javascript
javascript:(function(){var v=window.location.href;if(!v.includes('youtube.com/watch')){alert('Not a YouTube video!');return;}var t=document.title.replace(' - YouTube','');var c='';try{c=document.querySelector('ytd-channel-name a').textContent.trim();}catch(e){try{c=document.querySelector('.ytd-channel-name a').textContent.trim();}catch(e2){}}window.open('http://localhost:5000/grab?url='+encodeURIComponent(v)+'&title='+encodeURIComponent(t)+'&channel='+encodeURIComponent(c),'_blank','width=600,height=600,left=200,top=200');})();
```

### 4. Process the queue

```bash
python batch_processor.py --queue --limit 50 --provider anthropic
```

This generates Shorties, synthetic questions, and entities for each video.

### 5. Extract triples and build the knowledge graph

```bash
python batch_processor.py --queue --limit 50 --provider anthropic
python reindex_all.py
```

### 6. Start Ask Shorty

```bash
python ask_shorty_app.py
```

Open `http://localhost:5001/ask` to ask questions with the full multi-layer retrieval pipeline.

---

## How It Compares

| Feature | Typical RAG | Karpathy's LLM Wiki | Ask Shorty |
|---------|------------|---------------------|------------|
| Source handling | Chunk and embed | Compile into wiki pages | Compress into Shorties + extract entities + triples |
| Retrieval | Vector similarity | Index file scanning | 5-layer fusion (vector + BM25 + reranker + graph + global graph) |
| Cross-source reasoning | None | Wiki cross-references | Global knowledge graph with multi-hop BFS |
| Knowledge structure | Flat chunks | Interlinked markdown | SQLite + vector index + BM25 + knowledge graph |
| Quality measurement | None | Manual review / lint | Evaluation framework (Recall@K, MRR against golden answers) |
| Self-healing | No | LLM lint pass | Entity normalization + deduplication + agreement scoring |

---

## Scale Notes

Designed to handle tens of thousands of videos on a single machine. Supports both cloud LLMs (Anthropic) and local models (Ollama with Qwen, Llama, etc.) for processing. For significantly larger deployments, consider migrating from SQLite to PostgreSQL and from Chroma to a dedicated vector service.

---

## Roadmap

- [x] Dense Shorty compression layer
- [x] Multi-representation retrieval (Shorty + chunks + synthetic questions)
- [x] Entity and triple extraction
- [x] BM25 keyword search + cross-encoder reranking
- [x] Per-video graph reasoning
- [x] Global cross-video knowledge graph
- [x] HSC query routing
- [x] Reciprocal Rank Fusion across all layers
- [x] Evaluation framework
- [x] Course auto-generation from topic clusters
- [x] Knowledge explorer UI
- [x] Local LLM support (Ollama / vast.ai)
- [ ] Spaced repetition from knowledge graph
- [ ] Real-time ingest (process as you watch)
- [ ] Multi-user support and authentication
- [ ] Non-YouTube sources (podcasts, local files, meetings)
- [ ] PostgreSQL and external vector DB support at scale

---

## Vision

The AI industry spent 3 years building vector databases to search through messy document dumps. Retrieve a chunk, generate an answer, conversation ends, knowledge gone. RAG just searches the graveyard faster.

Ask Shorty takes a different approach: **compress once, index deeply, reason across sources**. Every video you add makes the system smarter. Facts connect across videos. Knowledge compounds instead of scattering.

Anyone should be able to ask questions across their entire video library — podcasts, lectures, meetings, research — and get precise answers instantly.

---

## Legal & Ethical Use

Ask Shorty is designed for **personal, non-commercial use** to index videos you have watched for research and reference purposes.

- **Fair Use**: Transcripts are compressed into transformative summaries. Original videos are always cited with links to creators.
- **No Scraping**: Uses YouTube's built-in transcript feature. The bookmarklet only receives the URL, title, and channel you provide.
- **Local Only**: All data stored on your machine (SQLite + Chroma). No cloud uploads, no third-party sharing.
- **Respect Creators**: Always link back to original videos. Do not redistribute transcripts or derivatives.

If you're a content creator and have concerns, please open an issue.

---

This project is licensed under the MIT License (see `LICENSE`).
