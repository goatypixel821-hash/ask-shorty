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
- Bookmarklet-based video capture
- Transcript extraction and vectorization
- Shorty generation
- Multi-angle query rewriting
- Hybrid search and retrieval

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

