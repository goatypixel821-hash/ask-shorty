# Ask Shorty: Dense Transcript Compression for High‑Recall RAG

## 1. Motivation

Modern LLM applications rely heavily on Retrieval‑Augmented Generation (RAG) to answer questions over external data such as documents, chats, videos, and podcasts. YouTube and podcast transcripts are a particularly important source: they contain high‑value technical discussions, news, and educational material, but are long, noisy, and difficult to search.

Typical “RAG over transcripts” systems run into several problems:

- **Low signal per chunk** – transcripts include filler, repetition, digressions, and incomplete sentences, which dilute embedding quality.
- **Fragmented facts** – key facts and causal chains are split across multiple chunks, so a single retrieved chunk often misses crucial context.
- **Recall failures** – even with good embeddings, important details (numbers, names, versions, dates) are easy to miss when they appear only once in hours of speech.
- **Scaling limits** – storing and searching only raw text chunks becomes expensive and slow as libraries grow.

Ask Shorty was built from a practical need: searching a personal YouTube watch history and being able to ask, “Where did I hear X?” and “What exactly did they say about Y?” without re‑watching entire videos. The central bet is a **dense, machine‑oriented compression layer** (the **Shorty**) that sits **alongside** chunk‑based RAG, plus optional structured artifacts (entities, synthetic questions, triples) and hybrid retrieval (dense + keyword + graph) where enabled.

## 2. Core Idea: The Shorty

At the center of Ask Shorty is the **Shorty**: a dense compression of a single transcript aimed at **retrieval and downstream LLM use**, not polished human prose.

Design targets (not guaranteed metrics unless separately measured on a benchmark):

- **Roughly 90–97% fewer tokens** than the raw transcript for the same video.
- **High retention of answer‑relevant signal**: entities, numbers, causal chains, “micro‑details,” and key claims—stated in prompts as a **quality goal**, not a formal proof of “~95% information retention.”

Implementation:

- Stored in SQLite as **`transcripts.shorty`** (see `transcript_database.py`).
- Indexed into Chroma as documents with metadata **`type="shorty"`**, one per video, in addition to transcript **chunks** and **synthetic questions**.

Shorties are not conventional summaries. The intended properties are:

- Preserve salient entities (people, organizations, systems).
- Preserve important numbers (dates, counts, sizes, versions).
- Preserve relationships and causal chains where the model follows the Shorty prompt.
- Optionally surface **MICRO‑DETAILS** (or equivalent sections in the prompt) for rare but important strings (product names, legal terms, etc.).

**Role relative to chunks:** Shorties **do not replace** chunk RAG. They provide a **whole‑video semantic surface** so a single embedding (or a small text) can match queries about “what this video is about” without relying on the one chunk that happened to contain the answer.

## 3. System Overview

### 3.1 Ingest

- A browser bookmarklet and **`video_grabber.py`** / **`start_grabber.py`** capture YouTube URLs and transcript text (e.g. **`youtube-transcript-api`**, **`simple_transcript_fetcher.py`**).
- Metadata and transcripts are stored in **SQLite** (`videos`, `transcripts`).

### 3.2 Enqueue and batch processing

- **`enqueue_backfill.py`** finds work to do (e.g. videos with transcripts missing Shorties, or specific **task** types such as **`triples`**) and inserts rows into **`processing_queue`**.
- **`batch_processor.py`** runs in **queue mode** (`--queue`, default): it **atomically claims** pending rows, runs the appropriate LLM or extraction step, and updates **`status`** (`pending` → `started` → `completed` or `failed`). Optional filters: **`--only-tasks`**, **`--exclude-tasks`** (e.g. one worker for everything except triples, another for **`triples` only**).
- Providers: **`--provider anthropic`** (default) or **`--provider openai-compatible`** with **`--base-url`** and **`--model`** for local or hosted OpenAI‑compatible endpoints.
- **Important:** In queue mode, **Chroma indexing is decoupled** from the hot path (some Chroma paths have been problematic for long‑running workers). After bulk LLM work, run **`reindex_all.py`** (or equivalent) to refresh vectors from SQLite.

**Typical queue `task` values** (see schema):

- **`shorty`** – generate or refresh Shorty text.
- **`synthetic_questions`** – generate stored questions for question–question matching.
- **`entities`** – extract structured entities.
- **`triples`** – extract **subject → relation → object** rows from the Shorty (and persist to **`facts`**).
- **`segments`** / **`events`** – optional **HSC** (Hierarchical Semantic Compression) artifacts when that pipeline is used.

### 3.3 Indexing for retrieval

- **`transcript_rag_enhanced.py`** (exported as **`TranscriptRAG`**) chunks transcripts and writes **Chroma** documents with embeddings from **`all-MiniLM-L6-v2`** (default). Metadata **`type`** is one of **`chunk`**, **`shorty`**, **`synthetic_question`** (and additional types if extended).
- **Chroma directory** defaults next to the DB (e.g. **`transcript_chroma_new`** beside an external **`transcripts.db`**); configurable via constructor args / env.
- **`bm25_index.py`** builds a **BM25** keyword index over the same logical documents (chunks, Shorties, entity strings, etc.) for hybrid retrieval. The index path is derived from the DB path.

### 3.4 Query and answer

- **`ask_shorty.py`** implements the main **Ask** pipeline (used by **`ask_shorty_app.py`**).
- **Baseline path:** metadata filtering (channel / date → candidate **`video_id`**s), **query rewriting** (Claude → multiple query strings), **multi‑representation vector search** over Chroma (chunks + Shorties + synthetic questions), context assembly, **Claude** answer with citations.
- **Optional extensions** (controlled by **environment variables**, see §5.5):
  - **BM25** + **Reciprocal Rank Fusion (RRF)** with dense hits.
  - **Graph retrieval** over **`facts`** (**`graph_search.py`**).
  - **Cross‑encoder reranking** (**`reranker.py`**).
  - **HSC** routing and extra segment/event retrieval (**`hsc/`**).

### 3.5 UIs and tooling

- **Ask UI** – **`ask_shorty_app.py`** (e.g. **`/ask`**).
- **Library / admin** – **`library_app.py`**.
- **Knowledge explorer** – templates / routes for graph and facts browsing (see repo).
- **Progress** – **`check_progress.py`** (`--db-path`): queue and triples (**`facts`**) counts.
- **Queue hygiene** – **`reset_stale.py`** (reset orphaned **`started`** rows to **`pending`**), **`reset_failed.py`** (retry **`failed`**).

## 4. Data Model

### 4.1 SQLite (canonical store)

Defined and migrated in **`transcript_database.py`**. Principal tables:

| Table | Role |
|--------|------|
| **videos** | `video_id` (PK), title, channel, url, watch metadata, etc. |
| **transcripts** | Full transcript text; **`shorty`**, **`shorty_generated_at`**. |
| **entities** | Per‑video entities with **`name`**, **`type`**, **`aliases`** (JSON). |
| **synthetic_questions** | Per‑video generated questions for retrieval. |
| **facts** | Per‑video **triples**: subject, relation, object (+ optional confidence / source). |
| **global_facts** | Normalized cross‑video fact store for graph‑scale views (rebuilt from per‑video facts as configured). |
| **fact_nodes**, **fact_edges**, **fact_frequency_meta** | Graph salience / structure around facts. |
| **entity_alias** | Alias normalization for graph and search. |
| **global_graph_meta** | Bookkeeping for global graph rebuilds. |
| **segments**, **events** | HSC‑style structured summaries when generated. |
| **processing_queue** | **`task`**, **`status`** (`pending`, `started`, `completed`, `failed`, `permanently_failed`), timestamps, **`error`**, **`attempts`**. |

SQLite is the **source of truth**; Chroma and BM25 are **derived** and can be rebuilt.

### 4.2 Chroma

- Collection **`transcripts`**, cosine similarity.
- Documents tagged by **`type`** and **`video_id`**; chunks include **`chunk_index`**.

### 4.3 BM25

- On‑disk index (see **`bm25_index.py`**) built from the same corpus family as vectors, enabling **exact token / name / acronym** matching that dense models often miss.

## 5. Retrieval Pipeline

### 5.1 Metadata filtering

Narrows **`video_id`** candidates when the question implies channel or date constraints (SQLite queries in **`ask_shorty.py`** / **`TranscriptDatabase`**).

### 5.2 Query rewriting

Claude generates **3–4 alternate phrasings** (JSON array of strings); each feeds retrieval, improving recall under lexical and semantic variation.

### 5.3 Multi‑representation dense retrieval

For each rewrite, search Chroma for:

- **`shorty`** – whole‑video compressed signal.
- **`synthetic_question`** – question–question alignment.
- **`chunk`** – local evidence and exact wording.

Hits are merged and ranked (with optional RRF and reranking as below).

### 5.4 Answer synthesis

Top Shorties, synthetic questions, and chunks are packed into a prompt; **Claude** produces the final answer with **video / channel / watch‑date** style citations per system prompt rules.

### 5.5 Optional layers (feature flags)

Set in the environment for **`ask_shorty.py`**:

| Variable | Effect |
|----------|--------|
| **`ASK_SHORTY_BM25=1`** | BM25 search; fused with dense lists via **RRF**. |
| **`ASK_SHORTY_GRAPH=1`** | Retrieval signals from **`facts`** via **`GraphSearch`**. |
| **`ASK_SHORTY_RERANK=1`** | Second‑stage **CrossEncoder** reranking over candidate groups. |
| **`ASK_SHORTY_HSC=1`** | HSC‑style routing and segment/event use where implemented. |

Defaults are conservative (often **off**) for latency and dependency cost; **BM25 in particular** has shown strong gains on **keyword‑sensitive** and **cross‑video** style queries in internal **`evaluate_rag.py`** runs when the index is built and the flag is on.

### 5.6 Evaluation

- **`evaluate_rag.py`** supports **baseline** and other modes, reporting **Recall@K**, **MRR**, **NDCG**, and optional answer metrics.
- **Empirical caveat:** On a fixed golden set, **within‑video “specific fact”** retrieval can reach very high recall when the right chunk exists; **cross‑video** retrieval remains harder for **dense‑only** configurations. **BM25 + fusion** often improves **cross‑video** recall when queries share surface forms with indexed text. Results depend on **corpus coverage**, **index freshness**, and **flags**—the README’s “five‑layer” story should be read as **capability when enabled and tuned**, not a single default code path.

## 6. Design Rationale

### 6.1 Why dense compression (Shorty)?

Mitigates **dilution** of embeddings across noisy chunks and **splits** across chunk boundaries for global questions. It is the **primary architectural bet** of Ask Shorty.

### 6.2 Why multi‑representation retrieval?

Chunks excel at **local** evidence; Shorties at **global** video semantics; synthetic questions at **intent** overlap. Together they reduce single‑representation blind spots.

### 6.3 Why triples and graph tables?

Structured **subject–relation–object** rows support **browsing**, **graph search**, and future **multi‑hop** reasoning. Quality depends on LLM extraction and normalization; they are **add‑ons** that compound with good Shorties, not a substitute for them.

### 6.4 Why SQLite + derived indexes?

Inspectability, durability, and simple **rebuild** loops (queue → SQLite → **`reindex_all.py`** → Chroma/BM25).

### 6.5 Why a queue?

LLM generation is slow and failure‑prone; **`processing_queue`** enables **resume**, **retry**, **parallel workers** with **`only-tasks` / `exclude-tasks`**, and clear **operations** (progress scripts, stale reset).

## 7. Implementation Notes

### 7.1 Key modules (non‑exhaustive)

| Module | Role |
|--------|------|
| **`transcript_database.py`** | Schema, migrations, CRUD, queue helpers, facts API. |
| **`shorty_generator.py`** | Shorty + synthetic question prompts (Anthropic path). |
| **`entity_extractor.py`** | Anthropic tool‑use or OpenAI‑compatible JSON parsing. |
| **`triple_extractor.py`** | Triple extraction from Shorties (OpenAI‑compatible path in batch). |
| **`transcript_rag_enhanced.py`** | Chunking, Chroma indexing, hybrid vector search. |
| **`bm25_index.py`** | BM25 build/search, RRF helpers. |
| **`graph_search.py`** | Query **`facts`** for retrieval hints. |
| **`reranker.py`** | Cross‑encoder reranking. |
| **`hsc/`** | Segments, events, routing, global graph helpers. |
| **`ask_shorty.py`** | End‑to‑end Ask pipeline and feature flags. |
| **`batch_processor.py`** | Queue and legacy batch processing. |
| **`enqueue_backfill.py`** | Enqueue backfill jobs. |
| **`reindex_all.py`** | Rebuild vectors from SQLite. |
| **`evaluate_rag.py`** | Retrieval / QA evaluation harness. |

### 7.2 Environment

- **`ANTHROPIC_API_KEY`** – Shorty pipeline on Anthropic, query rewrite, answers.
- **`OPENAI_*` / OpenAI‑compatible** – batch and local model paths in **`batch_processor.py`**.
- **`ASK_SHORTY_BM25`**, **`ASK_SHORTY_GRAPH`**, **`ASK_SHORTY_RERANK`**, **`ASK_SHORTY_HSC`** – optional retrieval behavior.
- **`ASK_SHORTY_DB_PATH`** / **`TranscriptRAG(..., transcript_db=...)`** – external DB layouts (e.g. dedicated **`transcripts.db`** per machine).

## 8. Current Status and Limitations

**Shipped in code today:** Shorties, synthetic questions, entities, triples → **`facts`**, Chroma multi‑type index, BM25 index, optional graph and rerank and HSC hooks, queue processing with OpenAI‑compatible and Anthropic backends, evaluation scripts, operational utilities (**`check_progress`**, **`reset_stale`**).

**Honest limitations:**

- **Shorty quality** is model‑ and prompt‑dependent; not every detail survives compression.
- **Cross‑video** questions stress **retrieval** more than single‑video questions; **dense embeddings alone** are often insufficient for entity‑heavy or lexically crisp queries—**BM25 (when built and enabled)** is an important practical complement.
- **Global graph** value grows with **coverage** (many videos with clean triples and aliases) and **fresh rebuilds**; it is not “free intelligence” on day one.
- **README “vision”** describes the **full stack when layers are on and data is complete**; default installs may run a **subset** until env flags and indexes are configured.

## 9. Future Work

- Tighter **Shorty schema** (fixed sections) and **regression tests** on golden videos.
- **Default‑on hybrid** (dense + BM25) once latency budgets are acceptable.
- Richer **graph** traversal and **agreement** scoring across **`global_facts`**.
- **User‑visible** retrieval diagnostics (scores, layer contributions).
- Scale‑out paths (Postgres, hosted vector DB) for very large libraries.

## 10. Conclusion

Ask Shorty is a **retrieval‑first** system for long‑form spoken content: **compress each video into a Shorty**, **index multiple representations**, and **optionally** fuse **keyword**, **graph**, and **reranking** signals. The Shorty remains the **core design choice**; other components are **force multipliers** whose measured impact depends on **data quality**, **index freshness**, and **configuration**—best validated with **`evaluate_rag.py`** and real queries rather than assumed from architecture diagrams alone.
