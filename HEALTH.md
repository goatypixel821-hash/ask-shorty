# Ask Shorty: Health Report

_Branch `claude/tests-and-health` (from `v2-hierarchical-rag`), 2026-11-04._

## TL;DR

- **Test safety net:** 176 pytest tests (before this, the only test was `test_annotations.py`, which can't be collected offline). They run in about 10 s with no network, no API keys and no real data. A 7-video synthetic corpus is indexed through the project's own code paths into a temp dir: SQLite, Chroma, V1 BM25, the V2 tables and the V2 BM25 pickles. LLM calls hit a fake client at the OpenRouter/Anthropic SDK boundary.
- **Bugs fixed:** 29 defects across 13 fix commits (plus the requirements fix), each with a regression test that fails before the fix and passes after. The most consequential:
  1. A fresh install crashed on its second start (WAL pragma inside a transaction).
  2. V1 vector search embedded queries with a different model than the documents.
  3. RRF fusion was computed but never used in the default path.
  4. Garbage or empty LLM output saved header-only Shorties and wiped existing facts while marking tasks `completed`.
  5. Job-state races between the worker thread and the polling endpoint.
- **V2 is the healthiest path.** With the fake LLM it cited the right fixture video for every test question, including the cross-video one. **V1 was badly broken** on any index not built with Chroma's default model; it is fixed now.
- **UI check:** 4 questions through the real Flask UIs in Chromium (3 on `/ask_v2`, 1 on `/ask`). All cite the expected fixture video. Screenshots are in `docs/screenshots/`.

## Running the tests

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt          # requirements.txt was missing flask/openai/anthropic/…
python -m pytest                             # 176 tests, ~10 s
ASK_SHORTY_TEST_REAL_MODELS=1 python -m pytest   # use real MiniLM + ms-marco cross-encoder

# UI check (Node Playwright + Chromium)
python tests/ui/serve_fixture_apps.py /tmp/ask-shorty-ui &   # :5001 ask app, :5002 library
NODE_PATH=$(npm root -g) node tests/ui/run_ui_check.js docs/screenshots
```

By default the tests use **stand-in models**: a hashed bag-of-words embedder and a token-overlap cross-encoder. That's because this sandbox could reach `huggingface.co` but not the weight CDN **`us.aws.cdn.hf.co`** (proxy 403, both the Xet and LFS paths). Chroma's own ONNX MiniLM (from S3) did download. With real models available, set `ASK_SHORTY_TEST_REAL_MODELS=1`. The retrieval assertions use distinctive planted facts, so they should hold either way, but **I have not run the suite with real models**.

`tests/fixtures/corpus.py` documents the planted facts: names, version numbers, dates, causal chains, and one fact spread across two videos (Kestrel 4.2 introduced Tidewater → the Tidewater bug was fixed by Marcus Oyelaran in 4.2.3).

---

## Architecture map

### Storage and indexing

**`transcript_database.py`**: The SQLite source of truth: `videos`, `transcripts` (Shorty lives in `transcripts.shorty`), `entities`, `synthetic_questions`, `processing_queue`, `facts`, `segments`, `events`, the global-graph tables and `video_annotations`. Migrations are additive `ALTER TABLE`s inside `ensure_db_exists()`. Fixed here: the WAL crash, `add_video` wiping metadata, and same-second "latest row" ambiguity.

**`v2_schema.py`**: DDL for the V2 tables (`video_signatures`, `segment_index`, `event_index`, `entity_postings`, `topic_postings`, `query_cache`). `query_cache` is created but never read or written.

**`populate_v2_indexes.py`**: Full refresh of the V2 tables from `videos`/`transcripts`/`segments`/`events`/`entities`. Run it, then **`v2_bm25.py`**, which builds three BM25 pickles (video routing text, segments, events) with a Porter-stemmed tokenizer. Both are required for V2; before this branch, missing pickles silently gave empty layers.

**`transcript_rag_enhanced.py`**: Chroma + SentenceTransformer indexing (`index_single_transcript`: chunks, Shorty and synthetic questions as one collection keyed by `type`). It now also has a `query()` that embeds with the indexing model. The default DB path is a hard-coded `C:\Users\number2\…`, and the default Chroma dir is `data/transcript_chroma_finetuned`. **`transcript_rag.py`** is a one-line alias (`TranscriptRAG = EnhancedTranscriptRAG`). Its `__main__` calls `rag.ask()`, which doesn't exist, and its docstring points at a non-existent `transcript_rag_v2.py`: **dead apart from the alias**.

**`bm25_index.py`**: The V1 BM25 (chunks, Shorties and entity names → one hit per video) plus `reciprocal_rank_fusion`. **`reindex_all.py` / `reindex_with_model.py` / `reindex_on_gpu.py`**: three Chroma rebuilders. `reindex_on_gpu.py` is deliberately standalone for vast.ai and carries a 5th copy of the chunker.

### Generation (LLM)

**`shorty_generator.py`**: The Shorty template prompt and generation through OpenRouter (default) or Anthropic, selected by `SHORTY_PROVIDER`. It now validates the body before prepending the header, and synthetic questions follow the provider. **`entity_extractor.py`**: entities through Anthropic tool-use, or a JSON-prompt path for OpenAI-compatible providers. `store_entities(replace=True)` is new. **`triple_extractor.py`**: subject→relation→object triples with a disk cache under `data/triple_llm_cache/` (empty results are no longer cached). **`hsc/segment_extractor.py`, `hsc/event_extractor.py`**: time-bounded segment summaries and causal events (HSC layers). The segment extractor has its own non-LLM fallback. **`anthropic_client.py`**: a singleton Anthropic client.

**`batch_processor.py`**: The queue worker. It atomically claims `processing_queue` rows (`UPDATE … RETURNING`, optional sharding), runs one task per row, and marks the row completed or failed; after 6 attempts a row becomes `permanently_failed`. It also has the legacy non-queue batch mode and the `--provider` mismatch guard. Queue mode never touches Chroma.

### Query pipelines

**`ask_shorty.py` (V1, 2.7k lines)**: The `AskShorty.answer_question` pipeline: metadata filter (LLM) → query rewrites (LLM) → three Chroma layers (chunk / Shorty / synthetic question) → optional RRF fusion with BM25, graph and HSC → optional cross-encoder rerank (`reranker.py`) → answer (LLM). It also contains a ~600-line agent tool loop (`_agent_run`) that **is not covered by tests**.

**`ask_shorty_v2.py` (V2)**: `AskShortyV2.retrieve_videos` runs: route (`query_router.py`) → LLM query expansion → video BM25 + RAM "instant" candidates (`v2_memory_maps.py`) → segment-BM25 rescue → segment BM25 within candidates → optional cross-encoder → watch-date sort for topic lookups. `answer()` builds Shorty+segment blocks, calls the LLM, then runs the non-blocking grounding audit and the optional verification excerpts. **V2 uses no Chroma at all.**

**`reranker.py`**: Groups hits by video and chunk neighbourhood, expands neighbour chunks from Chroma, then blends cross-encoder output (0.50), retrieval score (0.25), support count (0.15) and source diversity (0.10). **`graph_search.py`**: Per-video LIKE search over `facts` with one-hop expansion. **`hsc/`**: The HSC router (`hsc/query_router.py`), `hsc_search.hsc_retrieve` (SQL over segments/events/facts plus a vector rank, RRF-merged), the global cross-video graph (`global_graph*.py`, `graph_reasoner.py`, `fact_frequency.py`, `entity_normalizer.py`). Only `hsc_retrieve` and `_vector_rank` are exercised by tests.

### Apps and ingest

**`ask_shorty_app.py` (:5001)**: Flask app with `/ask` (V1), `/ask_v2`, `/agent`, the knowledge/observatory pages and courses. Questions become `ask_jobs` rows run on background threads, with SSE debug streams and polling. Job transitions are now compare-and-set. **`library_app.py` (:5002)**: Browse and edit videos, Shorties, questions and entities, and trigger LLM regeneration; it now honours `ASK_SHORTY_DB_PATH`. **`video_grabber.py` (:5000)**: Bookmarklet / paste ingest. It loads the embedding model, opens the DB and writes `data/grab_log.txt` **at import time**. **`ask_shorty_worker.py`**: A one-shot subprocess entry point. **Nothing calls it**; its stdout contract is fixed anyway.

**Courses, clusters, eval**: `build_clusters.py`, `label_clusters.py`, `relabel_clusters.py`, `build_courses.py`, `export_course.py` (courses UI), `evaluate_rag.py`, `build_eval_dataset.py`, `generate_*`, `review_eval_dataset.py`, `audit_eval_questions.py`, `evaluate_agent_sc.py`. These are offline tooling, untested here.

### Dead, duplicated or one-off code

| What | Notes |
|---|---|
| `transcript_rag.py` vs `transcript_rag_enhanced.py` | Not two implementations: `transcript_rag.py` is just an alias plus a broken REPL. Fold the alias into `transcript_rag_enhanced.py` (or rename that to `transcript_rag.py`) and drop the REPL. |
| Char chunker ×5 | `EnhancedTranscriptRAG._chunk_transcript`, `bm25_index._chunk_text`, `ask_shorty_v2._chunk_transcript_text`, an inline copy inside `AskShorty._apply_rrf_fusion`, and `reindex_on_gpu.py`. `tests/test_chunking.py` now guards that the three importable ones agree. |
| Two query routers | `query_router.py` (V2) and `hsc/query_router.py` (HSC) classify questions with different labels and regexes. |
| Three Chroma reindexers | `reindex_all.py`, `reindex_with_model.py`, `reindex_on_gpu.py`. |
| `ask_shorty_worker.py` | Unreferenced (the app uses in-process threads). |
| `v2_schema.query_cache` | Created, never used. |
| ~25 one-off scripts | `check_*.py`, `show_*.py`, `find_*.py`, `list_*.py`, `q.py`, `queue_video.py`, `save_pending.py`, `inspect_transcripts_db.py`: 4–20 lines each, most with a hard-coded `C:\Users\number2\…` DB path. 39 files hard-code a Windows user path. Move them to `scripts/adhoc/`, or delete. |

---

## Test coverage

| Area | File | Tests | What's checked |
|---|---|---:|---|
| Corpus smoke | `test_smoke_corpus.py` | 2 | every table / Chroma doc type populated |
| DB layer | `test_db.py` | 10 | reopen/WAL, re-grab keeps metadata, same-second latest row, embedding-id mapping, facts, queue, LIKE search |
| Chunking | `test_chunking.py` | 13 | window/overlap/coverage, all copies agree |
| BM25 (V1+V2) | `test_bm25.py` | 14 | top hit per planted fact, one hit/video, RRF maths, stemming, segment subset search |
| Vector + Shorty retrieval | `test_vector_retrieval.py` | 21 | ids/metadata, idempotent upsert, chunk/Shorty/synthetic-question layer top hits, indexing-model query, HSC vector rank |
| Fusion / weights | `test_fusion.py` | 5 | no per-layer weights (pinned), distance-domain scores, fused order applied, BM25 synth chunks, e2e context order |
| Query rewrite | `test_query_rewrite.py` | 6 | fallback never searches the template; LLM failure falls back; answer failure raises |
| Reranker | `test_reranker.py` | 11 | normalisation, grouping, buckets, neighbour expansion (real Chroma), blending, cap, rerank path e2e |
| Graph / triples | `test_graph_search.py` | 10 | top hits, cross-video one-hop, stopword crowding, zero confidence, LIKE wildcards |
| V2 e2e | `test_v2_pipeline.py` | 28 | each fixture question → right source, cross-video fact, events in cause_effect, cancel, LLM failures, missing/stale indexes, instant candidates, router |
| Grounding audit | `test_grounding_audit.py` | 18 | numbers, quotes (straight and curly), separator de-noise, empty context, question-not-evidence |
| Batch processor | `test_batch_processor.py` | 16 | provider guard (exit 2), happy path, entity dedupe, timeout/error/garbage/empty → failed with nothing corrupted, permanent failure after 6 attempts |
| Ask app jobs + worker | `test_app_jobs.py` | 11 | HTTP e2e, three race interleavings, cancel, stale/valid job-file recovery, worker stdout |
| Library app | `test_library_app.py` | 6 | DB path, listing, filters, detail, 404, edit round trip |
| V1 all layers | `test_v1_flags.py` | 5 | BM25+graph+HSC on, video_ids filter, empty question |

**Not covered:** the V1 agent tool loop (`_agent_run` and its ~15 `_agent_*` tools), the global graph and graph reasoner, course and cluster generation, the eval tooling, `video_grabber.py` ingest and YouTube/yt-dlp fetching, the Chrome extension, SSE streaming beyond job completion, and any behaviour at real-data scale (1.5 GB DB).

---

## Bugs found and fixed

Each fix is its own commit with a regression test that fails before and passes after.

| # | Where | Bug | Cause |
|---|---|---|---|
| 1 | `transcript_database.py` | **A fresh install crashed on its 2nd start**: `cannot change into wal mode from within a transaction` in `ask_shorty_app` and `video_grabber` at import | `PRAGMA journal_mode=WAL` ran while the implicit transaction from `INSERT OR IGNORE` was open. On a new DB it silently stayed in rollback mode, then every later open raised. Shaun's DB was already WAL, which makes the pragma a no-op, so he never hit it. |
| 2 | `ask_shorty.py`, `hsc/hsc_search.py` | **V1/HSC vector search returned near-random neighbours** unless the index was built with Chroma's default model | `collection.query(query_texts=…)` embeds with Chroma's built-in model, not the SentenceTransformer that indexed (fine-tuned via `chroma_model.txt`; the default dir is `transcript_chroma_finetuned`). `evaluate_rag.py` already knew this. Fixture repro: 16 of 21 vector-retrieval tests failed before (V1's first cited source was wrong for all 3 probed questions); all pass after. |
| 3 | `hsc/hsc_search.py` | Vector layer failures were invisible | Bare `except Exception: continue`. It now logs a warning. |
| 4 | `ask_shorty.py` | **RRF fusion had no effect on the default path**; with rerank on, the fusion signal was flattened | Fused score was `-rrf + 0.01*cos` (negative) and lists were never re-sorted, so `[:top_k]` kept cosine order. The reranker treats score as a distance (`1 - s/2` ≈ 1.01 for all). |
| 5 | `ask_shorty.py` | A restricted question ("limit to video IDs") could cite excluded videos | The filter was applied before fusion; BM25/graph/HSC hits for any video were fused back in. |
| 6 | `ask_shorty.py` | Non-JSON rewrite output searched **the prompt template** as a query; a rewrite timeout aborted the whole answer | The fallback returned `[user_prompt]`; no try/except around an optional step. |
| 7 | `reranker.py` | Groups whose hits were all >1.0 distance reported 1.0 and scored the wrong text | `best_retrieval_score` defaulted to 1.0 on a 0..2 scale. |
| 8 | `graph_search.py` | The real triple could be cut from results on a large facts table | Stopwords ("who", "is", "the") matched almost every row, then an **unordered** `LIMIT 200`. Rows are now ranked by matched tokens first. |
| 9 | `graph_search.py`, `transcript_database.py` | `%`/`_` in entity text acted as wildcards; confidence 0.0 became 1.0 | No LIKE escaping; `confidence or 1.0`. |
| 10 | `ask_shorty_v2.py` | Grounding audit **false positives**: "March 14 2021" became `142021` | The thousands-separator regex joined digits across any whitespace. |
| 11 | `ask_shorty_v2.py` | Curly-quoted fabrications were never audited; numbers from the question counted as evidence | Straight-quote regex only; the audit blob was the whole user prompt, including the question. |
| 12 | `ask_shorty_v2.py` | Missing V2 BM25 pickles gave **silently empty layers** | No check; it now emits a `warning` event and logs to stderr. |
| 13 | `v2_bm25.py`, `ask_shorty_v2.py` | Rebuilt V2 indexes were ignored until restart | Cache keyed by path only; the engine pinned index objects on first load. |
| 14 | `v2_memory_maps.py` | The last word of most questions never matched instant candidates; multi-word aliases were unreachable; candidate order varied between runs | `str.split()` kept `?`; only single-token alias lookup; returned `set` order (hash-seed dependent). |
| 15 | `query_router.py` | "What **caused** X?" routed to `general` and never used the event index | The regex matched `cause` but not `caused`, `causes` or `effects`. |
| 16 | `shorty_generator.py`, `batch_processor.py` | **Empty or garbage LLM output saved a header-only Shorty and marked the task completed** | The header was prepended to whatever came back, with no validation. |
| 17 | `batch_processor.py` | Garbage triple/segment JSON **deleted existing facts/segments** and marked the task completed | `[]` → `replace_*_for_video([])` does `DELETE` then nothing. Events now refuse to wipe existing rows. |
| 18 | `triple_extractor.py` | One bad response was cached forever as `[]` | `save_cached_triples` ran on empty results. |
| 19 | `entity_extractor.py`, `batch_processor.py`, `library_app.py` | Re-running `entities` duplicated every entity; library "regenerate" lost all entities if the LLM failed | `store_entities` only appended; the library deleted before extracting. |
| 20 | `shorty_generator.py` | Synthetic questions needed `ANTHROPIC_API_KEY` under the default `SHORTY_PROVIDER=openrouter` | Hard-wired Anthropic tool-use call. |
| 21 | `batch_processor.py` | The provider-mismatch guard exited with status 0 | A bare `return`; it now `SystemExit(2)`. |
| 22 | `transcript_database.py` | **Re-grabbing a video wiped `json_metadata`, `local_path`, `watch_date`, `created_at`** | `INSERT OR REPLACE` deletes the row first; it is now an upsert. |
| 23 | `transcript_database.py`, `transcript_rag_enhanced.py`, `library_app.py`, `video_grabber.py` | "Latest transcript" was ambiguous within one second; **synthetic-question → Chroma id mapping was reversed** | `ORDER BY created_at DESC` with 1-second timestamps and no tie-breaker (`id DESC` added). |
| 24 | `transcript_database.py` | A `db_path` with no directory crashed | `os.makedirs('')`. |
| 25 | `ask_shorty_app.py` | **Job-state races**: a timed-out or cancelled job flipped back to completed; a just-completed job was overwritten with error/timeout; an exception overwrote `cancelled` | Check-then-write across separate connections. `_update_job(expect=…)` is now a single compare-and-set UPDATE. |
| 26 | `ask_shorty_app.py` | An errored job could be served **another job's answer** | Recovery files lived in `<repo>/data/jobs`, were keyed only by job id, and were never cleared at start. They now live beside the DB and are cleared per job. |
| 27 | `ask_shorty_worker.py` | Its "one JSON object on stdout" output was unparseable | The engine prints progress to stdout; it now goes to stderr. |
| 28 | `library_app.py` | The library browsed a different DB than the Ask app | `DB_PATH` hard-coded; `ASK_SHORTY_DB_PATH` was ignored. |
| 29 | `requirements.txt` | A clean install couldn't run the apps | Missing flask, openai, anthropic, requests, httpx, youtube-transcript-api, yt-dlp. |

**SQL built by string concatenation:** I checked every `execute(f"…")` and `"…" + where_sql + "…"` site. All interpolate fixed clause text or `?` placeholder lists; values are always bound parameters. `queue_video.py` concatenates, but only hard-coded constants. **No injection found.**

**The claimed fusion weights (Shorty 2.0x, questions 1.5x, chunks 1.0x)** are not implemented anywhere. They appear as an unchecked ROADMAP Phase 2 item, not in the README. All three Chroma layers collapse into one list keyed by best cosine distance, and RRF weights every list equally. `test_layer_weights_claim_is_not_implemented` pins this.

---

## Risks and known issues (not fixed)

1. **`video_grabber.py` has heavy import-time side effects.** It constructs `TranscriptRAG()` (model download), creates `<repo>/data/transcripts.db` and writes `grab_log.txt`. The only pre-existing test (`test_annotations.py`) therefore can't be collected without the model.
2. **Job timeout fires on long, healthy jobs.** `JOB_RUNNING_TIMEOUT_SEC = 120` is measured from `updated_at`, which is only written on status changes, not on progress. A V2 query with a CPU cross-encoder, or a slow model, gets marked `error/timeout` while still working. Fix: touch `updated_at` from `emit`, or time out on a heartbeat.
3. **Orphaned queue rows.** A worker crash leaves rows in `started` forever (only the manual `reset_stale.py` clears them). Separately, `process_queue_tasks` resets `failed` → `pending` inside the same run, so a persistent LLM error is retried 6 times back-to-back, each one a paid call.
4. **Hard-coded Windows paths** are the defaults in `AskShorty.__init__`, `EnhancedTranscriptRAG.__init__` and `ask_shorty_app._FULL_DB` (39 files in total). On any other machine these silently point at a non-existent DB, which `TranscriptDatabase` would *create empty*.
5. **The apps bind `0.0.0.0` with no auth.** `library_app` has a fixed `secret_key` and POST routes that trigger paid LLM regeneration; anyone on the LAN can use them.
6. **Docs are stale.** The README's step 4 (`batch_processor.py --queue --provider anthropic`) now hits the mismatch guard unless `.env` sets `SHORTY_PROVIDER=anthropic`. The README and PROJECT_SUMMARY still describe Anthropic as the primary path and "five fused layers"; V2 (the active branch) uses none of the Chroma layers.
7. **V2 context has no relevance floor.** On a small library it sends every candidate (up to `LLM_VIDEOS=10`) to the LLM; all 7 fixture videos appeared in context for every question. On the real library that cap holds, but irrelevant blocks still cost tokens and invite confusion.
8. **In V2's skip-CE fast path**, when the cross-encoder is skipped *because* video BM25 is dominant, results are re-sorted by segment score, which can demote that dominant video. This may be intended; worth an eval check.
9. **Graph search now scans the full facts table** per query (ranking before LIMIT). That is correct, but on millions of rows consider an FTS5 table over `subject || relation || object`.
10. **Reranker grouping** uses `chunk_index // 3` buckets, not "within 3 of each other" as documented: adjacent chunks 2 and 3 split, 0 and 2 merge.
11. **28 `except Exception: pass` sites** remain (mostly logging, UI emits and best-effort writes). The retrieval-relevant ones are fixed; the rest deserve a pass.
12. `enrich_candidates.py` still uses `query_texts=` (same embedding-mismatch bug as #2, eval tooling only). `library_app` counts an empty-string Shorty as "has Shorty", and `?page=abc` causes a 500.

## What couldn't be tested here

- **Real models:** `us.aws.cdn.hf.co` is blocked, so all retrieval tests ran with stand-in embedder/cross-encoder. Score *magnitudes* (e.g. V2's `CE_SKIP_*` thresholds) are untested with the real ms-marco model.
- **Real LLMs:** there's no OpenRouter or Anthropic key, so prompt quality, Shorty quality and real JSON-compliance rates are untested.
- **Real data at scale:** the 1.5 GB DB isn't in the repo (correctly), so there is no latency or memory check of `V2MemoryMaps` or the BM25 pickles at size.
- **YouTube / yt-dlp ingest, the grabber bookmarklet and the Chrome extension.**
- **Windows-specific behaviour** (the Chroma `os._exit` crash the code works around, console encoding).
- **The V1 agent tool loop**, the global graph and graph reasoner, and the courses/clusters/eval scripts.

## The 5 most valuable next steps

1. **Run this suite on Shaun's machine with real models** (`ASK_SHORTY_TEST_REAL_MODELS=1`), then add 10–20 real-library regression questions (video ids only, no transcript text in the repo) to `evaluate_rag.py`. Re-measure V1 after fixes #2 and #4: its numbers were likely depressed by the embedding mismatch whenever the fine-tuned index was in use.
2. **Pick one pipeline.** V2 is cleaner, faster and better grounded. Either retire V1's Chroma path or make it a V2 layer. Then delete the alias module, the duplicate chunkers, the second router and the unused worker. This removes about a third of the retrieval code to maintain.
3. **Make configuration explicit and portable.** One `config.py` that reads `ASK_SHORTY_DB_PATH` / `ASK_SHORTY_CHROMA_PATH` / the provider, fails loudly when the DB doesn't exist (instead of creating an empty one), and replaces the 39 hard-coded `C:\Users\…` defaults. Remove import-time side effects from `video_grabber.py`.
4. **Harden the job and queue lifecycle.** Add a heartbeat `updated_at` for long jobs, reap `started` rows older than N minutes, and stop in-run auto-retry of `failed` rows (or add backoff and a per-run cap). Bind the apps to `127.0.0.1` by default.
5. **Run CI on every push.** The suite needs no network or keys and runs in about 10 s; a GitHub Actions job with `pip install -r requirements-dev.txt && pytest` (CPU torch) would have caught 1, 4, 16, 17 and 25 when they were introduced.
