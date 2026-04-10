#!/usr/bin/env python3
"""
Ask Shorty query pipeline.

Combines:
- Transcript chunks (RAG)
- Shorties
- Synthetic questions
- (Entities will be added via entity_extractor)

Uses Anthropic Claude for:
- Query rewriting into multiple angles
- Final answer generation from aggregated context
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING, Callable, Tuple
import logging
import os
import sqlite3
import time as _time

from anthropic_client import get_client
from transcript_database import TranscriptDatabase

if TYPE_CHECKING:
    # Only imported for type checking; runtime import is deferred to avoid
    # initializing Chroma / SentenceTransformer at startup.
    from transcript_rag import TranscriptRAG  # pragma: no cover


logger = logging.getLogger(__name__)

ANSWER_MODEL = "claude-sonnet-4-20250514"
MAX_SHORTIES_IN_CONTEXT = 10

# ---------------------------------------------------------------------------
# Reranker feature flags — set via environment variables
# ---------------------------------------------------------------------------

# Set ASK_SHORTY_RERANK=1 to enable second-stage CrossEncoder reranking.
ENABLE_RERANKING: bool = os.getenv("ASK_SHORTY_RERANK", "0").strip() not in ("", "0", "false", "False")

# Set ASK_SHORTY_RERANK_VERBOSE=1 to print per-group debug scores.
RERANK_VERBOSE: bool = os.getenv("ASK_SHORTY_RERANK_VERBOSE", "0").strip() not in ("", "0", "false", "False")

# How many hits per layer to collect when reranking (more input = better recall
# going into the reranker, at the cost of more CrossEncoder calls).
RERANK_TOP_K_INPUT: int = int(os.getenv("ASK_SHORTY_RERANK_TOP_K", "20"))

# How many top evidence groups to include in the final prompt.
RERANK_CONTEXT_GROUPS: int = int(os.getenv("ASK_SHORTY_RERANK_CONTEXT_N", "12"))

# BM25 hybrid (Reciprocal Rank Fusion with vector hits). ASK_SHORTY_BM25=1 to enable.
ENABLE_BM25: bool = os.getenv("ASK_SHORTY_BM25", "0").strip() not in ("", "0", "false", "False")

# Knowledge-graph retrieval over facts table. ASK_SHORTY_GRAPH=1 to enable.
ENABLE_GRAPH: bool = os.getenv("ASK_SHORTY_GRAPH", "0").strip() not in ("", "0", "false", "False")

# Hierarchical Semantic Compression (segments/events + routed RRF). ASK_SHORTY_HSC=1
ENABLE_HSC: bool = os.getenv("ASK_SHORTY_HSC", "0").strip() not in ("", "0", "false", "False")

QUERY_REWRITE_SYSTEM = """You are a query rewriting engine.

Given a single user question, you generate 3–4 alternative phrasings or angles
that are semantically equivalent but emphasize different aspects of the question.

Output ONLY a JSON array of strings, nothing else.
"""


QUERY_REWRITE_USER_TEMPLATE = """Rewrite this question into 3–4 alternate queries that highlight different angles.

Original question:
{question}
"""


ANSWER_SYSTEM_PROMPT = """You are Ask Shorty, an AI assistant that answers questions about indexed video and podcast content.

You have access to multiple types of context:
- SHORTY: A complete dense knowledge brief for an entire video. Treat each Shorty as a complete and sufficient knowledge source for that video. You do NOT need transcript chunks to answer questions about a video if its Shorty is present.
- CHUNK: A transcript excerpt from a specific video.
- SYNTHETIC_QUESTION: A pre-generated question that matched the user's query.

IMPORTANT RULES:
- Never say you lack context when Shorties are present. Shorties contain all key facts, entities, relationships, numbers, and details from the video.
- For cross-video questions, use ALL Shorties provided to identify themes, connections, and commonalities between videos.
- Always cite which video your answer comes from by title, and append the watch date in the format (watched: YYYY-MM-DD) immediately after each citation. The watch date is provided in the WATCHED field of each context block.
- ATTRIBUTION: When citing a video, always use the CHANNEL name as the creator. The CHANNEL field in each context block identifies who made the video. Never use the name of a person or subject discussed in the video as the creator.
- Be direct and confident in your answers.
"""


def _call_claude_json_array(system_prompt: str, user_prompt: str) -> List[str]:
    """Helper that expects Claude to return a JSON array of strings via tool use."""
    client = get_client()

    tools = [
        {
            "name": "rewrite_queries",
            "description": "Store alternate phrasings of the user query",
            "input_schema": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Rewritten query variants",
                    }
                },
                "required": ["queries"],
            },
        }
    ]

    resp = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=512,
        temperature=0.2,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=tools,
        tool_choice={"type": "tool", "name": "rewrite_queries"},
    )

    rewrites: List[str] = []
    for block in resp.content:
        btype = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
        if btype == "tool_use":
            name = getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
            if name != "rewrite_queries":
                continue
            tool_input = getattr(block, "input", None) if not isinstance(block, dict) else block.get("input")
            if isinstance(tool_input, dict):
                items = tool_input.get("queries", [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str):
                            q = item.strip()
                            if q:
                                rewrites.append(q)
            break

    if not rewrites:
        logger.warning("Query rewriting tool returned no queries; falling back to original.")
        return [user_prompt.strip()]
    return rewrites


def _call_claude_answer(system_prompt: str, user_prompt: str) -> str:
    """Helper to get final answer text from Claude."""
    client = get_client()
    resp = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=2048,
        temperature=0.3,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts: List[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


class AskShorty:
    def __init__(self):
        self.db = TranscriptDatabase()
        # Lazy-init RAG so that any heavy Chroma / SentenceTransformer setup
        # happens only on first real query, not at import time.
        self._rag: Optional["TranscriptRAG"] = None
        self._reranker = None  # lazy-loaded when reranking is enabled
        self._bm25_search = None  # lazy BM25 index

    def _get_bm25(self):
        """Lazy-load BM25 keyword index (may be empty if not built)."""
        if self._bm25_search is None:
            from bm25_index import BM25Search, default_bm25_index_path

            path = str(default_bm25_index_path(self.db.db_path))  # type: ignore[attr-defined]
            self._bm25_search = BM25Search(path)
        return self._bm25_search

    def _vector_video_rank(
        self,
        chunk_results: List[Dict[str, Any]],
        shorty_results: List[Dict[str, Any]],
        synq_results: List[Dict[str, Any]],
    ) -> List[str]:
        """Order video_ids by best Chroma distance (lower is better)."""
        by_video: Dict[str, float] = {}
        for r in chunk_results + shorty_results + synq_results:
            m = r.get("metadata") or {}
            vid = m.get("video_id")
            if not vid:
                continue
            sc = float(r.get("score", 1.0))
            if vid not in by_video or sc < by_video[vid]:
                by_video[vid] = sc
        ranked = sorted(by_video.items(), key=lambda x: x[1])
        return [v for v, _ in ranked]

    def _apply_rrf_fusion(
        self,
        question: str,
        chunk_results: List[Dict[str, Any]],
        shorty_results: List[Dict[str, Any]],
        synq_results: List[Dict[str, Any]],
        hsc_rank: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Merge vector layers with optional BM25 + graph using RRF; adjust scores for reranking.
        Returns (chunk_results, shorty_results, synq_results, fusion_debug_dict).
        """
        from bm25_index import reciprocal_rank_fusion

        vector_rank = self._vector_video_rank(chunk_results, shorty_results, synq_results)
        lists: List[List[str]] = []
        if hsc_rank:
            lists.append(hsc_rank)
        lists.append(vector_rank)

        bm25_hits: List[Dict[str, Any]] = []
        graph_hits: List[Dict[str, Any]] = []
        hsc_set: set = set(hsc_rank or [])

        if ENABLE_BM25:
            try:
                bm25 = self._get_bm25()
                bm25_hits = bm25.search(question, top_k=RERANK_TOP_K_INPUT)
                lists.append([h["video_id"] for h in bm25_hits])
            except Exception as exc:
                logger.warning("BM25 search failed: %s", exc)

        if ENABLE_GRAPH:
            try:
                from graph_search import GraphSearch

                gs = GraphSearch(self.db.db_path)  # type: ignore[attr-defined]
                graph_hits = gs.search(question, top_k=15)
                lists.append([h["video_id"] for h in graph_hits])
            except Exception as exc:
                logger.warning("Graph search failed: %s", exc)

        fused = reciprocal_rank_fusion(lists, k=60)
        rrf_map = {vid: sc for vid, sc in fused}

        vector_vids: set = set()
        for r in chunk_results + shorty_results + synq_results:
            vid = (r.get("metadata") or {}).get("video_id")
            if vid:
                vector_vids.add(vid)

        bm_set = {h["video_id"] for h in bm25_hits}
        g_set = {h["video_id"] for h in graph_hits}

        def _prov(vid: str) -> str:
            parts: List[str] = []
            if vid in vector_vids:
                parts.append("vector")
            if vid in hsc_set:
                parts.append("hsc")
            if vid in bm_set:
                parts.append("bm25")
            if vid in g_set:
                parts.append("graph")
            return ",".join(parts) if parts else "vector"

        for r in chunk_results + shorty_results + synq_results:
            m = dict(r.get("metadata") or {})
            vid = m.get("video_id")
            if vid and vid in rrf_map:
                r["score"] = -rrf_map[vid] + 0.01 * float(r.get("score", 1.0))
            if vid:
                m["fusion_source"] = _prov(vid)
                r["metadata"] = m

        synth_added: set = set()
        for h in bm25_hits:
            vid = h["video_id"]
            if vid in vector_vids or vid in synth_added or vid not in rrf_map:
                continue
            chunk_results.append(
                {
                    "id": f"{vid}:bm25:synth",
                    "text": h.get("text_preview") or "",
                    "score": -rrf_map[vid],
                    "metadata": {
                        "video_id": vid,
                        "type": "chunk",
                        "fusion_source": _prov(vid),
                    },
                    "query": question,
                }
            )
            synth_added.add(vid)

        for h in graph_hits:
            vid = h["video_id"]
            if vid in vector_vids or vid in synth_added or vid not in rrf_map:
                continue
            chunk_results.append(
                {
                    "id": f"{vid}:graph:synth",
                    "text": h.get("text_preview") or "",
                    "score": -rrf_map[vid],
                    "metadata": {
                        "video_id": vid,
                        "type": "chunk",
                        "fusion_source": _prov(vid),
                    },
                    "query": question,
                }
            )
            synth_added.add(vid)

        preview = [
            {"video_id": vid, "rrf": round(sc, 5), "sources": _prov(vid)}
            for vid, sc in fused[:12]
        ]
        return chunk_results, shorty_results, synq_results, {
            "rrf_top": preview,
            "bm25_hits": len(bm25_hits),
            "graph_hits": len(graph_hits),
            "hsc_rank_len": len(hsc_rank or []),
        }

    def _get_rag(self) -> "TranscriptRAG":
        """Lazily construct the TranscriptRAG instance on first use."""
        if self._rag is None:
            # Deferred import so that importing ask_shorty.py does not import
            # transcript_rag_enhanced or touch Chroma until needed.
            from transcript_rag import TranscriptRAG as _TranscriptRAG

            self._rag = _TranscriptRAG()
        return self._rag

    def _get_reranker(self):
        """Lazily import and construct the Reranker (loads CrossEncoder on first call)."""
        if self._reranker is None:
            from reranker import Reranker
            self._reranker = Reranker()
        return self._reranker

    def _load_video_title_map(self) -> Dict[str, str]:
        """Return {video_id: title} for all videos in the database."""
        try:
            conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
            cursor = conn.cursor()
            cursor.execute("SELECT video_id, title FROM videos")
            result = {row[0]: (row[1] or row[0]) for row in cursor.fetchall()}
            conn.close()
            return result
        except Exception as exc:
            logger.warning("Could not load video title map: %s", exc)
            return {}

    def _load_video_meta_map(self) -> Dict[str, Dict[str, str]]:
        """Return {video_id: {title, channel, watch_date}} for all videos."""
        try:
            conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
            cursor = conn.cursor()
            cursor.execute("SELECT video_id, title, channel, watch_date FROM videos")
            result: Dict[str, Dict[str, str]] = {}
            for row in cursor.fetchall():
                vid, title, channel, watch_date = row
                result[vid] = {
                    "title": title or vid,
                    "channel": channel or "",
                    "watch_date": (watch_date or "")[:10],  # keep YYYY-MM-DD only
                }
            conn.close()
            return result
        except Exception as exc:
            logger.warning("Could not load video meta map: %s", exc)
            return {}

    def _rewrite_query(self, question: str) -> List[str]:
        user_prompt = QUERY_REWRITE_USER_TEMPLATE.format(question=question.strip())
        rewrites = _call_claude_json_array(QUERY_REWRITE_SYSTEM, user_prompt)
        # Always include original question as first element
        base = [question.strip()]
        for q in rewrites:
            if q not in base:
                base.append(q)
        # Cap to 4 variants total
        return base[:4]

    def _search_layer(
        self,
        query_variants: List[str],
        type_filter: Optional[str] = None,
        top_k: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Generic search against Chroma with an optional type filter.
        Returns list of dicts: {text, score, metadata}.
        """
        results: List[Dict[str, Any]] = []
        where = {}
        if type_filter:
            where["type"] = type_filter

        rag = self._get_rag()
        for q in query_variants:
            res = rag.collection.query(
                query_texts=[q],
                n_results=top_k,
                where=where if where else None,
            )
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            scores = res.get("distances", [[]])[0]
            for i, doc in enumerate(docs):
                results.append(
                    {
                        "id": ids[i],
                        "text": doc,
                        "score": scores[i],
                        "metadata": metas[i],
                        "query": q,
                    }
                )
        # Sort best-first (cosine distance from Chroma is smaller=better)
        results.sort(key=lambda x: x.get("score", 1e9))
        return results

    def _filter_by_metadata(
        self,
        question: str,
        video_ids: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """
        Use Claude + SQLite to narrow candidate videos by channel / creator / date.

        Returns a list of video_ids to prefer. If it cannot infer anything
        useful, returns None to indicate "no metadata filter".
        """
        import sqlite3

        meta_system = """You are a metadata parser for video search.

Given a natural language question, extract:
- channel names or creator names, if any
- an optional date range (date_from, date_to) in ISO format YYYY-MM-DD
"""

        user_prompt = f"Question: {question.strip()}"

        client = get_client()
        tools = [
            {
                "name": "parse_metadata",
                "description": "Parse channel/creator names and optional date range from a question",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "channels": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "date_from": {"type": ["string", "null"]},
                        "date_to": {"type": ["string", "null"]},
                    },
                    "required": ["channels", "date_from", "date_to"],
                },
            }
        ]

        resp = client.messages.create(
            model=ANSWER_MODEL,
            max_tokens=256,
            temperature=0,
            system=meta_system,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "tool", "name": "parse_metadata"},
        )

        data: Dict[str, Any] = {"channels": [], "date_from": None, "date_to": None}
        for block in resp.content:
            btype = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
            if btype == "tool_use":
                name = getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
                if name != "parse_metadata":
                    continue
                tool_input = getattr(block, "input", None) if not isinstance(block, dict) else block.get("input")
                if isinstance(tool_input, dict):
                    data = tool_input
                break

        channels = [c.strip() for c in data.get("channels") or [] if isinstance(c, str) and c.strip()]
        date_from = (data.get("date_from") or "") or None
        date_to = (data.get("date_to") or "") or None

        if not channels and not date_from and not date_to:
            return None

        conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
        cursor = conn.cursor()

        where_clauses = []
        params: List[Any] = []

        if channels:
            # Match either channel or creator name stored in channel column
            placeholders = ",".join("?" for _ in channels)
            where_clauses.append(f"channel IN ({placeholders})")
            params.extend(channels)

        if date_from:
            where_clauses.append("watch_date >= ?")
            params.append(date_from)
        if date_to:
            where_clauses.append("watch_date <= ?")
            params.append(date_to)

        if video_ids:
            placeholders = ",".join("?" for _ in video_ids)
            where_clauses.append(f"video_id IN ({placeholders})")
            params.extend(video_ids)

        if not where_clauses:
            conn.close()
            return None

        where_sql = " AND ".join(where_clauses)
        cursor.execute(
            f"SELECT DISTINCT video_id FROM videos WHERE {where_sql}",
            params,
        )
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return None

        return [r[0] for r in rows]

    def _sqlite_shorty_keyword_search(
        self,
        question: str,
        video_ids: Optional[List[str]] = None,
        limit: int = MAX_SHORTIES_IN_CONTEXT,
    ) -> List[Dict[str, Any]]:
        """
        Fallback search that bypasses Chroma and uses SQLite + Shorties only.

        - Fetches videos that have a non-empty Shorty.
        - Does simple keyword matching against LOWER(shorty) and LOWER(title)
          using LIKE.
        - Scores by keyword overlap in Python and returns top matches.
        """
        import sqlite3
        import re

        text = question.lower()
        # Basic tokenization; ignore very short words
        words = [w for w in re.findall(r"\w+", text) if len(w) >= 3]
        if not words:
            return []

        conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
        cursor = conn.cursor()

        base_sql = """
            SELECT v.video_id, v.title, t.shorty
            FROM transcripts t
            JOIN videos v ON v.video_id = t.video_id
            WHERE t.shorty IS NOT NULL AND trim(t.shorty) != ''
        """
        params: List[Any] = []

        if video_ids:
            placeholders = ",".join("?" for _ in video_ids)
            base_sql += f" AND v.video_id IN ({placeholders})"
            params.extend(video_ids)

        # Build LIKE conditions for each keyword, across title and shorty
        like_clauses = []
        for w in words:
            like_clauses.append("LOWER(t.shorty) LIKE ?")
            params.append(f"%{w}%")
            like_clauses.append("LOWER(v.title) LIKE ?")
            params.append(f"%{w}%")

        if like_clauses:
            base_sql += " AND (" + " OR ".join(like_clauses) + ")"

        cursor.execute(base_sql, params)
        rows = cursor.fetchall()
        conn.close()

        results: List[Dict[str, Any]] = []
        for vid, title, shorty in rows:
            aggregate = (shorty or "") + " " + (title or "")
            lower = aggregate.lower()
            score = sum(1 for w in words if w in lower)
            if score > 0:
                results.append(
                    {
                        "video_id": vid,
                        "title": title or "",
                        "shorty": shorty or "",
                        "score": score,
                    }
                )

        # Higher score (more overlaps) is better
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def answer_question(
        self,
        question: str,
        video_ids: Optional[List[str]] = None,
        top_k_per_layer: int = 4,
        emit: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Main entrypoint for Ask Shorty.

        Returns:
        {
          "answer": "...",
          "used_context": [...],
          "sources": [...],
        }

        Optional *emit* callback receives structured debug events in real-time.
        Each event is a dict with at least {"type": str, "elapsed_ms": int}.
        """
        _t0 = _time.time()
        _debug_events: List[Dict[str, Any]] = []

        def _elapsed() -> int:
            return round((_time.time() - _t0) * 1000)

        def _emit(event_type: str, **data: Any) -> None:
            event = {"type": event_type, "elapsed_ms": _elapsed(), **data}
            _debug_events.append(event)
            label = data.get("label") or data.get("layer") or event_type
            print(f"[ask] [{_elapsed()}ms] {label}")
            if emit:
                try:
                    emit(event)
                except Exception:
                    pass

        _emit("step", label="starting answer_question")
        if not question or not question.strip():
            raise ValueError("Question is empty.")

        q = question.strip()

        # Use metadata parsing to narrow candidate videos (optional)
        _emit("step", label="metadata filter start")
        candidate_videos = self._filter_by_metadata(q, video_ids)
        _emit("step", label="metadata filter done",
              candidate_count=len(candidate_videos) if candidate_videos else 0)
        _emit("step", label="query rewrite start")
        rewrites = self._rewrite_query(q)
        _emit("rewrites", label="query rewrite done",
              original=q, variants=rewrites[1:])

        chunk_results: List[Dict[str, Any]] = []
        shorty_results: List[Dict[str, Any]] = []
        synq_results: List[Dict[str, Any]] = []

        # Chroma-based RAG search (chunks, shorties, synthetic questions)
        try:
            # Layer 1: transcript chunks
            _emit("step", label="RAG search start")
            chunk_where_type = "chunk"
            chunk_results = self._search_layer(
                rewrites,
                type_filter=chunk_where_type,
                top_k=top_k_per_layer,
            )

            # Layer 2: Shorties (global search or restricted by metadata)
            shorty_where: Dict[str, Any] = {"type": "shorty"}
            if candidate_videos:
                shorty_where["video_id"] = {"$in": candidate_videos}

            rag = self._get_rag()
            res = rag.collection.query(
                query_texts=rewrites,
                n_results=MAX_SHORTIES_IN_CONTEXT,
                where=shorty_where,
            )
            # Flatten results across rewrites
            all_ids = res.get("ids", [])
            all_docs = res.get("documents", [])
            all_metas = res.get("metadatas", [])
            all_scores = res.get("distances", [])
            for q_idx, docs in enumerate(all_docs):
                ids_row = all_ids[q_idx]
                metas_row = all_metas[q_idx]
                scores_row = all_scores[q_idx]
                for i, doc in enumerate(docs):
                    shorty_results.append(
                        {
                            "id": ids_row[i],
                            "text": doc,
                            "score": scores_row[i],
                            "metadata": metas_row[i],
                            "query": rewrites[q_idx],
                        }
                    )
            # Deduplicate by id and keep best score
            seen_shorties: Dict[str, Dict[str, Any]] = {}
            for r in shorty_results:
                rid = r["id"]
                if rid not in seen_shorties or r["score"] < seen_shorties[rid]["score"]:
                    seen_shorties[rid] = r
            shorty_results = sorted(seen_shorties.values(), key=lambda x: x["score"])[:MAX_SHORTIES_IN_CONTEXT]

            # Layer 3: synthetic questions
            synq_results = self._search_layer(
                rewrites,
                type_filter="synthetic_question",
                top_k=top_k_per_layer,
            )
            _emit("step", label="RAG search done",
                  chunk_count=len(chunk_results),
                  shorty_count=len(shorty_results),
                  synq_count=len(synq_results))
        except BaseException as e:
            _emit("error", label=f"RAG/Chroma search failed: {e!r}", message=str(e))
            raise

        # Optionally filter by video_ids if provided
        def _filter_by_videos(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not video_ids:
                return rows
            s = set(video_ids)
            out: List[Dict[str, Any]] = []
            for r in rows:
                vid = (r.get("metadata") or {}).get("video_id")
                if vid in s:
                    out.append(r)
            return out

        chunk_results = _filter_by_videos(chunk_results)
        shorty_results = _filter_by_videos(shorty_results)
        synq_results = _filter_by_videos(synq_results)

        fusion_debug: Dict[str, Any] = {}
        hsc_rank: Optional[List[str]] = None
        hsc_ctx: Optional[Dict[str, Any]] = None
        if ENABLE_HSC:
            try:
                from hsc.hsc_search import hsc_retrieve

                rag = self._get_rag()
                bm = self._get_bm25() if ENABLE_BM25 else None
                hsc_ctx = hsc_retrieve(
                    self.db.db_path,  # type: ignore[attr-defined]
                    q,
                    rag_collection=rag.collection,
                    bm25_search=bm,
                    enable_bm25=ENABLE_BM25,
                    enable_graph=ENABLE_GRAPH,
                )
                hsc_rank = hsc_ctx.get("ranked_video_ids") or []
            except Exception as exc:
                logger.warning("HSC retrieve failed; using non-HSC fusion: %s", exc)
                hsc_ctx = None
                hsc_rank = None

        if ENABLE_BM25 or ENABLE_GRAPH or ENABLE_HSC:
            chunk_results, shorty_results, synq_results, fusion_debug = self._apply_rrf_fusion(
                q, chunk_results, shorty_results, synq_results, hsc_rank=hsc_rank
            )
            if hsc_ctx:
                fusion_debug["query_type"] = hsc_ctx.get("query_type")
                fusion_debug["layer_used"] = hsc_ctx.get("layer_used")
                fusion_debug["event_hits"] = hsc_ctx.get("event_hits")
                fusion_debug["segment_hits"] = hsc_ctx.get("segment_hits")
                fusion_debug["reasoning_paths"] = hsc_ctx.get("reasoning_paths")
                fusion_debug["path_scores"] = hsc_ctx.get("path_scores")
                fusion_debug["reason_rank_len"] = hsc_ctx.get("reason_rank_len")
                fusion_debug["global_paths"] = hsc_ctx.get("global_paths")
                fusion_debug["global_path_scores"] = hsc_ctx.get("global_path_scores")
                fusion_debug["videos_in_path"] = hsc_ctx.get("videos_in_path")
                fusion_debug["global_rank_len"] = hsc_ctx.get("global_rank_len")
                fusion_debug["rarity_scores"] = hsc_ctx.get("rarity_scores")
                fusion_debug["node_frequencies_top"] = hsc_ctx.get("node_frequencies_top")
            _emit("fusion", label="RRF fusion", **fusion_debug)
            if hsc_ctx:
                _emit(
                    "hsc",
                    label="HSC",
                    query_type=hsc_ctx.get("query_type"),
                    layer_used=hsc_ctx.get("layer_used"),
                    event_hits=hsc_ctx.get("event_hits"),
                    segment_hits=hsc_ctx.get("segment_hits"),
                    route_reason=hsc_ctx.get("route_reason"),
                    reasoning_paths=hsc_ctx.get("reasoning_paths"),
                    path_scores=hsc_ctx.get("path_scores"),
                    query_entities=hsc_ctx.get("query_entities"),
                    reason_rank_len=hsc_ctx.get("reason_rank_len"),
                    global_paths=hsc_ctx.get("global_paths"),
                    global_path_scores=hsc_ctx.get("global_path_scores"),
                    videos_in_path=hsc_ctx.get("videos_in_path"),
                    global_rank_len=hsc_ctx.get("global_rank_len"),
                    rarity_scores=hsc_ctx.get("rarity_scores"),
                    node_frequencies_top=hsc_ctx.get("node_frequencies_top"),
                )

        # Load video metadata (title, channel, watch_date) for all videos once.
        video_meta = self._load_video_meta_map()

        def _enrich_hit(row: Dict[str, Any], layer: str) -> Dict[str, Any]:
            """Return a compact debug-friendly dict for one retrieval hit."""
            m = row.get("metadata") or {}
            vid = m.get("video_id", "")
            vm = video_meta.get(vid, {})
            text = row.get("text") or ""
            return {
                "video_id":    vid,
                "title":       vm.get("title", vid),
                "channel":     vm.get("channel", ""),
                "watch_date":  vm.get("watch_date", ""),
                "score":       round(float(row.get("score") or 0), 4),
                "type":        layer,
                "chunk_index": m.get("chunk_index"),
                "preview":     text[:160].replace("\n", " "),
                "fusion_source": m.get("fusion_source", ""),
            }

        _emit("hits", layer="chunk",
              count=len(chunk_results),
              hits=[_enrich_hit(r, "chunk") for r in chunk_results])
        _emit("hits", layer="shorty",
              count=len(shorty_results),
              hits=[_enrich_hit(r, "shorty") for r in shorty_results])
        _emit("hits", layer="synq",
              count=len(synq_results),
              hits=[_enrich_hit(r, "synq") for r in synq_results])

        def _meta_header(vid: str) -> str:
            """Build a one-line metadata header for a context block."""
            m = video_meta.get(vid, {})
            channel    = m.get("channel", "")
            watch_date = m.get("watch_date", "")
            parts = [f"video_id={vid}"]
            if channel:
                parts.append(f"CHANNEL={channel}")
            if watch_date:
                parts.append(f"WATCHED={watch_date}")
            return " ".join(parts)

        # ----------------------------------------------------------------
        # Branch A: Reranking path (ASK_SHORTY_RERANK=1)
        # ----------------------------------------------------------------
        if ENABLE_RERANKING:
            _emit("step", label="reranking enabled — grouping and scoring evidence")
            reranker = self._get_reranker()
            title_map = {vid: m["title"] for vid, m in video_meta.items()}

            all_hits = (
                reranker.normalize_flat_results(chunk_results,  title_map)
                + reranker.normalize_flat_results(shorty_results, title_map)
                + reranker.normalize_flat_results(synq_results,   title_map)
            )

            rag = self._get_rag()
            groups = reranker.group_hits(
                all_hits,
                collection=rag.collection,
                expand_neighbors=True,
            )
            ranked_groups = reranker.rerank_and_blend(
                q, groups, verbose=RERANK_VERBOSE
            )
            # Build context blocks with metadata headers injected
            raw_blocks = reranker.groups_to_context_blocks(
                ranked_groups, top_n=RERANK_CONTEXT_GROUPS
            )
            context_blocks = []
            for g, raw in zip(ranked_groups[:RERANK_CONTEXT_GROUPS], raw_blocks):
                header = _meta_header(g.video_id)
                # Replace the first line of raw block (which already has video_id)
                # with an enriched header that includes CHANNEL and WATCHED.
                lines = raw.split("\n", 1)
                enriched = header + ("\n" + lines[1] if len(lines) > 1 else "")
                context_blocks.append(enriched)
            _emit("step", label=f"reranking done — {len(groups)} groups, {len(context_blocks)} context blocks")
        else:
            # ----------------------------------------------------------------
            # Branch B: Original path (no reranking)
            # ----------------------------------------------------------------
            context_blocks = []

            def _fmt(row: Dict[str, Any], layer: str) -> str:
                m = row.get("metadata") or {}
                vid = m.get("video_id", "unknown_video")
                chunk_idx = m.get("chunk_index")
                header = (
                    f"[{layer}] {_meta_header(vid)}"
                    + (f" chunk={chunk_idx}" if chunk_idx is not None else "")
                )
                return f"{header}\n{row['text']}\n"

            for r in chunk_results[:top_k_per_layer]:
                context_blocks.append(_fmt(r, "chunk"))
            for r in shorty_results:
                context_blocks.append(_fmt(r, "shorty"))
            for r in synq_results[:top_k_per_layer]:
                context_blocks.append(_fmt(r, "synthetic_question"))

        if not context_blocks:
            answer_text = "I could not find any relevant information in your indexed videos to answer that."
            _emit("done", label="done (no context found)", total_elapsed_ms=_elapsed())
            return {
                "answer":       answer_text,
                "used_context": [],
                "sources":      [],
                "debug_events": _debug_events,
            }

        _emit("context", label=f"context assembled — {len(context_blocks)} blocks",
              count=len(context_blocks),
              blocks=context_blocks)

        merged_context = "\n---\n".join(context_blocks)
        user_prompt = f"User question:\n{q}\n\nContext passages:\n{merged_context}\n\nAnswer the question using ONLY the context above."
        _emit("step", label="calling Anthropic API")
        answer = _call_claude_answer(ANSWER_SYSTEM_PROMPT, user_prompt)
        _emit("step", label="got answer, saving")

        # Build a deduplicated sources list for the UI.
        sources: List[Dict[str, str]] = []
        seen_vids: set = set()
        for block in context_blocks:
            # Extract video_id from the header line
            import re as _re
            vid_match = _re.search(r"video_id=(\S+)", block.split("\n")[0])
            if not vid_match:
                continue
            vid = vid_match.group(1)
            if vid in seen_vids:
                continue
            seen_vids.add(vid)
            m = video_meta.get(vid, {})
            sources.append({
                "video_id":   vid,
                "title":      m.get("title", vid),
                "channel":    m.get("channel", ""),
                "watch_date": m.get("watch_date", ""),
            })

        _emit("done", label="done", total_elapsed_ms=_elapsed())

        return {
            "answer":       answer,
            "used_context": context_blocks,
            "sources":      sources,
            "debug_events": _debug_events,
        }


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1:
        q = " ".join(_sys.argv[1:])
    else:
        q = input("Question: ").strip()

    engine = AskShorty()
    result = engine.answer_question(q)
    print("\n=== ANSWER ===\n")
    print(result["answer"])

