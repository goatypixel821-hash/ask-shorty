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
from datetime import date, timedelta
import logging
import os
import sqlite3
from pathlib import Path
import time as _time
import json

from anthropic_client import get_client
from transcript_database import TranscriptDatabase

if TYPE_CHECKING:
    # Only imported for type checking; runtime import is deferred to avoid
    # initializing Chroma / SentenceTransformer at startup.
    from transcript_rag import TranscriptRAG  # pragma: no cover


def _load_shorty_dotenv() -> None:
    """Load ``shorty/.env`` so ASK_SHORTY_* vars apply (optional: pip install python-dotenv)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=True)


_load_shorty_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_ANSWER_MODEL = "claude-sonnet-4-20250514"
ANSWER_MODEL = (os.environ.get("ANSWER_MODEL") or "").strip() or DEFAULT_ANSWER_MODEL
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

# BM25 hybrid (Reciprocal Rank Fusion with vector hits). On by default; set ASK_SHORTY_BM25=0 to disable.
ENABLE_BM25: bool = os.getenv("ASK_SHORTY_BM25", "1").strip() not in ("", "0", "false", "False")

# Knowledge-graph retrieval over facts table. ASK_SHORTY_GRAPH=1 to enable.
ENABLE_GRAPH: bool = os.getenv("ASK_SHORTY_GRAPH", "0").strip() not in ("", "0", "false", "False")

# Hierarchical Semantic Compression (segments/events + routed RRF). ASK_SHORTY_HSC=1
ENABLE_HSC: bool = os.getenv("ASK_SHORTY_HSC", "0").strip() not in ("", "0", "false", "False")
ENABLE_AGENT: bool = os.getenv("ASK_SHORTY_AGENT", "0").strip().lower() in ("1", "true", "yes", "on")
AGENT_MAX_TOOL_CALLS = 8

QUERY_REWRITE_SYSTEM = """You are a query rewriting engine for a video transcript search system.

Given a user question, generate 4 search queries designed to find relevant videos:

1. A natural language paraphrase (slightly different wording, same meaning)
2. A technical/specific angle (use precise terminology the video might use)
3. A keyword-style query: 4-8 words you'd literally expect to find in a transcript about this topic — no question words, just the core nouns, verbs, and terms
4. Another semantic angle or a broader/narrower reformulation

The keyword query (item 3) is critical — it bridges the gap when users describe things in everyday language but transcripts use technical terms or specific product names.

Examples:
- User: "guy filming light moving across the wall" → keyword variant: "camera billion fps light travel room garage laser"
- User: "video about tiny computers running AI" → keyword variant: "edge AI inference microcontroller embedded neural network"
- User: "that trick where you fold paper to make it strong" → keyword variant: "corrugation origami cardboard structural strength fold"

Output ONLY a JSON array of 4 strings, nothing else.
"""


QUERY_REWRITE_USER_TEMPLATE = """Generate 4 search queries for this question: a paraphrase, a technical angle, a keyword query (literal transcript words), and a broader/narrower angle.

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

AGENT_SYSTEM_PROMPT = """You are Ask Shorty running in agent mode.

You can call tools to retrieve evidence from indexed videos and podcasts.
Use as many tool calls as needed up to the limit. If results are weak, try a
different angle before giving up.

IMPORTANT — conversation context:
The user may send short follow-up messages like "what is the video id" or
"tell me more about that". Earlier messages in this conversation are included
so you know what was just discussed. Use that context to resolve pronouns and
references (e.g. "that video", "the one you just mentioned"). You should still
call tools to verify facts rather than relying solely on prior turns.

IMPORTANT — only cite what you found:
Never invent or guess video titles, video IDs, or channel names. Only include
a video_id in your answer if a tool returned it. If no tool returned a match,
say so honestly instead of fabricating a plausible-sounding title. Every
video_id you cite MUST come from a tool result in this conversation.

When the user asks about a specific channel, creator, or what someone covers
across their videos, use get_videos_by_channel or get_videos_by_channel_with_shorties
first to list matching videos from the database (channel name is matched fuzzily,
e.g. "CNN" matches "CNN Breaking News"). Prefer get_videos_by_channel_with_shorties
for summarization or "what does X talk about" questions so you only get videos
that already have a Shorty; use get_videos_by_channel if you need every indexed
video from that channel including those without a Shorty yet.

For watch history, specific calendar ranges, or "what did I watch recently",
always call get_videos_by_date or get_recent_videos — never infer dates from
memory or training data. Use optional channel_name on those tools for combined
queries (e.g. CNN in December). get_recent_videos covers the last N days from
today on the machine running the app.

Never answer date, history, or channel questions from memory. Always call
the appropriate tool to retrieve fresh data from the database.

Video metadata in the database may include DESCRIPTION, TAGS, and CHAPTERS
(from YouTube json_metadata) when available; these are folded into V2 routing
text and Shorties for newly processed videos.

Citation requirements for the final answer:
- Cite sources using video title and channel.
- Include watch date as (watched: YYYY-MM-DD) when available.
- If quoting transcript details, include video_id and chunk hint when relevant.

You may use tools:
- search_vectors(query, type) where type is chunk|shorty|synthetic_question
- search_bm25(query) — hierarchical V2 retrieval (video + segment BM25, query
  expansion, segment rescue, cross-encoder rerank on general queries); prefer
  this for broad factual search across the library
- get_shorty(video_id)
- search_entities(name)
- get_transcript_chunk(video_id, timestamp_hint)
- search_segments(query, video_id optional) — search HSC segment summaries in SQLite; use when the user
  wants to know WHERE in a video something was discussed, needs a timestamp or time range, or asks
  "which part", "at what point", "when in the video". Optionally pass video_id to scope one video.
- search_events(query) — search structured causal events (title/cause/effect) in SQLite; use for questions
  like what caused something, why something happened, what happened after, consequences, or chains of events.
- get_videos_by_channel(channel_name, limit=10)
- get_videos_by_channel_with_shorties(channel_name, limit=10)
- get_videos_by_date(start_date, end_date, limit=20, channel_name optional)
- get_recent_videos(days=30, limit=20, channel_name optional)
- get_earliest_video(topic_query) — oldest watch_date video matching a topic (requires Shorty)
"""


def _anthropic_assistant_block_to_api_dict(block: Any) -> Optional[Dict[str, Any]]:
    """
    Turn an assistant content block from the Anthropic SDK into a JSON-serializable
    dict suitable for the next messages.create(..., messages=[...]) call.

    Needed so we do not drop extended-thinking blocks (e.g. ``thinking``) that must
    be echoed back before ``tool_use`` in tool loops — dropping them corrupts the
    conversation state the API expects and can yield unrelated answers.
    """
    if isinstance(block, dict):
        return dict(block)
    model_dump = getattr(block, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:
            return None
    return None


def _is_openrouter_model(model: str) -> bool:
    """
    Heuristic: OpenRouter models are typically namespace/name (contain '/').
    Anthropic models in this project are plain strings like 'claude-sonnet-...'.
    """
    m = (model or "").strip()
    return "/" in m


_openrouter_client = None


def _get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is not None:
        return _openrouter_client
    from openai import OpenAI

    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (required for ANSWER_MODEL via OpenRouter).")
    _openrouter_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return _openrouter_client


def _strip_json_fences(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if "```" in raw:
        import re as _re

        m = _re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            raw = m.group(1).strip()
    return raw


def _extract_json_array(text: str) -> Optional[List[str]]:
    """Best-effort extract a JSON array of strings from model output."""
    raw = _strip_json_fences(text)
    if not raw:
        return None
    # Slice outermost [...]
    s = raw.find("[")
    e = raw.rfind("]")
    if s != -1 and e != -1 and e > s:
        raw = raw[s : e + 1].strip()
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    out: List[str] = []
    for item in data:
        if isinstance(item, str):
            t = item.strip()
            if t:
                out.append(t)
    return out or None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extract a JSON object from model output."""
    raw = _strip_json_fences(text)
    if not raw:
        return None
    s = raw.find("{")
    e = raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        raw = raw[s : e + 1].strip()
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _call_answer_text(system_prompt: str, user_prompt: str, *, max_tokens: int, temperature: float) -> str:
    """
    Unified answer/rewrite call:
    - If ANSWER_MODEL is an OpenRouter model (contains '/'), use OpenAI-compatible client via OpenRouter.
    - Else use Anthropic client.
    """
    if _is_openrouter_model(ANSWER_MODEL):
        client = _get_openrouter_client()
        resp = client.chat.completions.create(
            model=ANSWER_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    client = get_client()
    resp = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
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


def _call_claude_json_array(system_prompt: str, user_prompt: str) -> List[str]:
    """Return a JSON array of strings using ANSWER_MODEL backend."""
    if _is_openrouter_model(ANSWER_MODEL):
        raw = _call_answer_text(system_prompt, user_prompt, max_tokens=512, temperature=0.2)
        arr = _extract_json_array(raw)
        if arr:
            return arr
        logger.warning("Query rewriting returned non-JSON; falling back to original question.")
        return [user_prompt.strip()]

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
    """Final answer generation using ANSWER_MODEL backend."""
    return _call_answer_text(system_prompt, user_prompt, max_tokens=2048, temperature=0.3)


class AskShorty:
    def __init__(self):
        self.db = TranscriptDatabase(
            os.environ.get("ASK_SHORTY_DB_PATH")
            or r"C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db"
        )
        # Lazy-init RAG so that any heavy Chroma / SentenceTransformer setup
        # happens only on first real query, not at import time.
        self._rag: Optional["TranscriptRAG"] = None
        self._reranker = None  # lazy-loaded when reranking is enabled
        self._bm25_search = None  # lazy BM25 index
        self._agent_messages: List[Dict[str, Any]] = []

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

        def _best_chunk_for_video(vid: str, q: str) -> str:
            """Return the transcript chunk most relevant to q for a BM25-only hit.

            Splits the full transcript into chunks and returns the one whose text
            has the most keyword overlap with q, falling back to text_preview.
            """
            try:
                import sqlite3 as _sqlite3
                conn2 = _sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
                cur2 = conn2.cursor()
                cur2.execute("SELECT text FROM transcripts WHERE video_id = ? LIMIT 1", (vid,))
                row2 = cur2.fetchone()
                conn2.close()
                if not row2 or not row2[0]:
                    return ""
                full_text = row2[0]
                # Split into ~800-char chunks with 200-char overlap (mirrors reindex_on_gpu)
                chunks: list = []
                start = 0
                while start < len(full_text):
                    end = min(start + 800, len(full_text))
                    chunks.append(full_text[start:end])
                    if end == len(full_text):
                        break
                    start = max(0, end - 200)
                if not chunks:
                    return full_text[:800]
                # Score by keyword overlap
                q_tokens = set(q.lower().split())
                best_chunk = max(chunks, key=lambda c: sum(1 for t in q_tokens if t in c.lower()))
                return best_chunk
            except Exception:
                return ""

        synth_added: set = set()
        for h in bm25_hits:
            vid = h["video_id"]
            if vid in vector_vids or vid in synth_added or vid not in rrf_map:
                continue
            best_text = _best_chunk_for_video(vid, question) or h.get("text_preview") or ""
            chunk_results.append(
                {
                    "id": f"{vid}:bm25:synth",
                    "text": best_text,
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

        data: Dict[str, Any] = {"channels": [], "date_from": None, "date_to": None}

        if _is_openrouter_model(ANSWER_MODEL):
            # OpenRouter path: ask for a JSON object (no Anthropic tool API).
            or_system = (
                meta_system
                + "\n\nRespond with ONLY a JSON object of the form "
                '{"channels": ["..."], "date_from": null, "date_to": null}. '
                "Use null when unknown. No markdown."
            )
            raw = _call_answer_text(or_system, user_prompt, max_tokens=256, temperature=0)
            parsed = _extract_json_object(raw)
            if parsed:
                data = parsed
            else:
                logger.warning(
                    "Metadata filter (OpenRouter) returned non-JSON; skipping filter."
                )
                return None
        else:
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

    def _agent_search_vectors(self, query: str, type_filter: str) -> Dict[str, Any]:
        hits = self._search_layer([query], type_filter=type_filter, top_k=6)[:6]
        video_meta = self._load_video_meta_map()
        payload: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        for h in hits:
            m = h.get("metadata") or {}
            vid = m.get("video_id", "")
            vm = video_meta.get(vid, {})
            chunk_idx = m.get("chunk_index")
            header = f"[{type_filter}] video_id={vid}"
            if vm.get("channel"):
                header += f" CHANNEL={vm['channel']}"
            if vm.get("watch_date"):
                header += f" WATCHED={vm['watch_date']}"
            if chunk_idx is not None:
                header += f" chunk={chunk_idx}"
            text = (h.get("text") or "").strip()
            context_blocks.append(f"{header}\n{text}\n")
            payload.append(
                {
                    "video_id": vid,
                    "title": vm.get("title", vid),
                    "channel": vm.get("channel", ""),
                    "watch_date": vm.get("watch_date", ""),
                    "score": round(float(h.get("score") or 0.0), 4),
                    "chunk_index": chunk_idx,
                    "text_preview": text[:260],
                }
            )
        return {"hits": payload, "context_blocks": context_blocks}

    def _agent_search_bm25(self, query: str) -> Dict[str, Any]:
        from ask_shorty_v2 import _shared_ask_shorty_v2_engine

        q = (query or "").strip()
        if not q:
            return {"hits": [], "context_blocks": []}

        eng = _shared_ask_shorty_v2_engine(str(self.db.db_path))  # type: ignore[attr-defined]
        vids, dbg = eng.retrieve_videos(q)
        scores_by_vid: Dict[str, float] = dict(dbg.get("rank_scores") or {})
        video_meta = self._load_video_meta_map()
        payload: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        for i, vid in enumerate(vids[:8]):
            vm = video_meta.get(vid, {})
            preview = ""
            if eng._maps:
                preview = (eng._maps.video_routing_text.get(vid) or "").strip()
            if not preview:
                conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT shorty FROM transcripts
                    WHERE video_id = ? AND shorty IS NOT NULL AND trim(shorty) != ''
                    ORDER BY id DESC LIMIT 1
                    """,
                    (vid,),
                )
                row = cur.fetchone()
                conn.close()
                if row and row[0]:
                    preview = (row[0] or "").strip()
            header = f"[v2_retrieval] video_id={vid}"
            if vm.get("channel"):
                header += f" CHANNEL={vm['channel']}"
            if vm.get("watch_date"):
                header += f" WATCHED={vm['watch_date']}"
            header += f" rank={i + 1}"
            context_blocks.append(f"{header}\n{preview[:260]}\n")
            payload.append(
                {
                    "video_id": vid,
                    "title": vm.get("title", vid),
                    "channel": vm.get("channel", ""),
                    "watch_date": vm.get("watch_date", ""),
                    "score": round(float(scores_by_vid.get(vid, 0.0)), 4),
                    "text_preview": preview[:260],
                }
            )
        return {"hits": payload, "context_blocks": context_blocks}

    def _agent_get_shorty(self, video_id: str) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
        cur = conn.cursor()
        cur.execute(
            """
            SELECT v.video_id, v.title, v.channel, v.watch_date, t.shorty
            FROM videos v
            LEFT JOIN transcripts t ON t.video_id = v.video_id
            WHERE v.video_id = ?
            ORDER BY t.created_at DESC
            LIMIT 1
            """,
            (video_id,),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return {"found": False, "context_blocks": []}
        vid, title, channel, watch_date, shorty = row
        text = (shorty or "").strip()
        if not text:
            return {
                "found": True,
                "video_id": vid,
                "title": title or vid,
                "channel": channel or "",
                "watch_date": (watch_date or "")[:10],
                "shorty": "",
                "context_blocks": [],
            }
        header = f"[shorty] video_id={vid}"
        if channel:
            header += f" CHANNEL={channel}"
        wd = (watch_date or "")[:10]
        if wd:
            header += f" WATCHED={wd}"
        return {
            "found": True,
            "video_id": vid,
            "title": title or vid,
            "channel": channel or "",
            "watch_date": wd,
            "shorty": text,
            "context_blocks": [f"{header}\n{text}\n"],
        }

    def _agent_search_entities(self, name: str) -> Dict[str, Any]:
        q = (name or "").strip().lower()
        if not q:
            return {"hits": [], "context_blocks": []}
        conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.video_id, e.name, e.type, e.aliases, v.title, v.channel, v.watch_date
            FROM entities e
            LEFT JOIN videos v ON v.video_id = e.video_id
            WHERE LOWER(e.name) LIKE ?
               OR LOWER(COALESCE(e.aliases, '')) LIKE ?
            LIMIT 20
            """,
            (f"%{q}%", f"%{q}%"),
        )
        rows = cur.fetchall()
        conn.close()
        hits: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        for row in rows:
            vid, ename, etype, aliases, title, channel, watch_date = row
            wd = (watch_date or "")[:10]
            hits.append(
                {
                    "video_id": vid,
                    "title": title or vid,
                    "channel": channel or "",
                    "watch_date": wd,
                    "name": ename,
                    "type": etype,
                    "aliases": aliases or "",
                }
            )
            header = f"[entity] video_id={vid}"
            if channel:
                header += f" CHANNEL={channel}"
            if wd:
                header += f" WATCHED={wd}"
            context_blocks.append(f"{header}\nENTITY={ename} TYPE={etype} ALIASES={aliases or ''}\n")
        return {"hits": hits, "context_blocks": context_blocks}

    def _agent_get_transcript_chunk(self, video_id: str, timestamp_hint: str) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
        cur = conn.cursor()
        cur.execute(
            """
            SELECT v.video_id, v.title, v.channel, v.watch_date, t.text
            FROM videos v
            LEFT JOIN transcripts t ON t.video_id = v.video_id
            WHERE v.video_id = ?
            ORDER BY t.created_at DESC
            LIMIT 1
            """,
            (video_id,),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return {"found": False, "context_blocks": []}
        vid, title, channel, watch_date, text = row
        text = (text or "").strip()
        if not text:
            return {"found": True, "video_id": vid, "text": "", "context_blocks": []}
        rag = self._get_rag()
        chunks = rag._chunk_transcript(text)  # reuses existing chunking logic
        hint = (timestamp_hint or "").strip().lower()
        idx = 0
        if hint:
            best_i = 0
            best_score = -1
            for i, ch in enumerate(chunks):
                score = ch.lower().count(hint)
                if score > best_score:
                    best_score = score
                    best_i = i
            idx = best_i
        chosen = chunks[idx] if chunks else text[:800]
        wd = (watch_date or "")[:10]
        header = f"[chunk] video_id={vid}"
        if channel:
            header += f" CHANNEL={channel}"
        if wd:
            header += f" WATCHED={wd}"
        header += f" chunk={idx}"
        return {
            "found": True,
            "video_id": vid,
            "title": title or vid,
            "channel": channel or "",
            "watch_date": wd,
            "chunk_index": idx,
            "text": chosen,
            "context_blocks": [f"{header}\n{chosen}\n"],
        }

    @staticmethod
    def _agent_channel_limit(raw: Any) -> int:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            n = 10
        return max(1, min(20, n))

    def _agent_get_videos_by_channel(self, channel_name: str, limit: int) -> Dict[str, Any]:
        q = (channel_name or "").strip()
        if not q:
            return {"videos": [], "context_blocks": []}
        like = f"%{q}%"
        lim = self._agent_channel_limit(limit)
        conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
        cur = conn.cursor()
        cur.execute(
            """
            SELECT v.video_id, v.title, v.channel,
                   EXISTS (
                       SELECT 1 FROM transcripts t
                       WHERE t.video_id = v.video_id
                         AND t.shorty IS NOT NULL AND TRIM(t.shorty) != ''
                   )
            FROM videos v
            WHERE LOWER(v.channel) LIKE LOWER(?)
            ORDER BY v.watch_date DESC
            LIMIT ?
            """,
            (like, lim),
        )
        rows = cur.fetchall()
        conn.close()
        videos: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        for vid, title, channel, has_shorty in rows:
            hs = bool(has_shorty)
            videos.append(
                {
                    "video_id": vid,
                    "title": title or vid,
                    "channel": channel or "",
                    "has_shorty": hs,
                }
            )
            context_blocks.append(
                f"[channel_list] video_id={vid} CHANNEL={channel or ''} "
                f"TITLE={title or vid} has_shorty={hs}\n"
            )
        return {"videos": videos, "context_blocks": context_blocks}

    def _agent_get_videos_by_channel_with_shorties(self, channel_name: str, limit: int) -> Dict[str, Any]:
        q = (channel_name or "").strip()
        if not q:
            return {"videos": [], "context_blocks": []}
        like = f"%{q}%"
        lim = self._agent_channel_limit(limit)
        conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT v.video_id, v.title, v.channel
            FROM videos v
            WHERE LOWER(v.channel) LIKE LOWER(?)
              AND EXISTS (
                  SELECT 1 FROM transcripts t
                  WHERE t.video_id = v.video_id
                    AND t.shorty IS NOT NULL AND TRIM(t.shorty) != ''
              )
            ORDER BY v.watch_date DESC
            LIMIT ?
            """,
            (like, lim),
        )
        rows = cur.fetchall()
        conn.close()
        videos: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        for vid, title, channel in rows:
            videos.append(
                {
                    "video_id": vid,
                    "title": title or vid,
                    "channel": channel or "",
                    "has_shorty": True,
                }
            )
            context_blocks.append(
                f"[channel_list+shorty] video_id={vid} CHANNEL={channel or ''} TITLE={title or vid}\n"
            )
        return {"videos": videos, "context_blocks": context_blocks}

    @staticmethod
    def _agent_history_limit(raw: Any) -> int:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            n = 20
        return max(1, min(50, n))

    def _agent_videos_in_watch_date_range(
        self,
        start_date: str,
        end_date: str,
        limit: int,
        channel_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        lim = self._agent_history_limit(limit)
        ch = (channel_name or "").strip()
        like = f"%{ch}%" if ch else None
        conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
        cur = conn.cursor()
        base_select = """
            SELECT v.video_id, v.title, v.channel, v.watch_date,
                   EXISTS (
                       SELECT 1 FROM transcripts t
                       WHERE t.video_id = v.video_id
                         AND t.shorty IS NOT NULL AND TRIM(t.shorty) != ''
                   )
            FROM videos v
        """
        if ch:
            cur.execute(
                base_select
                + """
            WHERE v.watch_date BETWEEN ? AND ?
              AND LOWER(v.channel) LIKE LOWER(?)
            ORDER BY v.watch_date DESC
            LIMIT ?
            """,
                (start_date, end_date, like, lim),
            )
        else:
            cur.execute(
                base_select
                + """
            WHERE v.watch_date BETWEEN ? AND ?
            ORDER BY v.watch_date DESC
            LIMIT ?
            """,
                (start_date, end_date, lim),
            )
        rows = cur.fetchall()
        conn.close()
        videos: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        for vid, title, channel, watch_date, has_shorty in rows:
            hs = bool(has_shorty)
            wd = (watch_date or "")[:10] if watch_date else ""
            videos.append(
                {
                    "video_id": vid,
                    "title": title or vid,
                    "channel": channel or "",
                    "watch_date": wd,
                    "has_shorty": hs,
                }
            )
            context_blocks.append(
                f"[watch_date] video_id={vid} CHANNEL={channel or ''} "
                f"TITLE={title or vid} WATCHED={wd} has_shorty={hs}\n"
            )
        return {"videos": videos, "context_blocks": context_blocks}

    def _agent_get_videos_by_date(
        self,
        start_date: str,
        end_date: str,
        limit: int,
        channel_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        s_raw = (start_date or "").strip()[:10]
        e_raw = (end_date or "").strip()[:10]
        if not s_raw or not e_raw:
            return {
                "error": "start_date and end_date are required (YYYY-MM-DD)",
                "videos": [],
                "context_blocks": [],
            }
        try:
            s_d = date.fromisoformat(s_raw)
            e_d = date.fromisoformat(e_raw)
        except ValueError:
            return {
                "error": "invalid_date_format_use_Yyyy_Mm_Dd",
                "videos": [],
                "context_blocks": [],
            }
        if s_d > e_d:
            s_d, e_d = e_d, s_d
        return self._agent_videos_in_watch_date_range(
            s_d.isoformat(), e_d.isoformat(), limit, channel_name
        )

    def _agent_get_recent_videos(
        self,
        days: Any,
        limit: int,
        channel_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            d = int(days)
        except (TypeError, ValueError):
            d = 30
        d = max(1, min(3650, d))
        end_d = date.today()
        start_d = end_d - timedelta(days=d - 1)
        return self._agent_videos_in_watch_date_range(
            start_d.isoformat(), end_d.isoformat(), limit, channel_name
        )

    def _agent_get_earliest_video(self, topic_query: str) -> Dict[str, Any]:
        """
        Return the earliest (oldest watch_date) video that matches a topic query.
        Uses watch_date + a non-empty Shorty requirement so results are usable immediately.
        """
        q = (topic_query or "").strip()
        if not q:
            return {"error": "topic_query_required", "video": None, "context_blocks": []}

        pat = f"%{q.lower()}%"
        sql = """
        SELECT
            v.video_id,
            v.title,
            v.channel,
            v.watch_date,
            t.shorty
        FROM videos v
        JOIN transcripts t ON t.video_id = v.video_id
        WHERE v.watch_date IS NOT NULL AND TRIM(v.watch_date) != ''
          AND t.shorty IS NOT NULL AND TRIM(t.shorty) != ''
          AND (
            lower(v.title) LIKE ?
            OR lower(v.channel) LIKE ?
            OR lower(t.shorty) LIKE ?
          )
        ORDER BY v.watch_date ASC
        LIMIT 1
        """
        import sqlite3

        conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
        cur = conn.cursor()
        cur.execute(sql, (pat, pat, pat))
        row = cur.fetchone()
        conn.close()
        if not row:
            return {"video": None, "context_blocks": []}
        vid, title, channel, watch_date, shorty = row
        header = f"video_id={vid} | watch_date={watch_date or ''} | channel={channel or ''} | title={title or ''}"
        block = header + "\n" + (shorty or "").strip()
        return {
            "video": {
                "video_id": vid,
                "title": title or "",
                "channel": channel or "",
                "watch_date": watch_date or "",
            },
            "context_blocks": [block],
        }

    @staticmethod
    def _format_segment_clock(seconds: Optional[float]) -> str:
        """Format segment time (seconds) as M:SS or H:MM:SS for display."""
        if seconds is None:
            return ""
        try:
            s = float(seconds)
        except (TypeError, ValueError):
            return ""
        if s < 0:
            s = 0.0
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(round(s % 60))
        if h > 0:
            return f"{h}:{m:02d}:{sec:02d}"
        return f"{m}:{sec:02d}"

    @staticmethod
    def _segment_early_mid_late(segment_index: int, total: int) -> str:
        if total <= 0:
            return "early"
        if total == 1:
            return "mid"
        third = total / 3.0
        if segment_index < third:
            return "early"
        if segment_index < 2 * third:
            return "mid"
        return "late"

    def _agent_search_segments(self, query: str, video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Search segments.summary with case-insensitive LIKE; optional filter by video_id.
        Returns up to 10 rows with timing or early/mid/late when start/end unset.
        """
        q = (query or "").strip()
        if not q:
            return {"error": "query_required", "results": [], "context_blocks": []}

        pat = f"%{q.lower()}%"
        db_path = self.db.db_path  # type: ignore[attr-defined]

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        if video_id and str(video_id).strip():
            vid_f = str(video_id).strip()
            cur.execute(
                """
                SELECT s.id, s.video_id, s.start_time, s.end_time, s.summary,
                       v.title, v.channel, v.watch_date
                FROM segments s
                JOIN videos v ON v.video_id = s.video_id
                WHERE lower(COALESCE(s.summary, '')) LIKE ?
                  AND s.video_id = ?
                ORDER BY COALESCE(s.start_time, 1e30), s.id
                LIMIT 10
                """,
                (pat, vid_f),
            )
        else:
            cur.execute(
                """
                SELECT s.id, s.video_id, s.start_time, s.end_time, s.summary,
                       v.title, v.channel, v.watch_date
                FROM segments s
                JOIN videos v ON v.video_id = s.video_id
                WHERE lower(COALESCE(s.summary, '')) LIKE ?
                ORDER BY COALESCE(v.watch_date, '') DESC, s.video_id,
                         COALESCE(s.start_time, 1e30), s.id
                LIMIT 10
                """,
                (pat,),
            )

        rows = cur.fetchall()

        order_cache: Dict[str, List[int]] = {}

        def segment_order_for_video(vid: str) -> List[int]:
            if vid not in order_cache:
                cur.execute(
                    """
                    SELECT id FROM segments
                    WHERE video_id = ?
                    ORDER BY COALESCE(start_time, id), id
                    """,
                    (vid,),
                )
                order_cache[vid] = [r[0] for r in cur.fetchall()]
            return order_cache[vid]

        results: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        for r in rows:
            seg_id, vid, st, et, summary, title, channel, watch_date = r
            st_ok = st is not None
            et_ok = et is not None
            if st_ok and et_ok:
                tr = (
                    f"{self._format_segment_clock(float(st))} - "
                    f"{self._format_segment_clock(float(et))}"
                )
            elif st_ok and not et_ok:
                tr = f"from {self._format_segment_clock(float(st))}"
            elif not st_ok and et_ok:
                tr = f"through {self._format_segment_clock(float(et))}"
            else:
                order_ids = segment_order_for_video(vid)
                try:
                    idx = order_ids.index(int(seg_id))
                except (ValueError, TypeError):
                    idx = 0
                tr = self._segment_early_mid_late(idx, len(order_ids))

            wd = (watch_date or "")[:10]
            results.append(
                {
                    "video_id": vid,
                    "title": title or "",
                    "channel": channel or "",
                    "watch_date": wd,
                    "start_time": st,
                    "end_time": et,
                    "summary": (summary or "")[:2000],
                    "time_range": tr,
                }
            )
            ch = channel or ""
            header = f"[segment] video_id={vid} CHANNEL={ch} WATCHED={wd} TIME_RANGE={tr}"
            context_blocks.append(f"{header}\n{(summary or '').strip()}\n")

        conn.close()
        return {"results": results, "context_blocks": context_blocks}

    def _agent_search_events(self, query: str) -> Dict[str, Any]:
        """Search events title/cause/effect with case-insensitive LIKE; up to 10 rows."""
        q = (query or "").strip()
        if not q:
            return {"error": "query_required", "results": [], "context_blocks": []}

        pat = f"%{q.lower()}%"
        db_path = self.db.db_path  # type: ignore[attr-defined]
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.id, e.video_id, e.title, e.cause, e.effect, e.systems,
                   v.title, v.channel, v.watch_date
            FROM events e
            JOIN videos v ON v.video_id = e.video_id
            WHERE lower(COALESCE(e.title, '')) LIKE ?
               OR lower(COALESCE(e.cause, '')) LIKE ?
               OR lower(COALESCE(e.effect, '')) LIKE ?
            ORDER BY COALESCE(v.watch_date, '') DESC, e.id
            LIMIT 10
            """,
            (pat, pat, pat),
        )
        rows = cur.fetchall()
        conn.close()

        results: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        for r in rows:
            _eid, vid, etitle, cause, effect, systems, vtitle, channel, watch_date = r
            wd = (watch_date or "")[:10]
            results.append(
                {
                    "video_id": vid,
                    "title": vtitle or "",
                    "channel": channel or "",
                    "watch_date": wd,
                    "event_title": etitle or "",
                    "cause": cause or "",
                    "effect": effect or "",
                    "systems": systems or "",
                }
            )
            ch = channel or ""
            header = f"[event] video_id={vid} CHANNEL={ch} WATCHED={wd} EVENT={(etitle or '')[:120]}"
            parts: List[str] = []
            if (cause or "").strip():
                parts.append(f"Cause: {cause.strip()}")
            if (effect or "").strip():
                parts.append(f"Effect: {effect.strip()}")
            if (systems or "").strip():
                parts.append(f"Systems: {systems.strip()}")
            context_blocks.append(f"{header}\n" + "\n".join(parts) + "\n")

        return {"results": results, "context_blocks": context_blocks}

    # ------------------------------------------------------------------
    # Hallucination guard — runs after agent produces its final answer
    # ------------------------------------------------------------------

    _VIDEO_ID_RE = None  # compiled lazily

    @classmethod
    def _get_video_id_re(cls):
        if cls._VIDEO_ID_RE is None:
            import re
            cls._VIDEO_ID_RE = re.compile(r"\b([a-zA-Z0-9_-]{11})\b")
        return cls._VIDEO_ID_RE

    def _verify_video_ids(self, candidate_ids: List[str]) -> Dict[str, bool]:
        """Return {video_id: exists} by checking the videos table."""
        if not candidate_ids:
            return {}
        try:
            conn = sqlite3.connect(self.db.db_path)  # type: ignore[attr-defined]
            cur = conn.cursor()
            placeholders = ",".join("?" for _ in candidate_ids)
            cur.execute(
                f"SELECT video_id FROM videos WHERE video_id IN ({placeholders})",
                candidate_ids,
            )
            found = {row[0] for row in cur.fetchall()}
            conn.close()
        except Exception:
            return {vid: True for vid in candidate_ids}
        return {vid: (vid in found) for vid in candidate_ids}

    def _hallucination_guard(
        self,
        answer: str,
        known_source_ids: set,
    ) -> str:
        """
        Post-process agent answer to catch hallucinated video references.

        1. Extract anything that looks like a YouTube video_id (11-char alphanumeric).
        2. Only flag IDs that the agent presented as citations (not random 11-char words).
           We look for IDs near context clues like parentheses, brackets, "video_id", or
           preceded by a watch URL pattern.
        3. Verify each against the videos table.
        4. Replace bogus IDs with a disclaimer note.
        5. If a title is in quotes/bold but has no video_id nearby, append an
           "[unverified]" tag.
        """
        import re

        pat = self._get_video_id_re()
        citation_pattern = re.compile(
            r"(?:"
            r"(?:video_id[=:\s]*)"                    # video_id= or video_id:
            r"|(?:youtube\.com/watch\?v=)"             # full URL
            r"|(?:youtu\.be/)"                         # short URL
            r"|(?:\[)"                                 # inside brackets [ID]
            r"|(?:\()"                                 # inside parens  (ID)
            r")"
            r"\s*([a-zA-Z0-9_-]{11})\b"
        )

        cited_ids: List[str] = []
        for m in citation_pattern.finditer(answer):
            vid = m.group(1)
            if vid not in cited_ids:
                cited_ids.append(vid)

        if not cited_ids:
            return answer

        valid_set = set(known_source_ids)
        unverified = [vid for vid in cited_ids if vid not in valid_set]
        if not unverified:
            return answer

        existence = self._verify_video_ids(unverified)
        bad_ids = [vid for vid in unverified if not existence.get(vid, True)]

        if not bad_ids:
            return answer

        for vid in bad_ids:
            answer = answer.replace(
                vid,
                f"~~{vid}~~ *(not found in library)*",
            )

        answer += (
            "\n\n---\n"
            "*Note: Some video IDs referenced above could not be verified in your library. "
            "I couldn't find a strong match for those — the information may be inaccurate.*"
        )
        return answer

    def _flag_unverified_titles(self, answer: str, known_source_ids: set) -> str:
        """
        If the answer mentions a video title in quotes or bold but there's no
        video_id within 200 chars, append [unverified] so the user knows it
        wasn't confirmed from the database.
        """
        import re

        title_pattern = re.compile(
            r'(?:'
            r'["\u201c]([^"\u201d]{8,80})["\u201d]'   # "Title Here" or \u201cTitle\u201d
            r'|'
            r'\*\*([^*]{8,80})\*\*'                     # **Title Here**
            r')'
        )
        vid_nearby = self._get_video_id_re()

        result_parts: List[str] = []
        last_end = 0

        for m in title_pattern.finditer(answer):
            title_text = m.group(1) or m.group(2)
            if not title_text:
                continue
            skip_words = {"unverified", "not found", "note:", "disclaimer"}
            if any(w in title_text.lower() for w in skip_words):
                continue
            start, end = m.start(), m.end()
            window = answer[max(0, start - 100):min(len(answer), end + 100)]
            nearby_ids = vid_nearby.findall(window)
            has_confirmed_id = any(vid in known_source_ids for vid in nearby_ids)
            if not has_confirmed_id and nearby_ids:
                pass
            elif not has_confirmed_id and not nearby_ids:
                result_parts.append(answer[last_end:end])
                result_parts.append(" [unverified]")
                last_end = end
                continue
            result_parts.append(answer[last_end:end])
            last_end = end

        if not result_parts:
            return answer
        result_parts.append(answer[last_end:])
        return "".join(result_parts)

    def _agent_run(
        self,
        question: str,
        emit: Optional[Callable[[Dict[str, Any]], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        tools = [
            {
                "name": "search_vectors",
                "description": "Search vector index by query and type.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["chunk", "shorty", "synthetic_question"],
                        },
                    },
                    "required": ["query", "type"],
                },
            },
            {
                "name": "search_bm25",
                "description": "Keyword search over indexed transcript corpus.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "get_shorty",
                "description": "Fetch one video's shorty from SQLite.",
                "input_schema": {
                    "type": "object",
                    "properties": {"video_id": {"type": "string"}},
                    "required": ["video_id"],
                },
            },
            {
                "name": "search_entities",
                "description": "Find matching entities by name/alias.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "get_transcript_chunk",
                "description": "Fetch a transcript chunk for a specific video.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string"},
                        "timestamp_hint": {"type": "string"},
                    },
                    "required": ["video_id", "timestamp_hint"],
                },
            },
            {
                "name": "get_videos_by_channel",
                "description": (
                    "List videos whose channel name contains the given substring (case-insensitive), "
                    'e.g. \"CNN\" matches \"CNN Breaking News\". Returns video_id, title, channel, and '
                    "has_shorty. Use when the user asks about a specific channel, creator, or what "
                    "someone covers across their videos — to see all matching indexed videos."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "channel_name": {"type": "string"},
                        "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 20},
                    },
                    "required": ["channel_name"],
                },
            },
            {
                "name": "get_videos_by_channel_with_shorties",
                "description": (
                    "Same as get_videos_by_channel but only videos that have a non-empty Shorty in SQLite. "
                    "Use when the user asks about a specific channel, creator, or cross-video themes — "
                    "especially for summarization or \"what does X cover\" where you need ready-made briefs."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "channel_name": {"type": "string"},
                        "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 20},
                    },
                    "required": ["channel_name"],
                },
            },
            {
                "name": "get_videos_by_date",
                "description": (
                    "List videos whose watch_date falls between start_date and end_date (inclusive), "
                    "strings YYYY-MM-DD, ordered by watch_date descending. Returns video_id, title, "
                    "channel, watch_date, has_shorty. Optional channel_name filters channel with a "
                    "case-insensitive substring match (e.g. December + CNN). Always use this for "
                    "calendar-range watch-history questions — never guess dates."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                        "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
                        "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50},
                        "channel_name": {
                            "type": "string",
                            "description": "Optional; if set, only videos whose channel contains this substring.",
                        },
                    },
                    "required": ["start_date", "end_date"],
                },
            },
            {
                "name": "get_recent_videos",
                "description": (
                    "Videos watched from today back N calendar days (inclusive), same fields as "
                    "get_videos_by_date. Optional channel_name for e.g. recent CNN watches. Use for "
                    "\"what did I watch recently\" or rolling windows — never infer from memory."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "days": {"type": "integer", "default": 30, "minimum": 1, "maximum": 3650},
                        "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 50},
                        "channel_name": {
                            "type": "string",
                            "description": "Optional channel substring filter (case-insensitive).",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "get_earliest_video",
                "description": (
                    "Find the oldest (earliest watch_date) video that matches a topic query. "
                    "Use when the user asks \"what was the first video I watched about X\". "
                    "Searches title/channel/shorty text; requires the video to have a non-empty Shorty."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"topic_query": {"type": "string"}},
                    "required": ["topic_query"],
                },
            },
            {
                "name": "search_segments",
                "description": (
                    "Search SQLite `segments.summary` (HSC segment summaries) with a case-insensitive "
                    "substring match. Optional `video_id` scopes to one video. Returns up to 10 rows: "
                    "video_id, title (from videos), start_time, end_time, summary, time_range — clock times "
                    "like \"8:32 - 12:45\" when start/end exist; if both are missing, early/mid/late by "
                    "order among that video's segments. Use when the user wants WHERE in a video something "
                    "was discussed, a timestamp, \"which part\", \"at what point\", or time in the video — "
                    "not as a substitute for vector/BM25 transcript search."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "video_id": {
                            "type": "string",
                            "description": "Optional; restrict matches to this video_id only.",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_events",
                "description": (
                    "Search SQLite `events` (structured causal events). Case-insensitive match on title, "
                    "cause, and effect. Returns up to 10 rows: video_id, video title, event_title, cause, "
                    "effect, systems (plus channel/watch_date). Use for causal questions: what caused X, "
                    "why something happened, what happened after, what led to, consequences — not for "
                    "generic \"what did they say\" retrieval."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        ]

        def _safe_json(v: Any) -> str:
            return json.dumps(v, ensure_ascii=False)

        def _tool_arg_snippet(inp: Dict[str, Any]) -> str:
            if inp.get("query"):
                q = str(inp.get("query") or "")
                vid = inp.get("video_id")
                if vid is not None and str(vid).strip():
                    return f"{q} (video_id={vid!r})"
                return q
            if inp.get("name"):
                return str(inp.get("name") or "")
            if inp.get("start_date") is not None and inp.get("end_date") is not None:
                s = f"{inp.get('start_date')}..{inp.get('end_date')}"
                cn = inp.get("channel_name")
                if cn:
                    s += f", channel={cn!r}"
                return s
            if "days" in inp:
                s = f"days={inp.get('days')}"
                cn = inp.get("channel_name")
                if cn:
                    s += f", channel={cn!r}"
                return s
            if inp.get("channel_name"):
                cn = str(inp.get("channel_name") or "")
                lim = inp.get("limit")
                if lim is not None:
                    return f"{cn}, limit={lim}"
                return cn
            if inp.get("video_id"):
                vid = str(inp.get("video_id") or "")
                th = inp.get("timestamp_hint")
                if th:
                    return f"{vid}, {th}"
                return vid
            return json.dumps(inp, ensure_ascii=False)[:200]

        def _tool_log(name: str, inp: Dict[str, Any]) -> None:
            print(f"[agent] {name}: {_tool_arg_snippet(inp)!r}")

        # Fresh transcript per top-level question; tool rounds stay in `messages` until loop ends.
        self._agent_messages = []
        messages: List[Dict[str, Any]] = []

        if conversation_history:
            for turn in conversation_history:
                role = turn.get("role", "")
                content = (turn.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": question})

        context_blocks: List[str] = []
        tool_calls = 0
        last_turn_text = ""
        _t0_agent = _time.time()
        tool_limit_reached = False
        use_openrouter = _is_openrouter_model(ANSWER_MODEL)
        oa_client = _get_openrouter_client() if use_openrouter else None
        anth_client = get_client() if not use_openrouter else None

        def _emit_agent(ev: Dict[str, Any]) -> None:
            if emit is None:
                return
            if "elapsed_ms" not in ev:
                ev = {**ev, "elapsed_ms": round((_time.time() - _t0_agent) * 1000)}
            try:
                emit(ev)
            except Exception:
                pass

        while True:
            if should_cancel is not None and should_cancel():
                _emit_agent({"type": "agent_step", "label": "[agent] cancelled/timeout flag detected; stopping"})
                last_turn_text = (
                    "Stopped early because this job was cancelled or timed out. "
                    "If you still want an answer, retry with a narrower question or specific video IDs."
                )
                break
            turn_text_parts: List[str] = []
            if use_openrouter:
                # OpenAI-compatible tool loop via OpenRouter
                tools_oa = [{"type": "function", "function": t} for t in tools]
                if tool_limit_reached:
                    resp = oa_client.chat.completions.create(  # type: ignore[union-attr]
                        model=ANSWER_MODEL,
                        max_tokens=1800,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": AGENT_SYSTEM_PROMPT + f"\n\nTool call limit reached ({AGENT_MAX_TOOL_CALLS}). Do NOT call tools. Produce the best possible final answer using existing tool results."},
                            *messages,
                        ],
                    )
                else:
                    resp = oa_client.chat.completions.create(  # type: ignore[union-attr]
                        model=ANSWER_MODEL,
                        max_tokens=1800,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                            *messages,
                        ],
                        tools=tools_oa,
                        tool_choice="auto",
                    )
                msg = resp.choices[0].message
                # tool_calls (OpenAI format)
                tcalls = getattr(msg, "tool_calls", None) or []
                if tcalls:
                    saw_tool_use = True
                    assistant_content = [{"type": "text", "text": (msg.content or "")}] if (msg.content or "").strip() else []
                    # Append assistant message (OpenAI schema)
                    messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc.model_dump() for tc in tcalls]})
                    tool_results: List[Dict[str, Any]] = []
                    for tc in tcalls:
                        name = tc.function.name
                        tool_id = tc.id
                        try:
                            tool_input = json.loads(tc.function.arguments or "{}")
                        except Exception:
                            tool_input = {}
                        if tool_calls >= AGENT_MAX_TOOL_CALLS:
                            tool_limit_reached = True
                            tool_results.append({"role": "tool", "tool_call_id": tool_id, "content": _safe_json({"error": "tool_call_limit_reached", "limit": AGENT_MAX_TOOL_CALLS})})
                            continue
                        _tool_log(str(name), tool_input)
                        _emit_agent({"type": "agent_tool", "label": f"[agent] {str(name)}: {_tool_arg_snippet(tool_input)!r}", "name": str(name), "input": tool_input})
                        tool_calls += 1
                        try:
                            if name == "search_vectors":
                                result = self._agent_search_vectors(query=str(tool_input.get("query") or ""), type_filter=str(tool_input.get("type") or "chunk"))
                            elif name == "search_bm25":
                                result = self._agent_search_bm25(query=str(tool_input.get("query") or ""))
                            elif name == "get_shorty":
                                result = self._agent_get_shorty(video_id=str(tool_input.get("video_id") or ""))
                            elif name == "search_entities":
                                result = self._agent_search_entities(name=str(tool_input.get("name") or ""))
                            elif name == "get_transcript_chunk":
                                result = self._agent_get_transcript_chunk(video_id=str(tool_input.get("video_id") or ""), timestamp_hint=str(tool_input.get("timestamp_hint") or ""))
                            elif name == "get_videos_by_channel":
                                result = self._agent_get_videos_by_channel(channel_name=str(tool_input.get("channel_name") or ""), limit=self._agent_channel_limit(tool_input.get("limit")))
                            elif name == "get_videos_by_channel_with_shorties":
                                result = self._agent_get_videos_by_channel_with_shorties(channel_name=str(tool_input.get("channel_name") or ""), limit=self._agent_channel_limit(tool_input.get("limit")))
                            elif name == "get_videos_by_date":
                                cn = tool_input.get("channel_name"); cn_s = str(cn).strip() if cn is not None else ""
                                result = self._agent_get_videos_by_date(start_date=str(tool_input.get("start_date") or ""), end_date=str(tool_input.get("end_date") or ""), limit=self._agent_history_limit(tool_input.get("limit")), channel_name=cn_s or None)
                            elif name == "get_recent_videos":
                                cn = tool_input.get("channel_name"); cn_s = str(cn).strip() if cn is not None else ""
                                result = self._agent_get_recent_videos(days=tool_input.get("days"), limit=self._agent_history_limit(tool_input.get("limit")), channel_name=cn_s or None)
                            elif name == "get_earliest_video":
                                result = self._agent_get_earliest_video(topic_query=str(tool_input.get("topic_query") or ""))
                            elif name == "search_segments":
                                vid_raw = tool_input.get("video_id")
                                vid_opt = str(vid_raw).strip() if vid_raw is not None else ""
                                result = self._agent_search_segments(
                                    query=str(tool_input.get("query") or ""),
                                    video_id=vid_opt or None,
                                )
                            elif name == "search_events":
                                result = self._agent_search_events(query=str(tool_input.get("query") or ""))
                            else:
                                result = {"error": f"unknown_tool:{name}"}
                        except Exception as exc:
                            result = {"error": f"{type(exc).__name__}: {exc}"}
                        extra_blocks = result.get("context_blocks") if isinstance(result, dict) else None
                        if isinstance(extra_blocks, list):
                            context_blocks.extend([b for b in extra_blocks if isinstance(b, str) and b.strip()])
                        tool_results.append({"role": "tool", "tool_call_id": tool_id, "content": _safe_json(result)})
                    # Append tool results and continue loop
                    messages.extend(tool_results)
                    continue
                # No tool calls: final text
                last_turn_text = (msg.content or "").strip()
                break

            # Anthropic tool loop
            client = anth_client
            # If tool limit was reached, request a final answer without tools.
            if tool_limit_reached:
                resp = client.messages.create(
                    model=ANSWER_MODEL,
                    max_tokens=1800,
                    temperature=0.2,
                    system=AGENT_SYSTEM_PROMPT
                    + f"\n\nTool call limit reached ({AGENT_MAX_TOOL_CALLS}). "
                    "Do NOT call tools. Produce the best possible final answer using existing tool results.",
                    messages=messages,
                )
            else:
                resp = client.messages.create(
                    model=ANSWER_MODEL,
                    max_tokens=1800,
                    temperature=0.2,
                    system=AGENT_SYSTEM_PROMPT,
                    messages=messages,
                    tools=tools,
                )

            assistant_content: List[Dict[str, Any]] = []
            tool_results_content: List[Dict[str, Any]] = []
            saw_tool_use = False

            for block in resp.content:
                btype = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
                if btype == "text":
                    text = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
                    if text:
                        dumped = _anthropic_assistant_block_to_api_dict(block)
                        if dumped:
                            assistant_content.append(dumped)
                        else:
                            assistant_content.append({"type": "text", "text": text})
                        turn_text_parts.append(text)
                    continue

                if btype == "tool_use":
                    dumped = _anthropic_assistant_block_to_api_dict(block)
                    if dumped:
                        assistant_content.append(dumped)
                    else:
                        name_fb = getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
                        tool_input_fb = getattr(block, "input", None) if not isinstance(block, dict) else block.get("input")
                        tool_id_fb = getattr(block, "id", None) if not isinstance(block, dict) else block.get("id")
                        if not isinstance(tool_input_fb, dict):
                            tool_input_fb = {}
                        assistant_content.append(
                            {
                                "type": "tool_use",
                                "id": tool_id_fb,
                                "name": name_fb,
                                "input": tool_input_fb,
                            }
                        )

                    saw_tool_use = True
                    name = getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
                    tool_input = getattr(block, "input", None) if not isinstance(block, dict) else block.get("input")
                    tool_id = getattr(block, "id", None) if not isinstance(block, dict) else block.get("id")
                    if not isinstance(tool_input, dict):
                        tool_input = {}

                    if tool_calls >= AGENT_MAX_TOOL_CALLS:
                        _emit_agent(
                            {
                                "type": "agent_tool",
                                "label": f"[agent] {name}: (skipped — tool call limit {AGENT_MAX_TOOL_CALLS})",
                                "name": name,
                                "input": tool_input,
                            }
                        )
                        tool_results_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": _safe_json({"error": "tool_call_limit_reached", "limit": AGENT_MAX_TOOL_CALLS}),
                            }
                        )
                        # Hard-stop: do not let the model keep looping once we've hit the cap.
                        tool_limit_reached = True
                        continue

                    if should_cancel is not None and should_cancel():
                        _emit_agent({"type": "agent_step", "label": "[agent] cancelled/timeout flag detected before tool; stopping"})
                        tool_limit_reached = True
                        tool_results_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": _safe_json({"error": "cancelled_or_timeout"}),
                            }
                        )
                        continue

                    _tool_log(str(name), tool_input)
                    _emit_agent(
                        {
                            "type": "agent_tool",
                            "label": f"[agent] {str(name)}: {_tool_arg_snippet(tool_input)!r}",
                            "name": str(name),
                            "input": tool_input,
                        }
                    )
                    tool_calls += 1

                    try:
                        if name == "search_vectors":
                            result = self._agent_search_vectors(
                                query=str(tool_input.get("query") or ""),
                                type_filter=str(tool_input.get("type") or "chunk"),
                            )
                        elif name == "search_bm25":
                            result = self._agent_search_bm25(query=str(tool_input.get("query") or ""))
                        elif name == "get_shorty":
                            result = self._agent_get_shorty(video_id=str(tool_input.get("video_id") or ""))
                        elif name == "search_entities":
                            result = self._agent_search_entities(name=str(tool_input.get("name") or ""))
                        elif name == "get_transcript_chunk":
                            result = self._agent_get_transcript_chunk(
                                video_id=str(tool_input.get("video_id") or ""),
                                timestamp_hint=str(tool_input.get("timestamp_hint") or ""),
                            )
                        elif name == "get_videos_by_channel":
                            result = self._agent_get_videos_by_channel(
                                channel_name=str(tool_input.get("channel_name") or ""),
                                limit=self._agent_channel_limit(tool_input.get("limit")),
                            )
                        elif name == "get_videos_by_channel_with_shorties":
                            result = self._agent_get_videos_by_channel_with_shorties(
                                channel_name=str(tool_input.get("channel_name") or ""),
                                limit=self._agent_channel_limit(tool_input.get("limit")),
                            )
                        elif name == "get_videos_by_date":
                            cn = tool_input.get("channel_name")
                            cn_s = str(cn).strip() if cn is not None else ""
                            result = self._agent_get_videos_by_date(
                                start_date=str(tool_input.get("start_date") or ""),
                                end_date=str(tool_input.get("end_date") or ""),
                                limit=self._agent_history_limit(tool_input.get("limit")),
                                channel_name=cn_s or None,
                            )
                        elif name == "get_recent_videos":
                            cn = tool_input.get("channel_name")
                            cn_s = str(cn).strip() if cn is not None else ""
                            result = self._agent_get_recent_videos(
                                days=tool_input.get("days"),
                                limit=self._agent_history_limit(tool_input.get("limit")),
                                channel_name=cn_s or None,
                            )
                        elif name == "get_earliest_video":
                            result = self._agent_get_earliest_video(
                                topic_query=str(tool_input.get("topic_query") or "")
                            )
                        elif name == "search_segments":
                            vid_raw = tool_input.get("video_id")
                            vid_opt = str(vid_raw).strip() if vid_raw is not None else ""
                            result = self._agent_search_segments(
                                query=str(tool_input.get("query") or ""),
                                video_id=vid_opt or None,
                            )
                        elif name == "search_events":
                            result = self._agent_search_events(query=str(tool_input.get("query") or ""))
                        else:
                            result = {"error": f"unknown_tool:{name}"}
                    except Exception as exc:
                        result = {"error": f"{type(exc).__name__}: {exc}"}

                    extra_blocks = result.get("context_blocks") if isinstance(result, dict) else None
                    if isinstance(extra_blocks, list):
                        context_blocks.extend([b for b in extra_blocks if isinstance(b, str) and b.strip()])

                    tool_results_content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": _safe_json(result),
                        }
                    )
                    continue

                dumped = _anthropic_assistant_block_to_api_dict(block)
                if dumped:
                    assistant_content.append(dumped)

            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

            if saw_tool_use and not tool_limit_reached:
                if tool_results_content:
                    messages.append({"role": "user", "content": tool_results_content})
                continue

            last_turn_text = "".join(turn_text_parts)
            break

        self._agent_messages = []

        # Deduplicate context while preserving order.
        dedup_context: List[str] = []
        seen = set()
        for b in context_blocks:
            if b in seen:
                continue
            seen.add(b)
            dedup_context.append(b)

        # Build source list from collected context headers.
        video_meta = self._load_video_meta_map()
        sources: List[Dict[str, str]] = []
        seen_vids: set = set()
        import re as _re
        for block in dedup_context:
            vid_match = _re.search(r"video_id=(\S+)", block.split("\n")[0])
            if not vid_match:
                continue
            vid = vid_match.group(1)
            if vid in seen_vids:
                continue
            seen_vids.add(vid)
            m = video_meta.get(vid, {})
            sources.append(
                {
                    "video_id": vid,
                    "title": m.get("title", vid),
                    "channel": m.get("channel", ""),
                    "watch_date": m.get("watch_date", ""),
                }
            )

        # --- Hallucination guard ---
        raw_answer = last_turn_text.strip() or "I could not find enough evidence to answer confidently."
        known_ids = set(seen_vids)
        guarded_answer = self._hallucination_guard(raw_answer, known_ids)
        guarded_answer = self._flag_unverified_titles(guarded_answer, known_ids)

        return {
            "answer": guarded_answer,
            "used_context": dedup_context,
            "sources": sources,
            "debug_events": [],
        }

    def answer_question(
        self,
        question: str,
        video_ids: Optional[List[str]] = None,
        top_k_per_layer: int = 4,
        emit: Optional[Callable[[Dict[str, Any]], None]] = None,
        agent_mode: Optional[bool] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
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

        *agent_mode*: if True, always run the agentic tool loop; if False, never;
        if None, use ASK_SHORTY_AGENT env (ENABLE_AGENT).

        *conversation_history*: prior user/assistant turns so the agent can
        handle follow-up questions (e.g. "what is the video id" after a previous
        answer).  Each entry is {"role": "user"|"assistant", "content": "..."}.
        """
        use_agent = (agent_mode is True) or (agent_mode is None and ENABLE_AGENT)
        if use_agent:
            return self._agent_run(
                question.strip(),
                emit=emit,
                should_cancel=should_cancel,
                conversation_history=conversation_history,
            )

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

            # Separate BM25-only synth chunks from normal vector chunks so they
            # always get their own context slots and don't compete for the same
            # top_k_per_layer positions (BM25-only hits have ids ending :bm25:synth).
            bm25_synth = [r for r in chunk_results if str(r.get("id", "")).endswith(":bm25:synth")]
            normal_chunks = [r for r in chunk_results if not str(r.get("id", "")).endswith(":bm25:synth")]
            for r in normal_chunks[:top_k_per_layer]:
                context_blocks.append(_fmt(r, "chunk"))
            for r in bm25_synth[:2]:
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

