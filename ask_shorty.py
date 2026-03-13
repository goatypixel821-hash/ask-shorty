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

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging
import os

from anthropic_client import get_client
from transcript_database import TranscriptDatabase

if TYPE_CHECKING:
    # Only imported for type checking; runtime import is deferred to avoid
    # initializing Chroma / SentenceTransformer at startup.
    from transcript_rag import TranscriptRAG  # pragma: no cover


logger = logging.getLogger(__name__)

ANSWER_MODEL = "claude-sonnet-4-20250514"
MAX_SHORTIES_IN_CONTEXT = 10

# Disable Chroma/TranscriptRAG usage entirely when this env var is set.
NO_CHROMA = os.getenv("ASK_SHORTY_NO_CHROMA", "").strip() not in ("", "0", "false", "False")


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
- Always cite which video your answer comes from by title.
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

    def _get_rag(self) -> "TranscriptRAG":
        """Lazily construct the TranscriptRAG instance on first use."""
        if self._rag is None:
            # Deferred import so that importing ask_shorty.py does not import
            # transcript_rag_enhanced or touch Chroma until needed.
            from transcript_rag import TranscriptRAG as _TranscriptRAG

            self._rag = _TranscriptRAG()
        return self._rag

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
    ) -> Dict[str, Any]:
        """
        Main entrypoint for Ask Shorty.

        Returns:
        {
          "answer": "...",
          "used_context": [...],
        }
        """
        print("[ask] Step A: starting answer_question")
        if not question or not question.strip():
            raise ValueError("Question is empty.")

        q = question.strip()

        # Use metadata parsing to narrow candidate videos (optional)
        print("[ask] Step B: metadata filter start")
        candidate_videos = self._filter_by_metadata(q, video_ids)
        print("[ask] Step B: metadata filter done")
        print("[ask] Step C: query rewrite start")
        rewrites = self._rewrite_query(q)
        print("[ask] Step C: query rewrite done")

        chunk_results: List[Dict[str, Any]] = []
        shorty_results: List[Dict[str, Any]] = []
        synq_results: List[Dict[str, Any]] = []

        # Try Chroma-based RAG search unless disabled
        used_chroma = False
        if not NO_CHROMA:
            try:
                # Layer 1: transcript chunks
                print("[ask] Step D: RAG search (chunks/shorties/synqs) start")
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
                print("[ask] Step D: RAG search done")
                used_chroma = True
            except BaseException as e:
                # Catch broad exceptions so we can fall back to SQLite keyword search
                print(f"[ask] Step D: RAG/Chroma search failed, falling back to SQLite-only search: {e!r}")

        if not used_chroma:
            print("[ask] Step D: using SQLite Shorty keyword fallback (no Chroma)")
            fallback = self._sqlite_shorty_keyword_search(q, video_ids=video_ids, limit=MAX_SHORTIES_IN_CONTEXT)
            shorty_results = []
            for item in fallback:
                shorty_results.append(
                    {
                        "id": f"{item['video_id']}:shorty",
                        "text": item["shorty"],
                        "score": -item["score"],  # lower is better later
                        "metadata": {
                            "video_id": item["video_id"],
                            "title": item["title"],
                            "type": "shorty",
                        },
                        "query": q,
                    }
                )

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

        # Build context blocks for Claude
        context_blocks: List[str] = []

        def _fmt(row: Dict[str, Any], layer: str) -> str:
            m = row.get("metadata") or {}
            vid = m.get("video_id", "unknown_video")
            chunk_idx = m.get("chunk_index")
            return (
                f"[{layer}] video_id={vid}"
                + (f" chunk={chunk_idx}" if chunk_idx is not None else "")
                + f"\n{row['text']}\n"
            )

        for r in chunk_results[:top_k_per_layer]:
            context_blocks.append(_fmt(r, "chunk"))
        # Always include ALL Shorties we selected above (already filtered by video_ids if provided)
        for r in shorty_results:
            context_blocks.append(_fmt(r, "shorty"))
        for r in synq_results[:top_k_per_layer]:
            context_blocks.append(_fmt(r, "synthetic_question"))

        if not context_blocks:
            answer_text = "I could not find any relevant information in your indexed videos to answer that."
            return {
                "answer": answer_text,
                "used_context": [],
            }

        merged_context = "\n---\n".join(context_blocks)
        user_prompt = f"User question:\n{q}\n\nContext passages:\n{merged_context}\n\nAnswer the question using ONLY the context above."
        print("[ask] Step E: calling Anthropic API")
        answer = _call_claude_answer(ANSWER_SYSTEM_PROMPT, user_prompt)
        print("[ask] Step F: got answer, saving")

        return {
            "answer": answer,
            "used_context": context_blocks,
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

