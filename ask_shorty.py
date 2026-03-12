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

from typing import List, Dict, Any, Optional
import logging

from anthropic_client import get_client
from transcript_rag import TranscriptRAG
from transcript_database import TranscriptDatabase


logger = logging.getLogger(__name__)

ANSWER_MODEL = "claude-sonnet-4-20250514"
MAX_SHORTIES_IN_CONTEXT = 10


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
        self.rag = TranscriptRAG()

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

        for q in query_variants:
            res = self.rag.collection.query(
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
        if not question or not question.strip():
            raise ValueError("Question is empty.")

        q = question.strip()

        # Use metadata parsing to narrow candidate videos (optional)
        candidate_videos = self._filter_by_metadata(q, video_ids)
        rewrites = self._rewrite_query(q)

        # Layer 1: transcript chunks
        chunk_where_type = "chunk"
        chunk_results = self._search_layer(
            rewrites,
            type_filter=chunk_where_type,
            top_k=top_k_per_layer,
        )

        # Layer 2: Shorties
        # For scale:
        # - If we have candidate_videos from metadata, restrict search to those.
        # - Otherwise, do a global similarity search over type="shorty".
        shorty_where: Dict[str, Any] = {"type": "shorty"}
        if candidate_videos:
            shorty_where["video_id"] = {"$in": candidate_videos}

        shorty_results: List[Dict[str, Any]] = []
        res = self.rag.collection.query(
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
        synq_results = self._search_layer(rewrites, type_filter="synthetic_question", top_k=top_k_per_layer)

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
        answer = _call_claude_answer(ANSWER_SYSTEM_PROMPT, user_prompt)

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

