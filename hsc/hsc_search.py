#!/usr/bin/env python3
"""
HSC retrieval: route query, run targeted SQL + graph + vector + BM25, merge with RRF.
"""

from __future__ import annotations

import sqlite3
import re
from typing import Any, Dict, List, Optional

from bm25_index import reciprocal_rank_fusion
from hsc.query_router import route_query


def _tokens(q: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", q) if len(t) >= 2]


def _rank_from_scores(scores: Dict[str, float]) -> List[str]:
    return [v for v, _ in sorted(scores.items(), key=lambda x: -x[1])]


def _search_facts_sql(conn: sqlite3.Connection, query: str, limit: int) -> Dict[str, float]:
    words = _tokens(query)[:12]
    if not words:
        return {}
    clauses = " OR ".join(
        "(LOWER(subject) LIKE ? OR LOWER(relation) LIKE ? OR LOWER(object) LIKE ?)"
        for _ in words
    )
    params: List[Any] = []
    for w in words:
        like = f"%{w}%"
        params.extend([like, like, like])
    cur = conn.cursor()
    try:
        cur.execute(
            f"SELECT video_id FROM facts WHERE {clauses} LIMIT 300",
            params,
        )
    except sqlite3.OperationalError:
        return {}
    scores: Dict[str, float] = {}
    for (vid,) in cur.fetchall():
        scores[vid] = scores.get(vid, 0.0) + 1.0
    return dict(sorted(scores.items(), key=lambda x: -x[1])[:limit])


def _search_events_sql(conn: sqlite3.Connection, query: str, limit: int) -> Dict[str, float]:
    words = _tokens(query)[:12]
    if not words:
        return {}
    clauses = " OR ".join(
        "(LOWER(title) LIKE ? OR LOWER(cause) LIKE ? OR LOWER(effect) LIKE ? OR LOWER(systems) LIKE ?)"
        for _ in words
    )
    params: List[Any] = []
    for w in words:
        like = f"%{w}%"
        params.extend([like, like, like, like])
    cur = conn.cursor()
    try:
        cur.execute(
            f"SELECT video_id FROM events WHERE {clauses} LIMIT 300",
            params,
        )
    except sqlite3.OperationalError:
        return {}
    scores: Dict[str, float] = {}
    for (vid,) in cur.fetchall():
        scores[vid] = scores.get(vid, 0.0) + 1.0
    return dict(sorted(scores.items(), key=lambda x: -x[1])[:limit])


def _search_segments_sql(conn: sqlite3.Connection, query: str, limit: int) -> Dict[str, float]:
    words = _tokens(query)[:12]
    if not words:
        return {}
    clauses = " OR ".join("(LOWER(summary) LIKE ?)" for _ in words)
    params = [f"%{w}%" for w in words]
    cur = conn.cursor()
    try:
        cur.execute(
            f"SELECT video_id, summary FROM segments WHERE {clauses} LIMIT 400",
            params,
        )
    except sqlite3.OperationalError:
        return {}
    scores: Dict[str, float] = {}
    for row in cur.fetchall():
        vid, summ = row[0], row[1] or ""
        overlap = sum(1 for w in words if w in summ.lower())
        scores[vid] = scores.get(vid, 0.0) + 1.0 + 0.1 * overlap
    return dict(sorted(scores.items(), key=lambda x: -x[1])[:limit])


def _vector_rank(
    collection,
    question: str,
    n: int = 20,
    types: Optional[List[str]] = None,
) -> List[str]:
    if collection is None:
        return []
    types = types or ["chunk", "shorty", "synthetic_question"]
    by_vid: Dict[str, float] = {}
    for st in types:
        try:
            res = collection.query(
                query_texts=[question],
                n_results=n,
                where={"type": st},
            )
        except Exception:
            continue
        metas = (res.get("metadatas") or [[]])[0]
        scores = (res.get("distances") or [[]])[0]
        for i in range(len(metas)):
            m = metas[i] if i < len(metas) else {}
            vid = (m or {}).get("video_id")
            if not vid:
                continue
            sc = float(scores[i]) if i < len(scores) else 1.0
            if vid not in by_vid or sc < by_vid[vid]:
                by_vid[vid] = sc
    ranked = sorted(by_vid.items(), key=lambda x: x[1])
    return [v for v, _ in ranked]


def hsc_retrieve(
    db_path: str,
    question: str,
    *,
    rag_collection=None,
    bm25_search=None,
    enable_bm25: bool = False,
    enable_graph: bool = False,
) -> Dict[str, Any]:
    """
    Classify query, gather ranked lists from HSC tables + vector + optional BM25/graph, RRF-merge.
    """
    routed = route_query(question)
    qtype = routed["type"]
    layer_used = qtype

    hsc_scores: Dict[str, float] = {}
    event_samples: List[Dict[str, Any]] = []
    segment_samples: List[Dict[str, Any]] = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if qtype == "fact":
            hsc_scores.update(_search_facts_sql(conn, question, 40))
        elif qtype == "event":
            hsc_scores.update(_search_events_sql(conn, question, 40))
        elif qtype == "summary":
            hsc_scores.update(_search_segments_sql(conn, question, 40))
        else:
            hsc_scores.update(_search_segments_sql(conn, question, 20))

        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT video_id, title, cause, effect FROM events ORDER BY id DESC LIMIT 8"
            )
            event_samples = [dict(r) for r in cur.fetchall()]
        except sqlite3.OperationalError:
            pass
        try:
            cur.execute(
                "SELECT video_id, summary FROM segments ORDER BY id DESC LIMIT 8"
            )
            segment_samples = [dict(r) for r in cur.fetchall()]
        except sqlite3.OperationalError:
            pass

    hsc_rank = _rank_from_scores(hsc_scores)
    vector_rank = _vector_rank(rag_collection, question, n=20)

    lists: List[List[str]] = []
    if hsc_rank:
        lists.append(hsc_rank)
    if vector_rank:
        lists.append(vector_rank)

    bm25_rank: List[str] = []
    if enable_bm25 and bm25_search is not None:
        try:
            bh = bm25_search.search(question, top_k=20)
            bm25_rank = [h["video_id"] for h in bh]
            if bm25_rank:
                lists.append(bm25_rank)
        except Exception:
            pass

    graph_rank: List[str] = []
    if enable_graph and qtype in ("fact", "event"):
        try:
            from graph_search import GraphSearch

            gh = GraphSearch(db_path).search(question, top_k=15)
            graph_rank = [h["video_id"] for h in gh]
            if graph_rank:
                lists.append(graph_rank)
        except Exception:
            pass

    # Multi-hop reasoning over facts triples (additive; independent of GraphSearch)
    reason_rank: List[str] = []
    reasoning_paths: List[Any] = []
    path_scores: List[float] = []
    query_entities: List[str] = []
    rarity_scores: List[float] = []
    node_frequencies_top: List[Any] = []
    global_rank: List[str] = []
    global_paths: List[Any] = []
    global_path_scores: List[float] = []
    videos_in_path: List[Any] = []
    if qtype in ("fact", "event"):
        try:
            from hsc.graph_reasoner import graph_reason

            gr = graph_reason(question, db_path, max_depth=3, top_n=20)
            reason_rank = gr.get("reason_rank") or []
            reasoning_paths = gr.get("reasoning_paths") or []
            path_scores = gr.get("path_scores") or []
            query_entities = gr.get("query_entities") or []
            rarity_scores = gr.get("rarity_scores") or []
            node_frequencies_top = gr.get("node_frequencies_top") or []
            if reason_rank:
                lists.append(reason_rank)
        except Exception:
            pass
        try:
            from hsc.global_graph import global_graph_reason

            ggr = global_graph_reason(question, db_path, max_depth=3, top_n=20)
            global_rank = ggr.get("global_rank") or []
            global_paths = ggr.get("global_paths") or []
            global_path_scores = ggr.get("global_path_scores") or []
            videos_in_path = ggr.get("videos_in_path") or []
            if global_rank:
                lists.append(global_rank)
        except Exception:
            pass

    if not lists:
        ranked_video_ids = []
    elif len(lists) == 1:
        ranked_video_ids = lists[0][:40]
    else:
        fused = reciprocal_rank_fusion(lists, k=60)
        ranked_video_ids = [v for v, _ in fused[:40]]

    return {
        "query_type": qtype,
        "layer_used": layer_used,
        "route_reason": routed.get("reason", ""),
        "ranked_video_ids": ranked_video_ids,
        "event_hits": len(hsc_rank) if qtype == "event" else 0,
        "segment_hits": len(hsc_rank) if qtype in ("summary", "raw") else 0,
        "graph_hits": len(graph_rank),
        "bm25_hits": len(bm25_rank),
        "reason_rank_len": len(reason_rank),
        "global_rank_len": len(global_rank),
        "reasoning_paths": reasoning_paths,
        "path_scores": path_scores,
        "query_entities": query_entities,
        "rarity_scores": rarity_scores,
        "node_frequencies_top": node_frequencies_top,
        "global_paths": global_paths,
        "global_path_scores": global_path_scores,
        "videos_in_path": videos_in_path,
        "hsc_target_rank": hsc_rank[:20],
        "vector_rank": vector_rank[:20],
        "event_samples": event_samples,
        "segment_samples": segment_samples,
    }
