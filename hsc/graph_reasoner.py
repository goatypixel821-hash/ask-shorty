#!/usr/bin/env python3
"""
Multi-hop graph reasoning over subject–relation–object triples (facts table).

Additive to graph_search.py — does not replace it.
"""

from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

TripleRow = Tuple[str, str, str, str]  # subject, relation, object, video_id


@dataclass
class TripleGraph:
    """Triples + adjacency + node frequency for scoring."""

    triples: List[TripleRow] = field(default_factory=list)
    forward: Dict[str, List[Tuple[str, str, str]]] = field(default_factory=dict)
    # subject -> [(relation, object, video_id)]
    backward: Dict[str, List[Tuple[str, str, str]]] = field(default_factory=dict)
    # object -> [(subject, relation, video_id)]
    node_freq: Dict[str, int] = field(default_factory=dict)
    nodes: Set[str] = field(default_factory=set)


def load_triples(db_path: str) -> Tuple[List[TripleRow], Dict[str, List[Tuple[str, str, str]]]]:
    """
    Load all rows from facts.

    Returns:
        triples: list of (subject, relation, object, video_id)
        graph: adjacency graph[subject] = [(relation, object, video_id)]
    """
    tg = _load_triple_graph(db_path)
    return tg.triples, tg.forward


def _load_triple_graph(db_path: str) -> TripleGraph:
    tg = TripleGraph()
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT subject, relation, object, video_id FROM facts "
                "WHERE subject IS NOT NULL AND relation IS NOT NULL AND object IS NOT NULL"
            )
            rows = cur.fetchall()
    except sqlite3.OperationalError:
        return tg

    freq: Counter[str] = Counter()
    for subj, rel, obj, vid in rows:
        s = (subj or "").strip()
        r = (rel or "").strip()
        o = (obj or "").strip()
        v = (vid or "").strip()
        if not s or not r or not o:
            continue
        tg.triples.append((s, r, o, v))
        tg.forward.setdefault(s, []).append((r, o, v))
        tg.backward.setdefault(o, []).append((s, r, v))
        for n in (s, o):
            key = n.lower()
            freq[key] += 1
            tg.nodes.add(s)
            tg.nodes.add(o)
    tg.node_freq = dict(freq)
    return tg


def _canonical_nodes(tg: TripleGraph) -> Dict[str, str]:
    """Map lowercase node string -> one canonical casing from graph."""
    canon: Dict[str, str] = {}
    for n in tg.nodes:
        k = n.lower()
        if k not in canon or len(n) > len(canon[k]):
            canon[k] = n
    return canon


def extract_query_entities(query: str, tg: TripleGraph) -> List[str]:
    """Match query text to graph nodes (substring + token overlap)."""
    if not query.strip() or not tg.nodes:
        return []
    canon = _canonical_nodes(tg)
    ql = query.lower()
    found: List[str] = []
    seen: Set[str] = set()

    # Longer nodes first to prefer multi-word entities
    sorted_nodes = sorted(canon.values(), key=lambda x: -len(x))
    for node in sorted_nodes:
        nl = node.lower()
        if len(nl) < 2:
            continue
        if nl in ql:
            if nl not in seen:
                seen.add(nl)
                found.append(node)

    # Word tokens: match single-word nodes
    words = re.findall(r"[A-Za-z0-9]+", query)
    for w in words:
        if len(w) < 3:
            continue
        wl = w.lower()
        if wl in canon and wl not in seen:
            seen.add(wl)
            found.append(canon[wl])

    return found[:12]


def find_paths(
    query_entities: List[str],
    tg: TripleGraph,
    max_depth: int = 3,
    max_paths: int = 200,
) -> List[List[List[str]]]:
    """
    BFS from seed entities along forward and backward edges.

    Each path is a list of triples [subject, relation, object] (strings).

    max_depth: maximum number of triples in a path.
    """
    if not query_entities or max_depth < 1:
        return []

    # Map loose names to actual graph keys
    canon = _canonical_nodes(tg)
    seeds: List[str] = []
    for e in query_entities:
        k = e.strip().lower()
        if k in canon:
            seeds.append(canon[k])
    seeds = list(dict.fromkeys(seeds))
    if not seeds:
        return []

    paths_out: List[List[List[str]]] = []
    seen_path_sig: Set[Tuple[Tuple[str, str, str], ...]] = set()

    iterations = 0
    max_iter = 50000
    for start in seeds:
        # queue: (focus_node, list of triples [[s,r,o], ...], depth)
        q: deque = deque()
        q.append((start, [], 0))

        while q and len(paths_out) < max_paths and iterations < max_iter:
            iterations += 1
            node, path_triples, depth = q.popleft()
            if depth >= max_depth:
                continue

            # Forward: node as subject
            for r, o, vid in tg.forward.get(node, []):
                tr = [node, r, o]
                new_path = path_triples + [tr]
                sig = tuple(tuple(t) for t in new_path)
                if sig not in seen_path_sig:
                    seen_path_sig.add(sig)
                    paths_out.append(new_path)
                if depth + 1 < max_depth:
                    q.append((o, new_path, depth + 1))

            # Backward: node as object -> (subject, relation, node)
            for s, r, vid in tg.backward.get(node, []):
                tr = [s, r, node]
                new_path = path_triples + [tr]
                sig = tuple(tuple(t) for t in new_path)
                if sig not in seen_path_sig:
                    seen_path_sig.add(sig)
                    paths_out.append(new_path)
                if depth + 1 < max_depth:
                    q.append((s, new_path, depth + 1))

    return paths_out


def _freq_lookup(part: str, node_freq: Dict[str, int]) -> int:
    k = part.strip().lower()
    return max(1, int(node_freq.get(k, 1)))


def score_path(
    path: List[List[str]],
    query: str,
    node_freq: Optional[Dict[str, int]] = None,
    *,
    rare_threshold: int = 3,
    rare_bonus: float = 2.0,
) -> float:
    """
    Salience-aware score (HSC Phase 3):

        rarity_score = sum(1 / log(1 + freq(node))) over nodes in path
        score = (query_overlap * 2.0) * (rarity_score * 1.5) - (path_length * 0.5)
        + rare_bonus if any node has freq < rare_threshold
    """
    if not path:
        return 0.0
    node_freq = node_freq or {}
    q = query.lower()
    path_length = len(path)

    chain = " ".join(f"{t[0]} {t[1]} {t[2]}" for t in path).lower()
    q_words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) >= 3]
    query_overlap = float(sum(1 for w in q_words if w in chain))
    if not q_words:
        query_overlap = max(0.5, query_overlap)

    rarity_score = 0.0
    seen_nodes: Set[str] = set()
    has_rare = False
    for tr in path:
        for part in tr:
            f = _freq_lookup(part, node_freq)
            rarity_score += 1.0 / math.log(1.0 + float(f))
            kk = part.strip().lower()
            if kk not in seen_nodes:
                seen_nodes.add(kk)
                if f < rare_threshold:
                    has_rare = True

    bonus = rare_bonus if has_rare else 0.0
    qo = max(0.5, query_overlap)
    final = (qo * 2.0) * (rarity_score * 1.5) - (path_length * 0.5) + bonus
    return max(final, 0.01)


def _path_rarity_detail(
    path: List[List[str]],
    node_freq: Dict[str, int],
) -> Tuple[float, Dict[str, int]]:
    """Rarity sum and per-node frequencies for debug."""
    rarity_score = 0.0
    freqs: Dict[str, int] = {}
    for tr in path:
        for part in tr:
            f = _freq_lookup(part, node_freq)
            rarity_score += 1.0 / math.log(1.0 + float(f))
            freqs[part] = f
    return rarity_score, freqs


def path_to_sentence(path: List[List[str]]) -> str:
    """Collapse a path into one readable sentence."""
    if not path:
        return ""
    parts: List[str] = []
    for i, tr in enumerate(path):
        s, r, o = tr[0], tr[1], tr[2]
        if i == 0:
            parts.append(f"{s} {r} {o}")
        else:
            prev_o = path[i - 1][2]
            if s.strip().lower() == prev_o.strip().lower():
                parts.append(f"then {r} {o}")
            elif o.strip().lower() == path[i - 1][0].strip().lower():
                parts.append(f"then {s} {r} {o}")
            else:
                parts.append(f"{s} {r} {o}")
    return ", ".join(parts) + "."


def _path_video_id(path: List[List[str]], tg: TripleGraph) -> str:
    """Most common video_id among triples that match path edges."""
    if not path:
        return ""
    counts: Counter[str] = Counter()
    for tr in path:
        s, r, o = tr[0], tr[1], tr[2]
        for row in tg.triples:
            if row[0] == s and row[1] == r and row[2] == o:
                counts[row[3]] += 1
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def graph_reason(
    query: str,
    db_path: str,
    *,
    max_depth: int = 3,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Extract entities from query, find multi-hop paths, score, return top N.

    Uses persisted fact_nodes frequencies when available (see hsc.fact_frequency).
    """
    from hsc.fact_frequency import load_node_frequency

    empty = {
        "paths": [],
        "scores": [],
        "reason_rank": [],
        "synthetic_rows": [],
        "reasoning_paths": [],
        "path_scores": [],
        "query_entities": [],
        "rarity_scores": [],
        "node_frequencies_top": [],
    }

    tg = _load_triple_graph(db_path)
    if not tg.triples:
        return dict(empty)

    persisted = load_node_frequency(db_path)
    merged_freq: Dict[str, int] = dict(tg.node_freq)
    for k, v in persisted.items():
        merged_freq[k] = max(int(merged_freq.get(k, 0)), int(v))

    entities = extract_query_entities(query, tg)
    raw_paths = find_paths(entities, tg, max_depth=max_depth)
    scored: List[Tuple[List[List[str]], float]] = []
    for p in raw_paths:
        sc = score_path(p, query, merged_freq)
        scored.append((p, sc))

    scored.sort(key=lambda x: -x[1])
    top = scored[:top_n]

    paths = [p for p, _ in top]
    scores = [s for _, s in top]

    reason_rank: List[str] = []
    seen_v: Set[str] = set()
    synthetic_rows: List[Dict[str, Any]] = []
    for p, sc in top:
        vid = _path_video_id(p, tg)
        text = path_to_sentence(p)
        short_txt = " → ".join(f"{t[1]} {t[2]}" for t in p[:3])
        if len(p) == 1:
            short_txt = f"{p[0][0]} {p[0][1]} {p[0][2]}"
        row = {
            "video_id": vid,
            "text": text or short_txt,
            "source": "graph_reasoning",
            "score": round(sc, 5),
            "path": p,
        }
        synthetic_rows.append(row)
        if vid and vid not in seen_v:
            seen_v.add(vid)
            reason_rank.append(vid)

    top3 = paths[:3]
    top3_sc = scores[:3]
    rarity_dbg: List[float] = []
    node_freq_dbg: List[Dict[str, int]] = []
    for p in top3:
        rsum, fmap = _path_rarity_detail(p, merged_freq)
        rarity_dbg.append(round(rsum, 5))
        node_freq_dbg.append(fmap)

    return {
        "paths": paths,
        "scores": scores,
        "reason_rank": reason_rank,
        "synthetic_rows": synthetic_rows,
        "reasoning_paths": top3,
        "path_scores": [round(s, 5) for s in top3_sc],
        "query_entities": entities,
        "rarity_scores": rarity_dbg,
        "node_frequencies_top": node_freq_dbg,
    }


def load_node_frequency(db_path: str) -> Dict[str, int]:
    """Load persisted node frequencies from fact_nodes (rebuilds if stale)."""
    from hsc.fact_frequency import load_node_frequency as _load_persisted

    return _load_persisted(db_path)
