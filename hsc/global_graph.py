"""
Cross-video global graph: normalized triples, multi-hop reasoning, scoring.
"""
from __future__ import annotations

import math
import re
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from hsc.entity_normalizer import load_aliases_from_db, normalize_entity
from hsc.global_graph_builder import ensure_global_graph_fresh
from hsc.graph_reasoner import score_path

EdgeStep = Tuple[str, str, str, str]  # subject_norm, relation, object_norm, video_id


@dataclass
class GlobalGraphData:
    forward_norm: Dict[str, List[Tuple[str, str, str]]] = field(default_factory=dict)
    # subject_norm -> [(relation, object_norm, video_id)]
    backward_norm: Dict[str, List[Tuple[str, str, str]]] = field(default_factory=dict)
    # object_norm -> [(subject_norm, relation, video_id)]
    edge_support: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    nodes: Set[str] = field(default_factory=set)
    node_display: Dict[str, str] = field(default_factory=dict)


def load_global_graph(db_path: str) -> Dict[str, Any]:
    """
    Load normalized adjacency from global_facts (rebuilds if stale).

    Returns dict with forward_norm, backward_norm, edge_support, nodes, node_display.
    """
    ensure_global_graph_fresh(db_path)
    gg = GlobalGraphData()
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT subject_norm, relation, object_norm, subject_raw, object_raw, video_id
                FROM global_facts
                """
            )
            rows = cur.fetchall()
            cur.execute(
                """
                SELECT subject_norm, relation, object_norm, COUNT(*) AS c
                FROM global_facts
                GROUP BY subject_norm, relation, object_norm
                """
            )
            for s, r, o, c in cur.fetchall():
                gg.edge_support[(s, r, o)] = int(c)
    except sqlite3.OperationalError:
        return {
            "forward_norm": {},
            "backward_norm": {},
            "edge_support": {},
            "nodes": set(),
            "node_display": {},
        }

    for subject_norm, relation, object_norm, sraw, oraw, vid in rows:
        r = (relation or "").strip().lower()
        sn = subject_norm or ""
        on = object_norm or ""
        v = (vid or "").strip()
        if not sn or not on or not r:
            continue
        gg.forward_norm.setdefault(sn, []).append((r, on, v))
        gg.backward_norm.setdefault(on, []).append((sn, r, v))
        gg.nodes.add(sn)
        gg.nodes.add(on)
        if sn not in gg.node_display and sraw:
            gg.node_display[sn] = str(sraw).strip()
        if on not in gg.node_display and oraw:
            gg.node_display[on] = str(oraw).strip()

    return {
        "forward_norm": gg.forward_norm,
        "backward_norm": gg.backward_norm,
        "edge_support": gg.edge_support,
        "nodes": gg.nodes,
        "node_display": gg.node_display,
    }


def _merged_norm_frequency(db_path: str) -> Dict[str, int]:
    """Combine fact_nodes (normalized) with global_facts node counts."""
    from hsc.fact_frequency import load_node_frequency

    out: Dict[str, int] = {}
    try:
        persisted = load_node_frequency(db_path)
        for k, v in persisted.items():
            ns = normalize_entity(k)["normalized"]
            if ns:
                out[ns] = max(out.get(ns, 0), int(v))
    except Exception:
        pass
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT subject_norm, object_norm FROM global_facts")
            for s, o in cur.fetchall():
                if s:
                    out[s] = out.get(s, 0) + 1
                if o:
                    out[o] = out.get(o, 0) + 1
    except sqlite3.OperationalError:
        pass
    for k in list(out.keys()):
        out[k] = max(1, int(out[k]))
    return out


def extract_global_entities(
    query: str,
    nodes: Set[str],
    extra_aliases: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Match query text to normalized graph nodes."""
    if not query.strip() or not nodes:
        return []
    extra_aliases = extra_aliases or {}
    found: List[str] = []
    seen: Set[str] = set()
    ql = query.lower().replace("_", " ")
    for n in sorted(nodes, key=lambda x: -len(x)):
        if len(n) < 2:
            continue
        if n in ql:
            if n not in seen:
                seen.add(n)
                found.append(n)
    words = re.findall(r"[A-Za-z0-9]+", query)
    for w in words:
        if len(w) < 2:
            continue
        norm = normalize_entity(w, extra_aliases=extra_aliases)["normalized"]
        if norm in nodes and norm not in seen:
            seen.add(norm)
            found.append(norm)
    return found[:12]


def find_global_paths(
    seeds: List[str],
    forward_norm: Dict[str, List[Tuple[str, str, str]]],
    backward_norm: Dict[str, List[Tuple[str, str, str]]],
    *,
    max_depth: int = 3,
    max_paths: int = 200,
) -> List[List[EdgeStep]]:
    if not seeds or max_depth < 1:
        return []
    seeds = list(dict.fromkeys(seeds))
    paths_out: List[List[EdgeStep]] = []
    seen_sig: Set[Tuple[Tuple[str, str, str, str], ...]] = set()
    iterations = 0
    max_iter = 50000

    for start in seeds:
        q: deque = deque()
        q.append((start, [], 0))
        while q and len(paths_out) < max_paths and iterations < max_iter:
            iterations += 1
            node, path_edges, depth = q.popleft()
            if depth >= max_depth:
                continue

            for r, o, vid in forward_norm.get(node, []):
                edge: EdgeStep = (node, r, o, vid)
                new_path = path_edges + [edge]
                sig = tuple(new_path)
                if sig not in seen_sig:
                    seen_sig.add(sig)
                    paths_out.append(new_path)
                if depth + 1 < max_depth:
                    q.append((o, new_path, depth + 1))

            for s, r, vid in backward_norm.get(node, []):
                edge = (s, r, node, vid)
                new_path = path_edges + [edge]
                sig = tuple(new_path)
                if sig not in seen_sig:
                    seen_sig.add(sig)
                    paths_out.append(new_path)
                if depth + 1 < max_depth:
                    q.append((s, new_path, depth + 1))

    return paths_out


def score_global_path(
    path: List[EdgeStep],
    query: str,
    node_freq: Dict[str, int],
    edge_support: Dict[Tuple[str, str, str], int],
) -> float:
    """Rarity base + multi-video bonus + log agreement on normalized edges."""
    if not path:
        return 0.0
    triples = [[s, r, o] for s, r, o, _ in path]
    base = score_path(triples, query, node_freq)
    vids = [e[3] for e in path]
    multi_bonus = 1.5 if len(set(vids)) > 1 else 0.0
    agree_bonus = 0.0
    for s, r, o, _ in path:
        cnt = edge_support.get((s, r, o), 1)
        agree_bonus += math.log(1.0 + float(cnt))
    return max(0.01, base + multi_bonus + agree_bonus)


def _load_video_labels(db_path: str, vids: Set[str]) -> Dict[str, str]:
    if not vids:
        return {}
    out: Dict[str, str] = {}
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            qmarks = ",".join("?" * len(vids))
            cur.execute(
                f"SELECT video_id, title FROM videos WHERE video_id IN ({qmarks})",
                tuple(vids),
            )
            for vid, title in cur.fetchall():
                if vid:
                    t = (title or "").strip() or str(vid)[:16]
                    out[str(vid)] = f"video {t[:48]}"
    except sqlite3.OperationalError:
        pass
    for v in vids:
        if v not in out:
            out[v] = f"video {v[:16]}"
    return out


def global_path_to_sentence(
    path: List[EdgeStep],
    *,
    node_display: Optional[Dict[str, str]] = None,
    video_labels: Optional[Dict[str, str]] = None,
) -> str:
    """Human-readable explanation with optional video labels."""
    if not path:
        return ""
    nd = node_display or {}
    vl = video_labels or {}
    parts: List[str] = []
    prev_o: Optional[str] = None
    for i, (s, r, o, vid) in enumerate(path):
        sd = nd.get(s, s)
        od = nd.get(o, o)
        label = vl.get(vid, vid[:16])
        if i == 0:
            parts.append(f"{sd} {r} {od} ({label})")
        else:
            if prev_o == s:
                parts.append(f"then {r} {od} ({label})")
            else:
                parts.append(f"{sd} {r} {od} ({label})")
        prev_o = o
    return ", ".join(parts) + "."


def global_graph_reason(
    query: str,
    db_path: str,
    *,
    max_depth: int = 3,
    top_n: int = 20,
) -> Dict[str, Any]:
    """
    Cross-video multi-hop reasoning over global_facts.

    Returns global_paths (top 3 for debug), global_path_scores, videos_in_path,
    global_rank (video_ids for RRF), global_explanations, global_query_entities.
    """
    empty: Dict[str, Any] = {
        "global_paths": [],
        "global_path_scores": [],
        "videos_in_path": [],
        "global_rank": [],
        "global_explanations": [],
        "global_query_entities": [],
    }

    gdata = load_global_graph(db_path)
    forward_norm = gdata["forward_norm"]
    backward_norm = gdata["backward_norm"]
    edge_support = gdata["edge_support"]
    nodes: Set[str] = gdata["nodes"]
    node_display: Dict[str, str] = gdata["node_display"]

    if not nodes:
        return dict(empty)

    extra_aliases = load_aliases_from_db(db_path)
    entities = extract_global_entities(query, nodes, extra_aliases=extra_aliases)
    if not entities:
        for w in re.findall(r"[A-Za-z0-9]+", query):
            if len(w) < 3:
                continue
            n = normalize_entity(w, extra_aliases=extra_aliases)["normalized"]
            if n in nodes:
                entities.append(n)
                if len(entities) >= 8:
                    break
    node_freq = _merged_norm_frequency(db_path)

    raw_paths = find_global_paths(
        entities,
        forward_norm,
        backward_norm,
        max_depth=max_depth,
    )
    scored: List[Tuple[List[EdgeStep], float]] = []
    for p in raw_paths:
        sc = score_global_path(p, query, node_freq, edge_support)
        scored.append((p, sc))

    scored.sort(key=lambda x: -x[1])
    top = scored[:top_n]
    paths = [p for p, _ in top]
    scores = [s for _, s in top]

    all_vids_in_top: Set[str] = set()
    for p, _ in top:
        for e in p:
            all_vids_in_top.add(e[3])
    video_labels = _load_video_labels(db_path, all_vids_in_top)

    global_rank: List[str] = []
    seen_r: Set[str] = set()
    for p, _ in top:
        for e in p:
            vid = e[3]
            if vid and vid not in seen_r:
                seen_r.add(vid)
                global_rank.append(vid)

    top3 = paths[:3]
    top3_sc = scores[:3]
    # JSON-serializable paths (tuples break json.dumps in SSE)
    global_paths_json: List[List[List[Any]]] = [
        [[s, r, o, v] for s, r, o, v in p] for p in top3
    ]
    videos_in_path: List[List[str]] = [
        list(dict.fromkeys(e[3] for e in p)) for p in top3
    ]
    explanations = [
        global_path_to_sentence(
            p,
            node_display=node_display,
            video_labels=video_labels,
        )
        for p in top3
    ]

    return {
        "global_paths": global_paths_json,
        "global_path_scores": [round(s, 5) for s in top3_sc],
        "videos_in_path": videos_in_path,
        "global_rank": global_rank,
        "global_explanations": explanations,
        "global_query_entities": entities,
        "paths": paths,
        "scores": scores,
    }
