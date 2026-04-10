#!/usr/bin/env python3
"""
Demonstrate cross-video global graph reasoning (HSC Phase 4).

Inserts a small demo chain across three real video_ids (facts table), rebuilds
global_facts, then runs global_graph_reason + hsc_retrieve so you can see
multi-video paths and explanations.

Remove demo rows: DELETE FROM facts WHERE source = 'cross_video_demo';
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "transcripts.db"
DEMO = "cross_video_demo"

# Three different videos -> one chain: PyTorch -> CUDA -> Driver 535 -> memory corruption
DEMO_TRIPLES = [
    ("08LGucX0mwM", "PyTorch", "uses", "CUDA"),
    ("0vvVo0Um1HY", "CUDA", "affected_by", "Driver 535"),
    ("5fm_aAmwXRk", "Driver 535", "causes", "memory corruption"),
]

QUERIES = [
    "What affected PyTorch?",  # routes to event; entities: pytorch, cuda, ...
    "How is PyTorch related to memory corruption?",
    "What bugs keep showing up across videos?",
]


def seed_demo(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("DELETE FROM facts WHERE source = ?", (DEMO,))
    for vid, s, r, o in DEMO_TRIPLES:
        cur.execute(
            """
            INSERT INTO facts (video_id, subject, relation, object, confidence, source)
            VALUES (?, ?, ?, ?, 1.0, ?)
            """,
            (vid, s, r, o, DEMO),
        )
    try:
        cur.execute(
            "INSERT OR IGNORE INTO global_graph_meta (id, stale) VALUES (1, 1)"
        )
        cur.execute("UPDATE global_graph_meta SET stale = 1 WHERE id = 1")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    print(f"[seed] Inserted {len(DEMO_TRIPLES)} demo facts (source={DEMO}).")


def main() -> None:
    os.chdir(ROOT)
    if not DB.exists():
        print("DB not found:", DB)
        sys.exit(1)

    conn = sqlite3.connect(str(DB))
    seed_demo(conn)
    conn.close()

    from hsc.global_graph_builder import build_global_graph
    from hsc.global_graph import global_graph_reason
    from hsc.query_router import route_query
    from hsc.hsc_search import hsc_retrieve

    n = build_global_graph(str(DB))
    print(f"[build_global_graph] rows in global_facts: {n}\n")

    print("=" * 70)
    print("GLOBAL GRAPH REASONING (cross-video paths)")
    print("=" * 70)
    for q in QUERIES:
        rt = route_query(q)
        print(f"\nQ: {q}")
        print(f"   route: {rt['type']} ({rt.get('reason', '')})")
        out = global_graph_reason(q, str(DB), max_depth=4, top_n=12)
        paths = out.get("global_paths") or []
        scores = out.get("global_path_scores") or []
        vids = out.get("videos_in_path") or []
        expl = out.get("global_explanations") or []
        ents = out.get("global_query_entities") or []
        print(f"   entities: {ents}")
        if not paths:
            print("   (no paths — graph may be empty or entities not in graph)")
            continue
        for i, p in enumerate(paths):
            sc = scores[i] if i < len(scores) else None
            vi = vids[i] if i < len(vids) else []
            print(f"   path[{i+1}] score={sc} videos_in_path={vi}")
            print(f"      steps: {p}")
        for i, ex in enumerate(expl):
            print(f"   sentence[{i+1}]: {ex}")

    print("\n" + "=" * 70)
    print("HSC RETRIEVE (fact/event gets SQL + local graph + global graph RRF)")
    print("=" * 70)
    q = "What affected PyTorch?"
    ctx = hsc_retrieve(
        str(DB),
        q,
        rag_collection=None,
        bm25_search=None,
        enable_bm25=False,
        enable_graph=False,
    )
    print(f"Query: {q}")
    print(f"query_type={ctx.get('query_type')} layer_used={ctx.get('layer_used')}")
    print(f"ranked_video_ids (top 15): {ctx.get('ranked_video_ids', [])[:15]}")
    print(f"global_paths (debug): {ctx.get('global_paths')}")
    print(f"global_path_scores: {ctx.get('global_path_scores')}")
    print(f"videos_in_path: {ctx.get('videos_in_path')}")
    print(f"reasoning_paths (local): {ctx.get('reasoning_paths')}")

    print("\nDone. To remove demo data:")
    print(f"  sqlite3 data/transcripts.db \"DELETE FROM facts WHERE source = '{DEMO}';\"")


if __name__ == "__main__":
    main()
