#!/usr/bin/env python3
"""
Persistent triple/node and edge frequencies for graph salience (HSC Phase 3).

Rebuild from facts table; does not alter facts schema.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from typing import Dict, Tuple


def rebuild_fact_frequency(db_path: str) -> None:
    """
    Scan facts, count node (subject + object) and edge (subject, relation, object) frequencies.
    Writes fact_nodes and fact_edges; clears stale flag.
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT subject, relation, object FROM facts "
                "WHERE subject IS NOT NULL AND relation IS NOT NULL AND object IS NOT NULL"
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return

        node_counts: Counter[str] = Counter()
        edge_counts: Counter[Tuple[str, str, str]] = Counter()

        for subj, rel, obj in rows:
            s = (subj or "").strip()
            r = (rel or "").strip()
            o = (obj or "").strip()
            if not s or not r or not o:
                continue
            # Nodes: normalized lowercase key for stable PK
            node_counts[s.lower()] += 1
            node_counts[o.lower()] += 1
            edge_counts[(s, r, o)] += 1

        cur.execute("DELETE FROM fact_nodes")
        cur.execute("DELETE FROM fact_edges")
        for node, freq in node_counts.items():
            cur.execute(
                "INSERT INTO fact_nodes (node, frequency) VALUES (?, ?)",
                (node, int(freq)),
            )
        for (s, r, o), freq in edge_counts.items():
            cur.execute(
                """
                INSERT INTO fact_edges (subject, relation, object, frequency)
                VALUES (?, ?, ?, ?)
                """,
                (s, r, o, int(freq)),
            )
        cur.execute(
            "INSERT OR IGNORE INTO fact_frequency_meta (id, stale) VALUES (1, 0)"
        )
        cur.execute("UPDATE fact_frequency_meta SET stale = 0 WHERE id = 1")
        conn.commit()


def _is_stale(db_path: str) -> bool:
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT stale FROM fact_frequency_meta WHERE id = 1")
            row = cur.fetchone()
            if row is None:
                return True
            return int(row[0]) != 0
    except sqlite3.OperationalError:
        return True


def load_node_frequency(db_path: str) -> Dict[str, int]:
    """
    Return node -> frequency from fact_nodes (rebuilds first if stale).

    Keys are lowercase for lookup; values are total counts (subject + object).
    """
    if _is_stale(db_path):
        rebuild_fact_frequency(db_path)

    out: Dict[str, int] = {}
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT node, frequency FROM fact_nodes")
            for node, freq in cur.fetchall():
                if node:
                    out[str(node).lower()] = int(freq)
    except sqlite3.OperationalError:
        return {}
    return out


def load_edge_frequency(db_path: str) -> Dict[Tuple[str, str, str], int]:
    """Optional: (subject, relation, object) -> count."""
    if _is_stale(db_path):
        rebuild_fact_frequency(db_path)

    out: Dict[Tuple[str, str, str], int] = {}
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT subject, relation, object, frequency FROM fact_edges"
            )
            for s, r, o, f in cur.fetchall():
                out[(s, r, o)] = int(f)
    except sqlite3.OperationalError:
        return {}
    return out
