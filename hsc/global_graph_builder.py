"""
Build global_facts from facts (normalized triples, deduped per video).
"""
from __future__ import annotations

import sqlite3

from hsc.entity_normalizer import load_aliases_from_db, normalize_entity


def build_global_graph(db_path: str) -> int:
    """
    Read all facts, normalize subject/object, insert into global_facts.
    Deduplicates identical (subject_norm, relation, object_norm, video_id).

    Returns:
        Number of rows inserted into global_facts (after clear).
    """
    from transcript_database import TranscriptDatabase

    TranscriptDatabase(db_path).ensure_db_exists()
    extra_aliases = load_aliases_from_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM global_facts")
        cur.execute(
            "SELECT video_id, subject, relation, object FROM facts"
        )
        rows = cur.fetchall()
        seen: set[tuple[str, str, str, str]] = set()
        for video_id, subj, rel, obj in rows:
            subj = subj or ""
            obj = obj or ""
            rel_s = (rel or "").strip().lower()
            ns = normalize_entity(subj, extra_aliases=extra_aliases)["normalized"]
            no = normalize_entity(obj, extra_aliases=extra_aliases)["normalized"]
            if not ns or not no or not rel_s:
                continue
            key = (ns, rel_s, no, str(video_id))
            if key in seen:
                continue
            seen.add(key)
            cur.execute(
                """
                INSERT OR IGNORE INTO global_facts
                (subject_norm, relation, object_norm, subject_raw, object_raw, video_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ns, rel_s, no, subj.strip(), obj.strip(), str(video_id)),
            )
        cur.execute("SELECT COUNT(*) FROM global_facts")
        total = int(cur.fetchone()[0])
        try:
            cur.execute(
                "INSERT OR IGNORE INTO global_graph_meta (id, stale) VALUES (1, 0)"
            )
            cur.execute("UPDATE global_graph_meta SET stale = 0 WHERE id = 1")
        except sqlite3.OperationalError:
            pass
        conn.commit()
    return total


def ensure_global_graph_fresh(db_path: str) -> None:
    """Rebuild global_facts if global_graph_meta.stale is set."""
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT stale FROM global_graph_meta WHERE id = 1")
            row = cur.fetchone()
            stale = row[0] if row else 1
    except sqlite3.OperationalError:
        stale = 1
    if stale:
        build_global_graph(db_path)
