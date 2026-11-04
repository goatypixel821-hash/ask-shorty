#!/usr/bin/env python3
"""
Lightweight graph search over stored subject–relation–object facts (SQLite).
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple


# Question words / function words matched almost every fact row ("is", "the",
# "who", "of"), so with an unordered LIMIT the real match could be cut off.
_STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i if in "
    "into is it its me my of on one or our that the their them then there these they this those "
    "to us was we were what when where which who whom why will with would you your about".split()
)


def _tokens(q: str) -> List[str]:
    out: List[str] = []
    for t in re.findall(r"[A-Za-z0-9]+", q):
        t = t.lower()
        if len(t) >= 2 and t not in _STOPWORDS and t not in out:
            out.append(t)
    return out


def _like_escape(s: str) -> str:
    """Escape LIKE wildcards so entity text is matched literally (ESCAPE '\\')."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _conf(v: Any) -> float:
    # `float(v or 1.0)` turned an explicit 0.0 confidence into 1.0.
    return 1.0 if v is None else float(v)


class GraphSearch:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find videos related to query via fact rows.
        1. Match tokens against subject / relation / object
        2. One-hop: include videos sharing entities with matched rows
        """
        words = _tokens(query)
        if not words:
            return []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            try:
                cur.execute("SELECT 1 FROM facts LIMIT 1")
            except sqlite3.OperationalError:
                return []

            # Rank candidate rows by how many query tokens they contain *before*
            # applying the LIMIT, so the cap drops the weakest matches, not arbitrary ones.
            blob = "LOWER(subject || ' ' || relation || ' ' || object)"
            hit_expr = " + ".join(f"({blob} LIKE ? ESCAPE '\\')" for _ in words)
            params: List[Any] = [f"%{_like_escape(w)}%" for w in words]

            cur.execute(
                f"""
                SELECT video_id, subject, relation, object, confidence
                FROM (
                    SELECT video_id, subject, relation, object, confidence, ({hit_expr}) AS hits
                    FROM facts
                )
                WHERE hits > 0
                ORDER BY hits DESC, confidence DESC
                LIMIT 200
                """,
                params,
            )
            rows = [dict(r) for r in cur.fetchall()]

        scores: Dict[str, float] = {}
        snippets: Dict[str, str] = {}
        entities: Set[str] = set()

        for r in rows:
            vid = r["video_id"]
            sub = (r["subject"] or "").strip()
            rel = (r["relation"] or "").strip()
            obj = (r["object"] or "").strip()
            line = f"{sub} — {rel} — {obj}"
            overlap = sum(1 for w in words if w in (sub + " " + rel + " " + obj).lower())
            sc = _conf(r["confidence"]) * (1.0 + overlap)
            scores[vid] = scores.get(vid, 0.0) + sc
            if vid not in snippets or len(line) > len(snippets.get(vid, "")):
                snippets[vid] = line
            if sub:
                entities.add(sub.lower())
            if obj:
                entities.add(obj.lower())

        # One-hop expansion: other facts sharing subject/object tokens
        if entities and rows:
            eh = list(entities)[:40]
            sub_clauses = " OR ".join(
                "(LOWER(subject) LIKE ? ESCAPE '\\' OR LOWER(object) LIKE ? ESCAPE '\\')" for _ in eh
            )
            flat: List[str] = []
            for e in eh:
                like = f"%{_like_escape(e)}%"
                flat.extend([like, like])
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(
                    f"""
                    SELECT video_id, subject, relation, object, confidence
                    FROM facts
                    WHERE {sub_clauses}
                    LIMIT 300
                    """,
                    flat,
                )
                for r in cur.fetchall():
                    vid = r["video_id"]
                    sub = (r["subject"] or "").strip()
                    rel = (r["relation"] or "").strip()
                    obj = (r["object"] or "").strip()
                    line = f"{sub} — {rel} — {obj}"
                    scores[vid] = scores.get(vid, 0.0) + 0.3 * _conf(r["confidence"])
                    if vid not in snippets:
                        snippets[vid] = line

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        out: List[Dict[str, Any]] = []
        for vid, sc in ranked:
            out.append(
                {
                    "video_id": vid,
                    "score": sc,
                    "text_preview": snippets.get(vid, ""),
                }
            )
        return out

    def find_connections(self, entity1: str, entity2: str) -> List[List[str]]:
        """Return simple subject–relation–object chains that mention both entities (best-effort)."""
        e1 = (entity1 or "").strip().lower()
        e2 = (entity2 or "").strip().lower()
        if not e1 or not e2:
            return []
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT subject, relation, object FROM facts
                    WHERE (
                        (LOWER(subject) LIKE ? OR LOWER(object) LIKE ? OR LOWER(relation) LIKE ?)
                        AND (LOWER(subject) LIKE ? OR LOWER(object) LIKE ? OR LOWER(relation) LIKE ?)
                    )
                    """,
                    (f"%{e1}%", f"%{e1}%", f"%{e1}%", f"%{e2}%", f"%{e2}%", f"%{e2}%"),
                )
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                return []
        paths: List[List[str]] = []
        for sub, rel, obj in rows:
            line = f"{sub} {rel} {obj}".lower()
            if e1 in line and e2 in line:
                paths.append([sub or "", rel or "", obj or ""])
        return paths[:50]

    def get_video_facts(self, video_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT id, subject, relation, object, confidence, source
                    FROM facts WHERE video_id = ? ORDER BY id
                    """,
                    (video_id,),
                )
                return [dict(r) for r in cur.fetchall()]
            except sqlite3.OperationalError:
                return []
