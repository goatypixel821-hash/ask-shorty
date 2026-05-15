#!/usr/bin/env python3
"""
Populate V2 index tables from existing videos, transcripts, entities, segments, events.

Usage:
  python populate_v2_indexes.py [--db-path PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from v2_schema import ensure_v2_tables, table_row_counts

_TS_ANY = re.compile(
    r"(?:\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})|\b\d{1,2}\s*(?:min|minutes|sec|seconds)\b",
    re.I,
)


def _bigrams(tokens: Sequence[str], min_len: int = 3) -> List[Tuple[str, str]]:
    t = [x.lower() for x in tokens if len(x) >= min_len]
    return [(t[i], t[i + 1]) for i in range(len(t) - 1)]


def _tokenize_words(text: str) -> List[str]:
    if not text:
        return []
    return [x.lower() for x in re.findall(r"[A-Za-z0-9]+", text) if len(x) >= 2]


def _routing_metadata_suffix(json_metadata: Any) -> str:
    """Append DESCRIPTION / TAGS / CHAPTERS from videos.json_metadata."""
    raw = json_metadata if isinstance(json_metadata, str) else ""
    raw = (raw or "").strip()
    if not raw:
        return ""
    try:
        meta = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(meta, dict):
        return ""

    parts: List[str] = []

    desc = ""
    for key in ("description", "shortDescription", "videoDescription", "desc"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            desc = v.strip()
            break
    if desc:
        if len(desc) > 500:
            desc = desc[:500]
        parts.append(f"DESCRIPTION: {desc}")

    tags = meta.get("tags")
    if isinstance(tags, list):
        tag_strs = [str(t).strip() for t in tags if t is not None and str(t).strip()]
        if tag_strs:
            parts.append("TAGS: " + ", ".join(tag_strs))

    chapters = meta.get("chapters")
    if chapters:
        titles: List[str] = []
        if isinstance(chapters, list):
            for ch in chapters:
                if isinstance(ch, dict):
                    t = (ch.get("title") or "").strip()
                    if t:
                        titles.append(t)
                elif isinstance(ch, str) and ch.strip():
                    titles.append(ch.strip())
        if titles:
            parts.append("CHAPTERS: " + ", ".join(titles))

    return " ".join(parts)


def _keywords_json(text: str, max_terms: int = 24) -> str:
    toks = _tokenize_words(text)
    c = Counter(toks)
    top = [w for w, _ in c.most_common(max_terms)]
    return json.dumps(top, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Populate Ask Shorty V2 index tables")
    ap.add_argument(
        "--db-path",
        default=None,
        help="transcripts.db path (default: ASK_SHORTY_DB_PATH or data/transcripts.db)",
    )
    args = ap.parse_args()

    import os

    db_path = args.db_path or os.environ.get("ASK_SHORTY_DB_PATH") or "data/transcripts.db"
    print(f"Using database: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_v2_tables(conn)
    cur = conn.cursor()

    # Full refresh
    for tbl in (
        "video_signatures",
        "segment_index",
        "event_index",
        "entity_postings",
        "topic_postings",
    ):
        cur.execute(f"DELETE FROM {tbl}")
    conn.commit()

    # --- video_signatures ---
    cur.execute(
        """
        SELECT v.video_id, v.title, v.channel, v.watch_date, v.json_metadata,
               t.shorty, t.text AS ttext
        FROM videos v
        JOIN (
            SELECT video_id, MAX(id) AS tid
            FROM transcripts
            WHERE shorty IS NOT NULL AND trim(shorty) != ''
            GROUP BY video_id
        ) pick ON pick.video_id = v.video_id
        JOIN transcripts t ON t.id = pick.tid
        """
    )
    video_rows = cur.fetchall()
    print(f"Videos with Shorty: {len(video_rows)}")

    for row in video_rows:
        vid = row["video_id"]
        title = (row["title"] or "").strip()
        channel = (row["channel"] or "").strip()
        shorty = (row["shorty"] or "").strip()
        raw_t = (row["ttext"] or "").strip()
        has_ts = 1 if (_TS_ANY.search(shorty) or _TS_ANY.search(raw_t)) else 0

        cur.execute(
            """
            SELECT question FROM synthetic_questions WHERE video_id = ?
            """,
            (vid,),
        )
        synqs = [r[0] for r in cur.fetchall() if r[0]]

        cur.execute(
            """
            SELECT name, aliases FROM entities WHERE video_id = ?
            """,
            (vid,),
        )
        ent_names: List[str] = []
        for er in cur.fetchall():
            name = (er["name"] or "").strip()
            if name:
                ent_names.append(name)
            try:
                als = json.loads(er["aliases"] or "[]")
            except Exception:
                als = []
            if isinstance(als, list):
                for a in als[:20]:
                    if isinstance(a, str) and a.strip():
                        ent_names.append(a.strip())
        ent_counter = Counter([e.lower() for e in ent_names])
        top_entities = [e for e, _ in ent_counter.most_common(12)]
        top_entities_json = json.dumps(top_entities, ensure_ascii=False)

        topic_scores: Counter[str] = Counter()
        for bg in _bigrams(_tokenize_words(shorty + " " + " ".join(synqs))):
            topic_scores["%s %s" % bg] += 1
        top_topics = [p for p, _ in topic_scores.most_common(20)]
        top_topics_json = json.dumps(top_topics, ensure_ascii=False)

        routing = " ".join(
            [
                title,
                channel,
                shorty,
                " ".join(synqs),
                " ".join(top_topics),
            ]
        ).strip()
        meta_suffix = _routing_metadata_suffix(row["json_metadata"])
        if meta_suffix:
            routing = f"{routing} {meta_suffix}".strip()

        cur.execute(
            """
            INSERT INTO video_signatures (
                video_id, routing_text, top_entities, top_topics,
                has_timestamps, watch_date, channel
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                vid,
                routing,
                top_entities_json,
                top_topics_json,
                has_ts,
                row["watch_date"],
                channel,
            ),
        )

    # --- segment_index ---
    cur.execute(
        """
        SELECT id, video_id, start_time, end_time, summary
        FROM segments
        """
    )
    seg_rows = cur.fetchall()
    for sr in seg_rows:
        summ = (sr["summary"] or "").strip()
        clean = summ
        cur.execute(
            """
            INSERT INTO segment_index (
                segment_id, video_id, summary, clean_text,
                start_s, end_s, keywords_json, entities_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sr["id"],
                sr["video_id"],
                summ,
                clean,
                sr["start_time"],
                sr["end_time"],
                _keywords_json(summ),
                json.dumps([], ensure_ascii=False),
            ),
        )

    # --- event_index ---
    cur.execute(
        """
        SELECT id, video_id, title, cause, effect, systems
        FROM events
        """
    )
    ev_rows = cur.fetchall()
    for er in ev_rows:
        cur.execute(
            """
            INSERT INTO event_index (
                event_id, video_id, title, cause, effect, systems, start_s, end_s
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                er["id"],
                er["video_id"],
                er["title"],
                er["cause"],
                er["effect"],
                er["systems"],
            ),
        )

    # --- entity_postings ---
    cur.execute(
        """
        SELECT video_id, name, aliases FROM entities
        """
    )
    for er in cur.fetchall():
        vid = er["video_id"]
        name = (er["name"] or "").strip()
        if not name:
            continue
        cur.execute(
            """
            INSERT INTO entity_postings (entity_name, alias, video_id, segment_id, count)
            VALUES (?, ?, ?, NULL, 1)
            """,
            (name, name, vid),
        )
        try:
            als = json.loads(er["aliases"] or "[]")
        except Exception:
            als = []
        if isinstance(als, list):
            for a in als:
                if isinstance(a, str) and a.strip():
                    al = a.strip()
                    cur.execute(
                        """
                        INSERT INTO entity_postings (entity_name, alias, video_id, segment_id, count)
                        VALUES (?, ?, ?, NULL, 1)
                        """,
                        (name, al, vid),
                    )

    # --- topic_postings (video-level from Shorty bigrams; segment-level from summaries) ---
    cur.execute(
        """
        SELECT video_id, routing_text FROM video_signatures
        """
    )
    for vr in cur.fetchall():
        vid = vr["video_id"]
        rt = vr["routing_text"] or ""
        c_bg = Counter(_bigrams(_tokenize_words(rt)))
        for (bg, sc) in c_bg.most_common(80):
            phrase = "%s %s" % bg
            cur.execute(
                """
                INSERT INTO topic_postings (topic_phrase, video_id, segment_id, score)
                VALUES (?, ?, NULL, ?)
                """,
                (phrase, vid, float(sc)),
            )

    cur.execute("SELECT segment_id, video_id, clean_text FROM segment_index")
    for sr in cur.fetchall():
        sid = sr["segment_id"]
        vid = sr["video_id"]
        ct = sr["clean_text"] or ""
        c_bg = Counter(_bigrams(_tokenize_words(ct)))
        for (bg, sc) in c_bg.most_common(40):
            phrase = "%s %s" % bg
            cur.execute(
                """
                INSERT INTO topic_postings (topic_phrase, video_id, segment_id, score)
                VALUES (?, ?, ?, ?)
                """,
                (phrase, vid, sid, float(sc)),
            )

    conn.commit()

    counts = table_row_counts(conn)
    print("Row counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
