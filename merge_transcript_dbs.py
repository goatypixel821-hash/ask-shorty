#!/usr/bin/env python3
"""
Merge youtube-history-viewer-copy transcripts.db into shorty transcripts.db (canonical).

- Default: timestamped backups (unless --no-backup), then dry-run counts only (no DB writes).
- --execute: same backups, then merge viewer rows into canonical, then summary stats.

Rules (videos in both DBs):
- Field-by-field: never replace a non-empty canonical value with an empty viewer value.
- watch_date, transcript text, and shorty (on transcript rows) follow the same rule when overlaying viewer onto canonical.
- When both sides have a non-empty value for the same column, keep canonical (shorty-enriched source of truth).
- "More non-null columns": if viewer has strictly more non-null columns than canonical for that video row,
  use viewer as the base row for the merge, then overlay the other side for any fields still empty on the chosen base
  (same non-empty-wins rules; canonical still wins ties on conflicts).

Child tables: insert viewer rows that are not already represented in canonical (dedupe keys per table).
Does not touch Chroma. Does not merge fact_nodes / fact_edges (rebuild from facts separately if needed).
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


CANONICAL_DEFAULT = Path(r"C:\Users\number2\Desktop\shorty\data\transcripts.db")
VIEWER_DEFAULT = Path(r"C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db")
BACKUP_DIR_DEFAULT = Path(r"C:\Users\number2\Desktop\shorty\data\db_merge_backups")


# Preferred column order when both DBs share the same schema (transcript_database.py).
VIDEO_COLS_PREFERRED = [
    "video_id",
    "title",
    "channel",
    "url",
    "has_transcript",
    "transcript_fetched_at",
    "watch_date",
    "local_path",
    "json_metadata",
    "created_at",
]


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_nonempty(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, (int, float)) and val == 0:
        return True  # 0 is meaningful
    if isinstance(val, bool):
        return True
    s = str(val).strip()
    return len(s) > 0


def non_null_count(row: Dict[str, Any]) -> int:
    n = 0
    for c, v in row.items():
        if c == "video_id":
            continue
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        n += 1
    return n


def fetch_video_row(conn: sqlite3.Connection, video_id: str) -> Dict[str, Any]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,))
    r = cur.fetchone()
    if not r:
        return {}
    cols = [d[0] for d in cur.description]
    return {cols[i]: r[i] for i in range(len(cols))}


def pragma_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def ordered_video_columns(canonical: sqlite3.Connection) -> List[str]:
    have = set(pragma_columns(canonical, "videos"))
    out: List[str] = [c for c in VIDEO_COLS_PREFERRED if c in have]
    for c in sorted(have):
        if c not in out:
            out.append(c)
    return out


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    )
    return cur.fetchone() is not None


def backup_file(src: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / f"{src.stem}_{_ts()}{src.suffix}"
    shutil.copy2(src, dst)
    return dst


def open_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


@dataclass
class DryCounts:
    videos_insert: int = 0
    videos_update: int = 0
    videos_skip_viewer: int = 0  # viewer row subset / no-op merge
    transcripts_new_rows: int = 0
    entities_insert: int = 0
    synthetic_insert: int = 0
    facts_insert: int = 0
    global_facts_insert: int = 0
    processing_queue_insert: int = 0
    segments_insert: int = 0
    events_insert: int = 0


def merge_video_rows(can: Dict[str, Any], view: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two video rows. Start from the row with more non-null columns (base),
    then fill empty fields from the other row. When both sides are non-empty for a
    column, keep the base row's value (richer row wins conflicts).
    Never produces an empty value where base had a non-empty value (overlay only fills gaps).
    """
    if not can:
        return dict(view)
    if not view:
        return dict(can)
    nn_can = non_null_count(can)
    nn_view = non_null_count(view)
    if nn_view > nn_can:
        base, overlay = dict(view), dict(can)
    else:
        base, overlay = dict(can), dict(view)
    out = dict(base)
    keys = sorted(set(out) | set(overlay))
    for c in keys:
        if c == "video_id":
            continue
        b, o = out.get(c), overlay.get(c)
        if not is_nonempty(b) and is_nonempty(o):
            out[c] = o
        elif is_nonempty(b) and not is_nonempty(o):
            out[c] = b
        elif is_nonempty(b) and is_nonempty(o):
            out[c] = b  # base (richer) wins on conflicts
        else:
            out[c] = b if b is not None else o
    out["video_id"] = can.get("video_id") or view.get("video_id")
    return out


def transcript_row_key(video_id: str, text: Optional[str]) -> Tuple[str, str]:
    return (video_id, (text or "").strip())


def load_transcript_keys(conn: sqlite3.Connection) -> Set[Tuple[str, str]]:
    if not table_exists(conn, "transcripts"):
        return set()
    cur = conn.cursor()
    cur.execute("SELECT video_id, text FROM transcripts")
    return {(str(r[0]), (r[1] or "").strip()) for r in cur.fetchall()}


def load_entity_keys(conn: sqlite3.Connection) -> Set[Tuple[str, str, str]]:
    if not table_exists(conn, "entities"):
        return set()
    cur = conn.cursor()
    cur.execute("SELECT video_id, name, type FROM entities")
    return {(str(r[0]), (r[1] or "").strip(), (r[2] or "").strip()) for r in cur.fetchall()}


def load_question_keys(conn: sqlite3.Connection) -> Set[Tuple[str, str]]:
    if not table_exists(conn, "synthetic_questions"):
        return set()
    cur = conn.cursor()
    cur.execute("SELECT video_id, question FROM synthetic_questions")
    return {(str(r[0]), (r[1] or "").strip()) for r in cur.fetchall()}


def load_fact_keys(conn: sqlite3.Connection) -> Set[Tuple[str, str, str, str]]:
    if not table_exists(conn, "facts"):
        return set()
    cur = conn.cursor()
    cur.execute("SELECT video_id, subject, relation, object FROM facts")
    return {
        (str(r[0]), (r[1] or "").strip(), (r[2] or "").strip(), (r[3] or "").strip())
        for r in cur.fetchall()
    }


def load_global_fact_keys(conn: sqlite3.Connection) -> Set[Tuple[str, str, str, str]]:
    if not table_exists(conn, "global_facts"):
        return set()
    cur = conn.cursor()
    cur.execute(
        "SELECT video_id, subject_norm, relation, object_norm FROM global_facts"
    )
    return {
        (str(r[0]), (r[1] or "").strip(), (r[2] or "").strip(), (r[3] or "").strip())
        for r in cur.fetchall()
    }


def load_processing_keys(conn: sqlite3.Connection) -> Set[Tuple[str, str, str, str]]:
    if not table_exists(conn, "processing_queue"):
        return set()
    cur = conn.cursor()
    cur.execute(
        "SELECT video_id, task, IFNULL(status,''), IFNULL(created_at,'') FROM processing_queue"
    )
    return {(str(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in cur.fetchall()}


def dry_run(canonical: Path, viewer: Path) -> DryCounts:
    c = open_ro(canonical)
    v = open_ro(viewer)
    counts = DryCounts()
    try:
        video_cols = ordered_video_columns(c)

        def _norm_cell(x: Any) -> Optional[str]:
            if x is None:
                return None
            s = str(x).strip()
            return s if s else None

        cur = v.cursor()
        cur.execute("SELECT video_id FROM videos")
        viewer_ids = [r[0] for r in cur.fetchall()]
        for vid in viewer_ids:
            can = fetch_video_row(c, vid)
            view = fetch_video_row(v, vid)
            if not can:
                counts.videos_insert += 1
            else:
                merged = merge_video_rows(can, view)
                merged = {k: merged.get(k) for k in video_cols}
                can_f = {k: can.get(k) for k in video_cols}
                if all(_norm_cell(merged.get(k)) == _norm_cell(can_f.get(k)) for k in video_cols):
                    counts.videos_skip_viewer += 1
                else:
                    counts.videos_update += 1

        t_can = load_transcript_keys(c)
        if table_exists(v, "transcripts"):
            cur.execute("SELECT video_id, text FROM transcripts")
            for video_id, text in cur.fetchall():
                k = transcript_row_key(str(video_id), text)
                if k not in t_can:
                    counts.transcripts_new_rows += 1

        e_can = load_entity_keys(c)
        if table_exists(v, "entities"):
            cur.execute("SELECT video_id, name, type FROM entities")
            for row in cur.fetchall():
                k = (str(row[0]), (row[1] or "").strip(), (row[2] or "").strip())
                if k not in e_can:
                    counts.entities_insert += 1

        q_can = load_question_keys(c)
        if table_exists(v, "synthetic_questions"):
            cur.execute("SELECT video_id, question FROM synthetic_questions")
            for row in cur.fetchall():
                k = (str(row[0]), (row[1] or "").strip())
                if k not in q_can:
                    counts.synthetic_insert += 1

        f_can = load_fact_keys(c)
        if table_exists(v, "facts"):
            cur.execute("SELECT video_id, subject, relation, object FROM facts")
            for row in cur.fetchall():
                k = (str(row[0]), (row[1] or "").strip(), (row[2] or "").strip(), (row[3] or "").strip())
                if k not in f_can:
                    counts.facts_insert += 1

        g_can = load_global_fact_keys(c)
        if table_exists(v, "global_facts"):
            cur.execute(
                "SELECT video_id, subject_norm, relation, object_norm FROM global_facts"
            )
            for row in cur.fetchall():
                k = (str(row[0]), (row[1] or "").strip(), (row[2] or "").strip(), (row[3] or "").strip())
                if k not in g_can:
                    counts.global_facts_insert += 1

        p_can = load_processing_keys(c)
        if table_exists(v, "processing_queue"):
            cur.execute(
                "SELECT video_id, task, IFNULL(status,''), IFNULL(created_at,'') FROM processing_queue"
            )
            for row in cur.fetchall():
                k = (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
                if k not in p_can:
                    counts.processing_queue_insert += 1

        # segments / events: loose dedupe by content fingerprint
        def seg_key(vid, st, et, sm):
            return (vid, st, et, (sm or "")[:200])

        s_keys: Set[Tuple] = set()
        if table_exists(c, "segments"):
            cur = c.cursor()
            cur.execute(
                "SELECT video_id, start_time, end_time, summary FROM segments"
            )
            for r in cur.fetchall():
                s_keys.add(seg_key(str(r[0]), r[1], r[2], r[3]))
        if table_exists(v, "segments"):
            cur = v.cursor()
            cur.execute(
                "SELECT video_id, start_time, end_time, summary FROM segments"
            )
            for r in cur.fetchall():
                k = seg_key(str(r[0]), r[1], r[2], r[3])
                if k not in s_keys:
                    counts.segments_insert += 1

        ev_keys: Set[Tuple] = set()
        if table_exists(c, "events"):
            cur = c.cursor()
            cur.execute("SELECT video_id, title, cause, effect FROM events")
            for r in cur.fetchall():
                ev_keys.add(
                    (str(r[0]), (r[1] or "").strip(), (r[2] or "").strip(), (r[3] or "").strip())
                )
        if table_exists(v, "events"):
            cur = v.cursor()
            cur.execute("SELECT video_id, title, cause, effect FROM events")
            for r in cur.fetchall():
                k = (str(r[0]), (r[1] or "").strip(), (r[2] or "").strip(), (r[3] or "").strip())
                if k not in ev_keys:
                    counts.events_insert += 1

    finally:
        c.close()
        v.close()
    return counts


def execute_merge(canonical: Path, viewer: Path) -> None:
    c = sqlite3.connect(str(canonical))
    v = open_ro(viewer)
    try:
        cur_c = c.cursor()
        cur_v = v.cursor()
        vcols = ordered_video_columns(c)

        # --- videos (schema-safe column list from canonical) ---
        cur_v.execute("SELECT * FROM videos")
        vdesc = [d[0] for d in cur_v.description]
        for row in cur_v.fetchall():
            view = {vdesc[i]: row[i] for i in range(len(vdesc))}
            vid = view.get("video_id")
            if not vid:
                continue
            can = fetch_video_row(c, str(vid))
            merged = merge_video_rows(can, view) if can else dict(view)
            merged = {k: merged.get(k) for k in vcols}
            if not can:
                placeholders = ", ".join(["?"] * len(vcols))
                cur_c.execute(
                    f"INSERT INTO videos ({', '.join(vcols)}) VALUES ({placeholders})",
                    [merged.get(k) for k in vcols],
                )
            else:
                sets = ", ".join(f"{k} = ?" for k in vcols if k != "video_id")
                vals = [merged.get(k) for k in vcols if k != "video_id"] + [vid]
                cur_c.execute(f"UPDATE videos SET {sets} WHERE video_id = ?", vals)

        # transcripts (all viewer rows not in canonical by video_id+text)
        t_keys = load_transcript_keys(c)
        if table_exists(v, "transcripts"):
            cur_v.execute(
                "SELECT video_id, text, language, confidence, shorty, shorty_generated_at, created_at FROM transcripts"
            )
            tdesc = [d[0] for d in cur_v.description]
            for row in cur_v.fetchall():
                d = {tdesc[i]: row[i] for i in range(len(tdesc))}
                k = transcript_row_key(str(d["video_id"]), d.get("text"))
                if k in t_keys:
                    continue
                t_keys.add(k)
                cur_c.execute(
                    """
                    INSERT INTO transcripts (video_id, text, language, confidence, shorty, shorty_generated_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        d.get("video_id"),
                        d.get("text"),
                        d.get("language"),
                        d.get("confidence"),
                        d.get("shorty"),
                        d.get("shorty_generated_at"),
                        d.get("created_at"),
                    ),
                )

        e_keys = load_entity_keys(c)
        if table_exists(v, "entities"):
            cur_v.execute("SELECT video_id, name, type, aliases, created_at FROM entities")
            edesc = [d[0] for d in cur_v.description]
            for row in cur_v.fetchall():
                er = {edesc[i]: row[i] for i in range(len(edesc))}
                vid, name, typ, aliases, created_at = (
                    er.get("video_id"),
                    er.get("name"),
                    er.get("type"),
                    er.get("aliases"),
                    er.get("created_at"),
                )
                k = (str(vid), (name or "").strip(), (typ or "").strip())
                if k in e_keys:
                    continue
                e_keys.add(k)
                cur_c.execute(
                    "INSERT INTO entities (video_id, name, type, aliases, created_at) VALUES (?,?,?,?,?)",
                    (vid, name, typ, aliases, created_at),
                )

        q_keys = load_question_keys(c)
        if table_exists(v, "synthetic_questions"):
            cur_v.execute(
                "SELECT video_id, question, embedding_id, created_at FROM synthetic_questions"
            )
            for row in cur_v.fetchall():
                vid, q, emb, created_at = row
                k = (str(vid), (q or "").strip())
                if k in q_keys:
                    continue
                q_keys.add(k)
                cur_c.execute(
                    "INSERT INTO synthetic_questions (video_id, question, embedding_id, created_at) VALUES (?,?,?,?)",
                    (vid, q, emb, created_at),
                )

        f_keys = load_fact_keys(c)
        if table_exists(v, "facts"):
            cur_v.execute(
                "SELECT video_id, subject, relation, object, confidence, source, created_at FROM facts"
            )
            for row in cur_v.fetchall():
                vid, subj, rel, obj, conf, src, created_at = row
                k = (str(vid), (subj or "").strip(), (rel or "").strip(), (obj or "").strip())
                if k in f_keys:
                    continue
                f_keys.add(k)
                cur_c.execute(
                    "INSERT INTO facts (video_id, subject, relation, object, confidence, source, created_at) VALUES (?,?,?,?,?,?,?)",
                    (vid, subj, rel, obj, conf, src, created_at),
                )

        g_keys = load_global_fact_keys(c)
        if table_exists(v, "global_facts"):
            cur_v.execute(
                "SELECT subject_norm, relation, object_norm, subject_raw, object_raw, video_id FROM global_facts"
            )
            for row in cur_v.fetchall():
                sn, rel, on_, sr, oraw, vid = row
                k = (str(vid), (sn or "").strip(), (rel or "").strip(), (on_ or "").strip())
                if k in g_keys:
                    continue
                g_keys.add(k)
                cur_c.execute(
                    "INSERT OR IGNORE INTO global_facts (subject_norm, relation, object_norm, subject_raw, object_raw, video_id) VALUES (?,?,?,?,?,?)",
                    (sn, rel, on_, sr, oraw, vid),
                )

        p_keys = load_processing_keys(c)
        if table_exists(v, "processing_queue"):
            cur_v.execute(
                "SELECT video_id, task, status, created_at, started_at, completed_at, error, attempts FROM processing_queue"
            )
            for row in cur_v.fetchall():
                vid, task, status, created_at, started_at, completed_at, err, attempts = row
                k = (str(vid), str(task), str(status or ""), str(created_at or ""))
                if k in p_keys:
                    continue
                p_keys.add(k)
                cur_c.execute(
                    """
                    INSERT INTO processing_queue (video_id, task, status, created_at, started_at, completed_at, error, attempts)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (vid, task, status, created_at, started_at, completed_at, err, attempts),
                )

        # segments
        s_keys: Set[Tuple] = set()
        if table_exists(c, "segments"):
            cur_c.execute("SELECT video_id, start_time, end_time, summary FROM segments")
            for r in cur_c.fetchall():
                s_keys.add((str(r[0]), r[1], r[2], (r[3] or "")[:200]))
        if table_exists(v, "segments"):
            cur_v.execute(
                "SELECT video_id, start_time, end_time, summary, embedding, created_at FROM segments"
            )
            for r in cur_v.fetchall():
                k = (str(r[0]), r[1], r[2], (r[3] or "")[:200])
                if k in s_keys:
                    continue
                s_keys.add(k)
                if table_exists(c, "segments"):
                    cur_c.execute(
                        "INSERT INTO segments (video_id, start_time, end_time, summary, embedding, created_at) VALUES (?,?,?,?,?,?)",
                        (r[0], r[1], r[2], r[3], r[4], r[5]),
                    )

        ev_keys: Set[Tuple] = set()
        if table_exists(c, "events"):
            cur_c.execute("SELECT video_id, title, cause, effect FROM events")
            for r in cur_c.fetchall():
                ev_keys.add(
                    (str(r[0]), (r[1] or "").strip(), (r[2] or "").strip(), (r[3] or "").strip())
                )
        if table_exists(v, "events"):
            cur_v.execute(
                "SELECT video_id, title, cause, effect, systems, raw_json, created_at FROM events"
            )
            for r in cur_v.fetchall():
                k = (str(r[0]), (r[1] or "").strip(), (r[2] or "").strip(), (r[3] or "").strip())
                if k in ev_keys:
                    continue
                ev_keys.add(k)
                if table_exists(c, "events"):
                    cur_c.execute(
                        "INSERT INTO events (video_id, title, cause, effect, systems, raw_json, created_at) VALUES (?,?,?,?,?,?,?)",
                        (r[0], r[1], r[2], r[3], r[4], r[5], r[6]),
                    )

        c.commit()
    finally:
        v.close()
        c.close()


def final_stats(canonical: Path) -> Dict[str, int]:
    conn = sqlite3.connect(str(canonical))
    cur = conn.cursor()
    out: Dict[str, int] = {}
    cur.execute("SELECT COUNT(*) FROM videos")
    out["videos_total"] = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(DISTINCT v.video_id) FROM videos v "
        "INNER JOIN transcripts t ON t.video_id = v.video_id "
        "WHERE t.text IS NOT NULL AND length(trim(t.text))>0"
    )
    out["with_transcript"] = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(DISTINCT v.video_id) FROM videos v "
        "INNER JOIN transcripts t ON t.video_id = v.video_id "
        "WHERE t.shorty IS NOT NULL AND length(trim(t.shorty))>0"
    )
    out["with_shorty"] = cur.fetchone()[0]
    cur.execute(
        """
        SELECT COUNT(*) FROM videos v
        WHERE NOT EXISTS (
            SELECT 1 FROM transcripts t
            WHERE t.video_id = v.video_id
            AND t.text IS NOT NULL AND length(trim(t.text))>0
        )
        AND NOT EXISTS (
            SELECT 1 FROM transcripts t2
            WHERE t2.video_id = v.video_id
            AND t2.shorty IS NOT NULL AND length(trim(t2.shorty))>0
        )
        """
    )
    out["missing_both"] = cur.fetchone()[0]
    conn.close()
    return out


def print_dry(counts: DryCounts) -> None:
    print("\n=== DRY RUN (no changes to canonical) ===\n")
    print("videos:")
    print(f"  insert (new video_id from viewer):     {counts.videos_insert}")
    print(f"  update (merge fields into canonical): {counts.videos_update}")
    print(f"  skip (viewer row would not change canonical): {counts.videos_skip_viewer}")
    print("\ntranscripts:")
    print(f"  new transcript rows (viewer-only text): {counts.transcripts_new_rows}")
    print("\nentities:                    ", counts.entities_insert)
    print("synthetic_questions:         ", counts.synthetic_insert)
    print("facts:                       ", counts.facts_insert)
    print("global_facts:                ", counts.global_facts_insert)
    print("processing_queue:          ", counts.processing_queue_insert)
    print("segments (extra, optional): ", counts.segments_insert)
    print("events (extra, optional):  ", counts.events_insert)
    print("\nNote: fact_nodes / fact_edges are NOT merged (rebuild from facts if needed).")
    print("\nNext: review the plan, then run with --execute to backup and merge.")


def main() -> int:
    p = argparse.ArgumentParser(description="Merge viewer transcripts.db into shorty canonical DB.")
    p.add_argument("--canonical", type=Path, default=CANONICAL_DEFAULT)
    p.add_argument("--viewer", type=Path, default=VIEWER_DEFAULT)
    p.add_argument("--backup-dir", type=Path, default=BACKUP_DIR_DEFAULT)
    p.add_argument(
        "--execute",
        action="store_true",
        help="After dry-run output, merge viewer into canonical (still creates backups unless --no-backup).",
    )
    p.add_argument("--no-backup", action="store_true", help="Skip file backups (not recommended).")
    args = p.parse_args()

    canonical: Path = args.canonical
    viewer: Path = args.viewer
    if not canonical.is_file():
        print("Canonical DB not found:", canonical, file=sys.stderr)
        return 1
    if not viewer.is_file():
        print("Viewer DB not found:", viewer, file=sys.stderr)
        return 1

    if not args.no_backup:
        b1 = backup_file(canonical, args.backup_dir)
        b2 = backup_file(viewer, args.backup_dir)
        print("Backups created:")
        print(" ", b1)
        print(" ", b2)
    else:
        print("(Skipping backups due to --no-backup)")

    counts = dry_run(canonical, viewer)
    print_dry(counts)

    if not args.execute:
        return 0

    print("\n=== EXECUTE MERGE ===\n")
    execute_merge(canonical, viewer)
    stats = final_stats(canonical)
    print("=== AFTER MERGE ===\n")
    print(f"total videos:        {stats['videos_total']}")
    print(f"with transcript:     {stats['with_transcript']}")
    print(f"with shorty:         {stats['with_shorty']}")
    print(f"missing both:        {stats['missing_both']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
