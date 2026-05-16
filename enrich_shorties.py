#!/usr/bin/env python3
"""
Prepend DESCRIPTION / TAGS / CHAPTERS from videos.json_metadata onto existing Shorties.

Skips videos that already contain DESCRIPTION: or TAGS: in the Shorty body.

Usage:
  python enrich_shorties.py
  python enrich_shorties.py --dry-run --limit 10
  python enrich_shorties.py --db-path data/transcripts.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from batch_processor import _format_json_metadata_prompt_prefix

_DEFAULT_DB = Path(__file__).resolve().parent / "data" / "transcripts.db"

_SKIP_MARKERS = ("DESCRIPTION:", "TAGS:")


def _shorty_already_enriched(shorty: str) -> bool:
    return any(m in shorty for m in _SKIP_MARKERS)


def _meta_has_enrichable_fields(meta: Dict[str, Any]) -> bool:
    return bool(_format_json_metadata_prompt_prefix(meta).strip())


def _prepend_metadata(shorty: str, prefix: str) -> str:
    body = (shorty or "").strip()
    pre = (prefix or "").strip()
    if not pre:
        return body
    if not body:
        return pre
    return f"{pre}\n\n{body}"


def _fetch_rows(conn: sqlite3.Connection, limit: int) -> List[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    sql = """
        SELECT t.id AS transcript_id,
               v.video_id,
               t.shorty,
               v.json_metadata
        FROM videos v
        INNER JOIN (
            SELECT video_id, MAX(id) AS tid
            FROM transcripts
            WHERE shorty IS NOT NULL AND trim(shorty) != ''
            GROUP BY video_id
        ) pick ON pick.video_id = v.video_id
        INNER JOIN transcripts t ON t.id = pick.tid
        WHERE v.json_metadata IS NOT NULL AND trim(v.json_metadata) != ''
        ORDER BY v.video_id
    """
    if limit > 0:
        sql += f" LIMIT {int(limit)}"
    return list(conn.execute(sql).fetchall())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepend json_metadata (DESCRIPTION/TAGS/CHAPTERS) to existing Shorties."
    )
    ap.add_argument(
        "--db-path",
        type=Path,
        default=_DEFAULT_DB,
        help=f"Path to transcripts.db (default: {_DEFAULT_DB})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to the database.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max videos to process (0 = all).",
    )
    args = ap.parse_args()

    db_path = args.db_path.resolve()
    if not db_path.is_file():
        print(f"Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    rows = _fetch_rows(conn, args.limit)

    scanned = 0
    skipped_enriched = 0
    skipped_no_fields = 0
    skipped_bad_meta = 0
    updated = 0
    errors = 0

    print(f"Database: {db_path}")
    print(f"Mode    : {'dry-run' if args.dry_run else 'write'}")
    print(f"Candidates (has Shorty + json_metadata): {len(rows)}\n")

    for row in rows:
        scanned += 1
        tid = row["transcript_id"]
        vid = row["video_id"]
        shorty = row["shorty"] or ""

        if _shorty_already_enriched(shorty):
            skipped_enriched += 1
            if scanned % 500 == 0:
                print(f"  … scanned {scanned} (updated {updated}, skipped enriched {skipped_enriched})")
            continue

        try:
            meta = json.loads(row["json_metadata"] or "{}")
        except json.JSONDecodeError:
            skipped_bad_meta += 1
            continue

        if not isinstance(meta, dict):
            skipped_bad_meta += 1
            continue

        if not _meta_has_enrichable_fields(meta):
            skipped_no_fields += 1
            continue

        prefix = _format_json_metadata_prompt_prefix(meta)
        new_shorty = _prepend_metadata(shorty, prefix)

        if args.dry_run:
            updated += 1
            if updated <= 3:
                print(f"[dry-run] {vid} (+{len(prefix)} chars prefix)")
        else:
            try:
                conn.execute(
                    "UPDATE transcripts SET shorty = ? WHERE id = ?",
                    (new_shorty, tid),
                )
                updated += 1
            except sqlite3.Error as exc:
                errors += 1
                print(f"[ERROR] {vid}: {exc}", file=sys.stderr)

        if scanned % 500 == 0:
            print(f"  … scanned {scanned} (updated {updated}, skipped enriched {skipped_enriched})")

    if not args.dry_run and updated:
        conn.commit()
    conn.close()

    print("\nDone.")
    print(f"  Scanned              : {scanned}")
    print(f"  Updated              : {updated}")
    print(f"  Skipped (already had): {skipped_enriched}")
    print(f"  Skipped (no fields)  : {skipped_no_fields}")
    print(f"  Skipped (bad meta)   : {skipped_bad_meta}")
    if errors:
        print(f"  Errors               : {errors}")


if __name__ == "__main__":
    main()
