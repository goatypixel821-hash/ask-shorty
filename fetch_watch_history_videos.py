#!/usr/bin/env python3
"""
Fill stub videos (watch-history import: video_id + watch_date + url only) with
metadata (yt-dlp), transcript (youtube-transcript-api), optional Chroma index,
and processing_queue tasks — modeled on youtube-history-viewer-copy's
batch_liked_to_main_db.py (pacing, session cap, jitter, logs).

Default: 50 videos per run, ~3 min + jitter between videos, 5 h session cap.

Uses fetch_transcript(video_id, url) only — NOT fetch_transcript_from_url —
so INSERT OR REPLACE in add_video cannot wipe watch_date on existing rows.

Examples:
  python -u fetch_watch_history_videos.py --dry-run
  python -u fetch_watch_history_videos.py
  python -u fetch_watch_history_videos.py --limit 20 --no-vectorize
  python -u fetch_watch_history_videos.py --aggressive --limit 100
  python -u fetch_watch_history_videos.py --remote root@1.2.3.4:40166 --sync-up --sync-down

Remote (Vast): use --sync-up before the run and --sync-down after to scp transcripts.db to
/workspace/transcripts.db and merge new rows back locally (scp -v shows progress). --dry-run
skips all SCP and fetching.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_VIEWER_DB = Path(
    r"C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db"
)
DATA = ROOT / "data"
FAIL_LOG = DATA / "watch_history_stub_fetch_failures.log"
RUN_LOG = DATA / "watch_history_stub_fetch.log"
SKIP_FILE = DATA / "watch_history_no_transcript.json"

DEFAULT_DELAY_MINUTES = 3.0
DEFAULT_SESSION_MINUTES = 300.0
DEFAULT_JITTER_SECONDS = 90.0
DEFAULT_LIMIT = 50
AGGRESSIVE_DELAY_SECONDS = 15.0
REMOTE_DB_PATH = "/workspace/transcripts.db"


def default_db_path() -> Path:
    env = os.environ.get("ASK_SHORTY_DB_PATH", "").strip()
    if env:
        return Path(env)
    return DEFAULT_VIEWER_DB


def default_chroma_dir(db_path: Path) -> Path:
    env = os.environ.get("ASK_SHORTY_CHROMA_PATH", "").strip()
    if env:
        return Path(env)
    for name in ("transcript_chroma_new", "transcript_chroma", "transcript_chroma_finetuned"):
        cand = db_path.parent / name
        if cand.is_dir():
            return cand
    return db_path.parent / "transcript_chroma_new"


def log_run(msg: str, also_print: bool = True) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().isoformat()} {msg}\n"
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line)
    if also_print:
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def log_failure(video_id: str, title: str, error: str) -> None:
    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FAIL_LOG, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now().isoformat()}\t{video_id}\t{title[:120]!r}\t{error}\n"
        )


def load_skip_ids() -> set[str]:
    if not SKIP_FILE.exists():
        return set()
    try:
        return set(
            json.load(open(SKIP_FILE, encoding="utf-8")).get("no_transcript", [])
        )
    except Exception:
        return set()


def add_skip_id(video_id: str, title: str, reason: str) -> None:
    skip: Dict[str, Any] = {}
    if SKIP_FILE.exists():
        try:
            skip = json.load(open(SKIP_FILE, encoding="utf-8"))
        except Exception:
            skip = {}
    ids: List[str] = list(skip.get("no_transcript", []))
    details: Dict[str, Any] = dict(skip.get("details", {}))
    if video_id not in ids:
        ids.append(video_id)
        details[video_id] = {
            "title": title[:120],
            "reason": reason,
            "skipped_at": datetime.now().isoformat(),
        }
        skip["no_transcript"] = ids
        skip["details"] = details
        SKIP_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SKIP_FILE, "w", encoding="utf-8") as f:
            json.dump(skip, f, indent=2, ensure_ascii=False)
        log_run(
            f"  [SKIP] Added {video_id} to permanent no-transcript list ({len(ids)} total)"
        )


def ensure_json_metadata_column(db_path: Path) -> None:
    if not db_path.exists():
        return
    with sqlite3.connect(str(db_path)) as conn:
        c = conn.cursor()
        c.execute("PRAGMA table_info(videos)")
        cols = [row[1] for row in c.fetchall()]
        if "json_metadata" not in cols:
            c.execute("ALTER TABLE videos ADD COLUMN json_metadata TEXT")
            conn.commit()


STUB_SELECT = """
SELECT v.video_id, v.url, v.watch_date, v.title
FROM videos v
WHERE v.url IS NOT NULL AND trim(v.url) != ''
  AND (
    v.title IS NULL OR trim(v.title) = '' OR v.title = 'Untitled'
  )
  AND (v.has_transcript IS NULL OR v.has_transcript = 0)
  AND NOT EXISTS (
    SELECT 1 FROM transcripts t
    WHERE t.video_id = v.video_id
      AND t.text IS NOT NULL AND trim(t.text) != ''
  )
ORDER BY (v.watch_date IS NULL), v.watch_date DESC
"""


def count_stubs(db_path: Path) -> int:
    with sqlite3.connect(str(db_path)) as conn:
        c = conn.cursor()
        c.execute(f"SELECT COUNT(*) FROM ({STUB_SELECT}) AS _stubq")
        return int(c.fetchone()[0])


def load_stub_batch(
    db_path: Path, limit: Optional[int], skip_ids: set[str]
) -> List[Tuple[str, str, Optional[str], Optional[str]]]:
    with sqlite3.connect(str(db_path)) as conn:
        c = conn.cursor()
        c.execute(STUB_SELECT)
        rows = c.fetchall()
    out: List[Tuple[str, str, Optional[str], Optional[str]]] = []
    for video_id, url, watch_date, title in rows:
        if video_id in skip_ids:
            continue
        out.append((video_id, url or "", watch_date, title))
        if limit is not None and len(out) >= limit:
            break
    return out


def parse_remote(remote: str) -> Tuple[str, int]:
    """
    Parse --remote like root@192.168.1.1:40166 -> (root@192.168.1.1, 40166).
    If no trailing :port, SSH/SCP default port 22.
    """
    s = (remote or "").strip()
    if not s:
        raise ValueError("empty --remote")
    if s.count(":") == 1:
        host_part, maybe_port = s.rsplit(":", 1)
        if maybe_port.isdigit():
            return host_part, int(maybe_port)
    return s, 22


def _run_scp(cmd: List[str]) -> None:
    """Run scp with inherited stdio so -v progress lines appear on the console."""
    log_run(f"SCP: {' '.join(cmd)}")
    r = subprocess.run(cmd, shell=False)
    if r.returncode != 0:
        raise RuntimeError(f"scp failed with exit code {r.returncode}")


def scp_upload_db(local_db: Path, remote_user_host: str, port: int, dest: str) -> None:
    cmd = [
        "scp",
        "-v",
        "-P",
        str(port),
        str(local_db.resolve()),
        f"{remote_user_host}:{dest}",
    ]
    _run_scp(cmd)


def scp_download_db(remote_user_host: str, port: int, remote_path: str, dest_local: Path) -> None:
    cmd = [
        "scp",
        "-v",
        "-P",
        str(port),
        f"{remote_user_host}:{remote_path}",
        str(dest_local.resolve()),
    ]
    _run_scp(cmd)


def _nonempty(s: Any) -> bool:
    return s is not None and str(s).strip() != ""


def merge_remote_into_local(local_db: Path, remote_copy: Path) -> Dict[str, int]:
    """
    Merge rows from remote_copy (fresh DB from Vast) into local_db.
    - Transcripts: insert remote rows whose (video_id, text) pair is not already present.
    - Videos: for each video_id present in both, fill NULL/empty local fields from remote.
    """
    stats = {"transcripts_inserted": 0, "videos_updated": 0}
    rp = str(remote_copy.resolve()).replace("\\", "/").replace("'", "''")
    conn = sqlite3.connect(str(local_db))
    try:
        conn.execute(f"ATTACH DATABASE '{rp}' AS rdb")
        cur = conn.cursor()
        cur.execute("PRAGMA main.table_info(transcripts)")
        t_cols = [row[1] for row in cur.fetchall() if row[1] != "id"]
        cur.execute("PRAGMA rdb.table_info(transcripts)")
        r_cols = [row[1] for row in cur.fetchall() if row[1] != "id"]
        common = [c for c in t_cols if c in r_cols]
        if common and "video_id" in common and "text" in common:
            cur.execute("SELECT COUNT(*) FROM main.transcripts")
            t_before = int(cur.fetchone()[0])
            col_list = ", ".join(common)
            where_dup = (
                "NOT EXISTS (SELECT 1 FROM main.transcripts t "
                "WHERE t.video_id = r.video_id "
                "AND ifnull(trim(t.text), '') = ifnull(trim(r.text), ''))"
            )
            cur.execute(
                f"""
                INSERT INTO main.transcripts ({col_list})
                SELECT {col_list}
                FROM rdb.transcripts r
                WHERE r.text IS NOT NULL AND trim(r.text) != ''
                  AND {where_dup}
                """
            )
            cur.execute("SELECT COUNT(*) FROM main.transcripts")
            t_after = int(cur.fetchone()[0])
            stats["transcripts_inserted"] = max(0, t_after - t_before)

        cur.execute("PRAGMA main.table_info(videos)")
        v_cols = [row[1] for row in cur.fetchall()]
        cur.execute("SELECT video_id FROM rdb.videos")
        remote_ids = [row[0] for row in cur.fetchall()]

        for vid in remote_ids:
            cur.execute("SELECT * FROM main.videos WHERE video_id = ?", (vid,))
            loc = cur.fetchone()
            cur.execute("SELECT * FROM rdb.videos WHERE video_id = ?", (vid,))
            rem = cur.fetchone()
            if not loc or not rem:
                continue
            cur.execute("PRAGMA main.table_info(videos)")
            loc_names = [r[1] for r in cur.fetchall()]
            loc_d = dict(zip(loc_names, loc))
            cur.execute("PRAGMA rdb.table_info(videos)")
            rem_names = [r[1] for r in cur.fetchall()]
            rem_d = dict(zip(rem_names, rem))

            updates: Dict[str, Any] = {}
            for col in v_cols:
                if col == "video_id" or col not in rem_d:
                    continue
                lv, rv = loc_d.get(col), rem_d.get(col)
                if col in ("title", "channel", "url", "json_metadata", "local_path"):
                    if not _nonempty(lv) and _nonempty(rv):
                        updates[col] = rv
                elif col == "has_transcript":
                    if (lv in (None, 0, False, "")) and rv:
                        updates[col] = 1
                elif col in ("transcript_fetched_at", "watch_date", "created_at"):
                    if not _nonempty(lv) and _nonempty(rv):
                        updates[col] = rv
            if updates:
                sets = ", ".join(f"{k} = ?" for k in updates)
                vals = list(updates.values()) + [vid]
                cur.execute(f"UPDATE main.videos SET {sets} WHERE video_id = ?", vals)
                stats["videos_updated"] += 1

        conn.commit()
    finally:
        try:
            conn.execute("DETACH DATABASE rdb")
        except sqlite3.OperationalError:
            pass
        conn.close()
    return stats


def apply_metadata_to_row(
    db_path: Path, video_id: str, meta: Dict[str, Any]
) -> None:
    import json as _json

    title = (meta.get("title") or "").strip()
    channel = (meta.get("channel") or "").strip()
    blob = _json.dumps(meta, ensure_ascii=False)
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            UPDATE videos
            SET title = ?, channel = ?, json_metadata = ?
            WHERE video_id = ?
            """,
            (title, channel, blob, video_id),
        )
        conn.commit()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stub watch-history videos -> metadata + transcript + queue (gentle pacing)"
    )
    ap.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="transcripts.db path (default: ASK_SHORTY_DB_PATH env or viewer copy)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max videos this run (default {DEFAULT_LIMIT}; 0 = unlimited)",
    )
    ap.add_argument(
        "--delay-minutes",
        type=float,
        default=DEFAULT_DELAY_MINUTES,
        help=f"Base pause between videos (default {DEFAULT_DELAY_MINUTES})",
    )
    ap.add_argument(
        "--session-minutes",
        type=float,
        default=DEFAULT_SESSION_MINUTES,
        help=f"Stop after this many minutes (default {DEFAULT_SESSION_MINUTES})",
    )
    ap.add_argument(
        "--jitter-seconds",
        type=float,
        default=DEFAULT_JITTER_SECONDS,
        help=f"Random 0..N s after each pause (default {DEFAULT_JITTER_SECONDS}; 0 = off)",
    )
    ap.add_argument(
        "--delay-seconds",
        type=float,
        default=None,
        help="If set, fixed delay in seconds instead of --delay-minutes",
    )
    ap.add_argument("--dry-run", action="store_true", help="Count stubs only; no network/DB writes")
    ap.add_argument(
        "--no-vectorize",
        action="store_true",
        help="Skip Chroma indexing (same idea as batch_liked --no-vectorize)",
    )
    ap.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip yt-dlp metadata (transcript + queue only)",
    )
    ap.add_argument(
        "--aggressive",
        action="store_true",
        help=f"Vast / disposable-IP mode: {AGGRESSIVE_DELAY_SECONDS:.0f}s delay, no jitter "
        "(overrides --delay-minutes / --jitter-seconds unless --delay-seconds is set explicitly).",
    )
    ap.add_argument(
        "--remote",
        type=str,
        default=None,
        help="SCP target, e.g. root@203.0.113.7:40166 (optional trailing :port for ssh -P / scp -P).",
    )
    ap.add_argument(
        "--remote-path",
        type=str,
        default=REMOTE_DB_PATH,
        help=f"Path on remote for transcripts.db (default {REMOTE_DB_PATH}).",
    )
    ap.add_argument(
        "--sync-up",
        action="store_true",
        help="Before processing: scp local DB to remote (needs --remote). Uses scp -v for progress.",
    )
    ap.add_argument(
        "--sync-down",
        action="store_true",
        help="After processing: scp remote DB to a temp file, merge transcripts + fill empty video fields locally.",
    )
    args = ap.parse_args()

    _delay_explicit = any(
        a == "--delay-seconds" or a.startswith("--delay-seconds=") for a in sys.argv
    )
    if args.aggressive and not _delay_explicit:
        args.delay_seconds = AGGRESSIVE_DELAY_SECONDS
    if args.aggressive:
        args.jitter_seconds = 0.0

    if (args.sync_up or args.sync_down) and not (args.remote or "").strip():
        print(
            "ERROR: --sync-up and --sync-down require --remote (e.g. root@host:port).",
            file=sys.stderr,
        )
        sys.exit(1)

    db_path = (args.db_path or default_db_path()).expanduser().resolve()

    print(
        "fetch_watch_history_videos: starting (if this is the first line, Python is loading)...",
        flush=True,
    )

    if not db_path.is_file():
        log_run(f"ERROR: database not found: {db_path}")
        sys.exit(1)

    ensure_json_metadata_column(db_path)
    skip_ids = load_skip_ids()
    if skip_ids:
        print(
            f"Permanent no-transcript skip list: {len(skip_ids)} ids ({SKIP_FILE.name}).",
            flush=True,
        )

    total_stubs = count_stubs(db_path)
    limit = args.limit if args.limit and args.limit > 0 else None
    to_do = load_stub_batch(db_path, limit, skip_ids)

    log_run("=" * 60)
    log_run(
        f"Stub videos (matching query, excluding skip list): total eligible ~{total_stubs} | "
        f"this run will try: {len(to_do)}"
    )

    if args.dry_run:
        log_run(f"[DRY RUN] Stub rows matching filters: {total_stubs}")
        for row in to_do[:15]:
            vid, url, wd, _t = row
            log_run(f"  would process: {vid}  watch_date={wd!r}  url={url[:60]}...")
        if len(to_do) > 15:
            log_run(f"  ... and {len(to_do) - 15} more in this batch (limit={args.limit})")
        log_run("")
        log_run("When run for real, defaults resemble batch_liked_to_main_db.py:")
        log_run(
            f"  pause ~{args.delay_minutes} min between videos"
            + (
                ""
                if args.delay_seconds is None
                else f"  (overridden: {args.delay_seconds}s)"
            )
        )
        log_run(f"  session cap ~{args.session_minutes} min - run again for next batch")
        if args.jitter_seconds > 0:
            log_run(f"  + random 0-{args.jitter_seconds:.0f}s jitter")
        log_run(f"  failures -> {FAIL_LOG.name}")
        if args.aggressive:
            log_run(
                f"[DRY RUN] --aggressive: would use ~{AGGRESSIVE_DELAY_SECONDS:.0f}s between videos, no jitter."
            )
        if args.sync_up or args.sync_down:
            log_run("[DRY RUN] Skipping --sync-up / --sync-down (no SCP).")
        return

    remote_host, remote_port = ("", 22)
    if args.remote:
        remote_host, remote_port = parse_remote(args.remote.strip())

    if args.sync_up:
        try:
            log_run(f"Sync UP: local -> {remote_host}:{args.remote_path} (scp -v)")
            scp_upload_db(db_path, remote_host, remote_port, args.remote_path)
        except Exception as e:
            log_run(f"ERROR: --sync-up failed: {e}")
            sys.exit(1)

    if not to_do:
        log_run(
            "Nothing to do - no stub videos left (or all skipped / already have transcripts)."
        )
        if args.sync_down:
            try:
                tmp_dir = Path(tempfile.mkdtemp(prefix="wh_sync_down_"))
                tmp_db = tmp_dir / "transcripts_remote.sqlite"
                log_run(
                    f"Sync DOWN: {remote_host}:{args.remote_path} -> merge into {db_path.name}"
                )
                scp_download_db(remote_host, remote_port, args.remote_path, tmp_db)
                stats = merge_remote_into_local(db_path, tmp_db)
                log_run(
                    f"Merge done: +{stats['transcripts_inserted']} transcript row(s), "
                    f"{stats['videos_updated']} video row(s) updated from remote."
                )
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as e:
                log_run(f"ERROR: --sync-down failed: {e}")
                sys.exit(1)
        return

    from simple_transcript_fetcher import SimpleTranscriptFetcher
    from transcript_database import TranscriptDatabase
    from video_downloader import VideoDownloader

    db_str = str(db_path)
    fetcher = SimpleTranscriptFetcher(db_str)
    db = TranscriptDatabase(db_str)
    downloader = VideoDownloader(str(ROOT / "downloads"))

    rag = None
    if not args.no_vectorize:
        print(
            "Loading vector search (sentence-transformers + Chroma). "
            "First run can take 1-4 minutes...",
            flush=True,
        )
        try:
            from transcript_rag import TranscriptRAG

            chroma_dir = default_chroma_dir(db_path)
            rag = TranscriptRAG(transcript_db=db_str, chroma_dir=str(chroma_dir))
            print("Vector search ready.", flush=True)
        except Exception as e:
            log_run(f"[WARN] Could not load RAG for vectorize: {e} - continuing without vectorize")
            rag = None

    session_start = time.time()
    session_deadline = session_start + args.session_minutes * 60.0

    log_run(
        f"Pacing: {'delay-seconds=' + str(args.delay_seconds) if args.delay_seconds is not None else str(args.delay_minutes) + ' min between videos'}"
        f" | session cap {args.session_minutes} min | jitter 0..{args.jitter_seconds:.0f}s"
    )
    min_pause_floor = 0.5 if args.aggressive else 15.0

    ok = 0
    fail = 0
    processed = 0

    for i, (vid, url, watch_date, _old_title) in enumerate(to_do, 1):
        if time.time() >= session_deadline:
            log_run(
                f"Session cap ({args.session_minutes:.0f} min) reached - stopping. "
                "Run again later; completed videos drop out of the stub query."
            )
            break

        log_run(f"[{i}/{len(to_do)}] {vid}  watch_date={watch_date!r} ...")

        display_title = vid
        try:
            if not args.no_metadata:
                try:
                    meta = downloader.fetch_metadata(url, quiet=True)
                    if meta:
                        apply_metadata_to_row(db_path, vid, meta)
                        display_title = (meta.get("title") or vid)[:80]
                        log_run(
                            f"  OK metadata: title={display_title[:50]!r} "
                            f"desc_chars={len(meta.get('description', '') or '')}"
                        )
                    else:
                        log_run("  WARN metadata: yt-dlp returned None")
                except Exception as e:
                    log_run(f"  WARN metadata: {e}")

            result = fetcher.fetch_transcript(vid, url)
            transcript_ok = result.get("success", False)

            if not transcript_ok:
                err = result.get("error", "unknown")
                log_run(f"  FAIL transcript: {err}")
                log_failure(vid, display_title, err)
                fail += 1
                err_lower = err.lower()
                if any(
                    phrase in err_lower
                    for phrase in (
                        "no transcript available",
                        "subtitles are disabled",
                        "no captions",
                        "transcripts are disabled",
                        "could not retrieve a transcript",
                    )
                ):
                    add_skip_id(vid, display_title, err[:200])
            else:
                ok += 1
                if result.get("cached"):
                    log_run("  OK transcript (cached)")
                else:
                    log_run("  OK transcript saved")

                try:
                    db.enqueue_processing_tasks(vid)
                    log_run("  OK enqueued processing_queue (full LLM chain)")
                except Exception as e:
                    log_run(f"  WARN enqueue: {e}")
                    log_failure(vid, display_title, f"enqueue: {e}")

                if rag and result.get("transcript"):
                    try:
                        rag.index_single_transcript(vid, result["transcript"])
                        log_run("  OK vectorized")
                    except Exception as e:
                        log_run(f"  WARN vectorize: {e}")

        except Exception as e:
            log_run(f"  ERROR: {e}")
            log_failure(vid, display_title, str(e))
            fail += 1

        processed += 1

        if i >= len(to_do):
            break
        if time.time() >= session_deadline:
            log_run(
                f"Session cap ({args.session_minutes:.0f} min) reached after this video - stopping."
            )
            break

        if args.delay_seconds is not None:
            base_sleep = args.delay_seconds
        else:
            base_sleep = args.delay_minutes * 60.0
        jitter = (
            random.uniform(0, args.jitter_seconds) if args.jitter_seconds > 0 else 0.0
        )
        nap = base_sleep + jitter
        remaining = session_deadline - time.time()
        nap = min(nap, max(0.0, remaining - 2.0))
        if nap < min_pause_floor:
            log_run("Not enough session time left for a full pause - stopping.")
            break
        if args.delay_seconds is not None and args.delay_seconds < 120:
            log_run(f"  Pausing {nap:.1f} s before next video...")
        else:
            log_run(f"  Pausing {nap / 60.0:.1f} min before next video...")
        time.sleep(nap)

    log_run("=" * 60)
    log_run(
        f"Session ended. This run: processed {processed} | OK: {ok} | Failed: {fail} | "
        f"Failures log: {FAIL_LOG.name}"
    )

    if args.sync_down:
        try:
            tmp_dir = Path(tempfile.mkdtemp(prefix="wh_sync_down_"))
            tmp_db = tmp_dir / "transcripts_remote.sqlite"
            log_run(
                f"Sync DOWN: {remote_host}:{args.remote_path} -> merge into {db_path.name}"
            )
            scp_download_db(remote_host, remote_port, args.remote_path, tmp_db)
            stats = merge_remote_into_local(db_path, tmp_db)
            log_run(
                f"Merge done: +{stats['transcripts_inserted']} transcript row(s), "
                f"{stats['videos_updated']} video row(s) updated from remote."
            )
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            log_run(f"ERROR: --sync-down failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
