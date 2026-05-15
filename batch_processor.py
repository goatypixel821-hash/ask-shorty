#!/usr/bin/env python3
"""
Batch processor for Ask Shorty.

Scans all videos that:
- have at least one transcript row
- do NOT yet have a Shorty

Then, in batches of 10:
- generates Shorty
- generates synthetic questions
- extracts entities
- indexes everything into Chroma

Features:
- Resume-safe (skips videos that already have Shorties)
- --limit N to cap how many videos to process
- --retry-failed to reprocess only failed video_ids listed in failed_videos.txt
- --db-path to point at an external transcripts.db
- 1 second pause between batches to rate-limit Anthropic calls
- Cost estimation using Haiku pricing before full run and per-batch
- ``--worker-id`` / ``--worker-count`` to split the same queue filter across terminals
  (same FIFO order; worker k claims rows where index i satisfies ``i % N == k``).
  Example: three terminals on segments only::

    python batch_processor.py --only-tasks segments --worker-id 0 --worker-count 3
    python batch_processor.py --only-tasks segments --worker-id 1 --worker-count 3
    python batch_processor.py --only-tasks segments --worker-id 2 --worker-count 3

Usage:
  python batch_processor.py
  python batch_processor.py --limit 100
  python batch_processor.py --retry-failed
"""

import argparse
import builtins
import os
import re
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable, Set

import sqlite3

from transcript_database import TranscriptDatabase
from shorty_generator import (
    generate_shorty,
    generate_synthetic_questions,
    SHORTY_SYSTEM_PROMPT,
    SHORTY_USER_PROMPT_TEMPLATE,
    SYNTHETIC_Q_SYSTEM_PROMPT,
    SYNTHETIC_Q_USER_PROMPT_TEMPLATE,
)
from entity_extractor import (
    extract_entities,
    store_entities,
    parse_entities_from_json,
    ENTITY_JSON_SYSTEM_PROMPT,
    ENTITY_JSON_USER_TEMPLATE,
)
from transcript_rag import TranscriptRAG


def _log(*args: Any, sep: str = " ", end: str = "\n", flush: bool = True) -> None:
    """Print to stdout with a local timestamp prefix."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    builtins.print(f"[{ts}]", *args, sep=sep, end=end, flush=flush)


def _format_chapter_timestamp(seconds: Any) -> str:
    """Format seconds as M:SS or H:MM:SS for chapter labels."""
    try:
        s = int(float(seconds or 0))
    except (TypeError, ValueError):
        s = 0
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def _format_json_metadata_prompt_prefix(meta: Any) -> str:
    """
    Build DESCRIPTION / TAGS / CHAPTERS block from videos.json_metadata for Shorty prompts.
    Only non-empty fields are included.
    """
    if not isinstance(meta, dict):
        return ""

    lines: List[str] = []

    desc = ""
    for key in ("description", "shortDescription", "videoDescription", "desc"):
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            desc = v.strip()
            break
    if desc:
        if len(desc) > 500:
            desc = desc[:500]
        lines.append(f"DESCRIPTION: {desc}")

    tags = meta.get("tags")
    if isinstance(tags, list):
        tag_strs = [str(t).strip() for t in tags if t is not None and str(t).strip()]
        if tag_strs:
            lines.append("TAGS: " + ", ".join(tag_strs))

    chapters = meta.get("chapters")
    if chapters and isinstance(chapters, list):
        ch_parts: List[str] = []
        for ch in chapters:
            if isinstance(ch, dict):
                title = (ch.get("title") or "").strip()
                if title:
                    ch_parts.append(
                        f"{_format_chapter_timestamp(ch.get('start_time', 0))} {title}"
                    )
            elif isinstance(ch, str) and ch.strip():
                ch_parts.append(ch.strip())
        if ch_parts:
            lines.append("CHAPTERS: " + ", ".join(ch_parts))

    return "\n".join(lines)


def _transcript_for_shorty(transcript_text: str, meta: Any) -> str:
    """Prepend json_metadata fields before transcript text for Shorty generation."""
    prefix = _format_json_metadata_prompt_prefix(meta)
    text = (transcript_text or "").strip()
    if prefix:
        return f"{prefix}\n\n{text}"
    return text


BATCH_SIZE = 10

# SQLite: wait on busy locks + WAL for concurrent workers / Flask app.
_SQLITE_BUSY_TIMEOUT_SEC = 30.0


def _sqlite_connect(database_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(database_path, timeout=_SQLITE_BUSY_TIMEOUT_SEC)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# After each failed API/network attempt, wait before retrying (same task, no queue status change).
NETWORK_RETRY_BACKOFF_SEC: Tuple[int, ...] = (30, 60, 120, 240, 480)
CONNECTIVITY_POLL_SEC = 60.0
CONNECTIVITY_SOCKET_TIMEOUT_SEC = 5.0


def _openai_transient_types() -> Tuple[type, ...]:
    try:
        from openai import APIConnectionError, APITimeoutError

        return (APIConnectionError, APITimeoutError)
    except ImportError:
        return ()


def _httpx_transient_types() -> Tuple[type, ...]:
    try:
        import httpx

        return (
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ConnectTimeout,
            httpx.RemoteProtocolError,
            httpx.NetworkError,
        )
    except ImportError:
        return ()


def _requests_transient_types() -> Tuple[type, ...]:
    try:
        import requests

        return (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        )
    except ImportError:
        return ()


def _urllib3_transient_types() -> Tuple[type, ...]:
    try:
        import urllib3.exceptions as u3e

        return (
            u3e.MaxRetryError,
            u3e.NewConnectionError,
            u3e.ConnectTimeoutError,
            u3e.ReadTimeoutError,
            u3e.ProtocolError,
        )
    except ImportError:
        return ()


def _anthropic_transient_types() -> Tuple[type, ...]:
    try:
        from anthropic import APIConnectionError

        return (APIConnectionError,)
    except ImportError:
        return ()


def _exception_matches_transient_type(e: BaseException, t: object) -> bool:
    if isinstance(t, type) and isinstance(e, t):
        return True
    if callable(t) and not isinstance(t, type):
        try:
            return bool(t(e))
        except Exception:
            return False
    return False


def _is_transient_network_error(exc: BaseException) -> bool:
    """True for connection loss, timeouts, urllib3/requests/httpx/OpenAI transport errors, and HTTP 5xx."""
    pieces: List[object] = [
        ConnectionError,
        TimeoutError,
        BrokenPipeError,
    ]
    pieces.extend(_anthropic_transient_types())
    pieces.extend(_openai_transient_types())
    pieces.extend(_httpx_transient_types())
    pieces.extend(_requests_transient_types())
    pieces.extend(_urllib3_transient_types())
    transient_types: Tuple[object, ...] = tuple(pieces)

    seen: Set[int] = set()

    def walk(e: BaseException) -> bool:
        if id(e) in seen:
            return False
        seen.add(id(e))
        try:
            import urllib3.exceptions as u3e

            if isinstance(e, u3e.HTTPError):
                st = getattr(e, "status", None)
                if st is not None and 500 <= int(st) < 600:
                    return True
        except ImportError:
            pass
        try:
            from openai import APIStatusError

            if isinstance(e, APIStatusError):
                code = getattr(e, "status_code", None)
                if code is not None and 500 <= int(code) < 600:
                    return True
        except ImportError:
            pass
        try:
            import httpx

            if isinstance(e, httpx.HTTPStatusError):
                resp = getattr(e, "response", None)
                if resp is not None:
                    code = getattr(resp, "status_code", None)
                    if code is not None and 500 <= int(code) < 600:
                        return True
        except ImportError:
            pass
        try:
            import requests

            if isinstance(e, requests.exceptions.HTTPError):
                resp = getattr(e, "response", None)
                if resp is not None:
                    code = getattr(resp, "status_code", None)
                    if code is not None and 500 <= int(code) < 600:
                        return True
        except ImportError:
            pass
        for t in transient_types:
            if _exception_matches_transient_type(e, t):
                return True
        if getattr(e, "__cause__", None) is not None and walk(e.__cause__):  # type: ignore[arg-type]
            return True
        if (
            getattr(e, "__context__", None) is not None
            and getattr(e, "__cause__", None) is None
            and walk(e.__context__)  # type: ignore[arg-type]
        ):
            return True
        return False

    return walk(exc)


def _quick_connectivity_check(timeout: float = CONNECTIVITY_SOCKET_TIMEOUT_SEC) -> bool:
    """Lightweight check: TCP connect to public resolvers (no API key)."""
    for host, port in (("1.1.1.1", 443), ("8.8.8.8", 443)):
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except OSError:
            continue
    try:
        urllib.request.urlopen(
            "https://www.cloudflare.com/cdn-cgi/trace",
            timeout=timeout,
        )
        return True
    except (OSError, urllib.error.URLError):
        return False


def _wait_for_internet(poll_sec: float = CONNECTIVITY_POLL_SEC) -> None:
    """Block until `_quick_connectivity_check` succeeds; log and poll every `poll_sec` seconds."""
    first = True
    while not _quick_connectivity_check():
        if first:
            _log(
                "[network] connectivity check failed (no route to internet). "
                "Pausing %.0fs and retrying until connection is restored…" % poll_sec
            )
            first = False
        else:
            _log("[network] still offline; next check in %.0fs…" % poll_sec)
        time.sleep(poll_sec)
    if not first:
        _log("[network] connectivity restored; resuming.")


def _call_with_connection_retries(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Call ``fn``; on transient network/5xx errors retry with backoff (30s … 480s), same task.
    After 5 failed retries (6 attempts total), re-raise so the task can be marked failed.
    """
    max_attempts = 1 + len(NETWORK_RETRY_BACKOFF_SEC)
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except Exception as e:
            if not _is_transient_network_error(e) or attempt >= max_attempts - 1:
                raise
            delay = NETWORK_RETRY_BACKOFF_SEC[attempt]
            _log(
                "  [network] temporary issue (%s: %s); will retry same task (%d/%d) after %ds…"
                % (type(e).__name__, e, attempt + 1, len(NETWORK_RETRY_BACKOFF_SEC), delay)
            )
            time.sleep(float(delay))
    raise RuntimeError("_call_with_connection_retries: unreachable")

# Haiku pricing (USD per 1M tokens)
INPUT_PRICE_PER_M = 1.00
OUTPUT_PRICE_PER_M = 5.00


def estimate_video_tokens(db: TranscriptDatabase, video_id: str) -> Tuple[int, int]:
    """
    Roughly estimate input and output tokens for a single video.
    - input_tokens ≈ len(transcript) / 4
    - Shorty output ≈ 15% of input
    - synthetic questions output ≈ +500
    - entity extraction output ≈ +300
    """
    transcript = db.get_transcript(video_id)
    if not transcript:
        return 0, 0

    input_tokens = int(len(transcript) / 4)
    shorty_out = int(input_tokens * 0.15)
    synthetic_out = 500
    entity_out = 300
    output_tokens = shorty_out + synthetic_out + entity_out
    return input_tokens, output_tokens


def estimate_batch_cost(
    db: TranscriptDatabase,
    videos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Estimate total tokens and cost for a list of videos.
    Uses Haiku pricing: $1/M input, $5/M output.
    """
    total_in = 0
    total_out = 0
    for v in videos:
        vid = v["video_id"]
        inp, out = estimate_video_tokens(db, vid)
        total_in += inp
        total_out += out

    cost_input = (total_in / 1_000_000.0) * INPUT_PRICE_PER_M
    cost_output = (total_out / 1_000_000.0) * OUTPUT_PRICE_PER_M
    total_cost = cost_input + cost_output

    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_input": cost_input,
        "cost_output": cost_output,
        "total_cost": total_cost,
    }


def format_token_count(n: int) -> str:
    """Format token count with ~ and commas."""
    return f"~{n:,}"


def format_cost(c: float) -> str:
    return f"~${c:0.2f}"


def _parse_task_filter(s: Optional[str]) -> Optional[List[str]]:
    """Comma-separated task names, e.g. 'triples' or 'shorty,synthetic_questions'."""
    if not s or not str(s).strip():
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _normalize_worker_shard(worker_id: Optional[int], worker_count: Optional[int]) -> Tuple[int, int]:
    """
    Return (worker_id, worker_count) for queue sharding.

    Pending rows are ordered FIFO (created_at ASC, id ASC) and assigned index i = 0,1,2,...
    Worker k (with N workers) claims rows where (i % N) == k.

    Defaults: single worker (0, 1) = claim all rows in order.
    """
    if worker_id is None and worker_count is None:
        return (0, 1)
    if worker_id is None or worker_count is None:
        raise ValueError("Use both --worker-id and --worker-count together, or neither.")
    wc = int(worker_count)
    wid = int(worker_id)
    if wc < 1:
        raise ValueError("--worker-count must be >= 1")
    if not (0 <= wid < wc):
        raise ValueError("--worker-id must be >= 0 and < --worker-count")
    return (wid, wc)


def get_videos_needing_shorties(db: TranscriptDatabase, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Return list of video records that have transcripts but no Shorty yet."""

    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()

    sql = """
        SELECT v.video_id, v.title, v.channel
        FROM videos v
        JOIN transcripts t ON t.video_id = v.video_id
        WHERE t.shorty IS NULL
        GROUP BY v.video_id
        ORDER BY v.created_at ASC
    """
    if limit is not None:
        sql += " LIMIT ?"
        cursor.execute(sql, (limit,))
    else:
        cursor.execute(sql)

    rows = cursor.fetchall()
    conn.close()

    videos: List[Dict[str, Any]] = []
    for vid, title, channel in rows:
        videos.append(
            {
                "video_id": vid,
                "title": title,
                "channel": channel,
            }
        )
    return videos


def get_videos_from_failed(db: TranscriptDatabase, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Read failed_videos.txt and return video records for those IDs."""
    path = "failed_videos.txt"
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        _log("No failed_videos.txt found; nothing to retry.")
        return []

    failed_ids: List[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: video_id\tTitle\tError
        parts = line.split("\t")
        if parts:
            failed_ids.append(parts[0])

    if not failed_ids:
        _log("failed_videos.txt is empty or has no valid entries.")
        return []

    if limit is not None:
        failed_ids = failed_ids[:limit]

    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in failed_ids)
    cursor.execute(
        f"""
        SELECT video_id, title, channel
        FROM videos
        WHERE video_id IN ({placeholders})
        """,
        failed_ids,
    )
    rows = cursor.fetchall()
    conn.close()

    videos: List[Dict[str, Any]] = []
    for vid, title, channel in rows:
        videos.append(
            {
                "video_id": vid,
                "title": title,
                "channel": channel,
            }
        )
    return videos


def write_failed_videos(failures: List[Dict[str, Any]]) -> None:
    """Write failed video IDs and errors to failed_videos.txt."""
    path = "failed_videos.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# video_id\ttitle\terror\n")
        for item in failures:
            vid = item.get("video_id", "")
            title = item.get("title", "").replace("\t", " ")
            err = item.get("error", "").replace("\n", " ").replace("\t", " ")
            f.write(f"{vid}\t{title}\t{err}\n")
    _log(f"Failed videos saved to: {path}")


def get_pending_queue_tasks(
    db: TranscriptDatabase,
    limit: Optional[int] = None,
    video_id: Optional[str] = None,
    worker_id: Optional[int] = None,
    worker_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch pending processing_queue tasks in FIFO order.
    If video_id is set, only tasks for that video are returned.
    With ``worker_count`` > 1, only rows whose global FIFO index i satisfies (i % worker_count) == worker_id.

    Returns list of dicts: {id, video_id, task}.
    """
    wid, wcnt = _normalize_worker_shard(worker_id, worker_count)

    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()

    if wcnt <= 1:
        sql = """
            SELECT id, video_id, task
            FROM processing_queue
            WHERE status = ?
        """
        params: List[Any] = ["pending"]
        if video_id is not None:
            sql += " AND video_id = ?"
            params.append(video_id)
        sql += " ORDER BY created_at ASC, id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
    else:
        where_extra = ""
        params_base: List[Any] = ["pending"]
        if video_id is not None:
            where_extra = " AND video_id = ?"
            params_base.append(video_id)
        sql = """
            WITH ranked AS (
                SELECT id, video_id, task,
                       ROW_NUMBER() OVER (ORDER BY created_at ASC, id ASC) AS rn
                FROM processing_queue
                WHERE status = ?""" + where_extra + """
            )
            SELECT id, video_id, task FROM ranked
            WHERE ((rn - 1) % ? = ?)
            ORDER BY rn ASC
        """
        params = params_base + [wcnt, wid]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    tasks: List[Dict[str, Any]] = []
    for row in rows:
        tasks.append({"id": row[0], "video_id": row[1], "task": row[2]})
    return tasks


def claim_next_queue_tasks(
    db: TranscriptDatabase,
    limit: int,
    video_id: Optional[str] = None,
    only_tasks: Optional[List[str]] = None,
    exclude_tasks: Optional[List[str]] = None,
    worker_id: Optional[int] = None,
    worker_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Atomically claim up to ``limit`` pending rows (sets status=started, bumps attempts).

    FIFO order is ``created_at ASC, id ASC``. With ``worker_count`` > 1, only rows whose
    0-based FIFO index i satisfies ``(i % worker_count) == worker_id`` are eligible,
    so multiple processes can share the same ``only_tasks`` / ``exclude_tasks`` filter
    without claiming the same rows.

    Requires SQLite 3.35+ for RETURNING (Python 3.13+ ok).
    """
    if only_tasks and exclude_tasks:
        raise ValueError("Use only one of only_tasks and exclude_tasks")

    wid, wcnt = _normalize_worker_shard(worker_id, worker_count)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clauses = ["status = 'pending'"]
    params_inner: List[Any] = []
    if video_id is not None:
        clauses.append("video_id = ?")
        params_inner.append(video_id)
    if only_tasks:
        clauses.append("task IN (%s)" % ",".join("?" * len(only_tasks)))
        params_inner.extend(only_tasks)
    elif exclude_tasks:
        clauses.append("task NOT IN (%s)" % ",".join("?" * len(exclude_tasks)))
        params_inner.extend(exclude_tasks)

    where_sql = " AND ".join(clauses)

    if wcnt <= 1:
        sql = """
            UPDATE processing_queue
            SET status = 'started', started_at = ?, attempts = COALESCE(attempts, 0) + 1
            WHERE id IN (
                SELECT id FROM processing_queue
                WHERE """ + where_sql + """
                ORDER BY created_at ASC, id ASC
                LIMIT ?
            )
            RETURNING id, video_id, task
        """
        params: List[Any] = [now] + params_inner + [limit]
    else:
        sql = """
            WITH ranked AS (
                SELECT id, ROW_NUMBER() OVER (ORDER BY created_at ASC, id ASC) AS rn
                FROM processing_queue
                WHERE """ + where_sql + """
            )
            UPDATE processing_queue
            SET status = 'started', started_at = ?, attempts = COALESCE(attempts, 0) + 1
            WHERE id IN (
                SELECT id FROM ranked
                WHERE ((rn - 1) % ? = ?)
                ORDER BY rn ASC
                LIMIT ?
            )
            RETURNING id, video_id, task
        """
        # Placeholders appear left-to-right: CTE filters, then SET started_at, then modulo, LIMIT.
        params = params_inner + [now] + [wcnt, wid, limit]

    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    try:
        cursor = conn.cursor()

        # Read-only debug (same connection, before claim transaction)
        cursor.execute(
            "SELECT COUNT(*) FROM processing_queue WHERE " + where_sql,
            params_inner,
        )
        total_pending_for_filter = int(cursor.fetchone()[0])

        dist_clauses = ["status = 'pending'"]
        dist_params: List[Any] = []
        if video_id is not None:
            dist_clauses.append("video_id = ?")
            dist_params.append(video_id)
        dist_where = " AND ".join(dist_clauses)
        cursor.execute(
            """
            SELECT task, COUNT(*) FROM processing_queue
            WHERE """ + dist_where + """
            GROUP BY task ORDER BY COUNT(*) DESC
            """,
            dist_params,
        )
        dist_rows = cursor.fetchall()
        dist_summary = ", ".join("%s=%d" % (repr(r[0]), int(r[1])) for r in dist_rows[:40])
        if len(dist_rows) > 40:
            dist_summary += " ..."

        _log(
            "[DEBUG] claim_next_queue_tasks: only_tasks=%s exclude_tasks=%s video_id=%s"
            % (only_tasks, exclude_tasks, video_id)
        )
        _log(
            "[DEBUG] claim_next_queue_tasks: (1) pending rows matching full filter (pre-modulo): %d"
            % total_pending_for_filter
        )
        _log(
            "[DEBUG] claim_next_queue_tasks: pending `task` column (same status/video scope, no only/exclude): %s"
            % (dist_summary or "(none)")
        )
        if only_tasks:
            present = {str(r[0]) if r[0] is not None else "" for r in dist_rows}
            for ot in only_tasks:
                ok = ot in present
                _log(
                    "[DEBUG] claim_next_queue_tasks: (3) only_tasks value %r matches pending `task` column: %s"
                    % (
                        ot,
                        "yes" if ok else "NO — no pending row has exactly this task string",
                    )
                )
        elif exclude_tasks:
            present = {str(r[0]) if r[0] is not None else "" for r in dist_rows}
            for et in exclude_tasks:
                _log(
                    "[DEBUG] claim_next_queue_tasks: (3) exclude_tasks value %r present in pending tasks: %s"
                    % (et, "yes" if et in present else "no")
                )

        if wcnt > 1:
            shard_count_sql = (
                """
                WITH ranked AS (
                    SELECT ROW_NUMBER() OVER (ORDER BY created_at ASC, id ASC) AS rn
                    FROM processing_queue
                    WHERE """ + where_sql + """
                )
                SELECT COUNT(*) FROM ranked WHERE ((rn - 1) % ? = ?)
                """
            )
            cursor.execute(shard_count_sql, params_inner + [wcnt, wid])
            shard_eligible = int(cursor.fetchone()[0])
            _log(
                "[DEBUG] claim_next_queue_tasks: (2) modulo shard worker_id=%d worker_count=%d "
                "=> ((rn - 1) %% %d) == %d; rows in this slice: %d"
                % (wid, wcnt, wcnt, wid, shard_eligible)
            )
        else:
            _log("[DEBUG] claim_next_queue_tasks: (2) single worker (worker_count=1); no modulo")

        conn.execute("BEGIN IMMEDIATE")
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    _log(
        "[DEBUG] claim_next_queue_tasks: claimed %d row(s) (only=%s exclude=%s worker=%s/%s)"
        % (len(rows), only_tasks, exclude_tasks, wid, wcnt)
    )
    return [{"id": r[0], "video_id": r[1], "task": r[2]} for r in rows]


def update_queue_task_status(
    db: TranscriptDatabase,
    task_id: int,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Update a single queue task's status and timestamps."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()

    if status == "started":
        cursor.execute(
            """
            UPDATE processing_queue
            SET status = ?, started_at = ?, attempts = COALESCE(attempts, 0) + 1
            WHERE id = ?
            """,
            (status, now, task_id),
        )
    elif status == "completed":
        cursor.execute(
            """
            UPDATE processing_queue
            SET status = 'completed', completed_at = ?, error = ?
            WHERE id = ?
            """,
            (now, error, task_id),
        )
    elif status == "failed":
        # After 5+ attempts, mark permanently_failed so auto-reset never retries it
        cursor.execute(
            """
            UPDATE processing_queue
            SET status = CASE WHEN COALESCE(attempts, 0) >= 6 THEN 'permanently_failed' ELSE 'failed' END,
                completed_at = ?, error = ?
            WHERE id = ?
            """,
            (now, error, task_id),
        )
    else:
        cursor.execute(
            "UPDATE processing_queue SET status = ? WHERE id = ?",
            (status, task_id),
        )

    conn.commit()
    conn.close()


def reset_failed_queue_tasks(db: TranscriptDatabase) -> int:
    """Reset only status='failed' to 'pending'. Never touch completed or permanently_failed."""
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    # Explicit: only rows with status exactly 'failed' are reset (never completed/started/permanently_failed)
    cursor.execute(
        "UPDATE processing_queue SET status = 'pending' WHERE status = ?",
        ("failed",),
    )
    n = cursor.rowcount
    conn.commit()
    conn.close()
    return n


# Chroma reindexing of Shorties disabled on Windows due to os._exit() crash - SQLite search used instead.

def process_batch(
    db: TranscriptDatabase,
    rag: TranscriptRAG,
    batch: List[Dict[str, Any]],
    start_index: int,
    total: int,
    totals: Dict[str, Any],
    shorty_fn: Callable[..., str],
    synth_q_fn: Callable[..., List[str]],
    entity_fn: Callable[..., List[Dict[str, Any]]],
) -> None:
    """Process a single batch of videos."""

    batch_input_est = 0
    batch_output_est = 0
    batch_success = 0
    batch_failures: List[Dict[str, Any]] = []

    for offset, video in enumerate(batch):
        idx = start_index + offset + 1
        video_id = video["video_id"]
        title = video.get("title") or "Untitled Video"
        channel = video.get("channel") or "Unknown Channel"

        _log(f"\n=== Processing video {idx} of {total} ===")
        _log(f"ID: {video_id}")
        _log(f"Title: {title}")
        _log(f"Channel: {channel}")

        # Double-check if Shorty already exists (resume safety)
        info = db.get_transcript_and_shorty(video_id)
        if info and info.get("shorty"):
            _log("  -> Shorty already present, skipping.")
            continue

        transcript_text = (info or {}).get("text") or db.get_transcript(video_id)
        if not transcript_text:
            _log("  -> No transcript found, skipping.")
            continue

        # Accumulate estimated cost for this video into batch & totals
        vin, vout = estimate_video_tokens(db, video_id)
        batch_input_est += vin
        batch_output_est += vout

        # Pull metadata for Shorty header
        video_info = db.get_video_info(video_id) or {}
        meta = (video_info.get("metadata") or {}) if isinstance(video_info, dict) else {}
        title_meta = video_info.get("title") if isinstance(video_info, dict) else None
        channel_meta = video_info.get("channel") if isinstance(video_info, dict) else None
        upload_date = meta.get("upload_date") if isinstance(meta, dict) else None

        final_title = title_meta or title
        final_channel = channel_meta or channel

        try:
            _log("  -> Generating Shorty...")
            shorty_text = shorty_fn(
                _transcript_for_shorty(transcript_text, meta),
                title=final_title,
                channel=final_channel,
                upload_date=upload_date,
            )
            saved = db.save_shorty(video_id, shorty_text)
            if not saved:
                msg = "Failed to save Shorty"
                _log(f"  ! {msg}")
                batch_failures.append({"video_id": video_id, "title": final_title, "error": msg})
                totals["videos_failed"].append({"video_id": video_id, "title": final_title, "error": msg})
                continue
            _log("  ✓ Shorty saved.")

            _log("  -> Generating synthetic questions...")
            questions = synth_q_fn(transcript_text, title=final_title)
            if questions:
                conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
                cursor = conn.cursor()
                for q in questions:
                    cursor.execute(
                        """
                        INSERT INTO synthetic_questions (video_id, question, embedding_id)
                        VALUES (?, ?, NULL)
                        """,
                        (video_id, q),
                    )
                conn.commit()
                conn.close()
                _log(f"  ✓ Stored {len(questions)} synthetic questions.")
            else:
                _log("  ! No synthetic questions generated.")
                questions = []

            _log("  -> Extracting entities...")
            entities = entity_fn(transcript_text, title=final_title)
            if entities:
                count = store_entities(video_id, entities, db=db)
                _log(f"  ✓ Stored {count} entities.")
            else:
                _log("  ! No entities extracted.")

            _log("  -> Indexing into Chroma...")
            rag.index_single_transcript(
                video_id,
                transcript_text,
                shorty=shorty_text,
                synthetic_questions=questions if questions else None,
            )
            _log("  ✓ Indexing complete.")

            batch_success += 1
            totals["videos_processed"] += 1

        except Exception as e:
            msg = str(e)
            _log(f"  !! Error processing video {video_id}: {msg}")
            batch_failures.append({"video_id": video_id, "title": final_title, "error": msg})
            totals["videos_failed"].append({"video_id": video_id, "title": final_title, "error": msg})

    # Update global token and cost totals using the estimated batch values
    totals["total_input_tokens"] += batch_input_est
    totals["total_output_tokens"] += batch_output_est
    batch_cost_in = (batch_input_est / 1_000_000.0) * INPUT_PRICE_PER_M
    batch_cost_out = (batch_output_est / 1_000_000.0) * OUTPUT_PRICE_PER_M
    batch_cost = batch_cost_in + batch_cost_out
    totals["total_cost"] += batch_cost

    # Batch summary
    _log("\nBatch complete.")
    _log(f"  ✓ {batch_success} Shorties generated")
    if batch_failures:
        _log(f"  ! {len(batch_failures)} failed:")
        for f in batch_failures:
            _log(f"    - {f['video_id']} - {f['title']} - {f['error']}")
    else:
        _log("  ! 0 failures in this batch")

    _log(
        f"Actual tokens used this batch (approx): "
        f"{format_token_count(batch_input_est)} input / {format_token_count(batch_output_est)} output"
    )
    _log(f"Running total cost: {format_cost(totals['total_cost'])}")


# When triple extraction stores 0 rows but Shorty is long, dump raw LLM text here for local inspection (no extra API).
TRIPLE_DEBUG_MIN_SHORTY_CHARS = 400


def _write_triple_zero_debug(
    video_id: str,
    title: str,
    shorty_len: int,
    raw: str,
) -> Path:
    root = Path(__file__).resolve().parent / "data" / "triple_debug"
    root.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", video_id).strip("_")[:120] or "unknown"
    path = root / f"{safe}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"video_id={video_id}\n")
        f.write(f"title={title}\n")
        f.write(f"shorty_chars={shorty_len}\n")
        f.write("--- raw model response ---\n")
        f.write(raw or "")
    return path


def process_queue_tasks(
    db: TranscriptDatabase,
    rag: TranscriptRAG,
    shorty_fn: Callable[..., str],
    synth_q_fn: Callable[..., List[str]],
    entity_fn: Callable[..., List[Dict[str, Any]]],
    triples_fn: Callable[[str, str], List[Dict[str, Any]]],
    limit: Optional[int],
    video_id: Optional[str] = None,
    only_tasks: Optional[List[str]] = None,
    exclude_tasks: Optional[List[str]] = None,
    debug_zero_triples: bool = False,
    triple_raw_sink: Optional[List[str]] = None,
    hsc_chat_fn: Optional[Callable[[str, str, int, float], str]] = None,
    worker_id: Optional[int] = None,
    worker_count: Optional[int] = None,
) -> None:
    """
    Process pending tasks from processing_queue in FIFO order.

    Rows are claimed atomically (safe for two processes if you use disjoint
    ``only_tasks`` / ``exclude_tasks``, e.g. one worker ``--exclude-tasks triples``
    and another ``--only-tasks triples``).

    With ``worker_count`` > 1, each process claims a disjoint slice of the same
    pending set (see ``_normalize_worker_shard``), e.g. three terminals on
    ``--only-tasks segments`` with ``--worker-id`` 0, 1, 2 and ``--worker-count`` 3.

    Each queue row represents exactly one task: shorty, synthetic_questions, entities, triples, etc.
    Transcript chunks are assumed to be already vectorized on grab.
    """
    wid, wcnt = _normalize_worker_shard(worker_id, worker_count)
    if wcnt > 1:
        _log(
            "Queue worker shard: id=%d count=%d (FIFO index i where (i %% %d) == %d)"
            % (wid, wcnt, wcnt, wid)
        )

    processed_count = 0
    batch_number = 0

    while True:
        batch_number += 1
        if limit is not None:
            remaining = max(limit - processed_count, 0)
            if remaining == 0:
                _log("\n[DEBUG] Loop stopping: reached --limit (processed_count=%d, limit=%d)." % (processed_count, limit))
                break
            batch_limit = min(BATCH_SIZE, remaining)
        else:
            batch_limit = BATCH_SIZE  # fetch in batches when no limit

        _wait_for_internet()

        tasks = claim_next_queue_tasks(
            db,
            batch_limit,
            video_id=video_id,
            only_tasks=only_tasks,
            exclude_tasks=exclude_tasks,
            worker_id=worker_id,
            worker_count=worker_count,
        )
        _log("\n[DEBUG] Batch #%d: requested batch_limit=%d, claimed %d tasks (processed_count so far=%d, limit=%s)." % (
            batch_number, batch_limit, len(tasks), processed_count, limit if limit is not None else "None"))

        if not tasks:
            n_failed = reset_failed_queue_tasks(db)
            if n_failed > 0:
                _log("\nResetting %d failed task(s) and continuing..." % n_failed)
                continue
            if processed_count == 0:
                _log("[DEBUG] Loop stopping: no pending tasks in queue.")
            else:
                _log("[DEBUG] Loop stopping: no more pending tasks (processed %d this run)." % processed_count)
            break

        _log(f"\n=== Processing {len(tasks)} queued tasks ===")
        for task in tasks:
            task_id = task["id"]
            current_video_id = task["video_id"]  # do not overwrite video_id param (used as filter for next fetch)
            kind = task["task"]

            _log(f"\nTask #{task_id} | video {current_video_id} | type={kind}")
            _log("[DEBUG] Task #%d status -> started (claimed)" % task_id)

            try:
                info = db.get_transcript_and_shorty(current_video_id)
                transcript_text = (info or {}).get("text") or db.get_transcript(current_video_id) or ""
                if kind != "triples" and not transcript_text.strip():
                    msg = "No transcript found; skipping task."
                    _log(f"  ! {msg}")
                    update_queue_task_status(db, task_id, "failed", msg)
                    _log("[DEBUG] Task #%d status -> failed" % task_id)
                    continue

                # Metadata for Shorty header or entity context
                video_info = db.get_video_info(current_video_id) or {}
                meta = (video_info.get("metadata") or {}) if isinstance(video_info, dict) else {}
                title_meta = video_info.get("title") if isinstance(video_info, dict) else None
                channel_meta = video_info.get("channel") if isinstance(video_info, dict) else None
                upload_date = meta.get("upload_date") if isinstance(meta, dict) else None

                final_title = title_meta or (video_info.get("title") if isinstance(video_info, dict) else "Untitled Video")
                final_channel = channel_meta or (video_info.get("channel") if isinstance(video_info, dict) else "Unknown Channel")

                if kind == "shorty":
                    _log("  -> Generating Shorty...")
                    shorty_text = _call_with_connection_retries(
                        shorty_fn,
                        _transcript_for_shorty(transcript_text, meta),
                        title=final_title,
                        channel=final_channel,
                        upload_date=upload_date,
                    )
                    saved = db.save_shorty(current_video_id, shorty_text)
                    if not saved:
                        msg = "Failed to save Shorty."
                        _log(f"  ! {msg}")
                        update_queue_task_status(db, task_id, "failed", msg)
                        _log("[DEBUG] Task #%d status -> failed" % task_id)
                        continue
                    _log("  ✓ Shorty saved.")
                    update_queue_task_status(db, task_id, "completed", None)
                    _log("[DEBUG] Task #%d status -> completed (processed_count now=%d)" % (task_id, processed_count + 1))
                    processed_count += 1

                elif kind == "synthetic_questions":
                    _log("  -> Generating synthetic questions...")
                    questions = _call_with_connection_retries(synth_q_fn, transcript_text, title=final_title)
                    if not questions:
                        # Local models often return non-JSON prose; failing the row blocks the whole pipeline.
                        msg = "No synthetic questions generated (skipped)."
                        _log(f"  ! {msg}")
                        update_queue_task_status(db, task_id, "completed", None)
                        _log("[DEBUG] Task #%d status -> completed (0 questions; processed_count now=%d)" % (task_id, processed_count + 1))
                        processed_count += 1
                        continue

                    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
                    cursor = conn.cursor()
                    for q in questions:
                        cursor.execute(
                            """
                            INSERT INTO synthetic_questions (video_id, question, embedding_id)
                            VALUES (?, ?, NULL)
                            """,
                            (current_video_id, q),
                        )
                    conn.commit()
                    conn.close()
                    _log(f"  ✓ Stored {len(questions)} synthetic questions.")
                    update_queue_task_status(db, task_id, "completed", None)
                    _log("[DEBUG] Task #%d status -> completed (processed_count now=%d)" % (task_id, processed_count + 1))
                    processed_count += 1

                elif kind == "entities":
                    _log("  -> Extracting entities...")
                    entities = _call_with_connection_retries(entity_fn, transcript_text, title=final_title)
                    if entities:
                        count = store_entities(current_video_id, entities, db=db)
                        _log(f"  ✓ Stored {count} entities.")
                    else:
                        _log("  ! No entities extracted.")

                    update_queue_task_status(db, task_id, "completed", None)
                    _log("[DEBUG] Task #%d status -> completed (processed_count now=%d)" % (task_id, processed_count + 1))
                    processed_count += 1

                elif kind == "triples":
                    shorty_text = (info or {}).get("shorty") or ""
                    if not shorty_text.strip():
                        msg = "No Shorty; cannot extract triples."
                        _log(f"  ! {msg}")
                        update_queue_task_status(db, task_id, "failed", msg)
                        _log("[DEBUG] Task #%d status -> failed" % task_id)
                        continue
                    _log("  -> Extracting triples...")
                    if triple_raw_sink is not None:
                        triple_raw_sink.clear()
                    triples = _call_with_connection_retries(triples_fn, shorty_text, final_title)
                    n_stored = db.replace_facts_for_video(current_video_id, triples)
                    _log(f"  ✓ Stored {n_stored} triples.")
                    slen = len(shorty_text.strip())
                    if (
                        debug_zero_triples
                        and n_stored == 0
                        and slen >= TRIPLE_DEBUG_MIN_SHORTY_CHARS
                    ):
                        if triple_raw_sink:
                            raw = triple_raw_sink[-1] if triple_raw_sink else ""
                            out_path = _write_triple_zero_debug(
                                current_video_id,
                                final_title or "",
                                slen,
                                raw,
                            )
                            _log(
                                "  [DEBUG] Zero triples — raw response saved (%d chars): %s"
                                % (len(raw or ""), out_path)
                            )
                        else:
                            _log(
                                "  [DEBUG] Zero triples — use --provider openai-compatible "
                                "with --debug-zero-triples to save raw responses under data/triple_debug/"
                            )
                    update_queue_task_status(db, task_id, "completed", None)
                    _log("[DEBUG] Task #%d status -> completed (processed_count now=%d)" % (task_id, processed_count + 1))
                    processed_count += 1

                elif kind == "segments":
                    _log("  -> HSC segment summaries...")
                    from hsc.segment_extractor import extract_segments

                    dur_sec: Optional[float] = None
                    if isinstance(meta, dict):
                        try:
                            d0 = float(meta.get("duration") or 0)
                            if d0 > 0:
                                dur_sec = d0
                        except (TypeError, ValueError):
                            dur_sec = None

                    segs = _call_with_connection_retries(
                        extract_segments,
                        transcript_text,
                        None,
                        duration_seconds=dur_sec,
                        title=final_title,
                        chat_fn=hsc_chat_fn,
                    )
                    rows = [
                        {
                            "start_time": s.get("start_time"),
                            "end_time": s.get("end_time"),
                            "summary": (s.get("summary") or "").strip(),
                        }
                        for s in segs
                        if (s.get("summary") or "").strip()
                    ]
                    n_seg = db.replace_segments_for_video(current_video_id, rows)
                    _log(f"  ✓ Stored {n_seg} segments.")
                    update_queue_task_status(db, task_id, "completed", None)
                    _log("[DEBUG] Task #%d status -> completed (processed_count now=%d)" % (task_id, processed_count + 1))
                    processed_count += 1

                elif kind == "events":
                    _log("  -> HSC event extraction...")
                    from hsc.event_extractor import extract_events

                    shorty_text = (info or {}).get("shorty") or ""
                    evs = _call_with_connection_retries(
                        extract_events,
                        transcript_text,
                        final_title,
                        shorty_text,
                        chat_fn=hsc_chat_fn,
                    )
                    n_ev = db.replace_events_for_video(current_video_id, evs)
                    _log(f"  ✓ Stored {n_ev} events.")
                    update_queue_task_status(db, task_id, "completed", None)
                    _log("[DEBUG] Task #%d status -> completed (processed_count now=%d)" % (task_id, processed_count + 1))
                    processed_count += 1

                else:
                    msg = f"Unknown task type: {kind}"
                    _log(f"  ! {msg}")
                    update_queue_task_status(db, task_id, "failed", msg)
                    _log("[DEBUG] Task #%d status -> failed" % task_id)
                    continue

                if limit is not None and processed_count >= limit:
                    _log("[DEBUG] Breaking for loop: reached limit (%d)." % limit)
                    break
                _log("[DEBUG] Loop iteration complete, continuing...")

            except Exception as e:
                msg = str(e)
                _log(f"  !! Error processing task {task_id} for video {current_video_id}: {type(e).__name__}: {msg}")
                update_queue_task_status(db, task_id, "failed", msg)
                _log("[DEBUG] Task #%d status -> failed (exception); continuing to next task." % task_id)
            except BaseException:
                # KeyboardInterrupt, SystemExit, etc. - do not swallow
                raise


def main():
    parser = argparse.ArgumentParser(description="Batch-process videos for Ask Shorty Shorties.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of videos to process (for testing).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only videos listed in failed_videos.txt.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "openai-compatible", "openrouter"],
        default="anthropic",
        help="Which LLM provider to use for generation. Default: anthropic. "
             "openrouter uses OPENROUTER_API_KEY and https://openrouter.ai/api/v1 (OpenAI SDK).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for openai-compatible endpoint (e.g. http://host:8000/v1). "
             "Ignored for openrouter (fixed to https://openrouter.ai/api/v1). "
             "Only used when --provider openai-compatible.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model id: openai-compatible (e.g. qwen2.5-14b), openrouter "
             "(e.g. qwen/qwen2.5-72b-instruct), or anthropic override for HSC "
             "segment/event tasks (default: claude-sonnet-4-20250514 from hsc.segment_extractor).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to transcripts.db (e.g. C:\\Users\\number2\\Desktop\\youtube-history-viewer-copy\\data\\transcripts.db). "
             "If not provided, uses the default DB for this project.",
    )
    parser.add_argument(
        "--queue",
        action="store_true",
        default=True,
        help="Process tasks from processing_queue (default behavior).",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Only process queue tasks for this video_id (e.g. dQw4w9WgXcQ).",
    )
    parser.add_argument(
        "--only-tasks",
        type=str,
        default=None,
        help="Comma-separated task types to claim (e.g. triples). For a second GPU worker.",
    )
    parser.add_argument(
        "--exclude-tasks",
        type=str,
        default=None,
        help="Comma-separated task types to skip (e.g. triples). Use on main worker while a second machine runs --only-tasks triples.",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=None,
        help="With --worker-count, only claim pending rows whose FIFO index i satisfies (i %% N) == id. Use 0..N-1 on N parallel terminals with the same --only-tasks / filters.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Split the pending queue across N workers by FIFO index; use with --worker-id.",
    )
    parser.add_argument(
        "--debug-zero-triples",
        action="store_true",
        help="When triple extraction stores 0 rows but Shorty is substantial, save raw LLM text "
        "to data/triple_debug/{video_id}.txt (openai-compatible / openrouter only; for local diagnosis).",
    )
    args = parser.parse_args()

    only_tasks_f = _parse_task_filter(args.only_tasks)
    exclude_tasks_f = _parse_task_filter(args.exclude_tasks)
    if only_tasks_f and exclude_tasks_f:
        _log("Error: use only one of --only-tasks and --exclude-tasks.")
        return

    try:
        _normalize_worker_shard(args.worker_id, args.worker_count)
    except ValueError as e:
        _log("Error: %s" % e)
        return

    # Allow pointing at an external transcripts.db (e.g. youtube-history-viewer-copy)
    if args.db_path:
        db = TranscriptDatabase(args.db_path)
        _dbp = Path(args.db_path).resolve()
        _chroma = _dbp.parent / "transcript_chroma_new"
        rag = TranscriptRAG(transcript_db=str(_dbp), chroma_dir=str(_chroma))
    else:
        db = TranscriptDatabase()
        rag = TranscriptRAG()

    # Select provider-specific generation functions
    provider = args.provider
    triple_raw_sink: Optional[List[str]] = None
    if args.debug_zero_triples and provider not in ("openai-compatible", "openrouter"):
        _log(
            "Note: --debug-zero-triples only saves raw triple responses with "
            "--provider openai-compatible or openrouter."
        )

    if provider == "anthropic":
        shorty_fn = generate_shorty
        synth_q_fn = generate_synthetic_questions
        entity_fn = extract_entities
        from triple_extractor import extract_triples as triples_fn

        def hsc_chat_fn(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> str:
            from anthropic_client import get_client
            from hsc.segment_extractor import MODEL as _HSC_MODEL

            client = get_client()
            mdl = (args.model or "").strip() or _HSC_MODEL
            resp = client.messages.create(
                model=mdl,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            parts: List[str] = []
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts).strip()
    else:
        # OpenAI-compatible or OpenRouter (same SDK; different base URL / API key env)
        import os
        from openai import OpenAI

        if provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            model = args.model or "openai/gpt-4o-mini"
            api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
            if not api_key:
                _log("Error: OPENROUTER_API_KEY is not set (required for --provider openrouter).")
                return
        else:
            base_url = args.base_url or "http://localhost:8000/v1"
            model = args.model or "gpt-3.5-turbo"
            api_key = os.environ.get("OPENAI_API_KEY") or "no-key"
        oa_client = OpenAI(base_url=base_url, api_key=api_key)

        def _openai_chat(system_prompt: str, user_prompt: str, max_tokens: int = 4096, temperature: float = 0.2) -> str:
            resp = oa_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()

        def shorty_fn(
            transcript_text: str,
            title: Optional[str] = None,
            channel: Optional[str] = None,
            upload_date: Optional[str] = None,
        ) -> str:
            if not transcript_text or not transcript_text.strip():
                raise ValueError("Transcript text is empty; cannot generate Shorty.")
            safe_title = title or "Untitled Video"
            safe_channel = channel or "Unknown channel"
            safe_date = upload_date or "unknown"
            user_prompt = SHORTY_USER_PROMPT_TEMPLATE.format(
                title=safe_title,
                channel=safe_channel,
                upload_date=safe_date,
                transcript=transcript_text.strip(),
            )
            body = _openai_chat(SHORTY_SYSTEM_PROMPT, user_prompt, max_tokens=4096, temperature=0.2)
            header = (
                f"SOURCE: {safe_title}\n"
                f"CHANNEL: {safe_channel}\n"
                f"DATE: {safe_date}\n"
                f"CREATOR: {safe_channel}\n\n"
            )
            return header + body.lstrip()

        def synth_q_fn(
            transcript_text: str,
            title: Optional[str] = None,
            n: int = 10,
        ) -> List[str]:
            if not transcript_text or not transcript_text.strip():
                raise ValueError("Transcript text is empty; cannot generate questions.")
            safe_title = title or "Untitled Video"
            user_prompt = SYNTHETIC_Q_USER_PROMPT_TEMPLATE.format(
                title=safe_title,
                transcript=transcript_text.strip(),
            )
            raw = _openai_chat(SYNTHETIC_Q_SYSTEM_PROMPT, user_prompt, max_tokens=2048, temperature=0.2)

            import json
            import re

            questions: List[str] = []

            def _slice_json_array(text: str) -> Optional[str]:
                t = text.strip()
                if "```" in t:
                    for sep in ("```json", "```"):
                        if t.startswith(sep):
                            t = t[len(sep) :].lstrip()
                        idx = t.find("```")
                        if idx != -1:
                            t = t[:idx].strip()
                        break
                start = t.find("[")
                end = t.rfind("]")
                if start != -1 and end != -1 and end > start:
                    return t[start : end + 1]
                return None

            blob = _slice_json_array(raw) or raw.strip()
            try:
                data = json.loads(blob)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            q = item.strip()
                            if q:
                                questions.append(q)
            except Exception:
                for line in raw.splitlines():
                    line = line.strip().lstrip("-").strip()
                    line = re.sub(r"^\d+[\).\]]\s*", "", line)
                    if not line:
                        continue
                    if line.endswith("?"):
                        questions.append(line)
                        continue
                    low = line.lower()
                    if re.match(
                        r"^(what|how|why|when|where|who|which|is |are |does |did |can |could |would |should )\b",
                        low,
                    ):
                        questions.append(line)
            if len(questions) > n:
                questions = questions[:n]
            return questions

        def entity_fn(
            transcript_text: str,
            title: Optional[str] = None,
        ) -> List[Dict[str, Any]]:
            if not transcript_text or not transcript_text.strip():
                return []
            safe_title = title or "Untitled Video"
            user_prompt = ENTITY_JSON_USER_TEMPLATE.format(title=safe_title, transcript=transcript_text.strip())
            raw = _openai_chat(ENTITY_JSON_SYSTEM_PROMPT, user_prompt, max_tokens=2048, temperature=0.1)
            # Debug: show raw API response before parsing
            _preview = raw[:2000] + ("..." if len(raw) > 2000 else "")
            _log("[DEBUG] Entity API raw response (%d chars):\n%s" % (len(raw), _preview))
            try:
                entities = parse_entities_from_json(raw)
                _log("[DEBUG] parse_entities_from_json returned %d entities" % len(entities))
                if (
                    not entities
                    and len(raw.strip()) > 80
                    and '"name"' not in raw
                ):
                    _log(
                        "[DEBUG] Entity response had no {name,type,aliases} objects; "
                        "treating as empty (queue completes; re-run later if you switch models)."
                    )
                return entities
            except Exception as e:
                _log("[DEBUG] parse_entities_from_json raised: %s: %s" % (type(e).__name__, e))
                return []

        if args.debug_zero_triples:
            triple_raw_sink = []

        def triples_fn(
            shorty_text: str,
            title: str,
        ) -> List[Dict[str, Any]]:
            from triple_extractor import extract_triples_openai, extract_triples_openai_raw

            chat = lambda s, u: _openai_chat(s, u, max_tokens=2048, temperature=0.1)
            if triple_raw_sink is not None:
                trips, raw = extract_triples_openai_raw(shorty_text, title, chat)
                triple_raw_sink.clear()
                triple_raw_sink.append(raw)
                return trips
            return extract_triples_openai(shorty_text, title, chat)

        def hsc_chat_fn(
            system: str, user: str, max_tokens: int, temperature: float
        ) -> str:
            return _openai_chat(
                system, user, max_tokens=max_tokens, temperature=temperature
            )

    # New default: process from the processing_queue if requested (queue mode).
    if args.queue:
        process_queue_tasks(
            db=db,
            rag=rag,
            shorty_fn=shorty_fn,
            synth_q_fn=synth_q_fn,
            entity_fn=entity_fn,
            triples_fn=triples_fn,
            limit=args.limit,
            video_id=args.video_id,
            only_tasks=only_tasks_f,
            exclude_tasks=exclude_tasks_f,
            debug_zero_triples=args.debug_zero_triples,
            triple_raw_sink=triple_raw_sink,
            hsc_chat_fn=hsc_chat_fn,
            worker_id=args.worker_id,
            worker_count=args.worker_count,
        )
        return

    if args.retry_failed:
        videos = get_videos_from_failed(db, args.limit)
        mode_desc = "failed-only"
    else:
        videos = get_videos_needing_shorties(db, args.limit)
        mode_desc = "needing-Shorties"

    total = len(videos)
    if total == 0:
        _log(f"No videos to process in {mode_desc} mode. Nothing to do.")
        return

    _log(f"Found {total} videos needing Shorties.")

    # Full-run cost estimate
    est = estimate_batch_cost(db, videos)
    batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    _log("\nCOST ESTIMATE (full run):")
    _log(f"  Estimated input tokens:  {format_token_count(est['input_tokens'])}")
    _log(f"  Estimated output tokens: {format_token_count(est['output_tokens'])}")
    _log(f"  Estimated cost:          {format_cost(est['total_cost'])} (Haiku pricing)")
    _log(f"\nProcess in batches of {BATCH_SIZE}.")
    _log(f"Total batches: {batches}")

    choice = input("\nProceed with full run? (yes/no/limit): ").strip().lower()
    if choice == "no":
        _log("Exiting without processing.")
        return
    elif choice.startswith("limit"):
        # allow "limit 100" or "limit:100"
        parts = choice.replace(":", " ").split()
        if len(parts) >= 2 and parts[1].isdigit():
            limit_val = int(parts[1])
        else:
            raw = input("Enter numeric limit: ").strip()
            if not raw.isdigit():
                _log("Invalid limit. Exiting.")
                return
            limit_val = int(raw)
        videos = videos[:limit_val]
        total = len(videos)
        batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        _log(f"\nLimiting run to {total} videos ({batches} batches).")
    elif choice.isdigit():
        limit_val = int(choice)
        videos = videos[:limit_val]
        total = len(videos)
        batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        _log(f"\nLimiting run to {total} videos ({batches} batches).")
    elif choice != "yes":
        _log("Unrecognized option, exiting.")
        return

    # Running totals across session
    totals: Dict[str, Any] = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "videos_processed": 0,
        "videos_failed": [],
    }

    start = 0
    batch_index = 0

    while start < total:
        batch_index += 1
        batch = videos[start : start + BATCH_SIZE]
        batch_start_num = start + 1
        batch_end_num = min(start + len(batch), total)

        # Per-batch estimate (for display)
        est_batch = estimate_batch_cost(db, batch)

        _log(
            f"\n=== Batch {batch_index} of {batches} "
            f"(videos {batch_start_num}-{batch_end_num}) ==="
        )
        _log("Titles:")
        for v in batch:
            t = v.get("title") or "Untitled Video"
            c = v.get("channel") or "Unknown Channel"
            _log(f"  - {t} ({c})")

        _log(
            f"Estimated batch cost: {format_cost(est_batch['total_cost'])}\n"
            f"Tokens so far this run: "
            f"{format_token_count(totals['total_input_tokens'])} input / "
            f"{format_token_count(totals['total_output_tokens'])} output\n"
            f"Cost so far this run: {format_cost(totals['total_cost'])}"
        )

        ans = input("\nProceed with this batch? (yes/skip/stop): ").strip().lower()
        if ans == "skip":
            _log("Skipping this batch.")
            start += BATCH_SIZE
            continue
        elif ans == "stop":
            _log("Stopping before this batch.")
            break
        elif ans != "yes":
            _log("Unrecognized option, treating as 'skip'.")
            start += BATCH_SIZE
            continue

        # Process the batch
        process_batch(db, rag, batch, start, total, totals, shorty_fn, synth_q_fn, entity_fn)
        start += BATCH_SIZE

        if start < total:
            _log("\nSleeping 1s before next batch to rate-limit...")
            time.sleep(1.0)

    # Session summary
    _log("\n=== Session Complete ===")
    _log(f"Videos processed: {totals['videos_processed']}")
    _log(f"Videos failed: {len(totals['videos_failed'])}")
    _log(
        f"Total input tokens:  {format_token_count(totals['total_input_tokens'])}\n"
        f"Total output tokens: {format_token_count(totals['total_output_tokens'])}\n"
        f"Total cost:          {format_cost(totals['total_cost'])}"
    )

    if totals["videos_failed"]:
        write_failed_videos(totals["videos_failed"])


if __name__ == "__main__":
    main()

