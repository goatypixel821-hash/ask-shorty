#!/usr/bin/env python3
"""
Flask app exposing the Ask Shorty UI and API.

Routes:
- GET /ask               -> HTML UI
- POST /api/ask          -> enqueue question, return job_id (ref_id null until done)
- GET /ask_v2            -> V2 UI (requires ASK_SHORTY_V2=true)
- POST /api/ask_v2       -> V2 hierarchical RAG job queue (deep answer)
- POST /api/search_fast  -> V2 local retrieval only (no LLM)
- GET /api/ask/result/<job_id> -> poll for answer (includes ref_id when completed)
- GET /api/ask/ref/<ref_id> -> fetch completed job by human-readable ref_id
- GET /agent             -> Agent Mode UI
- POST /api/agent/ask    -> enqueue agent-mode question
- GET /api/agent/result/<job_id> -> poll (same as ask result)
- GET /api/agent/stream/<job_id> -> SSE (same stream as ask)
- POST /api/agent/cancel/<job_id> -> cancel a pending/running agent job
- DELETE /api/agent/job/<job_id> -> same as cancel
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context, after_this_request

import os
import json
import time
import queue as _queue
import secrets
import string
import shutil
import sqlite3
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import build_courses
import export_course

print("Step 1: importing AskShorty...")
from ask_shorty import AskShorty
from transcript_database import TranscriptDatabase
try:
    from filter_ad_videos import PendingVideo as _AdPendingVideo, looks_like_ad as _looks_like_ad
except Exception:
    _AdPendingVideo = None
    _looks_like_ad = None


app = Flask(__name__)

_engine = None


def get_engine() -> AskShorty:
    """
    Lazily construct the AskShorty engine so that any heavy RAG initialization
    only occurs on the first incoming query, not at app startup.
    """
    global _engine
    if _engine is None:
        print("Step 2: creating engine...")
        try:
            _engine = AskShorty()
            print("Step 3: engine ready")
        except Exception as e:
            # Print the error so it shows up in the console / logs before crashing
            print("Error during AskShorty() initialization:", repr(e))
            raise
    return _engine


_engine_v2 = None


def _v2_flag_enabled() -> bool:
    return os.getenv("ASK_SHORTY_V2", "").strip().lower() in ("1", "true", "yes", "on")


def get_engine_v2():
    """Lazy Ask Shorty V2 engine (separate from V1 AskShorty)."""
    global _engine_v2
    if _engine_v2 is None:
        from ask_shorty_v2 import AskShortyV2

        _engine_v2 = AskShortyV2()
        try:
            _engine_v2._lazy_load()
        except Exception as e:
            print(
                "Ask Shorty V2: warm preload failed (will retry on first query):",
                repr(e),
                flush=True,
            )
    return _engine_v2


db = TranscriptDatabase()

# SQLite: wait on busy locks + WAL for concurrent app + batch workers.
_SQLITE_BUSY_TIMEOUT_SEC = 30.0


def _sqlite_connect(database_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(database_path, timeout=_SQLITE_BUSY_TIMEOUT_SEC)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


_jobs_dir = Path(__file__).parent / "data" / "jobs"
_jobs_dir.mkdir(parents=True, exist_ok=True)


def _startup_storage_diagnostic() -> None:
    """
    Print effective storage locations at startup (similar to grabber diagnostic).
    """
    base_dir = Path(__file__).resolve().parent
    db_raw = getattr(db, "db_path", os.environ.get("ASK_SHORTY_DB_PATH") or "")
    db_path_eff = str(Path(str(db_raw)).resolve()) if str(db_raw).strip() else str((base_dir / "data" / "transcripts.db").resolve())
    chroma_raw = (os.environ.get("ASK_SHORTY_CHROMA_PATH") or "").strip()
    chroma_eff = chroma_raw or str((base_dir / "data" / "transcript_chroma_finetuned").resolve())
    line = (
        "[ask-shorty-app] db_path=%s | chroma_path=%s | cwd=%s"
        % (db_path_eff, chroma_eff, str(Path.cwd().resolve()))
    )
    print(line, flush=True)


_startup_storage_diagnostic()


def _warm_v2_in_background() -> None:
    """Preload V2 BM25 + memory maps without blocking Flask bind."""
    if not _v2_flag_enabled():
        return

    def _run() -> None:
        t0 = time.perf_counter()
        try:
            get_engine_v2()
            ms = round((time.perf_counter() - t0) * 1000.0)
            print(f"[Ask Shorty] V2 warmup complete in {ms}ms", flush=True)
        except Exception as e:
            print(
                f"[Ask Shorty] V2 warmup failed (will retry on first query): {e!r}",
                flush=True,
            )

    threading.Thread(target=_run, daemon=True, name="v2-warmup").start()


_warm_v2_in_background()

# Per-job SSE event queues.  Created in api_ask before the worker thread
# starts so a client can connect immediately and receive every event.
_job_events: Dict[int, _queue.Queue] = {}
_job_cancel_events: Dict[int, threading.Event] = {}

# Course generation SSE (same pattern as /api/ask/stream)
_course_job_events: Dict[int, _queue.Queue] = {}
_course_job_next_id = 1
_course_job_lock = threading.Lock()

# Agent jobs the user cancelled from the UI (worker checks this + DB before writing "completed").
_cancelled_agent_job_ids: Set[int] = set()
_cancel_agent_lock = threading.Lock()
# If a row stays "running" longer than this (see updated_at), polling marks timeout.
JOB_RUNNING_TIMEOUT_SEC = 120


def _parse_job_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip()[:19]
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.fromisoformat(str(value).replace("Z", ""))
        except Exception:
            return None


def _seconds_since_job_timestamp(ts: Optional[str]) -> Optional[float]:
    t = _parse_job_timestamp(ts)
    if t is None:
        return None
    return (datetime.now() - t).total_seconds()


def _should_skip_late_completion_write(job_id: int) -> bool:
    """True if DB already says cancelled or poll-side timeout — do not overwrite with completed."""
    try:
        conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
        cur = conn.cursor()
        cur.execute(
            "SELECT status, IFNULL(error, '') FROM ask_jobs WHERE id = ?",
            (job_id,),
        )
        row = cur.fetchone()
        conn.close()
    except Exception as exc:
        print(f"[ask_jobs] skip completion check failed for job_id={job_id}: {exc!r}")
        return False
    if not row:
        return True
    status, err = row[0], (row[1] or "").strip().lower()
    if status == "cancelled":
        return True
    if status == "error" and err == "timeout":
        return True
    return False


def _ensure_jobs_table() -> None:
    """Create ask_jobs table if it doesn't exist."""
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ask_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            video_ids TEXT,
            status TEXT NOT NULL,            -- pending, running, completed, error
            answer TEXT,
            error TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            ref_id TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def _ensure_ref_id_column() -> None:
    """Add ref_id to ask_jobs if upgrading from an older schema."""
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(ask_jobs)")
    cols = {row[1] for row in cursor.fetchall()}
    if "ref_id" not in cols:
        cursor.execute("ALTER TABLE ask_jobs ADD COLUMN ref_id TEXT")
        conn.commit()
    conn.close()


def _new_ask_ref_id() -> str:
    """ASK-YYYYMMDD-XXXX with XXXX = 4 random uppercase alphanumeric."""
    d = datetime.now().strftime("%Y%m%d")
    alphabet = string.ascii_uppercase + string.digits
    suffix = "".join(secrets.choice(alphabet) for _ in range(4))
    return f"ASK-{d}-{suffix}"


def _allocate_ref_id() -> str:
    """Pick a ref_id not already present in ask_jobs (collision retry)."""
    for _ in range(32):
        rid = _new_ask_ref_id()
        conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM ask_jobs WHERE ref_id = ? LIMIT 1", (rid,))
        taken = cur.fetchone() is not None
        conn.close()
        if not taken:
            return rid
    return _new_ask_ref_id() + secrets.choice(string.ascii_uppercase)


def _cleanup_stale_jobs() -> None:
    """
    On app startup, mark any jobs that were left in pending/running state
    as error, since the previous process likely crashed during generation.
    """
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        UPDATE ask_jobs
        SET status = 'error',
            error = 'Process crashed - please retry',
            updated_at = ?
        WHERE status IN ('pending', 'running')
        """,
        (now,),
    )
    conn.commit()
    conn.close()


_ensure_jobs_table()
_ensure_ref_id_column()
_cleanup_stale_jobs()


def _update_job(job_id: int, **fields) -> None:
    """Helper to update a job row safely from worker thread."""
    if not fields:
        return
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    fields["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [job_id]
    cursor.execute(f"UPDATE ask_jobs SET {cols} WHERE id = ?", values)
    conn.commit()
    conn.close()


def _make_emit(job_id: int):
    """Return an emit callable that pushes events onto the job's SSE queue."""
    def emit(event: dict) -> None:
        q = _job_events.get(job_id)
        if q is not None:
            q.put(event)
    return emit


def _run_job(job_id: int, question: str, video_ids):
    """Background worker that runs AskShorty and stores the result.

    IMPORTANT: We first write the answer to a JSON file under data/jobs so
    that if the process crashes before the SQLite write, the polling API can
    still recover the result from disk.
    """
    try:
        _update_job(job_id, status="running")
        engine = get_engine()
        result = engine.answer_question(
            question,
            video_ids=video_ids,
            emit=_make_emit(job_id),
            should_cancel=lambda: _job_cancel_events.get(job_id, threading.Event()).is_set(),
        )
        with _cancel_agent_lock:
            was_flagged = job_id in _cancelled_agent_job_ids
            if was_flagged:
                _cancelled_agent_job_ids.discard(job_id)
        if was_flagged or _should_skip_late_completion_write(job_id):
            print(f"[ask] job_id={job_id} skip completed write (cancelled or timeout)")
            return
        answer_json = json.dumps(
            {
                "answer":       result.get("answer", ""),
                "used_context": result.get("used_context", []),
                "sources":      result.get("sources", []),
                "debug_events": result.get("debug_events", []),
            }
        )
        # Write to disk first so it survives a later crash
        job_file = _jobs_dir / f"{job_id}.json"
        try:
            print(f"[ask] Step F1: writing job file for job_id={job_id} -> {job_file}")
            job_file.write_text(answer_json, encoding="utf-8")
            print(f"[ask] Step F1: job file written for job_id={job_id}")
        except Exception as file_err:
            # Log to console but continue to attempt DB write
            print(f"[ask_shorty] Failed to write job file {job_file}: {file_err!r}")

        print(f"[ask] Step F2: updating DB row for job_id={job_id}")
        ref_id = _allocate_ref_id()
        _update_job(job_id, status="completed", answer=answer_json, error=None, ref_id=ref_id)
        print(f"[ask] Step F3: DB updated for job_id={job_id}, worker done ref_id={ref_id}")
    except Exception as e:
        q = _job_events.get(job_id)
        if q is not None:
            q.put({"type": "error", "message": str(e), "elapsed_ms": 0})
        _update_job(job_id, status="error", error=str(e))
    finally:
        # Signal SSE generator to close, then clean up the queue after a delay
        q = _job_events.get(job_id)
        if q is not None:
            q.put({"type": "stream_end"})

        def _cleanup():
            import time
            time.sleep(30)
            _job_events.pop(job_id, None)

        threading.Thread(target=_cleanup, daemon=True).start()


def _run_job_v2(job_id: int, question: str, video_ids):
    """Background worker: Ask Shorty V2 hierarchical path only."""
    try:
        _update_job(job_id, status="running")
        engine = get_engine_v2()
        result = engine.answer(
            question,
            video_ids=video_ids,
            emit=_make_emit(job_id),
            should_cancel=lambda: _job_cancel_events.get(job_id, threading.Event()).is_set(),
            job_id=job_id,
        )
        if _should_skip_late_completion_write(job_id):
            print(f"[ask-v2] job_id={job_id} skip completed write (timeout)")
            return
        answer_json = json.dumps(
            {
                "answer":       result.get("answer", ""),
                "used_context": result.get("used_context", []),
                "sources":      result.get("sources", []),
                "debug_events": result.get("debug_events", []),
                "grounding_audit": result.get("grounding_audit", []),
                "verification_excerpts": result.get("verification_excerpts", []),
            }
        )
        job_file = _jobs_dir / f"{job_id}.json"
        try:
            job_file.write_text(answer_json, encoding="utf-8")
        except Exception as file_err:
            print(f"[ask_shorty] Failed to write job file {job_file}: {file_err!r}")
        ref_id = _allocate_ref_id()
        try:
            from ask_shorty_v2 import v2_log_patch_ref_id

            v2_log_patch_ref_id(job_id, ref_id)
        except Exception:
            pass
        _update_job(job_id, status="completed", answer=answer_json, error=None, ref_id=ref_id)
    except Exception as e:
        q = _job_events.get(job_id)
        if q is not None:
            q.put({"type": "error", "message": str(e), "elapsed_ms": 0})
        _update_job(job_id, status="error", error=str(e))
    finally:
        q = _job_events.get(job_id)
        if q is not None:
            q.put({"type": "stream_end"})

        def _cleanup():
            import time
            time.sleep(30)
            _job_events.pop(job_id, None)

        threading.Thread(target=_cleanup, daemon=True).start()


def _run_agent_job(job_id: int, question: str, video_ids, conversation_history=None):
    """Background worker: agentic tool loop (AskShorty with agent_mode=True)."""
    try:
        _update_job(job_id, status="running")
        engine = get_engine()
        result = engine.answer_question(
            question,
            video_ids=video_ids,
            emit=_make_emit(job_id),
            agent_mode=True,
            should_cancel=lambda: _job_cancel_events.get(job_id, threading.Event()).is_set(),
            conversation_history=conversation_history,
        )
        with _cancel_agent_lock:
            was_flagged = job_id in _cancelled_agent_job_ids
            if was_flagged:
                _cancelled_agent_job_ids.discard(job_id)
        if was_flagged or _should_skip_late_completion_write(job_id):
            print(f"[agent] job_id={job_id} skip completed write (cancelled or timeout)")
            return
        answer_json = json.dumps(
            {
                "answer":       result.get("answer", ""),
                "used_context": result.get("used_context", []),
                "sources":      result.get("sources", []),
                "debug_events": result.get("debug_events", []),
            }
        )
        job_file = _jobs_dir / f"{job_id}.json"
        try:
            print(f"[agent] Step F1: writing job file for job_id={job_id} -> {job_file}")
            job_file.write_text(answer_json, encoding="utf-8")
            print(f"[agent] Step F1: job file written for job_id={job_id}")
        except Exception as file_err:
            print(f"[ask_shorty] Failed to write agent job file {job_file}: {file_err!r}")

        print(f"[agent] Step F2: updating DB row for job_id={job_id}")
        ref_id = _allocate_ref_id()
        _update_job(job_id, status="completed", answer=answer_json, error=None, ref_id=ref_id)
        print(f"[agent] Step F3: DB updated for job_id={job_id}, worker done ref_id={ref_id}")
    except Exception as e:
        q = _job_events.get(job_id)
        if q is not None:
            q.put({"type": "error", "message": str(e), "elapsed_ms": 0})
        _update_job(job_id, status="error", error=str(e))
    finally:
        q = _job_events.get(job_id)
        if q is not None:
            q.put({"type": "stream_end"})

        def _cleanup():
            import time
            time.sleep(30)
            _job_events.pop(job_id, None)

        threading.Thread(target=_cleanup, daemon=True).start()


def _agent_sse_enabled() -> bool:
    return os.getenv("ASK_SHORTY_AGENT", "0").strip().lower() in ("1", "true", "yes", "on")


@app.route("/ask", methods=["GET"])
def ask_page():
    return render_template("ask.html")


@app.route("/ask_v2", methods=["GET"])
def ask_v2_page():
    if not _v2_flag_enabled():
        return (
            "<p>Ask Shorty V2 is disabled. Set <code>ASK_SHORTY_V2=true</code> in .env and restart.</p>",
            503,
        )
    return render_template("ask_v2.html")


@app.route("/agent", methods=["GET"])
def agent_page():
    return render_template("agent.html", agent_sse=_agent_sse_enabled())


# ── Knowledge Observatory ─────────────────────────────────────────────────────

_CLUSTERS_PATH = Path(__file__).parent / "data" / "clusters.json"
_FULL_DB = os.environ.get("ASK_SHORTY_DB_PATH") or "C:/Users/number2/Desktop/youtube-history-viewer-copy/data/transcripts.db"


def _load_clusters():
    """Load cached clusters.json, or return None if not built yet."""
    if _CLUSTERS_PATH.exists():
        try:
            return json.loads(_CLUSTERS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@app.route("/knowledge", methods=["GET"])
def knowledge_page():
    return render_template("knowledge.html")


@app.route("/api/observatory/overview", methods=["GET"])
def api_observatory_overview():
    data = _load_clusters()
    stats = data["stats"] if data else {}

    # Pull live counts from the full corpus DB
    try:
        conn = _sqlite_connect(str(_FULL_DB))
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM videos")
        total_videos = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM transcripts WHERE shorty IS NOT NULL AND length(trim(shorty))>0")
        total_shorties = c.fetchone()[0]
        c.execute("SELECT MIN(watch_date), MAX(watch_date) FROM videos WHERE watch_date IS NOT NULL")
        d_from, d_to = c.fetchone()
        c.execute("SELECT channel, COUNT(*) n FROM videos GROUP BY channel ORDER BY n DESC LIMIT 10")
        top_channels = c.fetchall()
        conn.close()
    except Exception as e:
        total_videos = stats.get("total_videos", 0)
        total_shorties = stats.get("clustered", 0)
        d_from = stats.get("date_from", "")
        d_to   = stats.get("date_to", "")
        top_channels = []

    # Ask log count from local DB
    try:
        conn2 = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
        c2 = conn2.cursor()
        c2.execute("SELECT COUNT(*) FROM ask_jobs WHERE status='completed'")
        ask_count = c2.fetchone()[0]
        conn2.close()
    except Exception:
        ask_count = 0

    return jsonify({
        "total_videos":   total_videos,
        "total_shorties": total_shorties,
        "cluster_count":  stats.get("cluster_count", 0),
        "date_from":      (d_from or "")[:10],
        "date_to":        (d_to   or "")[:10],
        "top_channels":   [{"channel": r[0] or "Unknown", "count": r[1]} for r in top_channels],
        "ask_count":      ask_count,
        "clusters_built": _CLUSTERS_PATH.exists(),
    })


@app.route("/api/observatory/clusters", methods=["GET"])
def api_observatory_clusters():
    data = _load_clusters()
    if not data:
        return jsonify({"error": "Clusters not built yet. Run: python build_clusters.py"}), 404
    # Strip shorty text from response (too heavy); keep coords + metadata
    return jsonify({
        "generated_at": data.get("generated_at"),
        "stats":        data.get("stats"),
        "clusters":     data.get("clusters", []),
        "noise_videos": data.get("noise_videos", []),
    })


@app.route("/api/observatory/timeline", methods=["GET"])
def api_observatory_timeline():
    data = _load_clusters()
    if not data:
        # Fall back to raw video dates from DB
        all_videos = []
        try:
            conn = _sqlite_connect(str(_FULL_DB))
            c = conn.cursor()
            c.execute("SELECT video_id, watch_date FROM videos WHERE watch_date IS NOT NULL")
            all_videos = [{"video_id": r[0], "watch_date": r[1][:10], "cluster_id": -2}
                          for r in c.fetchall()]
            conn.close()
        except Exception:
            pass
    else:
        all_videos = data.get("all_videos", [])

    # Bucket by YYYY-MM
    from collections import defaultdict
    month_total: dict = defaultdict(int)
    month_cluster: dict = defaultdict(lambda: defaultdict(int))

    for v in all_videos:
        wd = (v.get("watch_date") or "")[:7]  # YYYY-MM
        if not wd or len(wd) < 7:
            continue
        month_total[wd] += 1
        cid = v.get("cluster_id", -2)
        month_cluster[wd][str(cid)] += 1

    months = sorted(month_total.keys())

    # Collect all cluster ids present
    all_cids: set = set()
    for cd in month_cluster.values():
        all_cids.update(cd.keys())

    by_cluster = {cid: [month_cluster[m].get(cid, 0) for m in months] for cid in all_cids}

    # Build cluster color map
    cluster_colors = {}
    if data:
        for cl in data.get("clusters", []):
            cluster_colors[str(cl["id"])] = cl["color"]
        cluster_colors["-1"] = "#374151"
        cluster_colors["-2"] = "#1e293b"

    return jsonify({
        "months":        months,
        "totals":        [month_total[m] for m in months],
        "by_cluster":    by_cluster,
        "cluster_colors": cluster_colors,
    })


@app.route("/api/observatory/ask-log", methods=["GET"])
def api_observatory_ask_log():
    try:
        conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
        c = conn.cursor()
        c.execute("""
            SELECT id, question, status, created_at, answer, video_ids, ref_id
            FROM ask_jobs
            ORDER BY id DESC
            LIMIT 100
        """)
        rows = c.fetchall()
        conn.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result = []
    for row in rows:
        job_id, question, status, created_at, answer_json, video_ids_json, ref_id = row
        answer_snippet = ""
        if answer_json:
            try:
                answer_snippet = json.loads(answer_json).get("answer", "")[:200]
            except Exception:
                pass
        result.append({
            "id":             job_id,
            "question":       question or "",
            "status":         status or "",
            "created_at":     created_at or "",
            "answer_snippet": answer_snippet,
            "video_ids":      json.loads(video_ids_json) if video_ids_json else None,
            "ref_id":         ref_id or None,
        })

    return jsonify({"asks": result})


# ── Knowledge Courses ─────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent


def _new_course_job_id() -> int:
    global _course_job_next_id
    with _course_job_lock:
        jid = _course_job_next_id
        _course_job_next_id += 1
        return jid


def _resolve_course_path(course_id: str) -> Optional[Path]:
    cat = build_courses.load_catalog()
    if cat:
        for c in cat.get("courses", []):
            if c.get("id") == course_id:
                rel = c.get("file") or ""
                p = (_ROOT / rel).resolve()
                if p.is_file():
                    return p
    for p in sorted((_ROOT / "data" / "courses").glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("id") == course_id:
                return p
        except Exception:
            continue
    return None


def _run_course_generation(job_id: int, payload: dict) -> None:
    q = _course_job_events.get(job_id)

    def emit(event: dict) -> None:
        if q is not None:
            q.put(event)

    try:
        all_c = bool(payload.get("all"))
        missing_only = bool(payload.get("missing_only"))
        cids = payload.get("cluster_ids")
        if not all_c:
            if not cids:
                if q is not None:
                    q.put({"type": "error", "message": "Provide cluster_ids or all: true"})
                return
            cids = [int(x) for x in cids]
        build_courses.run_generation(
            cluster_ids=None if all_c else cids,
            all_clusters=all_c,
            emit=emit,
            missing_only=missing_only and all_c,
        )
        if q is not None:
            q.put({"type": "complete"})
    except Exception as e:
        if q is not None:
            q.put({"type": "error", "message": str(e)})
    finally:
        if q is not None:
            q.put({"type": "stream_end"})

        def _cleanup():
            import time
            time.sleep(30)
            _course_job_events.pop(job_id, None)

        threading.Thread(target=_cleanup, daemon=True).start()


@app.route("/courses", methods=["GET"])
def courses_catalog_page():
    cat = build_courses.load_catalog() or {}
    clusters = _load_clusters()
    return render_template("courses.html", catalog=cat, clusters=clusters or {})


@app.route("/courses/<course_id>", methods=["GET"])
def course_detail_page(course_id: str):
    path = _resolve_course_path(course_id)
    if not path or not path.exists():
        return render_template(
            "courses.html",
            catalog=build_courses.load_catalog() or {},
            clusters=_load_clusters() or {},
            error=f'Course "{course_id}" not found. Generate it from the catalog.',
        ), 404
    course = json.loads(path.read_text(encoding="utf-8"))
    return render_template("course_detail.html", course=course)


@app.route("/api/courses/status", methods=["GET"])
def api_courses_status():
    cat = build_courses.load_catalog()
    if not cat:
        # Try rebuilding from disk if JSON files exist but catalog is missing
        try:
            cat = build_courses.rebuild_catalog_from_disk()
        except Exception:
            cat = {}
    courses_out: List[dict] = []
    for c in cat.get("courses", []):
        rel = c.get("file") or ""
        fp = (_ROOT / rel) if rel else None
        mtime = ""
        exists = bool(fp and fp.is_file())
        if exists and fp is not None:
            mtime = datetime.fromtimestamp(fp.stat().st_mtime).isoformat()
        entry = dict(c)
        entry["file_exists"] = exists
        entry["file_mtime"] = mtime
        nvid = 0
        try:
            if exists and fp is not None:
                raw = json.loads(fp.read_text(encoding="utf-8"))
                for m in raw.get("modules") or []:
                    nvid += len(m.get("lessons") or [])
        except Exception:
            pass
        entry["video_count"] = nvid
        courses_out.append(entry)
    return jsonify(
        {
            "catalog_generated_at": cat.get("generated_at"),
            "total_courses": cat.get("total_courses", 0),
            "total_lessons": cat.get("total_lessons", 0),
            "total_hours": cat.get("total_hours", 0),
            "courses": courses_out,
        }
    )


@app.route("/api/courses/generate", methods=["POST"])
def api_courses_generate():
    data = request.get_json(force=True, silent=True) or {}
    all_c = bool(data.get("all"))
    cids = data.get("cluster_ids")
    if not all_c and not cids:
        return jsonify({"success": False, "error": "cluster_ids or all: true required"}), 400

    job_id = _new_course_job_id()
    _course_job_events[job_id] = _queue.Queue()
    thread = threading.Thread(
        target=_run_course_generation,
        args=(job_id, data),
        daemon=True,
    )
    thread.start()
    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/courses/stream/<int:job_id>", methods=["GET"])
def api_courses_stream(job_id: int):
    def generate():
        q = _course_job_events.get(job_id)
        if q is None:
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
            return
        while True:
            try:
                event = q.get(timeout=90)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "stream_end":
                    break
            except _queue.Empty:
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/courses/<course_id>/download", methods=["GET"])
def api_courses_download(course_id: str):
    fmt = (request.args.get("format") or "markdown").lower()
    path = _resolve_course_path(course_id)
    if not path or not path.exists():
        return jsonify({"error": "Course not found"}), 404
    course = json.loads(path.read_text(encoding="utf-8"))
    slug = course.get("slug") or "course"

    tmp = Path(tempfile.mkdtemp())
    try:
        if fmt in ("md", "markdown"):
            out = tmp / f"{slug}.md"
            export_course.export_markdown(course, out)
            data = out.read_bytes()

            @after_this_request
            def _rm(response):
                shutil.rmtree(tmp, ignore_errors=True)
                return response

            return Response(
                data,
                mimetype="text/markdown; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{slug}.md"',
                },
            )
        if fmt in ("html",):
            out = tmp / f"{slug}.html"
            export_course.export_html(course, out)
            data = out.read_bytes()

            @after_this_request
            def _rm2(response):
                shutil.rmtree(tmp, ignore_errors=True)
                return response

            return Response(
                data,
                mimetype="text/html; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{slug}.html"',
                },
            )
        if fmt in ("json",):
            @after_this_request
            def _rm3(response):
                shutil.rmtree(tmp, ignore_errors=True)
                return response

            return Response(
                path.read_bytes(),
                mimetype="application/json; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{slug}.json"',
                },
            )
        if fmt in ("pdf",):
            # Print-to-PDF from browser is primary; return printable HTML as fallback.
            out = tmp / f"{slug}-print.html"
            export_course.export_html(course, out)
            data = out.read_bytes()

            @after_this_request
            def _rm4(response):
                shutil.rmtree(tmp, ignore_errors=True)
                return response

            return Response(
                data,
                mimetype="text/html; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{slug}-print.html"',
                },
            )
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        return jsonify({"error": str(e)}), 500

    shutil.rmtree(tmp, ignore_errors=True)
    return jsonify({"error": "bad format"}), 400


@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    video_ids = data.get("video_ids") or None

    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400

    # Insert job row
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO ask_jobs (question, video_ids, status, created_at, updated_at)
        VALUES (?, ?, 'pending', ?, ?)
        """,
        (question, json.dumps(video_ids), now, now),
    )
    job_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # Create event queue before starting worker so the SSE client can
    # connect immediately without missing any early events.
    _job_events[job_id] = _queue.Queue()
    _job_cancel_events[job_id] = threading.Event()

    # Kick off background worker
    thread = threading.Thread(target=_run_job, args=(job_id, question, video_ids), daemon=True)
    thread.start()

    return jsonify({"success": True, "job_id": job_id, "ref_id": None})


@app.route("/api/ask_v2", methods=["POST"])
def api_ask_v2():
    if not _v2_flag_enabled():
        return (
            jsonify(
                {
                    "success": False,
                    "error": "ASK_SHORTY_V2 is not enabled",
                }
            ),
            503,
        )
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    video_ids = data.get("video_ids") or None

    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400

    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO ask_jobs (question, video_ids, status, created_at, updated_at)
        VALUES (?, ?, 'pending', ?, ?)
        """,
        (question, json.dumps(video_ids), now, now),
    )
    job_id = cursor.lastrowid
    conn.commit()
    conn.close()

    _job_events[job_id] = _queue.Queue()
    _job_cancel_events[job_id] = threading.Event()

    thread = threading.Thread(target=_run_job_v2, args=(job_id, question, video_ids), daemon=True)
    thread.start()

    return jsonify({"success": True, "job_id": job_id, "ref_id": None})


@app.route("/api/search_fast", methods=["POST"])
def api_search_fast():
    """
    Synchronous local retrieval (V2 BM25 + segments). No LLM, no agent, no CE load.
    """
    if not _v2_flag_enabled():
        return (
            jsonify(
                {
                    "success": False,
                    "error": "ASK_SHORTY_V2 is not enabled",
                }
            ),
            503,
        )
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or data.get("q") or "").strip()
    video_ids = data.get("video_ids") or None
    top_k = data.get("top_k") or data.get("limit")

    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400

    t0 = time.perf_counter()
    try:
        engine = get_engine_v2()
        sf_kwargs: Dict[str, object] = {"restrict_videos": video_ids}
        if top_k is not None:
            sf_kwargs["top_k"] = int(top_k)
        payload = engine.search_fast(question, **sf_kwargs)  # type: ignore[arg-type]
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0)
        q_log = question[:80].replace('"', "'")
        print(
            f'[Ask Shorty] fast_search query="{q_log}" failed after {elapsed_ms}ms: {e!r}',
            flush=True,
        )
        return jsonify({"success": False, "error": str(e)}), 500

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0)
    n_results = len(payload.get("results") or [])
    q_log = question[:80].replace('"', "'")
    print(
        f'[Ask Shorty] fast_search query="{q_log}" took {elapsed_ms}ms results={n_results}',
        flush=True,
    )
    return jsonify(
        {
            "success": True,
            "elapsed_ms": elapsed_ms,
            **payload,
        }
    )


@app.route("/api/agent/ask", methods=["POST"])
def api_agent_ask():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    video_ids = data.get("video_ids") or None
    raw_history = data.get("history")

    conversation_history = None
    if isinstance(raw_history, list):
        conversation_history = []
        for turn in raw_history[-20:]:
            if not isinstance(turn, dict):
                continue
            role = (turn.get("role") or "").strip()
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                conversation_history.append({"role": role, "content": content[:4000]})

    if not question:
        return jsonify({"success": False, "error": "Question is required"}), 400

    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        """
        INSERT INTO ask_jobs (question, video_ids, status, created_at, updated_at)
        VALUES (?, ?, 'pending', ?, ?)
        """,
        (question, json.dumps(video_ids), now, now),
    )
    job_id = cursor.lastrowid
    conn.commit()
    conn.close()

    _job_events[job_id] = _queue.Queue()
    _job_cancel_events[job_id] = threading.Event()

    thread = threading.Thread(
        target=_run_agent_job,
        args=(job_id, question, video_ids),
        kwargs={"conversation_history": conversation_history},
        daemon=True,
    )
    thread.start()

    return jsonify({"success": True, "job_id": job_id, "ref_id": None})


def _api_agent_cancel_impl(job_id: int):
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM ask_jobs WHERE id = ?", (job_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return jsonify({"success": False, "error": "Job not found", "status": "missing"}), 404
    status = row[0]
    if status not in ("pending", "running"):
        return jsonify(
            {
                "success": False,
                "error": "Job cannot be cancelled",
                "status": status,
            }
        ), 400
    with _cancel_agent_lock:
        _cancelled_agent_job_ids.add(job_id)
    ev = _job_cancel_events.get(job_id)
    if ev is not None:
        ev.set()
    _update_job(job_id, status="cancelled", error="cancelled")
    return jsonify({"success": True, "status": "cancelled", "job_id": job_id})


@app.route("/api/agent/cancel/<int:job_id>", methods=["POST"])
def api_agent_cancel_job(job_id: int):
    return _api_agent_cancel_impl(job_id)


@app.route("/api/agent/job/<int:job_id>", methods=["DELETE"])
def api_agent_delete_job(job_id: int):
    """Same as POST cancel (UI may use DELETE)."""
    return _api_agent_cancel_impl(job_id)


@app.route("/api/ask/ref/<ref_id>", methods=["GET"])
def api_ask_by_ref(ref_id: str):
    """Return a completed ask job by its stable ref_id (e.g. ASK-20260418-X7K2)."""
    rid = (ref_id or "").strip()
    if not rid:
        return jsonify({"success": False, "error": "ref_id is required"}), 400
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, status, answer, error, question, video_ids, ref_id
        FROM ask_jobs
        WHERE ref_id = ?
        LIMIT 1
        """,
        (rid,),
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        return jsonify({"success": False, "error": "Job not found", "status": "missing"}), 404
    _jid, status, answer_json, error_text, question, video_ids_json, row_ref = row
    if status != "completed":
        return jsonify(
            {
                "success": False,
                "status": status,
                "error": error_text or "Job not completed",
                "ref_id": row_ref,
            }
        ), 400
    try:
        payload = json.loads(answer_json or "{}")
    except Exception:
        payload = {"answer": "", "used_context": []}
    return jsonify(
        {
            "success": True,
            "status": status,
            "job_id": _jid,
            "ref_id": row_ref,
            "question": question or "",
            "video_ids": json.loads(video_ids_json) if video_ids_json else None,
            "answer": payload.get("answer", ""),
            "used_context": payload.get("used_context", []),
            "sources": payload.get("sources", []),
            "debug_events": payload.get("debug_events", []),
            "grounding_audit": payload.get("grounding_audit", []),
            "verification_excerpts": payload.get("verification_excerpts", []),
        }
    )


@app.route("/api/ask/result/<int:job_id>", methods=["GET"])
@app.route("/api/agent/result/<int:job_id>", methods=["GET"])
def api_ask_result(job_id: int):
    print(f"[ask_jobs] Polling result for job_id={job_id}")
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT status, answer, error, ref_id, updated_at
        FROM ask_jobs
        WHERE id = ?
        """,
        (job_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        print(f"[ask_jobs] job_id={job_id} not found in DB")
        return jsonify({"success": False, "error": "Job not found", "status": "missing"}), 404

    status, answer_json, error_text, ref_id, updated_at = row
    print(f"[ask_jobs] job_id={job_id} DB status={status!r}")

    if status == "running":
        age = _seconds_since_job_timestamp(updated_at)
        if age is not None and age > JOB_RUNNING_TIMEOUT_SEC:
            print(f"[ask_jobs] job_id={job_id} running timeout ({age:.0f}s) -> error")
            # Cooperative cancel: worker checks this and stops quickly.
            ev = _job_cancel_events.get(job_id)
            if ev is not None:
                ev.set()
            _update_job(job_id, status="error", error="timeout")
            return jsonify(
                {
                    "success": False,
                    "status": "error",
                    "error": "timeout",
                    "ref_id": ref_id,
                }
            )

    if status == "cancelled":
        return jsonify(
            {
                "success": False,
                "status": "cancelled",
                "error": "cancelled",
                "ref_id": ref_id,
            }
        )

    # If the worker crashed after writing the JSON file but before updating
    # SQLite, the row can be left as "error". Only then read the job file —
    # never when status is still "pending" or "running", or a stale file from
    # an earlier run (same job_id) can be mistaken for the current job.
    job_file = _jobs_dir / f"{job_id}.json"
    print(f"[ask_jobs] job_id={job_id} job_file={job_file} exists={job_file.exists()}")
    err_lower = (error_text or "").strip().lower()
    if status == "error" and job_file.exists() and err_lower != "timeout":
        try:
            file_json = job_file.read_text(encoding="utf-8")
            payload = json.loads(file_json or "{}")
        except Exception:
            payload = {"answer": "", "used_context": []}
        # Best-effort to sync DB state, but even if this fails we still return
        try:
            sync_ref = _allocate_ref_id()
            _update_job(
                job_id,
                status="completed",
                answer=file_json,
                error=None,
                ref_id=sync_ref,
            )
            ref_id = sync_ref
        except Exception as sync_err:
            print(f"[ask_shorty] Failed to sync job {job_id} from file to DB: {sync_err!r}")
        resp = {
            "success": True,
            "status": "completed",
            "answer": payload.get("answer", ""),
            "used_context": payload.get("used_context", []),
            "sources": payload.get("sources", []),
            "ref_id": ref_id,
            "grounding_audit": payload.get("grounding_audit", []),
            "verification_excerpts": payload.get("verification_excerpts", []),
        }
        print(f"[ask_jobs] job_id={job_id} returning completed (from file)")
        return jsonify(resp)

    if status in ("pending", "running"):
        print(f"[ask_jobs] job_id={job_id} still {status}, continuing to poll")
        return jsonify({"success": False, "status": status, "ref_id": ref_id})

    if status == "completed":
        raw = answer_json or "{}"
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("answer JSON is not an object")
        except Exception as parse_exc:
            print(f"[ask_jobs] job_id={job_id} invalid completed answer JSON: {parse_exc!r}")
            payload = {
                "answer": (
                    "Something went wrong reading this answer from storage (invalid or incomplete data). "
                    "Please run your question again."
                ),
                "used_context": [],
                "sources": [],
                "debug_events": [],
                "grounding_audit": [],
                "verification_excerpts": [],
            }
        resp = {
            "success": True,
            "status": status,
            "answer": payload.get("answer", ""),
            "used_context": payload.get("used_context", []),
            "sources": payload.get("sources", []),
            "ref_id": ref_id,
            "grounding_audit": payload.get("grounding_audit", []),
            "verification_excerpts": payload.get("verification_excerpts", []),
        }
        print(f"[ask_jobs] job_id={job_id} returning completed (from DB)")
        return jsonify(resp)

    # status == "error" and no file fallback (HTTP 200 so clients read JSON and stop polling)
    print(f"[ask_jobs] job_id={job_id} returning error: {error_text!r}")
    return jsonify(
        {
            "success": False,
            "status": status,
            "error": error_text or "Unknown error",
            "ref_id": ref_id,
        }
    )


@app.route("/api/ask/stream/<int:job_id>", methods=["GET"])
@app.route("/api/agent/stream/<int:job_id>", methods=["GET"])
def api_ask_stream(job_id: int):
    """Server-Sent Events stream for real-time pipeline debug output."""

    def generate():
        q = _job_events.get(job_id)
        if q is None:
            # Job already finished or never existed — emit a single done marker
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
            return

        while True:
            try:
                event = q.get(timeout=90)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "stream_end":
                    break
            except _queue.Empty:
                # Keep-alive ping so the connection doesn't time out
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/review/<video_id>", methods=["GET"])
def review_video(video_id: str):
    """Readable single-video review page for generated artifacts + transcript."""
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            v.title,
            v.channel,
            v.watch_date,
            t.shorty,
            t.text
        FROM videos v
        LEFT JOIN transcripts t ON t.video_id = v.video_id
        WHERE v.video_id = ?
        ORDER BY t.created_at DESC
        LIMIT 1
        """,
        (video_id,),
    )
    row = cursor.fetchone()
    if row:
        title, channel, watch_date, shorty, transcript_text = row
    else:
        conn.close()
        return render_template(
            "review_video.html",
            found=False,
            video_id=video_id,
            title=None,
            channel=None,
            watch_date=None,
            shorty=None,
            questions=[],
            entity_groups={},
            facts=[],
            transcript_text=None,
            bookmarklet_href="",
        )

    cursor.execute(
        """
        SELECT question
        FROM synthetic_questions
        WHERE video_id = ?
        ORDER BY created_at ASC
        """,
        (video_id,),
    )
    questions = [r[0] for r in cursor.fetchall() if (r[0] or "").strip()]

    cursor.execute(
        """
        SELECT name, type, aliases
        FROM entities
        WHERE video_id = ?
        ORDER BY name ASC
        """,
        (video_id,),
    )
    entity_rows = cursor.fetchall()

    cursor.execute(
        """
        SELECT subject, relation, object
        FROM facts
        WHERE video_id = ?
        ORDER BY id ASC
        """,
        (video_id,),
    )
    facts = [
        {
            "subject": (s or "").strip(),
            "relation": (r or "").strip(),
            "object": (o or "").strip(),
        }
        for (s, r, o) in cursor.fetchall()
        if (s or "").strip() and (r or "").strip() and (o or "").strip()
    ]
    conn.close()

    entity_groups = {
        "person": [],
        "organization": [],
        "location": [],
        "concept": [],
    }
    for name, raw_type, aliases_json in entity_rows:
        n = (name or "").strip()
        if not n:
            continue
        t = (raw_type or "").strip().lower()
        if t not in entity_groups:
            t = "concept"
        try:
            aliases_val = json.loads(aliases_json) if aliases_json else []
            if not isinstance(aliases_val, list):
                aliases_val = []
        except Exception:
            aliases_val = []
        aliases = [str(a).strip() for a in aliases_val if str(a).strip()]
        entity_groups[t].append({"name": n, "aliases": aliases})

    bookmarklet_href = (
        "javascript:(function(){var u=location.href,m=u.match(/[?&]v=([a-zA-Z0-9_-]{11})/)"
        "||u.match(/youtu\\.be\\/([a-zA-Z0-9_-]{11})/)"
        "||u.match(/\\/shorts\\/([a-zA-Z0-9_-]{11})/);"
        "if(!m){alert('No YouTube video ID found in URL.');return;}"
        "window.open('http://localhost:5001/review/'+m[1],'_blank');})();"
    )

    return render_template(
        "review_video.html",
        found=True,
        video_id=video_id,
        title=title,
        channel=channel,
        watch_date=watch_date,
        shorty=shorty,
        questions=questions,
        entity_groups=entity_groups,
        facts=facts,
        transcript_text=transcript_text,
        bookmarklet_href=bookmarklet_href,
    )


@app.route("/pending-review", methods=["GET"])
def pending_review():
    """Management page for pending shorty queue rows."""
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT DISTINCT
            pq.video_id,
            COALESCE(v.channel, '') AS channel,
            COALESCE(v.title, '') AS title
        FROM processing_queue pq
        LEFT JOIN videos v ON v.video_id = pq.video_id
        WHERE pq.task = 'shorty' AND pq.status = 'pending'
        ORDER BY channel ASC, title ASC, pq.video_id ASC
        """
    )
    rows = cursor.fetchall()
    conn.close()

    items = []
    for video_id, channel, title in rows:
        ad_flag = False
        if _looks_like_ad and _AdPendingVideo:
            try:
                ad_flag, _ = _looks_like_ad(
                    _AdPendingVideo(video_id=video_id, title=title, channel=channel)
                )
            except Exception:
                ad_flag = False
        items.append(
            {
                "video_id": video_id,
                "channel": channel,
                "title": title,
                "ad_flag": bool(ad_flag),
            }
        )

    return render_template("pending_review.html", videos=items)


@app.route("/api/pending-review/skip", methods=["POST"])
def api_pending_review_skip():
    """Mark pending shorty queue rows as permanently_failed for provided video IDs."""
    payload = request.get_json(silent=True) or {}
    raw_ids = payload.get("video_ids")
    if not isinstance(raw_ids, list):
        return jsonify({"success": False, "error": "video_ids must be a list"}), 400

    video_ids = []
    for v in raw_ids:
        s = str(v).strip()
        if s:
            video_ids.append(s)
    # preserve order, remove duplicates
    seen = set()
    video_ids = [v for v in video_ids if not (v in seen or seen.add(v))]

    if not video_ids:
        return jsonify({"success": True, "updated_rows": 0, "video_count": 0})

    placeholders = ",".join("?" for _ in video_ids)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reason = "Skipped from /pending-review"

    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()
    cursor.execute(
        f"""
        UPDATE processing_queue
        SET status = 'permanently_failed',
            completed_at = ?,
            error = ?
        WHERE task = 'shorty'
          AND status = 'pending'
          AND video_id IN ({placeholders})
        """,
        [now, reason, *video_ids],
    )
    updated = cursor.rowcount
    conn.commit()
    conn.close()

    return jsonify(
        {
            "success": True,
            "updated_rows": int(updated),
            "video_count": len(video_ids),
        }
    )


@app.route("/debug/videos", methods=["GET"])
def debug_videos():
    """List videos and whether they have Shorties, questions, and entities."""
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            v.video_id,
            v.title,
            v.channel,
            EXISTS (
                SELECT 1 FROM transcripts t
                WHERE t.video_id = v.video_id AND t.shorty IS NOT NULL
            ) AS has_shorty,
            (SELECT COUNT(*) FROM synthetic_questions sq WHERE sq.video_id = v.video_id) AS question_count,
            (SELECT COUNT(*) FROM entities e WHERE e.video_id = v.video_id) AS entity_count
        FROM videos v
        ORDER BY v.created_at DESC
        LIMIT 500
        """
    )
    rows = cursor.fetchall()
    conn.close()

    videos = [
        {
            "video_id": vid,
            "title": title,
            "channel": channel,
            "has_shorty": bool(has_shorty),
            "question_count": question_count,
            "entity_count": entity_count,
        }
        for (vid, title, channel, has_shorty, question_count, entity_count) in rows
    ]

    return render_template("debug_videos.html", videos=videos)


@app.route("/debug/video/<video_id>", methods=["GET"])
def debug_video(video_id: str):
    """Show Shorty, synthetic questions, and entities for a single video."""
    conn = _sqlite_connect(str(db.db_path))  # type: ignore[attr-defined]
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT v.title, v.channel, t.shorty
        FROM videos v
        LEFT JOIN transcripts t ON t.video_id = v.video_id
        WHERE v.video_id = ?
        ORDER BY t.created_at DESC
        LIMIT 1
        """,
        (video_id,),
    )
    row = cursor.fetchone()

    if row:
        title, channel, shorty = row
    else:
        title, channel, shorty = None, None, None

    cursor.execute(
        """
        SELECT question
        FROM synthetic_questions
        WHERE video_id = ?
        ORDER BY created_at ASC
        """,
        (video_id,),
    )
    questions = [r[0] for r in cursor.fetchall()]

    cursor.execute(
        """
        SELECT name, type, aliases
        FROM entities
        WHERE video_id = ?
        ORDER BY name ASC
        """,
        (video_id,),
    )
    entities = []
    for name, etype, aliases_json in cursor.fetchall():
        try:
            import json

            aliases = json.loads(aliases_json) if aliases_json else []
        except Exception:
            aliases = []
        entities.append(
            {
                "name": name,
                "type": etype,
                "aliases": aliases,
            }
        )

    conn.close()

    return render_template(
        "debug_video.html",
        video_id=video_id,
        title=title,
        channel=channel,
        shorty=shorty,
        questions=questions,
        entities=entities,
    )


def _log_registered_api_routes() -> None:
    """Startup diagnostic: confirm Fast Search and other API routes are on this app instance."""
    api_rules = sorted(
        {
            r.rule
            for r in app.url_map.iter_rules()
            if r.rule.startswith("/api/")
        }
    )
    has_fast = "/api/search_fast" in api_rules
    has_v2 = "/api/ask_v2" in api_rules
    here = Path(__file__).resolve()
    print(
        f"[ask-shorty-app] loaded {here}",
        flush=True,
    )
    print(
        f"[ask-shorty-app] /api/search_fast registered={has_fast} "
        f"| /api/ask_v2 registered={has_v2}",
        flush=True,
    )
    if not has_fast:
        print(
            "[ask-shorty-app] WARNING: /api/search_fast missing — "
            "stop the old server and restart: python ask_shorty_app.py",
            flush=True,
        )
    print(
        "[ask-shorty-app] API routes: " + ", ".join(api_rules),
        flush=True,
    )


@app.errorhandler(404)
def _api_json_not_found(e):
    """Return JSON (not HTML) for unknown /api/* paths."""
    path = (request.path or "").strip()
    if path.startswith("/api/"):
        return (
            jsonify(
                {
                    "success": False,
                    "error": f"API route not found: {path}",
                    "hint": (
                        "Restart Ask Shorty from ask_shorty_app.py if you "
                        "recently added routes. On startup you should see "
                        "/api/search_fast registered=True."
                    ),
                }
            ),
            404,
        )
    return e


_log_registered_api_routes()

if __name__ == "__main__":
    # Disable the Flask reloader on Windows to avoid noisy socket errors.
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False, threaded=True)

