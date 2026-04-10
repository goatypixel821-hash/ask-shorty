#!/usr/bin/env python3
"""
Flask app exposing the Ask Shorty UI and API.

Routes:
- GET /ask               -> HTML UI
- POST /api/ask          -> enqueue question, return job_id
- GET /api/ask/result/<job_id> -> poll for answer
"""

from flask import Flask, render_template, request, jsonify, Response, stream_with_context, after_this_request

import json
import queue as _queue
import shutil
import sqlite3
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import build_courses
import export_course

print("Step 1: importing AskShorty...")
from ask_shorty import AskShorty
from transcript_database import TranscriptDatabase


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


db = TranscriptDatabase()
_jobs_dir = Path(__file__).parent / "data" / "jobs"
_jobs_dir.mkdir(parents=True, exist_ok=True)

# Per-job SSE event queues.  Created in api_ask before the worker thread
# starts so a client can connect immediately and receive every event.
_job_events: Dict[int, _queue.Queue] = {}

# Course generation SSE (same pattern as /api/ask/stream)
_course_job_events: Dict[int, _queue.Queue] = {}
_course_job_next_id = 1
_course_job_lock = threading.Lock()


def _ensure_jobs_table() -> None:
    """Create ask_jobs table if it doesn't exist."""
    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
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
            updated_at TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def _cleanup_stale_jobs() -> None:
    """
    On app startup, mark any jobs that were left in pending/running state
    as error, since the previous process likely crashed during generation.
    """
    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
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
_cleanup_stale_jobs()


def _update_job(job_id: int, **fields) -> None:
    """Helper to update a job row safely from worker thread."""
    if not fields:
        return
    # Use a short timeout so we don't hang indefinitely on a locked DB.
    conn = sqlite3.connect(db.db_path, timeout=5.0)  # type: ignore[attr-defined]
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
            question, video_ids=video_ids, emit=_make_emit(job_id)
        )
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
        _update_job(job_id, status="completed", answer=answer_json, error=None)
        print(f"[ask] Step F3: DB updated for job_id={job_id}, worker done")
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


@app.route("/ask", methods=["GET"])
def ask_page():
    return render_template("ask.html")


# ── Knowledge Observatory ─────────────────────────────────────────────────────

_CLUSTERS_PATH = Path(__file__).parent / "data" / "clusters.json"
_FULL_DB = "C:/Users/number2/Desktop/youtube-history-viewer-copy/data/transcripts.db"


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
        conn = sqlite3.connect(_FULL_DB)
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
        conn2 = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
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
            conn = sqlite3.connect(_FULL_DB)
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
        conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
        c = conn.cursor()
        c.execute("""
            SELECT id, question, status, created_at, answer, video_ids
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
        job_id, question, status, created_at, answer_json, video_ids_json = row
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
    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
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

    # Kick off background worker
    thread = threading.Thread(target=_run_job, args=(job_id, question, video_ids), daemon=True)
    thread.start()

    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/ask/result/<int:job_id>", methods=["GET"])
def api_ask_result(job_id: int):
    print(f"[ask_jobs] Polling result for job_id={job_id}")
    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT status, answer, error
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

    status, answer_json, error_text = row
    print(f"[ask_jobs] job_id={job_id} DB status={status!r}")

    # If DB doesn't show a completed answer yet, but a job file exists,
    # treat it as completed (process likely crashed before DB write).
    job_file = _jobs_dir / f"{job_id}.json"
    print(f"[ask_jobs] job_id={job_id} job_file={job_file} exists={job_file.exists()}")
    if status in ("pending", "running", "error") and job_file.exists():
        try:
            file_json = job_file.read_text(encoding="utf-8")
            payload = json.loads(file_json or "{}")
        except Exception:
            payload = {"answer": "", "used_context": []}
        # Best-effort to sync DB state, but even if this fails we still return
        try:
            _update_job(
                job_id,
                status="completed",
                answer=file_json,
                error=None,
            )
        except Exception as sync_err:
            print(f"[ask_shorty] Failed to sync job {job_id} from file to DB: {sync_err!r}")
        resp = {
            "success": True,
            "status": "completed",
            "answer": payload.get("answer", ""),
            "used_context": payload.get("used_context", []),
            "sources": payload.get("sources", []),
        }
        print(f"[ask_jobs] job_id={job_id} returning completed (from file)")
        return jsonify(resp)

    if status in ("pending", "running"):
        print(f"[ask_jobs] job_id={job_id} still {status}, continuing to poll")
        return jsonify({"success": False, "status": status})

    if status == "completed":
        try:
            payload = json.loads(answer_json or "{}")
        except Exception:
            payload = {"answer": "", "used_context": []}
        resp = {
            "success": True,
            "status": status,
            "answer": payload.get("answer", ""),
            "used_context": payload.get("used_context", []),
            "sources": payload.get("sources", []),
        }
        print(f"[ask_jobs] job_id={job_id} returning completed (from DB)")
        return jsonify(resp)

    # status == "error" and no file fallback
    print(f"[ask_jobs] job_id={job_id} returning error: {error_text!r}")
    return jsonify({"success": False, "status": status, "error": error_text or "Unknown error"}), 500


@app.route("/api/ask/stream/<int:job_id>", methods=["GET"])
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


@app.route("/debug/videos", methods=["GET"])
def debug_videos():
    """List videos and whether they have Shorties, questions, and entities."""
    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
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
    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
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


if __name__ == "__main__":
    # Disable the Flask reloader on Windows to avoid noisy socket errors.
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False, threaded=True)

