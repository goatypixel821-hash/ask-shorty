#!/usr/bin/env python3
"""
Flask app exposing the Ask Shorty UI and API.

Routes:
- GET /ask               -> HTML UI
- POST /api/ask          -> enqueue question, return job_id
- GET /api/ask/result/<job_id> -> poll for answer
"""

from flask import Flask, render_template, request, jsonify

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

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


def _run_job(job_id: int, question: str, video_ids):
    """Background worker that runs AskShorty and stores the result.

    IMPORTANT: We first write the answer to a JSON file under data/jobs so
    that if the process crashes before the SQLite write, the polling API can
    still recover the result from disk.
    """
    try:
        _update_job(job_id, status="running")
        engine = get_engine()
        result = engine.answer_question(question, video_ids=video_ids)
        answer_json = json.dumps(
            {
                "answer": result.get("answer", ""),
                "used_context": result.get("used_context", []),
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
        _update_job(job_id, status="error", error=str(e))


@app.route("/ask", methods=["GET"])
def ask_page():
    return render_template("ask.html")


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
        }
        print(f"[ask_jobs] job_id={job_id} returning completed (from DB)")
        return jsonify(resp)

    # status == "error" and no file fallback
    print(f"[ask_jobs] job_id={job_id} returning error: {error_text!r}")
    return jsonify({"success": False, "status": status, "error": error_text or "Unknown error"}), 500


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
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)

