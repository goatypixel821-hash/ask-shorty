#!/usr/bin/env python3
"""
Flask app exposing the Ask Shorty UI and API.

Routes:
- GET /ask       -> HTML UI
- POST /api/ask  -> JSON answer from AskShorty
"""

from flask import Flask, render_template, request, jsonify

from ask_shorty import AskShorty
from transcript_database import TranscriptDatabase
import sqlite3


app = Flask(__name__)
engine = AskShorty()
db = TranscriptDatabase()


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

    try:
        result = engine.answer_question(question, video_ids=video_ids)
        return jsonify(
            {
                "success": True,
                "answer": result["answer"],
                "used_context": result.get("used_context", []),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


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

