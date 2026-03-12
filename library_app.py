#!/usr/bin/env python3
"""
Library browser and admin panel for Ask Shorty.

Runs as a standalone Flask app on port 5002.
Connects directly to the existing SQLite DB at:
  C:\\Users\\number2\\Desktop\\shorty\\data\\transcripts.db
"""

import os
import math
import sqlite3
from typing import List, Dict, Any, Optional

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)

from shorty_generator import generate_shorty
from shorty_generator import generate_synthetic_questions
from entity_extractor import extract_entities, store_entities


DB_PATH = os.path.join(os.path.dirname(__file__), "data", "transcripts.db")
PAGE_SIZE = 50


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_channels() -> List[str]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT channel FROM videos WHERE channel IS NOT NULL AND channel != '' ORDER BY channel")
    rows = cur.fetchall()
    conn.close()
    return [r["channel"] for r in rows]


def get_video_counts(filters: Dict[str, Any]) -> int:
    conn = get_db_connection()
    cur = conn.cursor()

    where_clauses = []
    params: List[Any] = []

    if filters.get("channel"):
        where_clauses.append("v.channel = ?")
        params.append(filters["channel"])

    status = filters.get("status")
    if status == "has_shorty":
        where_clauses.append(
            "EXISTS (SELECT 1 FROM transcripts t WHERE t.video_id = v.video_id AND t.shorty IS NOT NULL)"
        )
    elif status == "missing_shorty":
        where_clauses.append(
            "NOT EXISTS (SELECT 1 FROM transcripts t WHERE t.video_id = v.video_id AND t.shorty IS NOT NULL)"
        )

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    cur.execute(
        f"""
        SELECT COUNT(*)
        FROM videos v
        {where_sql}
        """,
        params,
    )
    count = cur.fetchone()[0]
    conn.close()
    return int(count)


def get_videos_page(filters: Dict[str, Any], page: int) -> List[sqlite3.Row]:
    conn = get_db_connection()
    cur = conn.cursor()

    where_clauses = []
    params: List[Any] = []

    if filters.get("channel"):
        where_clauses.append("v.channel = ?")
        params.append(filters["channel"])

    status = filters.get("status")
    if status == "has_shorty":
        where_clauses.append(
            "EXISTS (SELECT 1 FROM transcripts t WHERE t.video_id = v.video_id AND t.shorty IS NOT NULL)"
        )
    elif status == "missing_shorty":
        where_clauses.append(
            "NOT EXISTS (SELECT 1 FROM transcripts t WHERE t.video_id = v.video_id AND t.shorty IS NOT NULL)"
        )

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    offset = (page - 1) * PAGE_SIZE

    cur.execute(
        f"""
        SELECT
            v.video_id,
            v.title,
            v.channel,
            v.watch_date,
            (SELECT json_extract(v2.json_metadata, '$.upload_date')
             FROM videos v2 WHERE v2.video_id = v.video_id) AS upload_date,
            EXISTS (
                SELECT 1 FROM transcripts t
                WHERE t.video_id = v.video_id AND t.shorty IS NOT NULL
            ) AS has_shorty,
            (SELECT COUNT(*) FROM synthetic_questions sq WHERE sq.video_id = v.video_id) AS question_count,
            (SELECT COUNT(*) FROM entities e WHERE e.video_id = v.video_id) AS entity_count
        FROM videos v
        {where_sql}
        ORDER BY v.created_at DESC
        LIMIT ? OFFSET ?
        """,
        params + [PAGE_SIZE, offset],
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_video_detail(video_id: str) -> Dict[str, Any]:
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            v.video_id,
            v.title,
            v.channel,
            v.watch_date,
            json_extract(v.json_metadata, '$.upload_date') AS upload_date
        FROM videos v
        WHERE v.video_id = ?
        """,
        (video_id,),
    )
    video_row = cur.fetchone()

    cur.execute(
        """
        SELECT id, text, shorty
        FROM transcripts
        WHERE video_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (video_id,),
    )
    transcript_row = cur.fetchone()

    cur.execute(
        """
        SELECT id, question
        FROM synthetic_questions
        WHERE video_id = ?
        ORDER BY created_at ASC
        """,
        (video_id,),
    )
    question_rows = cur.fetchall()

    cur.execute(
        """
        SELECT id, name, type, aliases
        FROM entities
        WHERE video_id = ?
        ORDER BY name ASC
        """,
        (video_id,),
    )
    entity_rows = cur.fetchall()

    # Normalize entities so aliases are displayed as a simple comma-separated string
    import json

    entities: List[Dict[str, Any]] = []
    for row in entity_rows:
        raw_aliases = row["aliases"]
        display_aliases = ""
        if raw_aliases:
            try:
                parsed = json.loads(raw_aliases)
                if isinstance(parsed, list):
                    display_aliases = ", ".join(str(a) for a in parsed)
                else:
                    display_aliases = str(parsed)
            except Exception:
                display_aliases = str(raw_aliases)
        entities.append(
            {
                "id": row["id"],
                "name": row["name"],
                "type": row["type"],
                "aliases": display_aliases,
            }
        )

    conn.close()

    return {
        "video": video_row,
        "transcript": transcript_row,
        "questions": question_rows,
        "entities": entities,
    }


app = Flask(__name__, template_folder=os.path.join("templates", "library"))
app.secret_key = "ask-shorty-library-secret"  # For flash messages; can be any string.


@app.route("/", methods=["GET"])
def library_index():
    channel = request.args.get("channel") or ""
    status = request.args.get("status") or "all"
    page = max(int(request.args.get("page", "1") or "1"), 1)

    filters = {
        "channel": channel if channel else None,
        "status": status if status in ("has_shorty", "missing_shorty") else "all",
    }

    total_count = get_video_counts(filters)
    total_pages = max(math.ceil(total_count / PAGE_SIZE), 1)
    if page > total_pages:
        page = total_pages

    videos = get_videos_page(filters, page)
    channels = get_channels()

    return render_template(
        "index.html",
        videos=videos,
        channels=channels,
        current_channel=channel,
        current_status=status,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
    )


@app.route("/video/<video_id>", methods=["GET"])
def video_detail_view(video_id: str):
    detail = get_video_detail(video_id)
    if not detail["video"]:
        return f"Video {video_id} not found", 404

    return render_template(
        "video.html",
        video_id=video_id,
        video=detail["video"],
        transcript=detail["transcript"],
        questions=detail["questions"],
        entities=detail["entities"],
    )


@app.route("/video/<video_id>/save/metadata", methods=["POST"])
def save_metadata(video_id: str):
    title = (request.form.get("title") or "").strip()
    channel = (request.form.get("channel") or "").strip()
    upload_date = (request.form.get("upload_date") or "").strip()

    conn = get_db_connection()
    cur = conn.cursor()

    # Update core fields
    cur.execute(
        """
        UPDATE videos
        SET title = ?, channel = ?
        WHERE video_id = ?
        """,
        (title, channel, video_id),
    )

    # Update JSON metadata upload_date
    cur.execute(
        """
        SELECT json_metadata
        FROM videos
        WHERE video_id = ?
        """,
        (video_id,),
    )
    row = cur.fetchone()
    import json

    if row is not None:
        raw = row[0]
        try:
            meta = json.loads(raw) if raw else {}
        except Exception:
            meta = {}
        if upload_date:
            meta["upload_date"] = upload_date
        else:
            meta.pop("upload_date", None)
        cur.execute(
            """
            UPDATE videos
            SET json_metadata = ?
            WHERE video_id = ?
            """,
            (json.dumps(meta), video_id),
        )

    conn.commit()
    conn.close()

    flash("Metadata saved.", "success")
    return redirect(url_for("video_detail_view", video_id=video_id))


@app.route("/video/<video_id>/save/shorty", methods=["POST"])
def save_shorty_text(video_id: str):
    shorty_text = (request.form.get("shorty") or "").strip()

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE transcripts
        SET shorty = ?
        WHERE video_id = ?
        """,
        (shorty_text, video_id),
    )
    conn.commit()
    conn.close()

    flash("Shorty saved.", "success")
    return redirect(url_for("video_detail_view", video_id=video_id))


@app.route("/video/<video_id>/save/transcript", methods=["POST"])
def save_transcript(video_id: str):
    """Update the raw transcript text for this video."""
    transcript_id = request.form.get("transcript_id")
    transcript_text = (request.form.get("transcript_text") or "").strip()

    if not transcript_id:
        flash("Missing transcript identifier.", "error")
        return redirect(url_for("video_detail_view", video_id=video_id))

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE transcripts
        SET text = ?
        WHERE id = ? AND video_id = ?
        """,
        (transcript_text, transcript_id, video_id),
    )
    conn.commit()
    conn.close()

    flash("Transcript saved.", "success")
    return redirect(url_for("video_detail_view", video_id=video_id))


@app.route("/video/<video_id>/regenerate/shorty", methods=["POST"])
def regenerate_shorty(video_id: str):
    detail = get_video_detail(video_id)
    video = detail["video"]
    if not video:
        return f"Video {video_id} not found", 404

    title = video["title"] or "Untitled Video"
    channel = video["channel"] or "Unknown Channel"

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT text FROM transcripts WHERE video_id = ? ORDER BY created_at DESC LIMIT 1",
        (video_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row or not row["text"]:
        flash("No transcript found to regenerate Shorty.", "error")
        return redirect(url_for("video_detail_view", video_id=video_id))

    transcript_text = row["text"]

    # Regenerate Shorty using existing helper (Anthropic/Haiku)
    shorty_text = generate_shorty(transcript_text, title=title, channel=channel)

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE transcripts
        SET shorty = ?
        WHERE video_id = ?
        """,
        (shorty_text, video_id),
    )
    conn.commit()
    conn.close()

    flash("Shorty regenerated.", "success")
    return redirect(url_for("video_detail_view", video_id=video_id))


@app.route("/video/<video_id>/regenerate/questions", methods=["POST"])
def regenerate_questions(video_id: str):
    detail = get_video_detail(video_id)
    video = detail["video"]
    if not video:
        return f"Video {video_id} not found", 404

    title = video["title"] or "Untitled Video"

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT text FROM transcripts WHERE video_id = ? ORDER BY created_at DESC LIMIT 1",
        (video_id,),
    )
    row = cur.fetchone()
    if not row or not row["text"]:
        conn.close()
        flash("No transcript found to regenerate questions.", "error")
        return redirect(url_for("video_detail_view", video_id=video_id))

    transcript_text = row["text"]

    # Delete existing questions
    cur.execute("DELETE FROM synthetic_questions WHERE video_id = ?", (video_id,))

    # Generate and insert new questions
    questions = generate_synthetic_questions(transcript_text, title=title)
    for q in questions:
        cur.execute(
            """
            INSERT INTO synthetic_questions (video_id, question, embedding_id)
            VALUES (?, ?, NULL)
            """,
            (video_id, q),
        )

    conn.commit()
    conn.close()

    flash(f"Regenerated {len(questions)} synthetic questions.", "success")
    return redirect(url_for("video_detail_view", video_id=video_id))


@app.route("/video/<video_id>/question/save/<int:question_id>", methods=["POST"])
def save_question(video_id: str, question_id: int):
    """Update the text of a single synthetic question."""
    question_text = (request.form.get("question_text") or "").strip()

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE synthetic_questions
        SET question = ?
        WHERE id = ? AND video_id = ?
        """,
        (question_text, question_id, video_id),
    )
    conn.commit()
    conn.close()

    flash("Question saved.", "success")
    return redirect(url_for("video_detail_view", video_id=video_id))


@app.route("/video/<video_id>/question/delete/<int:question_id>", methods=["POST"])
def delete_question(video_id: str, question_id: int):
    """Delete a single synthetic question."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM synthetic_questions WHERE id = ? AND video_id = ?",
        (question_id, video_id),
    )
    conn.commit()
    conn.close()

    flash("Question deleted.", "success")
    return redirect(url_for("video_detail_view", video_id=video_id))


@app.route("/video/<video_id>/regenerate/entities", methods=["POST"])
def regenerate_entities(video_id: str):
    detail = get_video_detail(video_id)
    video = detail["video"]
    if not video:
        return f"Video {video_id} not found", 404

    title = video["title"] or "Untitled Video"

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT text FROM transcripts WHERE video_id = ? ORDER BY created_at DESC LIMIT 1",
        (video_id,),
    )
    row = cur.fetchone()
    if not row or not row["text"]:
        conn.close()
        flash("No transcript found to regenerate entities.", "error")
        return redirect(url_for("video_detail_view", video_id=video_id))

    transcript_text = row["text"]

    # Delete existing entities
    cur.execute("DELETE FROM entities WHERE video_id = ?", (video_id,))
    conn.commit()
    conn.close()

    # Extract and store new entities
    entities = extract_entities(transcript_text, title=title)
    if entities:
        store_entities(video_id, entities)
        flash(f"Regenerated {len(entities)} entities.", "success")
    else:
        flash("No entities extracted from transcript.", "warning")

    return redirect(url_for("video_detail_view", video_id=video_id))


@app.route("/video/<video_id>/entity/delete/<int:entity_id>", methods=["POST"])
def delete_entity(video_id: str, entity_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM entities WHERE id = ? AND video_id = ?", (entity_id, video_id))
    conn.commit()
    conn.close()

    flash("Entity deleted.", "success")
    return redirect(url_for("video_detail_view", video_id=video_id))


@app.route("/video/<video_id>/entity/add", methods=["POST"])
def add_entity(video_id: str):
    # Accept both old and new field names for compatibility.
    name = (request.form.get("entity_name") or request.form.get("name") or "").strip()
    etype = (request.form.get("entity_type") or request.form.get("type") or "").strip()
    aliases_raw = (request.form.get("entity_aliases") or request.form.get("aliases") or "").strip()

    if not name:
        flash("Entity name is required.", "error")
        return redirect(url_for("video_detail_view", video_id=video_id))

    aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()]
    import json

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO entities (video_id, name, type, aliases)
        VALUES (?, ?, ?, ?)
        """,
        (video_id, name, etype or "manual", json.dumps(aliases)),
    )
    conn.commit()
    conn.close()

    flash("Entity added.", "success")
    return redirect(url_for("video_detail_view", video_id=video_id))


@app.route("/video/<video_id>/entity/save/<int:entity_id>", methods=["POST"])
def save_entity(video_id: str, entity_id: int):
    """Update an existing entity's name, type, and aliases."""
    name = (request.form.get("name") or "").strip()
    etype = (request.form.get("type") or "").strip()
    aliases_raw = (request.form.get("aliases") or "").strip()

    aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()]
    import json

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE entities
        SET name = ?, type = ?, aliases = ?
        WHERE id = ? AND video_id = ?
        """,
        (name, etype or "manual", json.dumps(aliases), entity_id, video_id),
    )
    conn.commit()
    conn.close()

    flash("Entity saved.", "success")
    return redirect(url_for("video_detail_view", video_id=video_id))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=False)

