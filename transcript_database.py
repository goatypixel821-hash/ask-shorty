#!/usr/bin/env python3
"""
Simple Transcript Database Manager
Creates and manages SQLite database for YouTube video transcripts + Shorties + entities + synthetic questions
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any


class TranscriptDatabase:
    def __init__(self, db_path: str = None):
        """Initialize transcript database"""
        if db_path is None:
            db_path = os.environ.get("ASK_SHORTY_DB_PATH") or "data/transcripts.db"
        self.db_path = db_path
        self.ensure_db_exists()

    def ensure_db_exists(self):
        """Create database and tables if they don't exist; apply lightweight migrations safely."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Videos table - tracks which videos have transcripts
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    video_id TEXT PRIMARY KEY,
                    title TEXT,
                    channel TEXT,
                    url TEXT,
                    has_transcript BOOLEAN DEFAULT FALSE,
                    transcript_fetched_at TIMESTAMP,
                    watch_date TEXT,
                    local_path TEXT,
                    json_metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            # Ensure extra columns exist (migration) for older schemas
            cursor.execute("PRAGMA table_info(videos)")
            video_cols = [col[1] for col in cursor.fetchall()]
            if "watch_date" not in video_cols:
                cursor.execute("ALTER TABLE videos ADD COLUMN watch_date TEXT")
            if "local_path" not in video_cols:
                cursor.execute("ALTER TABLE videos ADD COLUMN local_path TEXT")
            if "json_metadata" not in video_cols:
                cursor.execute("ALTER TABLE videos ADD COLUMN json_metadata TEXT")

            # Transcripts table - now stores transcript + Shorty
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT,
                    text TEXT,
                    language TEXT DEFAULT 'en',
                    confidence REAL,
                    shorty TEXT,
                    shorty_generated_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (video_id)
                )
                """
            )

            # Lightweight migration: add shorty columns if an older table exists
            cursor.execute("PRAGMA table_info(transcripts)")
            transcript_cols = [col[1] for col in cursor.fetchall()]
            if "shorty" not in transcript_cols:
                cursor.execute("ALTER TABLE transcripts ADD COLUMN shorty TEXT")
            if "shorty_generated_at" not in transcript_cols:
                cursor.execute("ALTER TABLE transcripts ADD COLUMN shorty_generated_at TIMESTAMP")

            # Entities table: per-video named entities and aliases
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,        -- person, organization, system, protocol, software, location
                    aliases TEXT,              -- JSON array of aliases as text
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (video_id)
                )
                """
            )

            # Synthetic questions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS synthetic_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    embedding_id TEXT,          -- ID used in ChromaDB for this question
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos (video_id)
                )
                """
            )

            # Indexes for scale
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_videos_channel ON videos(channel)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_videos_title ON videos(title)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_transcripts_video_id ON transcripts(video_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_transcripts_shorty_not_null "
                "ON transcripts(video_id) WHERE shorty IS NOT NULL"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_video_id ON entities(video_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_synthetic_questions_video_id "
                "ON synthetic_questions(video_id)"
            )

            # Processing queue for deferred LLM work
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    task TEXT NOT NULL,          -- shorty, synthetic_questions, entities
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error TEXT,
                    attempts INTEGER DEFAULT 0,
                    FOREIGN KEY (video_id) REFERENCES videos (video_id)
                )
                """
            )
            # Migration: add attempts column if missing (existing DBs)
            cursor.execute("PRAGMA table_info(processing_queue)")
            pq_columns = [row[1] for row in cursor.fetchall()]
            if "attempts" not in pq_columns:
                cursor.execute("ALTER TABLE processing_queue ADD COLUMN attempts INTEGER DEFAULT 0")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_processing_queue_status_created "
                "ON processing_queue(status, created_at)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_processing_queue_video_id "
                "ON processing_queue(video_id)"
            )

            # Knowledge-graph triples (subject–relation–object) extracted from Shorties
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    source TEXT DEFAULT 'shorty',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_video_id ON facts(video_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object)"
            )

            # Graph salience: persisted node/edge frequencies (rebuilt from facts)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS fact_nodes (
                    node TEXT PRIMARY KEY,
                    frequency INTEGER NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS fact_edges (
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    frequency INTEGER NOT NULL,
                    PRIMARY KEY (subject, relation, object)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS fact_frequency_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    stale INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            cursor.execute(
                "INSERT OR IGNORE INTO fact_frequency_meta (id, stale) VALUES (1, 1)"
            )

            # Global normalized graph (cross-video); rebuilt from facts
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS global_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_norm TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object_norm TEXT NOT NULL,
                    subject_raw TEXT NOT NULL,
                    object_raw TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    UNIQUE (subject_norm, relation, object_norm, video_id)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_global_facts_subject ON global_facts(subject_norm)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_global_facts_object ON global_facts(object_norm)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_global_facts_video ON global_facts(video_id)"
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_alias (
                    alias TEXT PRIMARY KEY,
                    canonical TEXT NOT NULL
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS global_graph_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    stale INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            cursor.execute(
                "INSERT OR IGNORE INTO global_graph_meta (id, stale) VALUES (1, 1)"
            )

            # HSC: time-bounded segment summaries (optional embedding JSON)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    summary TEXT,
                    embedding TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_segments_video_id ON segments(video_id)"
            )

            # HSC: structured incidents (causal events)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    title TEXT,
                    cause TEXT,
                    effect TEXT,
                    systems TEXT,
                    raw_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (video_id) REFERENCES videos(video_id)
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_video_id ON events(video_id)"
            )

            conn.commit()
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[OK] Transcript database ready: {self.db_path}"
            )

    def add_video(self, video_id: str, title: str, channel: str, url: str) -> bool:
        """Add a video to the database.

        Note: this sets created_at (when the video record was first seen/indexed),
        not the user's watch date.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Use local time instead of UTC
                local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO videos (video_id, title, channel, url, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (video_id, title, channel, url, local_time),
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"[ERROR] Error adding video {video_id}: {e}")
            return False

    def has_transcript(self, video_id: str) -> bool:
        """Check if video has a transcript"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT has_transcript FROM videos WHERE video_id = ?", (video_id,))
            result = cursor.fetchone()
            return result[0] if result else False

    def save_transcript(
        self,
        video_id: str,
        text: str,
        language: str = "en",
        confidence: float = 1.0,
    ) -> bool:
        """Save transcript text to database (Shorty is added later via separate update)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Insert transcript (use local time)
                local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    """
                    INSERT INTO transcripts (video_id, text, language, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (video_id, text, language, confidence, local_time),
                )

                # Update video status (use local time). We do NOT touch watch_date here;
                # that is reserved for "when the user watched/grabbed the video" and is
                # set explicitly by the grabber.
                local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    """
                    UPDATE videos
                    SET has_transcript = TRUE, transcript_fetched_at = ?
                    WHERE video_id = ?
                    """,
                    (local_time, video_id),
                )

                conn.commit()
                return True
        except Exception as e:
            print(f"[ERROR] Error saving transcript for {video_id}: {e}")
            return False

    def enqueue_processing_tasks(
        self,
        video_id: str,
        tasks: Optional[List[str]] = None,
    ) -> None:
        """
        Enqueue LLM processing tasks for a video.

        Tasks are strings: 'shorty', 'synthetic_questions', 'entities',
        'segments', 'events', 'triples' (HSC segment/event layers).
        """
        if tasks is None:
            tasks = [
                "shorty",
                "synthetic_questions",
                "entities",
                "segments",
                "events",
                "triples",
            ]
        if not tasks:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                for task in tasks:
                    cursor.execute(
                        """
                        INSERT INTO processing_queue (video_id, task, status)
                        VALUES (?, ?, 'pending')
                        """,
                        (video_id, task),
                    )
                conn.commit()
        except Exception as e:
            print(f"[ERROR] Error enqueuing processing tasks for {video_id}: {e}")

    def set_watch_date(self, video_id: str, when: Optional[datetime] = None) -> None:
        """
        Set the watch_date for a video to the given datetime (or now).

        watch_date represents when the USER watched or grabbed the video,
        not the upload date.
        """
        if when is None:
            when = datetime.now()
        ts = when.strftime("%Y-%m-%d %H:%M:%S")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE videos
                    SET watch_date = ?
                    WHERE video_id = ?
                    """,
                    (ts, video_id),
                )
                conn.commit()
        except Exception as e:
            print(f"[ERROR] Error setting watch_date for {video_id}: {e}")

    def save_shorty(self, video_id: str, shorty_text: str) -> bool:
        """Attach a Shorty to the most recent transcript row for a video."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    """
                    UPDATE transcripts
                    SET shorty = ?, shorty_generated_at = ?
                    WHERE id = (
                        SELECT id FROM transcripts
                        WHERE video_id = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    )
                    """,
                    (shorty_text, local_time, video_id),
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"[ERROR] Error saving Shorty for {video_id}: {e}")
            return False

    def get_transcript(self, video_id: str) -> Optional[str]:
        """Get transcript text for a video (latest version)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT text FROM transcripts WHERE video_id = ? ORDER BY created_at DESC LIMIT 1",
                (video_id,),
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def get_transcript_and_shorty(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get both transcript text and Shorty for a video (latest row)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT text, shorty, shorty_generated_at
                FROM transcripts
                WHERE video_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (video_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "text": row[0],
                "shorty": row[1],
                "shorty_generated_at": row[2],
            }

    def update_local_path(self, video_id: str, local_path: str) -> bool:
        """Update the local file path for a video"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE videos
                    SET local_path = ?
                    WHERE video_id = ?
                    """,
                    (local_path, video_id),
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"[ERROR] Error updating local path for {video_id}: {e}")
            return False

    def save_metadata(self, video_id: str, metadata: Dict[str, Any]) -> bool:
        """Save JSON metadata for a video"""
        import json

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    UPDATE videos
                    SET json_metadata = ?
                    WHERE video_id = ?
                    """,
                    (json.dumps(metadata), video_id),
                )
                conn.commit()
                return True
        except Exception as e:
            print(f"[ERROR] Error saving metadata for {video_id}: {e}")
            return False

    def get_video_info(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get video information"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Check if columns exist (for backward compatibility)
            try:
                cursor.execute("SELECT local_path, json_metadata FROM videos LIMIT 1")
                has_extra_cols = True
            except sqlite3.OperationalError:
                has_extra_cols = False

            query = """
                SELECT video_id, title, channel, url, has_transcript, transcript_fetched_at
            """

            if has_extra_cols:
                query += ", local_path, json_metadata "

            query += "FROM videos WHERE video_id = ?"

            cursor.execute(query, (video_id,))
            result = cursor.fetchone()

            if result:
                info = {
                    "video_id": result[0],
                    "title": result[1],
                    "channel": result[2],
                    "url": result[3],
                    "has_transcript": bool(result[4]),
                    "transcript_fetched_at": result[5],
                }

                if has_extra_cols and len(result) > 6:
                    info["local_path"] = result[6]
                    if result[7]:
                        import json

                        try:
                            info["metadata"] = json.loads(result[7])
                        except Exception:
                            info["metadata"] = {}

                return info
            return None

    def replace_facts_for_video(
        self,
        video_id: str,
        triples: List[Dict[str, Any]],
        source: str = "shorty",
    ) -> int:
        """Replace all facts for a video with new triples. Each triple: subject, relation, object."""
        n = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM facts WHERE video_id = ?", (video_id,))
            for t in triples:
                sub = (t.get("subject") or "").strip()
                rel = (t.get("relation") or "").strip()
                obj = (t.get("object") or "").strip()
                if not sub or not rel or not obj:
                    continue
                conf = float(t.get("confidence", 1.0) or 1.0)
                cursor.execute(
                    """
                    INSERT INTO facts (video_id, subject, relation, object, confidence, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (video_id, sub, rel, obj, conf, source),
                )
                n += 1
            conn.commit()
        self._mark_fact_frequency_stale()
        self._mark_global_graph_stale()
        return n

    def _mark_global_graph_stale(self) -> None:
        """Invalidate global_facts-derived graph; next load rebuilds."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO global_graph_meta (id, stale) VALUES (1, 1)"
                )
                conn.execute("UPDATE global_graph_meta SET stale = 1 WHERE id = 1")
                conn.commit()
        except sqlite3.OperationalError:
            pass

    def _mark_fact_frequency_stale(self) -> None:
        """Invalidate cached fact_nodes / fact_edges; next load rebuilds."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO fact_frequency_meta (id, stale) VALUES (1, 1)"
                )
                conn.execute(
                    "UPDATE fact_frequency_meta SET stale = 1 WHERE id = 1"
                )
                conn.commit()
        except sqlite3.OperationalError:
            pass

    def count_facts_for_video(self, video_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM facts WHERE video_id = ?", (video_id,))
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def get_facts_for_video(self, video_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, subject, relation, object, confidence, source
                FROM facts WHERE video_id = ? ORDER BY id
                """,
                (video_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def replace_segments_for_video(
        self,
        video_id: str,
        segments: List[Dict[str, Any]],
    ) -> int:
        """Replace segment rows for a video. Each dict: start_time, end_time, summary, embedding (optional)."""
        import json as _json

        n = 0
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
            for seg in segments:
                st = seg.get("start_time")
                et = seg.get("end_time")
                summ = (seg.get("summary") or "").strip()
                if not summ:
                    continue
                emb = seg.get("embedding")
                emb_s = None
                if emb is not None:
                    emb_s = emb if isinstance(emb, str) else _json.dumps(emb)
                cur.execute(
                    """
                    INSERT INTO segments (video_id, start_time, end_time, summary, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        video_id,
                        float(st) if st is not None else None,
                        float(et) if et is not None else None,
                        summ,
                        emb_s,
                    ),
                )
                n += 1
            conn.commit()
        return n

    def replace_events_for_video(
        self,
        video_id: str,
        events: List[Dict[str, Any]],
    ) -> int:
        """Replace event rows for a video."""
        import json as _json

        n = 0
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM events WHERE video_id = ?", (video_id,))
            for ev in events:
                title = (ev.get("title") or "").strip()
                if not title:
                    continue
                cause = (ev.get("cause") or "").strip()
                effect = (ev.get("effect") or "").strip()
                systems = ev.get("systems") or []
                if isinstance(systems, list):
                    sys_s = _json.dumps(systems)
                else:
                    sys_s = str(systems)
                raw = ev.get("raw_json")
                raw_s = raw if isinstance(raw, str) else _json.dumps(ev)
                cur.execute(
                    """
                    INSERT INTO events (video_id, title, cause, effect, systems, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (video_id, title, cause, effect, sys_s, raw_s),
                )
                n += 1
            conn.commit()
        return n

    def count_segments_for_video(self, video_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT COUNT(*) FROM segments WHERE video_id = ?", (video_id,))
                row = cur.fetchone()
                return int(row[0]) if row else 0
            except sqlite3.OperationalError:
                return 0

    def count_events_for_video(self, video_id: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            try:
                cur.execute("SELECT COUNT(*) FROM events WHERE video_id = ?", (video_id,))
                row = cur.fetchone()
                return int(row[0]) if row else 0
            except sqlite3.OperationalError:
                return 0

    def get_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM videos")
            total_videos = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM videos WHERE has_transcript = TRUE")
            videos_with_transcripts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM transcripts")
            total_transcripts = cursor.fetchone()[0]

            return {
                "total_videos": total_videos,
                "videos_with_transcripts": videos_with_transcripts,
                "total_transcripts": total_transcripts,
            }

    def search_transcripts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search transcripts by text content"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT v.video_id, v.title, v.channel, t.text
                FROM videos v
                JOIN transcripts t ON v.video_id = t.video_id
                WHERE t.text LIKE ?
                ORDER BY v.title
                LIMIT ?
                """,
                (f"%{query}%", limit),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "video_id": row[0],
                        "title": row[1],
                        "channel": row[2],
                        "text": row[3],
                    }
                )
            return results


def main():
    """Test the transcript database"""
    print("🚀 Testing Transcript Database")
    print("=" * 50)

    # Initialize database (creates/updates schema)
    db = TranscriptDatabase()

    # Test adding a video
    test_video_id = "test123"
    db.add_video(
        test_video_id,
        "Test Video",
        "Test Channel",
        "https://youtube.com/watch?v=test123",
    )

    # Test saving transcript
    test_transcript = "This is a test transcript for the video."
    db.save_transcript(test_video_id, test_transcript)

    # Test Shorty save
    db.save_shorty(test_video_id, "COMPRESSED TRANSCRIPT — test shorty")

    # Test retrieval
    has_transcript = db.has_transcript(test_video_id)
    transcript_info = db.get_transcript_and_shorty(test_video_id)

    print(f"[OK] Video has transcript: {has_transcript}")
    print(f"[OK] Transcript text: {transcript_info['text'][:50]}...")
    print(f"[OK] Shorty: {transcript_info['shorty']}")

    # Get stats
    stats = db.get_stats()
    print(f"📊 Database stats: {stats}")

    print("🎉 Transcript database test completed!")


if __name__ == "__main__":
    main()

