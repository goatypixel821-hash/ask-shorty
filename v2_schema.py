#!/usr/bin/env python3
"""
V2 hierarchical RAG — SQLite DDL (additive; same DB as transcripts).
"""

from __future__ import annotations

import sqlite3
from typing import Any


def ensure_v2_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS video_signatures (
            video_id TEXT PRIMARY KEY,
            routing_text TEXT,
            top_entities TEXT,
            top_topics TEXT,
            has_timestamps INTEGER DEFAULT 0,
            watch_date TEXT,
            channel TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_video_signatures_channel ON video_signatures(channel)"
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS segment_index (
            segment_id INTEGER PRIMARY KEY,
            video_id TEXT NOT NULL,
            summary TEXT,
            clean_text TEXT,
            start_s REAL,
            end_s REAL,
            keywords_json TEXT,
            entities_json TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_segment_index_video_id ON segment_index(video_id)"
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS event_index (
            event_id INTEGER PRIMARY KEY,
            video_id TEXT NOT NULL,
            title TEXT,
            cause TEXT,
            effect TEXT,
            systems TEXT,
            start_s REAL,
            end_s REAL
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_event_index_video_id ON event_index(video_id)"
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS entity_postings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT NOT NULL,
            alias TEXT,
            video_id TEXT NOT NULL,
            segment_id INTEGER,
            count INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_entity_postings_name ON entity_postings(entity_name)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_entity_postings_alias ON entity_postings(alias)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_entity_postings_video ON entity_postings(video_id)"
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS topic_postings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_phrase TEXT NOT NULL,
            video_id TEXT NOT NULL,
            segment_id INTEGER,
            score REAL NOT NULL DEFAULT 1.0
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_topic_postings_phrase ON topic_postings(topic_phrase)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_topic_postings_video ON topic_postings(video_id)"
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash TEXT PRIMARY KEY,
            route_type TEXT,
            shortlist_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ttl_seconds INTEGER NOT NULL DEFAULT 3600
        )
        """
    )

    conn.commit()


def table_row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    names = [
        "video_signatures",
        "segment_index",
        "event_index",
        "entity_postings",
        "topic_postings",
        "query_cache",
    ]
    cur = conn.cursor()
    out: dict[str, int] = {}
    for n in names:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {n}")
            out[n] = int(cur.fetchone()[0])
        except Exception:
            out[n] = -1
    return out
