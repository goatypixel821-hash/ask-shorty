#!/usr/bin/env python3
"""Tests for watch-time annotation API (video_grabber /api/annotate, /api/tags)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Repo root on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import video_grabber as vg  # noqa: E402
from transcript_database import TranscriptDatabase  # noqa: E402


class AnnotationApiTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db_path = Path(self._tmpdir.name) / "test_annotations.db"
        self.db = TranscriptDatabase(str(self.db_path))
        vg.db = self.db
        self.client = vg.app.test_client()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_annotate_existing_video(self):
        vid = "existingVid01"
        self.db.add_video(vid, "T", "C", f"https://youtube.com/watch?v={vid}")
        r = self.client.post(
            "/api/annotate",
            json={
                "video_id": vid,
                "timestamp_seconds": 10.5,
                "note_text": "first mark",
                "tags": ["idea"],
            },
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue(data["success"])
        self.assertFalse(data["video_created"])
        self.assertEqual(self.db.count_annotations(vid), 1)

    def test_annotate_new_video_creates_row(self):
        vid = "brandNewVid99"
        self.assertFalse(self.db.video_exists(vid))
        r = self.client.post(
            "/api/annotate",
            json={
                "video_id": vid,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "title": "New Title",
                "channel": "New Chan",
                "timestamp_seconds": 0,
                "note_text": "hello",
                "tags": ["todo"],
            },
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(data["video_created"])
        self.assertTrue(self.db.video_exists(vid))
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT watch_date FROM videos WHERE video_id = ?", (vid,)
            ).fetchone()
        self.assertTrue(row and row[0])

    def test_annotate_twice_appends(self):
        vid = "appendVid0001"
        self.db.ensure_bare_video(vid, "T", "C", f"https://youtube.com/watch?v={vid}")
        for i in range(2):
            r = self.client.post(
                "/api/annotate",
                json={"video_id": vid, "timestamp_seconds": float(i * 10), "tags": ["a"]},
            )
            self.assertEqual(r.status_code, 200)
        self.assertEqual(self.db.count_annotations(vid), 2)

    def test_get_tags_distinct(self):
        v1, v2 = "tagsVid00001", "tagsVid00002"
        self.db.ensure_bare_video(v1, "T", "C", "u1")
        self.db.ensure_bare_video(v2, "T", "C", "u2")
        self.db.insert_annotation(v1, 1.0, tags=["idea", "todo"])
        self.db.insert_annotation(v2, 2.0, tags=["quote", "todo"])
        r = self.client.get("/api/tags")
        self.assertEqual(r.status_code, 200)
        tags = set(r.get_json()["tags"])
        self.assertEqual(tags, {"idea", "quote", "todo"})

    def test_malformed_missing_video_id(self):
        r = self.client.post(
            "/api/annotate",
            json={"timestamp_seconds": 5},
        )
        self.assertEqual(r.status_code, 400)
        self.assertFalse(r.get_json().get("success", True))
        self.assertEqual(self.db.count_annotations(), 0)

    def test_malformed_missing_timestamp(self):
        r = self.client.post(
            "/api/annotate",
            json={"video_id": "xYz12345678"},
        )
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.db.count_annotations(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
