#!/usr/bin/env python3
"""RAM maps for Ask Shorty V2 (entity aliases, topics, signatures, segment lists)."""

from __future__ import annotations

import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set


def _approx_obj_bytes(obj: object) -> int:
    return len(str(obj).encode("utf-8", errors="replace"))


@dataclass
class V2MemoryMaps:
    db_path: str
    alias_to_videos: Dict[str, Set[str]] = field(default_factory=dict)
    topic_to_videos: Dict[str, Set[str]] = field(default_factory=dict)
    video_routing_text: Dict[str, str] = field(default_factory=dict)
    video_to_segment_ids: Dict[str, List[int]] = field(default_factory=dict)
    load_time_s: float = 0.0
    approx_bytes: int = 0

    @classmethod
    def load(cls, db_path: str) -> "V2MemoryMaps":
        t0 = time.perf_counter()
        mm = cls(db_path=db_path)
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute(
                "SELECT alias, video_id FROM entity_postings WHERE alias IS NOT NULL"
            )
            for r in cur.fetchall():
                al = (r["alias"] or "").strip().lower()
                if not al:
                    continue
                mm.alias_to_videos.setdefault(al, set()).add(str(r["video_id"]))

            cur.execute(
                "SELECT DISTINCT topic_phrase, video_id FROM topic_postings"
            )
            for r in cur.fetchall():
                ph = (r["topic_phrase"] or "").strip().lower()
                if not ph:
                    continue
                mm.topic_to_videos.setdefault(ph, set()).add(str(r["video_id"]))

            cur.execute("SELECT video_id, routing_text FROM video_signatures")
            for r in cur.fetchall():
                mm.video_routing_text[str(r["video_id"])] = r["routing_text"] or ""

            cur.execute(
                "SELECT video_id, segment_id FROM segment_index ORDER BY video_id, segment_id"
            )
            for r in cur.fetchall():
                vid = str(r["video_id"])
                mm.video_to_segment_ids.setdefault(vid, []).append(int(r["segment_id"]))

        mm.load_time_s = time.perf_counter() - t0
        mm.approx_bytes = (
            _approx_obj_bytes(mm.alias_to_videos)
            + _approx_obj_bytes(mm.topic_to_videos)
            + sum(len(v.encode("utf-8")) for v in mm.video_routing_text.values())
            + _approx_obj_bytes(mm.video_to_segment_ids)
        )
        print(
            "[V2MemoryMaps] loaded",
            f"db={db_path}",
            f"aliases={len(mm.alias_to_videos)} topics={len(mm.topic_to_videos)}",
            f"videos={len(mm.video_routing_text)} seg_lists={len(mm.video_to_segment_ids)}",
            f"~{mm.approx_bytes / 1e6:.2f} MB (rough string size)",
            f"time={mm.load_time_s:.3f}s",
            file=sys.stderr,
        )
        return mm

    def instant_candidates(self, query_lower: str, max_per_channel: int = 200) -> List[str]:
        """Cheap lexical overlap: union alias + topic hits for query tokens."""
        cands: Set[str] = set()
        # Word tokens, keeping inner '.'/'-' (versions, "kestrel-migrate"). The old
        # str.split() kept trailing punctuation, so "tidewater?" never matched.
        all_toks = re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", (query_lower or "").lower())
        toks = [t for t in all_toks if len(t) >= 3][:24]
        for t in toks:
            if t in self.alias_to_videos:
                cands |= self.alias_to_videos[t]
        # phrase scan: 2-grams against topic phrases and multi-word aliases
        # ("marcus oyelaran" was unreachable via single-token lookups).
        for i in range(len(all_toks) - 1):
            bg = f"{all_toks[i]} {all_toks[i+1]}"
            if bg in self.topic_to_videos:
                cands |= self.topic_to_videos[bg]
            if bg in self.alias_to_videos:
                cands |= self.alias_to_videos[bg]
        # Sorted: set order varies with hash seed, which made the capped candidate
        # pool (and so answers) differ between runs.
        return sorted(cands)[:max_per_channel]
