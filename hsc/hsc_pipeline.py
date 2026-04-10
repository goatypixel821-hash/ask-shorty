#!/usr/bin/env python3
"""
End-to-end HSC processing for one video (segments + events + optional triples).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from transcript_database import TranscriptDatabase

logger = logging.getLogger(__name__)


def process_video_hsc(db: TranscriptDatabase, video_id: str) -> Dict[str, Any]:
    """
    1. Load transcript (and Shorty if present)
    2. Generate and store segments
    3. Extract and store events
    4. Ensure triples exist (extract if facts table empty for this video)
    """
    from hsc.segment_extractor import extract_segments
    from hsc.event_extractor import extract_events
    from triple_extractor import extract_triples

    info = db.get_transcript_and_shorty(video_id)
    if not info or not (info.get("text") or "").strip():
        raise ValueError(f"No transcript text for {video_id}")

    text = info["text"].strip()
    shorty = (info.get("shorty") or "").strip()
    vinfo = db.get_video_info(video_id) or {}
    title = (vinfo.get("title") or video_id) if isinstance(vinfo, dict) else video_id

    # Segments
    seg_list = extract_segments(text, timestamps=None)
    rows = [
        {
            "start_time": s.get("start_time"),
            "end_time": s.get("end_time"),
            "summary": s.get("summary") or "",
        }
        for s in seg_list
        if (s.get("summary") or "").strip()
    ]
    n_seg = db.replace_segments_for_video(video_id, rows)

    # Events (use Shorty when available for richer extraction)
    ev_list = extract_events(text, title=title, shorty_text=shorty)
    n_ev = db.replace_events_for_video(video_id, ev_list)

    # Triples if missing
    n_triples = 0
    if shorty and db.count_facts_for_video(video_id) == 0:
        try:
            triples = extract_triples(shorty, title)
            n_triples = db.replace_facts_for_video(video_id, triples)
        except Exception as exc:
            logger.warning("Triple extraction in HSC pipeline failed: %s", exc)

    return {
        "video_id": video_id,
        "segments_stored": n_seg,
        "events_stored": n_ev,
        "triples_stored": n_triples,
    }
