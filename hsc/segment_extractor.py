#!/usr/bin/env python3
"""
Segment-level dense summaries from transcripts (HSC).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from anthropic_client import get_client

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"

SEGMENT_SYSTEM = """You write extremely dense segment summaries for video transcripts.

Rules:
- 1–3 sentences per segment maximum.
- Capture topic, claims, names, numbers, and outcomes.
- No filler or repetition.
"""

SEGMENT_USER_TEMPLATE = """Transcript excerpt (may be mid-video):

{excerpt}

Reply with ONLY a JSON object:
{{"summary": "<1-3 dense sentences>"}}
"""


def _split_by_chars(text: str, max_chars: int = 3200, overlap: int = 200) -> List[str]:
    """Split transcript into ~500–1000 token windows (char-based heuristic)."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        parts.append(text[start:end].strip())
        if end >= n:
            break
        start = max(0, end - overlap)
    return [p for p in parts if p]


def _estimate_times(
    idx: int,
    n_parts: int,
    timestamps: Optional[Sequence[Tuple[float, float]]],
) -> Tuple[Optional[float], Optional[float]]:
    if timestamps and idx < len(timestamps):
        return float(timestamps[idx][0]), float(timestamps[idx][1])
    return None, None


def _summarize_excerpt(excerpt: str) -> str:
    client = get_client()
    user = SEGMENT_USER_TEMPLATE.format(excerpt=excerpt[:120000])
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        temperature=0.2,
        system=SEGMENT_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    raw = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            raw += block.text
        elif isinstance(block, dict) and block.get("type") == "text":
            raw += block.get("text", "")
    raw = raw.strip()
    if "```" in raw:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            raw = m.group(1).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("summary"):
            return str(data["summary"]).strip()
    except json.JSONDecodeError:
        pass
    return raw[:2000].strip() or excerpt[:500]


def extract_segments(
    transcript_text: str,
    timestamps: Optional[Sequence[Tuple[float, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Produce segment records: start_time, end_time, text (source window), summary.

    Without timestamps, start/end are null; summaries still stored for retrieval.
    """
    text = (transcript_text or "").strip()
    if not text:
        return []

    windows = _split_by_chars(text, max_chars=3200, overlap=200)
    out: List[Dict[str, Any]] = []
    n = len(windows)
    for i, win in enumerate(windows):
        st, et = _estimate_times(i, n, timestamps)
        try:
            summary = _summarize_excerpt(win)
        except Exception as exc:
            logger.warning("Segment summarize failed window %s: %s", i, exc)
            summary = win[:800] + ("…" if len(win) > 800 else "")
        row: Dict[str, Any] = {
            "start_time": st,
            "end_time": et,
            "text": win,
            "summary": summary,
        }
        out.append(row)
    return out
