#!/usr/bin/env python3
"""
Keyword-based query routing for HSC (extensible to LLM later).
"""

from __future__ import annotations

import re
from typing import Dict, Literal, Any

QueryRouteType = Literal["fact", "event", "summary", "raw"]


def route_query(query: str) -> Dict[str, Any]:
    """
    Classify query for targeted HSC retrieval.

    Returns:
        {"type": "fact" | "event" | "summary" | "raw", "reason": str}
    """
    q = (query or "").strip().lower()
    if not q:
        return {"type": "summary", "reason": "empty_default"}

    # Quote / verbatim
    if re.search(
        r"\b(exactly said|verbatim|quote|said word for word|transcript says)\b", q
    ):
        return {"type": "raw", "reason": "quote_keywords"}

    # Causal / why
    if re.search(
        r"\b(what caused|why did|because of|root cause|what led to|due to)\b", q
    ):
        return {"type": "fact", "reason": "causal_keywords"}

    # Events / incidents
    if re.search(
        r"\b(what happened|incident|bug|crash|failure|discovered|broke|affected)\b", q
    ):
        return {"type": "event", "reason": "event_keywords"}

    # Location / where in video (segment)
    if re.search(r"\b(where (was|did)|which part|timestamp|timecode)\b", q):
        return {"type": "summary", "reason": "location_segment_keywords"}

    # Default: shorty / vector
    return {"type": "summary", "reason": "default_shorty_vector"}
