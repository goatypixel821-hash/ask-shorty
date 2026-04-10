#!/usr/bin/env python3
"""
Structured event / incident extraction (causal bugs, actions, discoveries).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from anthropic_client import get_client

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"

EVENT_SYSTEM = """You extract structured EVENTS from video content.

Each event captures something noteworthy: bugs, failures, discoveries, actions, decisions.
Focus on CAUSAL relationships when present.

Return ONLY a JSON array. Each object:
{
  "title": "short label",
  "cause": "what led to it or empty string",
  "effect": "what happened as a result or empty string",
  "systems": ["optional", "tags", "like", "CUDA"]
}

Rules:
- 3–12 events per video when content supports it; fewer if sparse.
- systems: technologies, products, orgs mentioned (can be empty []).
- Never invent facts not supported by the text.
- Return [] if nothing clear.
"""

EVENT_USER_TEMPLATE = """Title: {title}

Transcript (excerpt):
{transcript}

---
Optional condensed summary (Shorty):
{shorty}
"""


def parse_events_json(raw: str) -> List[Dict[str, Any]]:
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if not title:
            continue
        systems = item.get("systems") or []
        if not isinstance(systems, list):
            systems = [str(systems)]
        out.append(
            {
                "title": title,
                "cause": (item.get("cause") or "").strip(),
                "effect": (item.get("effect") or "").strip(),
                "systems": [str(s).strip() for s in systems if s],
                "raw_json": json.dumps(item, ensure_ascii=False),
            }
        )
    return out


def extract_events(
    transcript_text: str,
    title: str = "",
    shorty_text: str = "",
) -> List[Dict[str, Any]]:
    """LLM extraction; returns rows ready for DB (includes raw_json)."""
    client = get_client()
    tr = (transcript_text or "")[:100000]
    sh = (shorty_text or "")[:80000]
    user = EVENT_USER_TEMPLATE.format(
        title=title or "Untitled",
        transcript=tr,
        shorty=sh or "(none)",
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0.15,
        system=EVENT_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    raw = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            raw += block.text
        elif isinstance(block, dict) and block.get("type") == "text":
            raw += block.get("text", "")
    events = parse_events_json(raw)
    for ev in events:
        if "raw_json" not in ev:
            ev["raw_json"] = json.dumps(
                {k: v for k, v in ev.items() if k != "raw_json"},
                ensure_ascii=False,
            )
    return events
