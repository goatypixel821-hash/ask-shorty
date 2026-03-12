#!/usr/bin/env python3
"""
Entity and alias extraction for Ask Shorty.

Uses Anthropic Claude to extract structured entities from a transcript and
stores them in the SQLite `entities` table.
"""

from typing import List, Dict, Any, Optional
import json
import logging

from anthropic_client import get_client
from transcript_database import TranscriptDatabase
import sqlite3


ENTITY_MODEL = "claude-3-haiku-20240307"

logger = logging.getLogger(__name__)


ENTITY_SYSTEM_PROMPT = """You extract structured entities from video transcripts.

For each distinct entity, return:
- name: canonical name
- type: a short label describing what it is, such as person, organization, company, agency, system, infrastructure, protocol, software, application, product, location, country, city, or concept.
- aliases: array of alternate names, abbreviations, codenames, or nicknames (can be empty)

Requirements:
- Prefer specific, concrete entities that are important to understanding the video.
- Merge clear aliases into the same entity (e.g., DCS-3000 and Red Hook).
- Be generous: include people, organizations, products, software, protocols, locations, and important concepts.
- Output ONLY a JSON array of entity objects, nothing else.
"""


ENTITY_USER_TEMPLATE = """Extract entities from this transcript.

Transcript metadata:
- Title: {title}

Transcript:
{transcript}
"""


def _call_claude_entities(system_prompt: str, user_prompt: str) -> List[Dict[str, Any]]:
    client = get_client()
    resp = client.messages.create(
        model=ENTITY_MODEL,
        max_tokens=2048,
        temperature=0.1,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    parts: List[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    raw = "\n".join(parts).strip()

    # Claude sometimes wraps JSON in markdown code fences like ```json ... ```.
    # Strip those fences before attempting to parse.
    if raw.startswith("```"):
        # Drop leading ```... line
        raw = raw.split("\n", 1)[-1]
        # Drop trailing ``` if present
        raw = raw.rsplit("```", 1)[0].strip()

    # Claude can also prepend explanatory text before the JSON array.
    # Locate the first '[' and last ']' and keep only that slice.
    start = raw.find('[')
    end = raw.rfind(']')
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]

    def _normalize_type(raw_type: str) -> Optional[str]:
        t = raw_type.strip().lower()
        if t in ["person", "per", "human", "individual"]:
            return "person"
        if t in ["organization", "organisation", "org", "company", "agency"]:
            return "organization"
        if t in ["system", "sys", "infrastructure", "platform"]:
            return "system"
        if t in ["protocol", "standard"]:
            return "protocol"
        if t in ["software", "app", "application", "program"]:
            return "software"
        if t in ["location", "place", "city", "country", "region"]:
            return "location"
        if t in ["concept", "idea", "topic", "theme"]:
            return "concept"
        return None

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            cleaned: List[Dict[str, Any]] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                etype_raw = str(item.get("type", "")).strip()
                aliases = item.get("aliases") or []
                if not name or not etype_raw:
                    continue
                etype = _normalize_type(etype_raw)
                if not etype:
                    continue
                aliases = [str(a).strip() for a in aliases if isinstance(a, str) and a.strip()]
                cleaned.append(
                    {
                        "name": name,
                        "type": etype,
                        "aliases": aliases,
                    }
                )
            if not cleaned:
                logger.info("Entity extractor returned an empty list after parsing. Raw payload: %s", raw)
            return cleaned
    except Exception as e:
        logger.warning("Failed to parse entity extractor response: %s; raw payload: %s", e, raw)
    return []


def extract_entities(transcript_text: str, title: Optional[str] = None) -> List[Dict[str, Any]]:
    """Public API: extract entity records from transcript text."""
    if not transcript_text or not transcript_text.strip():
        return []
    safe_title = title or "Untitled Video"
    user_prompt = ENTITY_USER_TEMPLATE.format(title=safe_title, transcript=transcript_text.strip())
    return _call_claude_entities(ENTITY_SYSTEM_PROMPT, user_prompt)


def store_entities(video_id: str, entities: List[Dict[str, Any]]) -> int:
    """
    Store entities into the SQLite `entities` table.

    Returns the number of entities stored.
    """
    if not entities:
        return 0

    db = TranscriptDatabase()
    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
    cursor = conn.cursor()

    count = 0
    for ent in entities:
        name = ent.get("name")
        etype = ent.get("type")
        aliases = ent.get("aliases") or []
        if not name or not etype:
            continue
        cursor.execute(
            """
            INSERT INTO entities (video_id, name, type, aliases)
            VALUES (?, ?, ?, ?)
            """,
            (video_id, name, etype, json.dumps(aliases)),
        )
        count += 1

    conn.commit()
    conn.close()
    return count


if __name__ == "__main__":
    import sys as _sys
    import sqlite3  # for type checker

    if len(_sys.argv) < 2:
        print("Usage: python entity_extractor.py VIDEO_ID")
        raise SystemExit(1)

    vid = _sys.argv[1]
    db = TranscriptDatabase()
    tx = db.get_transcript(vid)
    if not tx:
        print(f"No transcript for {vid}")
        raise SystemExit(1)

    ents = extract_entities(tx, title=None)
    stored = store_entities(vid, ents)
    print(f"Stored {stored} entities for {vid}")

