#!/usr/bin/env python3
"""
Extract subject–relation–object triples from Shorty text via LLM.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TRIPLE_MODEL = "claude-sonnet-4-20250514"

TRIPLE_SYSTEM_PROMPT = """Extract factual relationship triples from this video summary.
Return ONLY a JSON array of objects. No prose, no markdown, no explanation.

Each object has exactly three keys:
- "subject": the entity doing or being something
- "relation": the relationship (use active verbs: hacked, caused, developed, uses, affects)
- "object": what the subject relates to

Focus on causal, membership, action, technical, and people/role relationships.
Return 5-15 triples. Only include facts clearly stated in the text.
Never invent relationships. Return [] if no clear relationships exist.
Your response MUST start with [ and end with ]. Nothing else."""

TRIPLE_USER_TEMPLATE = """Video title: {title}

Summary:
{shorty}

IMPORTANT: Respond with ONLY a JSON array. Start with [ and end with ]. No other text."""


def _cache_path(shorty_text: str, title: str, base_dir: Path) -> Path:
    h = hashlib.sha256(f"{title}\n{shorty_text}".encode("utf-8")).hexdigest()
    d = base_dir / "data" / "triple_llm_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{h}.json"


def load_cached_triples(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception as exc:
        logger.debug("triple cache read failed: %s", exc)
    return None


def save_cached_triples(path: Path, triples: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)


def _triple_fields_from_dict(item: Dict[str, Any]) -> Optional[tuple]:
    """Accept subject/relation/object with case-insensitive keys; relation alias 'predicate'."""
    if not isinstance(item, dict):
        return None
    norm = {str(k).lower().strip(): v for k, v in item.items() if isinstance(k, str)}
    sub = norm.get("subject") or norm.get("subj")
    rel = norm.get("relation") or norm.get("rel") or norm.get("predicate")
    obj = norm.get("object") or norm.get("obj")
    if isinstance(sub, str) and isinstance(rel, str) and isinstance(obj, str):
        s, r, o = sub.strip(), rel.strip(), obj.strip()
        if s and r and o:
            return s, r, o
    return None


def parse_triples_json(raw: str) -> List[Dict[str, Any]]:
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()

    blobs: List[str] = [text]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        inner = text[start : end + 1]
        if inner not in blobs:
            blobs.append(inner)

    data: Any = None
    for blob in blobs:
        try:
            data = json.loads(blob)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        return []
    if not isinstance(data, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            triple = _triple_fields_from_dict(item)
            if triple:
                s, r, o = triple
                conf = item.get("confidence", 1.0)
                try:
                    c = float(conf if conf is not None else 1.0)
                except (TypeError, ValueError):
                    c = 1.0
                out.append(
                    {
                        "subject": s,
                        "relation": r,
                        "object": o,
                        "confidence": c,
                    }
                )
    return out


def extract_triples_anthropic(shorty_text: str, title: str) -> List[Dict[str, Any]]:
    from anthropic_client import get_client

    client = get_client()
    user = TRIPLE_USER_TEMPLATE.format(title=title or "Untitled", shorty=shorty_text.strip())
    resp = client.messages.create(
        model=TRIPLE_MODEL,
        max_tokens=2048,
        temperature=0.1,
        system=TRIPLE_SYSTEM_PROMPT.strip(),
        messages=[{"role": "user", "content": user}],
    )
    parts: List[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    raw = "\n".join(parts).strip()
    return parse_triples_json(raw)


def extract_triples(
    shorty_text: str,
    title: str,
    *,
    use_cache: bool = True,
    project_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Extract triples from Shorty text. Uses disk cache under data/triple_llm_cache/.
    """
    root = project_root or Path(__file__).resolve().parent
    cache_file = _cache_path(shorty_text, title, root)
    if use_cache:
        cached = load_cached_triples(cache_file)
        if cached is not None:
            return cached
    triples = extract_triples_anthropic(shorty_text, title)
    if use_cache:
        try:
            save_cached_triples(cache_file, triples)
        except Exception as exc:
            logger.warning("triple cache write failed: %s", exc)
    return triples


def extract_triples_openai(
    shorty_text: str,
    title: str,
    chat_fn,
) -> List[Dict[str, Any]]:
    """chat_fn(system_prompt, user_prompt) -> raw string (OpenAI-compatible)."""
    triples, _raw = extract_triples_openai_raw(shorty_text, title, chat_fn)
    return triples


def extract_triples_openai_raw(
    shorty_text: str,
    title: str,
    chat_fn,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Same as extract_triples_openai but returns (triples, raw_llm_string) for debugging.
    chat_fn(system_prompt, user_prompt) -> raw string (OpenAI-compatible).
    """
    user = TRIPLE_USER_TEMPLATE.format(title=title or "Untitled", shorty=shorty_text.strip())
    raw = chat_fn(TRIPLE_SYSTEM_PROMPT.strip(), user)
    return parse_triples_json(raw), raw if isinstance(raw, str) else str(raw)
