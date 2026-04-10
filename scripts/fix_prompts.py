#!/usr/bin/env python3
"""One-shot: tighten entity, synq, and triple prompts for small-model JSON compliance."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def patch(path: Path, old: str, new: str, label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        print(f"  [{label}] NOT FOUND in {path.name}")
        return False
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print(f"  [{label}] patched in {path.name}")
    return True


# ── entity_extractor.py ──────────────────────────────────────────────────

EE = ROOT / "entity_extractor.py"

patch(EE,
    'ENTITY_JSON_SYSTEM_PROMPT = """You extract structured entities from video transcripts.\n'
    '\n'
    'Respond with ONLY a valid JSON array of objects. No markdown, no code fences, no explanation.\n'
    'Each object must have:\n'
    '- "name": string (canonical name)\n'
    '- "type": string (one of: person, organization, system, protocol, software, location, concept, or product)\n'
    '- "aliases": array of strings (alternate names, abbreviations; can be empty [])\n'
    '\n'
    'Requirements:\n'
    '- Prefer specific, concrete entities important to understanding the video.\n'
    '- Merge clear aliases into the same entity.\n'
    '- Be generous: include people, organizations, products, software, protocols, locations, important concepts.\n'
    '"""',
    # replacement ──
    'ENTITY_JSON_SYSTEM_PROMPT = """You extract structured entities from video transcripts.\n'
    'You MUST respond with ONLY a valid JSON array. No prose, no markdown, no explanation, no code fences.\n'
    '\n'
    'Each object in the array has exactly three keys:\n'
    '- "name": string (canonical name)\n'
    '- "type": string (one of: person, organization, system, protocol, software, location, concept, product)\n'
    '- "aliases": array of strings (alternate names; can be empty [])\n'
    '\n'
    'Rules:\n'
    '- Prefer specific, concrete entities important to the video.\n'
    '- Merge clear aliases into one entity.\n'
    '- Include people, organizations, products, software, protocols, locations, concepts.\n'
    '- Output MUST start with [ and end with ]. Nothing else."""',
    "entity_system",
)

patch(EE,
    'ENTITY_JSON_USER_TEMPLATE = """Extract entities from this transcript. Reply with only a JSON array.\n'
    '\n'
    'Title: {title}\n'
    '\n'
    'Transcript:\n'
    '{transcript}\n'
    '"""',
    # replacement ──
    'ENTITY_JSON_USER_TEMPLATE = """Extract entities from this transcript.\n'
    '\n'
    'Title: {title}\n'
    '\n'
    'Transcript (excerpt):\n'
    '{transcript}\n'
    '\n'
    'IMPORTANT: Your entire response must be a JSON array starting with [ and ending with ]. No other text."""',
    "entity_user",
)


# ── shorty_generator.py (synthetic questions) ────────────────────────────

SG = ROOT / "shorty_generator.py"

patch(SG,
    'SYNTHETIC_Q_SYSTEM_PROMPT = """You generate likely user questions about a video.\n'
    '\n'
    'Given a transcript, produce 8\u201310 clear, specific questions a user might ask.\n'
    '\n'
    'Requirements:\n'
    '- Questions should be factual and answerable from the video.\n'
    '- Cover entities, systems, numbers, causal stories, and key claims.\n'
    '- Vary angle and level of abstraction.\n'
    '- Output ONLY a JSON array of strings, nothing else.\n'
    '"""',
    # replacement ──
    'SYNTHETIC_Q_SYSTEM_PROMPT = """You generate likely user questions about a video.\n'
    'Given a transcript, produce 8-10 clear, specific questions a user might ask.\n'
    '\n'
    'Rules:\n'
    '- Questions must be factual and answerable from the video.\n'
    '- Cover entities, systems, numbers, causal stories, and key claims.\n'
    '- Vary angle and level of abstraction.\n'
    '- Output ONLY a JSON array of strings.\n'
    '- Your response MUST start with [ and end with ]. No other text."""',
    "synq_system",
)


# ── triple_extractor.py ──────────────────────────────────────────────────

TE = ROOT / "triple_extractor.py"

patch(TE,
    'TRIPLE_SYSTEM_PROMPT = """\n'
    'Extract factual relationship triples from this video summary.\n'
    'Return ONLY a JSON array of triples. Each triple has:\n'
    '- subject: the entity doing or being something\n'
    '- relation: the relationship (use active verbs: hacked, caused, developed, uses, affects)\n'
    '- object: what the subject relates to\n'
    '\n'
    'Focus on:\n'
    '- Causal relationships (X caused Y, X led to Y)\n'
    '- Membership (X is part of Y, X belongs to Y)\n'
    '- Actions (X hacked Y, X developed Z)\n'
    '- Technical relationships (X uses Y, X requires Z)\n'
    '- People and their roles (X founded Y, X works at Z)\n'
    '\n'
    'Return 5-15 triples per video. Only include facts clearly stated in the text.\n'
    'Never invent relationships. Return [] if no clear relationships exist.\n'
    '"""',
    # replacement ──
    'TRIPLE_SYSTEM_PROMPT = """Extract factual relationship triples from this video summary.\n'
    'Return ONLY a JSON array of objects. No prose, no markdown, no explanation.\n'
    '\n'
    'Each object has exactly three keys:\n'
    '- "subject": the entity doing or being something\n'
    '- "relation": the relationship (use active verbs: hacked, caused, developed, uses, affects)\n'
    '- "object": what the subject relates to\n'
    '\n'
    'Focus on causal, membership, action, technical, and people/role relationships.\n'
    'Return 5-15 triples. Only include facts clearly stated in the text.\n'
    'Never invent relationships. Return [] if no clear relationships exist.\n'
    'Your response MUST start with [ and end with ]. Nothing else."""',
    "triple_system",
)

patch(TE,
    'TRIPLE_USER_TEMPLATE = """Video title: {title}\n'
    '\n'
    'Summary:\n'
    '{shorty}\n'
    '"""',
    # replacement ──
    'TRIPLE_USER_TEMPLATE = """Video title: {title}\n'
    '\n'
    'Summary:\n'
    '{shorty}\n'
    '\n'
    'IMPORTANT: Respond with ONLY a JSON array. Start with [ and end with ]. No other text."""',
    "triple_user",
)

print("\nDone. All prompts tightened for small-model JSON compliance.")
