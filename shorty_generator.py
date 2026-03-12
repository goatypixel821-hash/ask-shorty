#!/usr/bin/env python3
"""
Shorty and synthetic question generation for Ask Shorty.

Uses Anthropic Claude to:
- Generate a dense, machine-oriented Shorty for a transcript.
- Generate 8–10 likely user questions about a video.
"""

from typing import List, Optional, Dict, Any

from anthropic_client import get_client


SHORTY_MODEL = "claude-3-haiku-20240307"


SHORTY_SYSTEM_PROMPT = """You are a compression engine for video transcripts.

Your job is to produce a maximum-density knowledge brief called a Shorty.
It is NOT a summary. It is for machine consumption, not humans.

Requirements:
- Preserve ~95% of answerable information in ~90–97% fewer tokens.
- Preserve ALL entities, relationships, causal chains, claims, numbers,
  technical details, and named systems.
- Explicitly separate facts from commentary.
- Add a MICRO-DETAILS section for specific technical terms, exact names,
  quantities, and aliases.
- Add a TIMELINE section at the end capturing any chronological information
  mentioned in the video — dates, sequences of events, and when things happened.
- Use the same structure and style as the EXAMPLE below.
- Do NOT include any explanation of what you are doing.

EXAMPLE FORMAT (STRUCTURE TO COPY, NOT CONTENT):

COMPRESSED TRANSCRIPT — <Video Title or Topic>

<Tightly packed paragraphs, clearly separated incidents/sections.>

MICRO-DETAILS:
- <bullet of critical detail>
- <bullet of critical detail>

TIMELINE:
- <bullet of chronological detail>
- <bullet of chronological detail>
"""


SHORTY_USER_PROMPT_TEMPLATE = """Compress this transcript to maximum semantic density for LLM retrieval.

Video metadata (for your reference):
- Title: {title}
- Channel: {channel}
- Upload date: {upload_date}

Transcript:
{transcript}
"""


SYNTHETIC_Q_SYSTEM_PROMPT = """You generate likely user questions about a video.

Given a transcript, produce 8–10 clear, specific questions a user might ask.

Requirements:
- Questions should be factual and answerable from the video.
- Cover entities, systems, numbers, causal stories, and key claims.
- Vary angle and level of abstraction.
- Output ONLY a JSON array of strings, nothing else.
"""


SYNTHETIC_Q_USER_PROMPT_TEMPLATE = """Generate 8–10 likely questions a user might ask about this video.

Transcript metadata:
- Title: {title}

Transcript:
{transcript}
"""


def _call_claude(system_prompt: str, user_prompt: str) -> str:
    """Low-level helper to call Claude and return the text content."""
    client = get_client()
    resp = client.messages.create(
        model=SHORTY_MODEL,
        max_tokens=4096,
        temperature=0.2,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    # anthropic messages API returns a list of content blocks; we join text blocks
    parts: List[str] = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts).strip()


def generate_shorty(
    transcript_text: str,
    title: Optional[str] = None,
    channel: Optional[str] = None,
    upload_date: Optional[str] = None,
) -> str:
    """
    Generate a Shorty for a transcript.

    Returns the full Shorty text. Raises RuntimeError on configuration issues,
    and RuntimeError on hard API failures.
    """
    if not transcript_text or not transcript_text.strip():
        raise ValueError("Transcript text is empty; cannot generate Shorty.")

    safe_title = title or "Untitled Video"
    safe_channel = channel or "Unknown channel"
    safe_date = upload_date or "unknown"

    user_prompt = SHORTY_USER_PROMPT_TEMPLATE.format(
        title=safe_title,
        channel=safe_channel,
        upload_date=safe_date,
        transcript=transcript_text.strip(),
    )

    body = _call_claude(SHORTY_SYSTEM_PROMPT, user_prompt)

    header = (
        f"SOURCE: {safe_title}\n"
        f"CHANNEL: {safe_channel}\n"
        f"DATE: {safe_date}\n"
        f"CREATOR: {safe_channel}\n\n"
    )

    return header + body.lstrip()


def generate_synthetic_questions(
    transcript_text: str,
    title: Optional[str] = None,
    n: int = 10,
) -> List[str]:
    """
    Generate likely user questions about a video.

    Returns a list of question strings. If parsing fails, falls back to
    returning a best-effort list with basic splitting.
    """
    if not transcript_text or not transcript_text.strip():
        raise ValueError("Transcript text is empty; cannot generate questions.")

    safe_title = title or "Untitled Video"
    user_prompt = SYNTHETIC_Q_USER_PROMPT_TEMPLATE.format(
        title=safe_title,
        transcript=transcript_text.strip(),
    )

    raw = _call_claude(SYNTHETIC_Q_SYSTEM_PROMPT, user_prompt)

    # Try strict JSON parse first
    import json

    questions: List[str] = []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    q = item.strip()
                    if q:
                        questions.append(q)
    except Exception:
        # Fallback: split by lines and take question-like lines
        for line in raw.splitlines():
            line = line.strip().lstrip("-").strip()
            if not line:
                continue
            if not line.endswith("?"):
                continue
            questions.append(line)

    # Truncate or pad to n (padding not necessary; we just cap)
    if len(questions) > n:
        questions = questions[:n]

    return questions


def generate_shorty_and_questions_for_video(
    video_id: str,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience helper:
    - Pull transcript from TranscriptDatabase.
    - Generate Shorty + synthetic questions.

    Returns a dict with keys: video_id, transcript, shorty, questions.
    """
    from transcript_database import TranscriptDatabase

    db = TranscriptDatabase()
    transcript = db.get_transcript(video_id)
    if not transcript:
        raise ValueError(f"No transcript found in DB for video_id={video_id}")

    # Pull metadata for richer Shorty header
    info = db.get_video_info(video_id) or {}
    meta = (info.get("metadata") or {}) if isinstance(info, dict) else {}
    title_meta = info.get("title") if isinstance(info, dict) else None
    channel_meta = info.get("channel") if isinstance(info, dict) else None
    upload_date = meta.get("upload_date") if isinstance(meta, dict) else None

    final_title = title_meta or title

    shorty = generate_shorty(
        transcript,
        title=final_title,
        channel=channel_meta,
        upload_date=upload_date,
    )
    questions = generate_synthetic_questions(transcript, title=final_title)

    return {
        "video_id": video_id,
        "transcript": transcript,
        "shorty": shorty,
        "questions": questions,
    }

