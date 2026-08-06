#!/usr/bin/env python3
"""
Shorty and synthetic question generation for Ask Shorty.

Uses Anthropic Claude to:
- Generate a dense, machine-oriented Shorty for a transcript.
- Generate 8–10 likely user questions about a video.
"""

from typing import List, Optional, Dict, Any
import logging
import os
from pathlib import Path

from anthropic_client import get_client


logger = logging.getLogger(__name__)

# Load .env BEFORE resolving SHORTY_PROVIDER / SHORTY_MODEL.
# Other modules (e.g. transcript_rag) also call load_dotenv, but batch_processor
# imports this file first — so without this, SHORTY_MODEL freezes to the
# built-in fallback and .env is ignored.
def _load_shorty_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=True)


_load_shorty_dotenv()

# Explicit per-run selection via env. No silent provider fallback.
#   SHORTY_PROVIDER = anthropic | openrouter   (default: openrouter)
#   SHORTY_MODEL    = model id for that provider (optional override)
SHORTY_PROVIDER = (os.environ.get("SHORTY_PROVIDER") or "openrouter").strip().lower()
_DEFAULT_SHORTY_MODELS = {
    "anthropic": "claude-3-haiku-20240307",
    "openrouter": "anthropic/claude-3-haiku",
}
if SHORTY_PROVIDER not in _DEFAULT_SHORTY_MODELS:
    raise RuntimeError(
        f"Invalid SHORTY_PROVIDER={SHORTY_PROVIDER!r}. "
        f"Use one of: {sorted(_DEFAULT_SHORTY_MODELS)}"
    )
SHORTY_MODEL = (
    os.environ.get("SHORTY_MODEL") or _DEFAULT_SHORTY_MODELS[SHORTY_PROVIDER]
).strip()


SHORTY_SYSTEM_PROMPT = """You are a compression engine for video transcripts.

Your job is to produce a maximum-density knowledge brief called a Shorty.
It is NOT a summary. It is for machine consumption, not humans.

Generate a dense Shorty following this EXACT structure and these rules.

GROUNDING RULES (MUST OBEY — HIGHEST PRIORITY)
- Only state facts, names, dates, numbers, and claims that appear in the transcript or the
  provided video metadata (title, description, tags, chapters). Do not add facts from your
  own training knowledge, even if you recognize the topic, tool, event, or person being
  discussed.
- If a detail is unclear, garbled, not stated, or you are not confident it is correct, write
  [not stated] or [unclear] rather than guessing or substituting a plausible-sounding
  real name, date, number, or fact.
- If the transcript's speech-to-text is garbled, incomplete, or the speaker never
  actually names something (e.g. "a lake called lake," a mumbled place name, an
  unclear number), do NOT supply the real answer from your own knowledge, even if
  you're confident you know what it must be. Write [unclear] instead.
  This applies even when the surrounding topic makes the real answer easy to guess
  (e.g. a video about a specific storm, mountain, or historical event) — recognizing
  the topic is not the same as the speaker having stated the specific name, date,
  or figure. The one exception: correcting an obvious ASR misspelling of a word the
  speaker clearly DID say (e.g. "Elberus" → "Elbrus," "swalbard" → "Svalbard") is
  fine, since the referent was actually spoken — the line to hold is between fixing
  a garbled attempt at a word and inventing an answer to a gap the speaker never
  filled.
- If the transcript gives a date with only a year (e.g. "1939") or year and month
  (e.g. "January 2026"), preserve that same precision (YYYY or YYYY-MM). Do NOT invent a
  specific day to force YYYY-MM-DD format.
- If no source URL is provided in the metadata, write exactly: (URL not provided)
  Never construct, guess, or use a placeholder-style URL.
- Only attribute a creator/author to a tool, product, or piece of software if the transcript
  or metadata explicitly states who made it. If not stated, list the tool name alone.
- The STRUCTURE, COMPRESSION RULES, and any bracketed examples in THIS prompt (e.g.
  "RDP:3389", "package.json", "cache rack by Khan, OpenClaw") are illustrations of formatting
  only. They are never real content. Do not copy, reuse, or reference them in your output
  under any circumstances.
- When two or more entities in the transcript are discussed close together (e.g. two
  channels, two people, two products), double check which specific fact, number, or quote
  belongs to which entity before writing it down.

STRUCTURE (MUST FOLLOW EXACTLY)

HEADER
TITLE – <video title>
SOURCE: (<url, or exactly "URL not provided" if none given>)
CHANNEL: <channel>
DATE: <YYYY-MM-DD, or YYYY-MM, or YYYY — matching the precision actually available>

CONTEXT (2–3 sentences)
What this covers, why it matters. When the speaker states a core conceptual thesis,
motivating argument, or "why this exists / why it was designed this way" framing,
include that thesis here in compressed form — not only the technical subject or
that "claims were exaggerated."

If DESCRIPTION, TAGS, or CHAPTERS appear at the top of the transcript input, those
lines are video metadata (not spoken transcript). Use them to improve entity
extraction and topic labeling; do not copy them verbatim into the Shorty.

TOPICS (as many blocks as needed — do NOT cap at 1–3)
Include DIY, science, news, finance, commentary, tutorial, history, sports,
anecdote, analysis, or any other format. Prefer an extra TOPIC over demoting a
distinct segment into MICRO-DETAILS People/Organizations lists alone.
TOPIC 1 – <specific name or theme>
- What it is: <subject, claim, project, or storyline>
- Who/what is involved: <people, tools, orgs, products, places, systems>
- Key details: <numbers, steps, mechanisms, claims, comparisons, specs>
- Outcome or conclusion: <result, takeaway, verdict, recommendation, or open question>

(If the video covers additional distinct subjects, add TOPIC 2, TOPIC 3, TOPIC 4…
with the same bullet pattern.)

If the video opens with or returns to a substantial conceptual thesis, design
philosophy, or rhetorical "why" argument (analogy, comparison, rhetorical question —
e.g. linear vs rotary motion, "animals never evolved wheels," "gasoline is clean but
inefficient / diesel is efficient but dirty"), give that its own TOPIC block when it
is more than a one-sentence aside. Do not drop it because later sections are denser
with mechanisms, numbers, or build steps. Specs and mechanisms do not replace it.

NARRATIVE / ANECDOTE SEGMENTS (MUST CAPTURE ARC, NOT ONLY PUNCHLINE)
When the speaker tells a story, personal anecdote, or scene with a beginning,
middle, and payoff (setup → events → punchline or moral), give it its own TOPIC.
In Key details, preserve the connective tissue: setting, sequence of events,
turning point, and how the punchline lands. Do NOT keep only the final claim
or conspiracy list while dropping the story that gives it meaning.
When the speaker explicitly states why they are telling a story, or that they
are breaking a personal rule / usual practice to tell it, that stated reason
is part of the arc and must appear in the TOPIC — not just the events themselves.
The stated reason for telling the story must appear as its own dedicated
sentence in Key details in the format: "[Speaker] breaks [personal rule /
usual practice] because [specific stated reason]." It is not sufficient to
include the events that follow — the justification sentence itself must be present.

HISTORICAL / EDUCATIONAL EXPLAINER SEGMENTS
Multi-minute historical, institutional, or explainer blocks (trade histories,
past rules, decades-old cases, background that situates current news) are
first-class content. Give each distinct historical thread its own TOPIC with
named people, years, and causal links. Do not skip them because they are not
"breaking news" or denser sections follow.

SECONDARY NEWS / SEGMENT THRESHOLD
If the show spends a distinct beat on a subject (its own intro, claim, and
people), that subject earns a TOPIC even when shorter than the main story.
Listing names under MICRO-DETAILS People/Organizations is not enough for a
segment that had its own airtime.
Sponsor reads, ad segments, and promotional announcements are NOT topics.
Capture them in the SPONSORS / PROMOTIONS section instead. Do not create a
TOPIC block for any segment whose primary purpose is advertising or promoting
a product, service, or community — even if the host spends more than a minute
on it.

MOTIVE / SUBTEXT / ANALYTICAL THROUGH-LINES
When the speaker explicitly states a motive, theory, conspiracy framing, or
analytical through-line (e.g. "X is why they prolong Y," "this connects A to B"),
capture it as its own TOPIC (or a dedicated Key details + Outcome in the parent
TOPIC if inseparable). Do not flatten it to a single CONTEXT sentence or to an
entity list entry.
When the speaker names a specific person, file, scandal, or entity as the
stated reason for a behavior or through-line, that named entity must appear in
the TOPIC by name — do not paraphrase it into a generic motive description.
A stated motive or named entity may appear anywhere in the transcript — not only
adjacent to the main section where the behavior is discussed. Before finalizing
any TOPIC that covers a behavior, action, or pattern, scan the full transcript
for all statements the speaker makes about why that behavior exists. If a named
person, file, or scandal appears anywhere as an explicit stated reason, it must
be pulled into the relevant TOPIC even if it appears far from the primary
discussion of that TOPIC.

SPONSORS / PROMOTIONS (include only if present; omit section entirely if none)
- [Brand or product name]: [one-line description of what was promoted and any offer details stated — discount code, URL, price]

MICRO-DETAILS (critical for retrieval)
Dates: All dates in YYYY-MM-DD format where stated at that precision; otherwise YYYY-MM or YYYY
Numbers: ALL numbers with units preserved (~10GB, $50K, 97%), only as stated in source
Tools: Exact names as stated; include creator only if the source states it
Versions: Full version strings as stated (2.3.0, Python 3.11.2)
Technical: Protocols, files, commands — only ones actually mentioned in this transcript
People: Full names with roles/affiliations as stated
Organizations: Full names + acronyms as stated
People / Organizations lists support retrieval of names; they do NOT replace
TOPIC coverage for any segment that had its own narrative or news beat.

TIMELINE (4–8 key dates)
YYYY-MM-DD – Event description (or YYYY-MM / YYYY if that's the precision given)

COMPRESSION RULES (MUST OBEY)
- Drop filler words; keep ALL entities, numbers, names, and versions that are actually stated.
- Preserve causality (X led to Y, not just “X happened, Y happened”).
- Capture the video's core conceptual thesis, motivating argument, or "why" framing
  even when it is presented narratively or rhetorically rather than as a spec or
  number. This is often introduced early, before the technical breakdown. A one-line
  version belongs in CONTEXT; if substantial, also give it its own TOPIC block.
- Preserve narrative sequence for anecdotes and stories (setup → beats → payoff),
  not only the concluding claim.
- Historical explainer segments get TOPIC blocks with years and named actors;
  do not treat them as optional background.
- Do not use MICRO-DETAILS People/Organizations as a substitute for TOPICs when
  a subject had its own segment.
- Explicit speaker-stated motives, subtext, and analytical through-lines must
  appear as TOPIC-level content, not only as a CONTEXT aside.
- Keep technical specifics that enable precise queries (protocol names, ports, file names, commands, config keys) — only when present in the source.
- Maintain enough context to answer who / what / when / where / why / how.
- Never summarize specific details into vague descriptions (do NOT replace "Python 3.11.2" with "a newer Python version").
- When in doubt between fabricating a plausible detail and omitting it, always omit it or mark [not stated].

Do NOT include any explanation of what you are doing. Only output the Shorty in the structure above.
"""


SHORTY_USER_PROMPT_TEMPLATE = """Compress this transcript to maximum semantic density for LLM retrieval. Prefer covering every distinct segment (stories, history explainers, secondary news beats, and stated motives) as TOPICs over dropping them to name lists.

Video metadata (for your reference):
- Title: {title}
- Channel: {channel}
- Upload date: {upload_date}

Transcript:
{transcript}
"""


SYNTHETIC_Q_SYSTEM_PROMPT = """You generate likely user questions about a video.
Given a transcript, produce 8-10 clear, specific questions a user might ask.

Rules:
- Questions must be factual and answerable from the video.
- Cover entities, systems, numbers, causal stories, and key claims.
- Vary angle and level of abstraction.
- Output ONLY a JSON array of strings.
- Your response MUST start with [ and end with ]. No other text."""


SYNTHETIC_Q_USER_PROMPT_TEMPLATE = """Generate 8–10 likely questions a user might ask about this video.

Transcript metadata:
- Title: {title}

Transcript:
{transcript}
"""


def _call_claude(system_prompt: str, user_prompt: str) -> str:
    """Call the configured Shorty provider. No silent fallback between providers."""
    if SHORTY_PROVIDER == "anthropic":
        if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
            raise RuntimeError(
                "SHORTY_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                "Set the key, or set SHORTY_PROVIDER=openrouter "
                "(requires OPENROUTER_API_KEY)."
            )
        client = get_client()
        resp = client.messages.create(
            model=SHORTY_MODEL,
            max_tokens=8000,
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

    if SHORTY_PROVIDER == "openrouter":
        if not (os.environ.get("OPENROUTER_API_KEY") or "").strip():
            raise RuntimeError(
                "SHORTY_PROVIDER=openrouter but OPENROUTER_API_KEY is not set. "
                "Set the key, or set SHORTY_PROVIDER=anthropic "
                "(requires ANTHROPIC_API_KEY)."
            )
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"].strip(),
            base_url="https://openrouter.ai/api/v1",
        )
        resp = client.chat.completions.create(
            model=SHORTY_MODEL,
            max_tokens=8000,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    raise RuntimeError(f"Unsupported SHORTY_PROVIDER={SHORTY_PROVIDER!r}")


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

    client = get_client()

    tools = [
        {
            "name": "save_questions",
            "description": "Save the generated questions",
            "input_schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of questions",
                    }
                },
                "required": ["questions"],
            },
        }
    ]

    resp = client.messages.create(
        model=SHORTY_MODEL,
        max_tokens=1024,
        temperature=0.2,
        system=SYNTHETIC_Q_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        tools=tools,
        tool_choice={"type": "tool", "name": "save_questions"},
    )

    raw_questions: List[str] = []
    for block in resp.content:
        # Anthropic SDK: block may be an object or a dict
        btype = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
        if btype == "tool_use":
            name = getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
            if name != "save_questions":
                continue
            tool_input = getattr(block, "input", None) if not isinstance(block, dict) else block.get("input")
            if isinstance(tool_input, dict):
                items = tool_input.get("questions", [])
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, str):
                            q = item.strip()
                            if q:
                                raw_questions.append(q)
            break

    if not raw_questions:
        logger.warning("Structured synthetic questions tool returned no questions.")

    # Truncate to n (no padding needed)
    if len(raw_questions) > n:
        raw_questions = raw_questions[:n]

    return raw_questions


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

