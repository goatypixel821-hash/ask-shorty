#!/usr/bin/env python3
"""
Segment-level topical summaries from transcripts (HSC).

Produces time-bounded segments when video duration is known (from metadata),
otherwise start_time/end_time are None. Segments are topical windows from an LLM,
not fixed-size character chunks.

Debug: set environment variable ``SEGMENT_EXTRACTOR_DEBUG_LLM=1`` to print raw LLM
text from ``_call_llm_raw`` / topical parse (very noisy).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# (system_prompt, user_prompt, max_tokens, temperature) -> raw assistant text
HSCChatFn = Callable[[str, str, int, float], str]

from anthropic_client import get_client

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"

# Legacy: tiny per-window summary (only used when explicit timestamps are passed)
SEGMENT_SYSTEM_LEGACY = """You write extremely dense segment summaries for video transcripts.

Rules:
- 1–3 sentences per segment maximum.
- Capture topic, claims, names, numbers, and outcomes.
- No filler or repetition.
- Do NOT paste transcript text; synthesize in your own words.
"""

SEGMENT_USER_LEGACY = """Transcript excerpt (may be mid-video):

{excerpt}

Reply with ONLY a JSON object:
{{"summary": "<1-3 dense sentences>"}}
"""

# Topical segmentation: one JSON object per LLM call
SEGMENT_TOPICAL_SYSTEM = """You segment a video transcript into meaningful topical spans.

For each span you write a SHORT topical summary (2–3 sentences) that says what is discussed
and why it matters — in your own words.

Hard rules:
- Do NOT copy-paste or lightly paraphrase long runs from the transcript.
- No bullet lists inside summaries; use flowing prose.
- Summaries must be readable without the transcript.
- JSON only; no markdown fences unless wrapping the whole JSON block.
"""

MAX_SINGLE_PASS_CHARS = 95_000
CHUNK_CHARS = 48_000
CHUNK_OVERLAP = 1_200
LLM_MAX_TOKENS = 8192

# Set SEGMENT_EXTRACTOR_DEBUG_LLM=1 to print full LLM responses (noisy; for diagnosing parse failures).
# Checked at call time so you can set the env after load_dotenv() and before import if needed.
def _debug_llm_enabled() -> bool:
    return (os.environ.get("SEGMENT_EXTRACTOR_DEBUG_LLM") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _debug_print_llm_raw(caller: str, raw: Optional[str], max_preview: int = 16000) -> None:
    """Print raw model text before JSON / downstream parsing (see SEGMENT_EXTRACTOR_DEBUG_LLM)."""
    if not _debug_llm_enabled():
        return
    body = raw if isinstance(raw, str) else str(raw)
    truncated = len(body) > max_preview
    preview = body[:max_preview] if truncated else body
    print(f"\n[segment_extractor DEBUG {caller}] raw_chars={len(body)} truncated={truncated}", flush=True)
    print(preview, flush=True)
    if truncated:
        print(f"... ({len(body) - max_preview} more chars omitted from preview)", flush=True)
    print(f"[segment_extractor DEBUG {caller}] END\n", flush=True)


def _anthropic_response_text(resp: Any) -> str:
    raw = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            raw += block.text
        elif isinstance(block, dict) and block.get("type") == "text":
            raw += block.get("text", "")
    return raw.strip()


def _strip_json_fence(raw: str) -> str:
    text = (raw or "").strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    return text


def _extract_outer_json_object(text: str) -> str:
    """If the model adds preamble/postamble, take the outermost {...} block."""
    t = (text or "").strip()
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end > start:
        return t[start : end + 1].strip()
    return t


def _parse_topical_segments(raw: str) -> List[Dict[str, Any]]:
    """Parse {"segments":[{start_char,end_char,summary},...]} or bare array."""
    text = _strip_json_fence(raw)
    text = _extract_outer_json_object(text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        data = data["segments"]
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            sc = int(item.get("start_char", item.get("start", -1)))
            ec = int(item.get("end_char", item.get("end", -1)))
        except (TypeError, ValueError):
            continue
        summary = (item.get("summary") or "").strip()
        if not summary or sc < 0 or ec <= sc:
            continue
        out.append({"start_char": sc, "end_char": ec, "summary": summary})
    return out


def _summary_mirrors_transcript(summary: str, excerpt: str) -> bool:
    """Heuristic: model echoed transcript instead of summarizing."""
    if len(summary) < 50:
        return False
    s = re.sub(r"\s+", " ", summary.strip().lower())
    e = re.sub(r"\s+", " ", excerpt.strip().lower())
    probe = s[: min(120, len(s))]
    if len(probe) >= 40 and probe in e:
        return True
    # Long common prefix with excerpt start
    ex_head = e[: min(400, len(e))]
    if len(s) >= 80 and ex_head.startswith(s[:80]):
        return True
    return False


def _topical_user_prompt(
    title: Optional[str],
    excerpt: str,
    excerpt_offset: int,
    total_len: int,
) -> str:
    safe_title = (title or "Untitled").strip()
    return f"""Video title: {safe_title}

This string is excerpt characters [{excerpt_offset}:{excerpt_offset + len(excerpt)}] of a longer transcript
(total length {total_len} characters). Indices below are 0-based within THIS EXCERPT ONLY (0 .. {len(excerpt)}).

Transcript excerpt:
\"\"\"
{excerpt}
\"\"\"

Return ONLY a JSON object with this exact shape:
{{
  "segments": [
    {{"start_char": 0, "end_char": 1200, "summary": "2-3 sentences on what this span covers."}},
    ...
  ]
}}

Rules:
- start_char inclusive, end_char exclusive, relative to the excerpt string above.
- Segments sorted by start_char ascending.
- Cover from 0 through {len(excerpt)} with no gap larger than 500 characters (small overlaps OK).
- Prefer 6–18 segments for this excerpt; merge tiny chit-chat, split major topic shifts.
- Summaries: topical description only — never long quotes from the excerpt.
"""


def _call_llm_raw(
    system: str,
    user: str,
    chat_fn: Optional[HSCChatFn],
    max_tokens: int,
    temperature: float,
) -> str:
    """All segment LLM traffic goes here — use ``chat_fn`` when the caller provides it (e.g. OpenRouter from batch_processor)."""
    if chat_fn is not None:
        raw = chat_fn(system, user, max_tokens, temperature)
        _debug_print_llm_raw(
            f"_call_llm_raw(chat_fn max_tokens={max_tokens} temp={temperature})",
            raw if isinstance(raw, str) else str(raw),
        )
        out = (raw or "").strip()
        # OpenRouter / some models occasionally return null message.content on the first try.
        if not out and max_tokens >= 2048:
            logger.warning(
                "segment_extractor: empty completion from chat_fn (max_tokens=%s); retrying once.",
                max_tokens,
            )
            raw2 = chat_fn(system, user, max_tokens, temperature)
            _debug_print_llm_raw(
                f"_call_llm_raw(chat_fn RETRY max_tokens={max_tokens} temp={temperature})",
                raw2 if isinstance(raw2, str) else str(raw2),
            )
            out = (raw2 or "").strip()
        return out
    # Fall back to Anthropic only when no chat_fn (CLI / tests without OpenRouter wrapper)
    client = get_client()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw_anth = _anthropic_response_text(resp)
    _debug_print_llm_raw(
        f"_call_llm_raw(anthropic max_tokens={max_tokens} temp={temperature})",
        raw_anth,
    )
    return raw_anth


def _llm_topical_for_excerpt(
    excerpt: str,
    excerpt_offset: int,
    total_len: int,
    title: Optional[str],
    chat_fn: Optional[HSCChatFn],
) -> List[Dict[str, Any]]:
    user = _topical_user_prompt(title, excerpt, excerpt_offset, total_len)
    raw = _call_llm_raw(
        SEGMENT_TOPICAL_SYSTEM,
        user,
        chat_fn,
        LLM_MAX_TOKENS,
        0.15,
    )
    _debug_print_llm_raw(
        "_llm_topical_for_excerpt BEFORE parse (attempt 1, after strip from _call_llm_raw)",
        raw,
    )
    segs = _parse_topical_segments(raw)
    if not segs:
        logger.warning("Topical segment parse failed; retrying with stricter JSON instruction.")
        raw2 = _call_llm_raw(
            SEGMENT_TOPICAL_SYSTEM
            + "\nYour previous reply was not valid JSON. Reply with ONLY one JSON object.",
            user,
            chat_fn,
            LLM_MAX_TOKENS,
            0.0,
        )
        _debug_print_llm_raw(
            "_llm_topical_for_excerpt BEFORE parse (attempt 2 retry, after strip)",
            raw2,
        )
        segs = _parse_topical_segments(raw2)
    # Clip bounds to excerpt
    L = len(excerpt)
    fixed: List[Dict[str, Any]] = []
    for s in segs:
        sc = max(0, min(s["start_char"], L))
        ec = max(sc + 1, min(s["end_char"], L))
        summ = s["summary"]
        span = excerpt[sc:ec]
        if _summary_mirrors_transcript(summ, span):
            summ = _summarize_span_fallback(span, chat_fn)
        fixed.append({"start_char": sc, "end_char": ec, "summary": summ})
    return fixed


def _summarize_span_fallback(span: str, chat_fn: Optional[HSCChatFn]) -> str:
    """Short anti-copy summary for a span when main summary looked like a paste."""
    user = (
        "In 2-3 sentences, describe what this transcript span is about. "
        "Do not quote; synthesize.\n\n---\n"
        + span[:6000]
    )
    raw = _call_llm_raw(
        "You write concise topical summaries. No quotes from the source.",
        user,
        chat_fn,
        512,
        0.2,
    )
    m = re.search(r"\{[\s\S]*\"summary\"\s*:\s*\"([^\"]+)\"", raw)
    if m:
        return m.group(1).strip()
    if "```" in raw:
        raw = _strip_json_fence(raw)
    try:
        d = json.loads(raw)
        if isinstance(d, dict) and d.get("summary"):
            return str(d["summary"]).strip()
    except json.JSONDecodeError:
        pass
    line = raw.splitlines()[0].strip() if raw.strip() else ""
    return (line[:800] + "…") if len(line) > 800 else line


def _merge_overlapping(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge overlapping [char_start,char_end) ranges; concatenate distinct summaries."""
    if not items:
        return []
    items = sorted(items, key=lambda x: (x["char_start"], x["char_end"]))
    merged: List[Dict[str, Any]] = []
    for cur in items:
        if not merged:
            merged.append(cur)
            continue
        prev = merged[-1]
        if cur["char_start"] < prev["char_end"]:
            prev["char_end"] = max(prev["char_end"], cur["char_end"])
            if cur["summary"] and cur["summary"] not in prev["summary"]:
                prev["summary"] = (prev["summary"] + " " + cur["summary"]).strip()[:2000]
        else:
            merged.append(cur)
    return merged


def _char_span_to_times(
    start_c: int,
    end_c: int,
    total_chars: int,
    duration_seconds: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    if not duration_seconds or duration_seconds <= 0 or total_chars <= 0:
        return None, None
    st = duration_seconds * (start_c / total_chars)
    et = duration_seconds * (end_c / total_chars)
    if et <= st:
        et = min(duration_seconds, st + 1.0)
    return float(st), float(et)


def _split_by_chars(text: str, max_chars: int = 3200, overlap: int = 200) -> List[str]:
    """Legacy: split transcript into windows (char-based heuristic)."""
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


def _summarize_excerpt_legacy(excerpt: str, chat_fn: Optional[HSCChatFn]) -> str:
    user = SEGMENT_USER_LEGACY.format(excerpt=excerpt[:120000])
    raw = _call_llm_raw(SEGMENT_SYSTEM_LEGACY, user, chat_fn, 512, 0.2)
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
    cleaned = raw[:2000].strip()
    if cleaned and not _summary_mirrors_transcript(cleaned, excerpt):
        return cleaned
    return _summarize_span_fallback(excerpt[:4000], chat_fn)


def extract_segments(
    transcript_text: str,
    timestamps: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    duration_seconds: Optional[float] = None,
    title: Optional[str] = None,
    chat_fn: Optional[HSCChatFn] = None,
) -> List[Dict[str, Any]]:
    """
    Produce segment records: start_time, end_time, text (source window), summary.

    - When ``timestamps`` is provided (legacy): one summary per timestamp window
      over fixed-size transcript chunks (same count as timestamps).
    - Otherwise: topical segmentation via LLM; ``duration_seconds`` maps character
      spans to approximate wall-clock times (linear by transcript position).
    """
    text = (transcript_text or "").strip()
    if not text:
        return []

    total_len = len(text)

    # Legacy path: explicit per-window timestamps from caller
    if timestamps is not None and len(timestamps) > 0:
        windows = _split_by_chars(text, max_chars=3200, overlap=200)
        out: List[Dict[str, Any]] = []
        n = len(windows)
        for i, win in enumerate(windows):
            st, et = _estimate_times(i, n, timestamps)
            try:
                summary = _summarize_excerpt_legacy(win, chat_fn=chat_fn)
            except Exception as exc:
                logger.warning("Segment summarize failed window %s: %s", i, exc)
                summary = _summarize_span_fallback(win[:4000], chat_fn)
            out.append(
                {
                    "start_time": st,
                    "end_time": et,
                    "text": win,
                    "summary": summary,
                }
            )
        return out

    # Topical path
    chunks: List[Tuple[int, str]] = []
    if total_len <= MAX_SINGLE_PASS_CHARS:
        chunks.append((0, text))
    else:
        pos = 0
        while pos < total_len:
            end = min(total_len, pos + CHUNK_CHARS)
            chunks.append((pos, text[pos:end]))
            if end >= total_len:
                break
            pos = end - CHUNK_OVERLAP

    global_items: List[Dict[str, Any]] = []
    for offset, excerpt in chunks:
        try:
            local = _llm_topical_for_excerpt(excerpt, offset, total_len, title, chat_fn)
        except Exception as exc:
            logger.error("Topical segment LLM failed for offset %s: %s", offset, exc)
            continue
        for seg in local:
            gs = offset + seg["start_char"]
            ge = offset + seg["end_char"]
            global_items.append(
                {
                    "char_start": gs,
                    "char_end": ge,
                    "summary": seg["summary"],
                }
            )

    merged = _merge_overlapping(global_items)
    if not merged:
        raise RuntimeError(
            "HSC topical segmentation produced no segments (LLM unavailable, empty parse, or API error). "
            "If using batch_processor with --provider openrouter, check OPENROUTER_API_KEY; "
            "otherwise check ANTHROPIC_API_KEY or chat_fn errors in logs."
        )
    out2: List[Dict[str, Any]] = []
    for m in merged:
        st_t, en_t = _char_span_to_times(
            m["char_start"], m["char_end"], total_len, duration_seconds
        )
        span_text = text[m["char_start"] : m["char_end"]]
        out2.append(
            {
                "start_time": st_t,
                "end_time": en_t,
                "text": span_text,
                "summary": m["summary"],
            }
        )
    return out2
