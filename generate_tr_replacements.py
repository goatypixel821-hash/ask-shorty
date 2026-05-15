#!/usr/bin/env python3
"""
Generate fact_lookup replacement questions for tr_* eval artifacts.

Reads tr_*.json from an eval queries directory, calls Claude to write a
title-free factual question from expected_chunk_texts, and writes replacements/
with the same filenames (query + query_type updated only).

Usage:
  python generate_tr_replacements.py eval_results/20260514_065902/queries
  python generate_tr_replacements.py eval_results/20260514_065902/queries --limit 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
import os
from typing import Any, Callable, Dict, List, Optional

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
# OpenRouter slug (direct API uses ANTHROPIC_MODEL above).
OPENROUTER_MODEL = "anthropic/claude-sonnet-4"

SYSTEM_PROMPT = """You write evaluation questions for a personal video transcript search system.

Given transcript excerpts from ONE video, produce a single specific factual question that:
- Can be answered only using facts stated in those excerpts (not general knowledge)
- Names concrete details: numbers, names, tools, claims, mechanisms, outcomes, etc.
- Is at least 8 words long
- Does NOT mention the video title, channel name, creator name, or phrases like "in the video", "this video", "the speaker in the video"
- Does NOT quote or paraphrase the title closely enough that search would trivially match the title field
- Excerpts may be partial sentence fragments; still ask about a concrete fact they contain (names, numbers, claims, comparisons)
- Never refuse or say you lack information — always output one question grounded in whatever text is present

Return ONLY valid JSON: {"query": "your question here"}"""


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parent
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=True)


def _make_llm_caller(model_override: Optional[str] = None) -> tuple[Callable[..., str], str]:
    """
    Prefer direct Anthropic API; fall back to OpenRouter (anthropic/claude-sonnet-4).
    """
    anthropic_model = model_override or ANTHROPIC_MODEL
    or_model = model_override or OPENROUTER_MODEL

    if (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        from anthropic_client import get_client

        client = get_client()

        def _call(
            system_prompt: str,
            user_prompt: str,
            *,
            max_tokens: int,
            temperature: float,
        ) -> str:
            resp = client.messages.create(
                model=anthropic_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            parts: List[str] = []
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts).strip()

        return _call, f"anthropic:{anthropic_model}"

    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY in .env to generate replacements."
        )
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    def _call_or(
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        resp = client.chat.completions.create(
            model=or_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    return _call_or, f"openrouter:{or_model}"


_WORD_RE = re.compile(r"[a-z0-9]+", re.I)


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _title_words(title: str) -> List[str]:
    return [w for w in _WORD_RE.findall((title or "").lower()) if len(w) >= 4]


_REFUSAL_RE = re.compile(
    r"unable to generate|insufficient (?:complete )?information|cannot (?:write|create)|"
    r"i (?:can'?t|cannot) (?:write|generate|create)",
    re.I,
)


def _valid_eval_query(q: str) -> bool:
    q = (q or "").strip()
    if len(q) < 20 or "?" not in q:
        return False
    if _REFUSAL_RE.search(q):
        return False
    return True


def _parse_query_json(raw: str) -> Optional[str]:
    text = (raw or "").strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                q = (obj.get("query") or "").strip()
                if q and _valid_eval_query(q):
                    return q
        except json.JSONDecodeError:
            pass
    # Fallback: whole response as question if it looks like one line.
    for line in text.splitlines():
        line = line.strip().strip('"')
        if "?" in line:
            qpart = line.split("?")[0].strip() + "?"
            if _valid_eval_query(qpart):
                return qpart
    return None


def _title_leaks(
    query: str, title: str, chunk_texts: Optional[List[str]] = None
) -> bool:
    """True if query echoes the title, not topic words that also appear in excerpts."""
    q = (query or "").lower()
    t = (title or "").strip()
    if not q or not t:
        return False
    if t.lower() in q:
        return True
    tn = _normalize(title)
    qn = _normalize(query)
    if tn and len(tn) >= 12 and tn in qn:
        return True
    corpus = _normalize(" ".join(chunk_texts or []))

    def _in_corpus(word: str) -> bool:
        if not corpus:
            return False
        if word in corpus:
            return True
        return len(word) >= 5 and word[:5] in corpus

    tw = _title_words(title)
    if not tw:
        return False
    leaked = [w for w in tw if w in qn and not _in_corpus(w)]
    if len(leaked) >= 2:
        return True
    if len(tw) <= 3 and leaked:
        return True
    return False


def _generate_question(
    llm: Callable[..., str],
    *,
    title: str,
    chunk_texts: List[str],
    old_query: str,
    video_ids: List[str],
    gold_note: str = "",
) -> str:
    chunks_block = "\n\n---\n\n".join(
        f"[excerpt {i + 1}]\n{(t or '').strip()}"
        for i, t in enumerate(chunk_texts)
        if (t or "").strip()
    )
    if not chunks_block.strip():
        chunks_block = "(no excerpt text provided — use any factual detail you can infer is specific to this video, avoid the title)"

    user_prompt = f"""VIDEO TITLE (for your context only — must NOT appear in the question):
{title}

RELEVANT VIDEO ID(S): {", ".join(video_ids) or "unknown"}

TRANSCRIPT EXCERPTS:
{chunks_block[:12000]}

OLD EVAL QUESTION (title-leaking template — do not copy):
{old_query}
"""
    if gold_note.strip():
        user_prompt += f"\n\nHINT (ground-truth snippet — base the question on this if excerpts are thin):\n{gold_note.strip()[:1500]}"
    user_prompt += "\n\nWrite one new specific factual question answerable only from the excerpts."

    raw = llm(
        SYSTEM_PROMPT,
        user_prompt,
        max_tokens=256,
        temperature=0.3,
    )
    q = _parse_query_json(raw)
    if not q:
        raw2 = llm(
            SYSTEM_PROMPT,
            user_prompt + '\n\nReply with ONLY this JSON on one line: {"query": "..."}',
            max_tokens=128,
            temperature=0.1,
        )
        q = _parse_query_json(raw2)
    if not q:
        raw3 = llm(
            SYSTEM_PROMPT,
            user_prompt
            + "\n\nExcerpts are fragmented. Ask one specific question about a name, claim, or comparison visible in the text.",
            max_tokens=256,
            temperature=0.4,
        )
        q = _parse_query_json(raw3)
    if not q:
        raise ValueError(f"Could not parse query from model output: {raw[:200]!r}")
    if _title_leaks(q, title, chunk_texts):
        banned = [w for w in _title_words(title) if w not in _normalize(" ".join(chunk_texts))]
        extra = f" Do not use these title-only words: {', '.join(banned[:12])}." if banned else ""
        retry_prompt = (
            user_prompt
            + "\n\nYour previous attempt still leaked the title."
            + extra
            + " Rewrite with zero overlap with the video title."
        )
        raw2 = llm(
            SYSTEM_PROMPT,
            retry_prompt,
            max_tokens=256,
            temperature=0.2,
        )
        q2 = _parse_query_json(raw2)
        if q2 and not _title_leaks(q2, title, chunk_texts):
            return q2
        raise ValueError(f"Generated query still leaks title: {q!r}")
    return q


def _resolve_queries_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir() and path.name != "queries" and (path / "queries").is_dir():
        return path / "queries"
    return path


def main() -> None:
    _load_dotenv()

    ap = argparse.ArgumentParser(
        description="Generate title-free fact_lookup replacements for tr_* eval JSON files."
    )
    ap.add_argument(
        "queries_dir",
        type=Path,
        help="Eval queries directory (e.g. eval_results/20260514_065902/queries)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max files to process (0 = all)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in replacements/",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to sleep between API calls",
    )
    ap.add_argument(
        "--model",
        default=None,
        help=f"Override model (default: {ANTHROPIC_MODEL} or {OPENROUTER_MODEL} via OpenRouter)",
    )
    args = ap.parse_args()

    queries_dir = _resolve_queries_dir(args.queries_dir)
    if not queries_dir.is_dir():
        print(f"Not a directory: {queries_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = queries_dir / "replacements"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        llm, backend = _make_llm_caller(args.model)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    paths = sorted(queries_dir.glob("tr_*.json"))
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    if not paths:
        print(f"No tr_*.json files in {queries_dir}", file=sys.stderr)
        sys.exit(1)

    ok = 0
    skipped = 0
    failed = 0

    print(f"Input : {queries_dir}")
    print(f"Output: {out_dir}")
    print(f"Backend: {backend}")
    print(f"Files : {len(paths)}\n")

    for i, path in enumerate(paths, 1):
        out_path = out_dir / path.name
        if args.skip_existing and out_path.is_file():
            print(f"[{i}/{len(paths)}] skip (exists) {path.name}")
            skipped += 1
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[{i}/{len(paths)}] FAIL read {path.name}: {exc}")
            failed += 1
            continue

        title = (data.get("source_video_title") or "").strip()
        video_ids = list(data.get("relevant_video_ids") or [])
        chunks = list(data.get("expected_chunk_texts") or [])
        old_query = (data.get("query") or "").strip()

        try:
            new_query = _generate_question(
                llm,
                title=title,
                chunk_texts=chunks,
                old_query=old_query,
                video_ids=video_ids,
                gold_note=(data.get("gold_answer_note") or ""),
            )
        except Exception as exc:
            print(f"[{i}/{len(paths)}] FAIL {path.name}: {exc}")
            failed += 1
            continue

        out_data = dict(data)
        out_data["query"] = new_query
        out_data["query_type"] = "fact_lookup"
        if "category" in out_data:
            out_data["category"] = "specific_fact"

        out_path.write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"[{i}/{len(paths)}] OK {path.name}")
        print(f"    Q: {new_query[:90]}{'...' if len(new_query) > 90 else ''}")
        ok += 1

        if args.delay > 0 and i < len(paths):
            time.sleep(args.delay)

    print(f"\nDone: {ok} written, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
