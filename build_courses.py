#!/usr/bin/env python3
"""
Generate structured course packages from clusters.json + full corpus SQLite.

Uses claude-sonnet-4-20250514 with aggressive disk caching (data/course_llm_cache/).
Full corpus DB: set ASK_SHORTY_FULL_DB or use default path in DEFAULT_FULL_DB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DEFAULT_FULL_DB = "C:/Users/number2/Desktop/youtube-history-viewer-copy/data/transcripts.db"
CLUSTERS_PATH = ROOT / "data" / "clusters.json"
COURSES_DIR = ROOT / "data" / "courses"
CATALOG_PATH = ROOT / "data" / "course_catalog.json"
LLM_CACHE_DIR = ROOT / "data" / "course_llm_cache"

COURSE_MODEL = "claude-sonnet-4-20250514"

# Merge / split overrides (cluster ids from clusters.json)
CLUSTER_OVERRIDES: Dict[str, Any] = {
    "merge": [
        ([3, 10, 22], "DIY Engineering and Making"),
        ([1, 15, 16, 24, 30, 32, 33], "US Politics and Crisis"),
        ([12, 13, 19, 20, 21], "AI Revolution"),
    ],
    "split": [
        (
            11,
            [
                "DIY Science: Chemistry and Materials",
                "DIY Science: Physics and Engineering",
                "DIY Science: Biology and Nature",
            ],
        ),
    ],
}

EmitFn = Optional[Callable[[dict], None]]


def _emit(emit: EmitFn, event: dict) -> None:
    if emit:
        emit(event)


def slugify(text: str) -> str:
    s = (text or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "course"


def _stable_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[Any]:
    path = LLM_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_set(key: str, value: Any) -> None:
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = LLM_CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _llm_cached(call_kind: str, payload: Any, fn: Callable[[], Any]) -> Any:
    h = _stable_hash({"kind": call_kind, "model": COURSE_MODEL, "payload": payload})
    cached = _cache_get(h)
    if cached is not None and "result" in cached:
        return cached["result"]
    result = fn()
    _cache_set(h, {"result": result, "cached_at": datetime.now().isoformat()})
    return result


def _call_claude_tool(
    tool_name: str,
    tool_schema: dict,
    system: str,
    user: str,
    max_tokens: int = 4096,
) -> dict:
    from anthropic_client import get_client

    client = get_client()
    tools = [
        {
            "name": tool_name,
            "description": "Structured output for course builder",
            "input_schema": tool_schema,
        }
    ]
    resp = client.messages.create(
        model=COURSE_MODEL,
        max_tokens=max_tokens,
        temperature=0.3,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=tools,
        tool_choice={"type": "tool", "name": tool_name},
    )
    for block in resp.content:
        btype = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
        if btype == "tool_use":
            name = getattr(block, "name", None) if not isinstance(block, dict) else block.get("name")
            if name != tool_name:
                continue
            inp = getattr(block, "input", None) if not isinstance(block, dict) else block.get("input")
            if isinstance(inp, dict):
                return inp
    raise RuntimeError(f"Claude did not return tool_use for {tool_name}")


# ---------------------------------------------------------------------------
# DB loading
# ---------------------------------------------------------------------------


@dataclass
class VideoRow:
    video_id: str
    title: str
    channel: str
    watch_date: str
    upload_date: str
    transcript_len: int
    shorty: str
    synthetic_questions: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)


def _parse_upload_date(json_metadata: Optional[str]) -> str:
    if not json_metadata:
        return ""
    try:
        meta = json.loads(json_metadata)
        return str(meta.get("upload_date") or "").strip()
    except Exception:
        return ""


def fetch_videos(conn: sqlite3.Connection, video_ids: Sequence[str]) -> Dict[str, VideoRow]:
    if not video_ids:
        return {}
    placeholders = ",".join("?" * len(video_ids))
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT v.video_id, v.title, v.channel, v.watch_date, v.json_metadata,
               LENGTH(COALESCE(t.text, '')) AS tlen, COALESCE(t.shorty, '')
        FROM videos v
        LEFT JOIN transcripts t ON t.video_id = v.video_id
        WHERE v.video_id IN ({placeholders})
        """,
        tuple(video_ids),
    )
    rows = {r[0]: r for r in cur.fetchall()}

    sq: Dict[str, List[str]] = {vid: [] for vid in video_ids}
    cur.execute(
        f"""
        SELECT video_id, question FROM synthetic_questions
        WHERE video_id IN ({placeholders})
        ORDER BY id ASC
        """,
        tuple(video_ids),
    )
    for vid, q in cur.fetchall():
        if vid in sq and q:
            sq[vid].append(q.strip())

    ent: Dict[str, List[Dict[str, str]]] = {vid: [] for vid in video_ids}
    cur.execute(
        f"""
        SELECT video_id, name, type FROM entities
        WHERE video_id IN ({placeholders})
        ORDER BY name ASC
        """,
        tuple(video_ids),
    )
    for vid, name, etype in cur.fetchall():
        if vid in ent:
            ent[vid].append({"name": name or "", "type": etype or "concept"})

    out: Dict[str, VideoRow] = {}
    for vid in video_ids:
        r = rows.get(vid)
        if not r:
            continue
        _, title, channel, watch_date, jmeta, tlen, shorty = r
        out[vid] = VideoRow(
            video_id=vid,
            title=title or vid,
            channel=channel or "",
            watch_date=(watch_date or "")[:10],
            upload_date=_parse_upload_date(jmeta),
            transcript_len=int(tlen or 0),
            shorty=(shorty or "").strip(),
            synthetic_questions=sq.get(vid, []),
            entities=ent.get(vid, []),
        )
    return out


def _upload_sort_key(upload: str) -> Tuple[int, str]:
    u = (upload or "").strip()
    if not u:
        return (2, "")  # missing last
    digits = re.sub(r"\D", "", u)
    if len(digits) >= 8:
        return (0, digits[:8])
    return (1, u)


def sort_video_ids_by_upload(video_ids: Sequence[str], rows: Dict[str, VideoRow]) -> List[str]:
    def key(vid: str) -> Tuple[int, str]:
        r = rows.get(vid)
        if not r:
            return (2, vid)
        return _upload_sort_key(r.upload_date)

    return sorted(list(video_ids), key=key)


# ---------------------------------------------------------------------------
# Course specs (merge / split / plain)
# ---------------------------------------------------------------------------


@dataclass
class CourseSpec:
    """One course to generate."""

    cluster_ids: List[int]
    title: str
    color: str
    video_ids: List[str]
    file_tag: str  # prefix for filename e.g. "2" or "merged_3_10_22"
    split_subtitles: Optional[List[str]] = None  # if splitting cluster 11, titles for sub-courses


def _cluster_by_id(clusters: List[dict], cid: int) -> Optional[dict]:
    for c in clusters:
        if int(c.get("id", -1)) == cid:
            return c
    return None


def _video_ids_from_cluster(cl: dict) -> List[str]:
    return [v["video_id"] for v in cl.get("videos", []) if v.get("video_id")]


def build_course_specs(cluster_data: dict) -> List[CourseSpec]:
    clusters = cluster_data.get("clusters", [])
    by_id = {int(c["id"]): c for c in clusters if "id" in c}
    merged_sets = {tuple(sorted(a)): title for a, title in CLUSTER_OVERRIDES.get("merge", [])}
    split_map = {int(x[0]): x[1] for x in CLUSTER_OVERRIDES.get("split", [])}

    used: set = set()
    specs: List[CourseSpec] = []

    # Merged courses first
    for ids_tuple, mtitle in merged_sets.items():
        ids = list(ids_tuple)
        vids: List[str] = []
        color = "#6366f1"
        for cid in ids:
            cl = by_id.get(cid)
            if not cl:
                continue
            color = cl.get("color") or color
            for vid in _video_ids_from_cluster(cl):
                if vid not in vids:
                    vids.append(vid)
            used.add(cid)
        tag = "merged_" + "_".join(str(i) for i in sorted(ids))
        specs.append(
            CourseSpec(
                cluster_ids=sorted(ids),
                title=mtitle,
                color=color,
                video_ids=vids,
                file_tag=tag,
            )
        )

    # Split cluster(s)
    for scid, subtitles in split_map.items():
        cl = by_id.get(scid)
        if not cl:
            continue
        used.add(scid)
        vids = _video_ids_from_cluster(cl)
        specs.append(
            CourseSpec(
                cluster_ids=[scid],
                title=cl.get("label") or f"Cluster {scid}",
                color=cl.get("color") or "#8b5cf6",
                video_ids=vids,
                file_tag=str(scid),
                split_subtitles=subtitles,
            )
        )

    # Remaining clusters
    for cl in clusters:
        cid = int(cl["id"])
        if cid in used:
            continue
        specs.append(
            CourseSpec(
                cluster_ids=[cid],
                title=cl.get("label") or f"Cluster {cid}",
                color=cl.get("color") or "#3b82f6",
                video_ids=_video_ids_from_cluster(cl),
                file_tag=str(cid),
            )
        )

    return specs


def _split_video_ids_with_llm(
    course_title: str,
    subtitles: List[str],
    summaries: List[dict],
    emit: EmitFn,
) -> List[Tuple[str, List[str]]]:
    """Returns [(sub_title, video_ids), ...] using cached LLM."""

    payload = {"course_title": course_title, "subtitles": subtitles, "summaries": summaries}

    def run() -> List[Tuple[str, List[str]]]:
        system = (
            "You partition videos into sub-courses. Each video_id must appear exactly once. "
            "Align groups with the given sub-course titles thematically."
        )
        user = json.dumps(payload, ensure_ascii=False, indent=2)
        schema = {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "video_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["title", "video_ids"],
                    },
                }
            },
            "required": ["assignments"],
        }
        out = _call_claude_tool("split_assignments", schema, system, user, max_tokens=2048)
        assigns = out.get("assignments") or []
        result: List[Tuple[str, List[str]]] = []
        for i, a in enumerate(assigns):
            t = a.get("title") or (subtitles[i] if i < len(subtitles) else f"Part {i+1}")
            vids = [str(x) for x in (a.get("video_ids") or []) if x]
            result.append((t, vids))
        return result

    _emit(emit, {"type": "progress", "step": "split", "message": "Splitting large cluster via LLM (cached if seen before)"})
    return _llm_cached("split_cluster", payload, run)


# ---------------------------------------------------------------------------
# Module planning + course body
# ---------------------------------------------------------------------------


def _module_plan_llm(
    course_title: str,
    ordered_videos: List[dict],
    emit: EmitFn,
) -> List[dict]:
    payload = {"course_title": course_title, "videos": ordered_videos}

    def run() -> List[dict]:
        system = (
            "You design 3–6 course modules. Group thematically similar videos. "
            "Preserve a sensible learning order (foundational first). "
            "Every video_id from the input must appear exactly once."
        )
        user = json.dumps(payload, ensure_ascii=False, indent=2)
        schema = {
            "type": "object",
            "properties": {
                "modules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "video_ids": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["title", "video_ids"],
                    },
                }
            },
            "required": ["modules"],
        }
        out = _call_claude_tool("module_plan", schema, system, user, max_tokens=2048)
        return out.get("modules") or []

    _emit(emit, {"type": "progress", "step": "modules", "message": "Planning modules (LLM, cached)"})
    return _llm_cached("module_plan", payload, run)


def _normalize_modules(
    raw_modules: List[dict],
    all_vids: List[str],
) -> List[dict]:
    """Ensure every video appears once; fix duplicates / drops."""
    seen: set = set()
    cleaned: List[dict] = []
    for m in raw_modules:
        title = (m.get("title") or "Module").strip()
        vids = []
        for x in m.get("video_ids") or []:
            vid = str(x)
            if vid in all_vids and vid not in seen:
                vids.append(vid)
                seen.add(vid)
        if vids:
            cleaned.append({"title": title, "video_ids": vids})
    for vid in all_vids:
        if vid not in seen:
            seen.add(vid)
            if cleaned:
                cleaned[-1]["video_ids"].append(vid)
            else:
                cleaned.append({"title": "Core lessons", "video_ids": [vid]})
    return cleaned


def _module_narrative_llm(
    course_title: str,
    module_title: str,
    lesson_titles: List[str],
    shorty_excerpts: List[str],
    emit: EmitFn,
) -> dict:
    cap = 800
    excerpts = [s[:cap] for s in shorty_excerpts]
    payload = {
        "course_title": course_title,
        "module_title": module_title,
        "lesson_titles": lesson_titles,
        "shorty_excerpts": excerpts,
    }

    def run() -> dict:
        system = (
            "Write educational module content. "
            "introduction: 200–300 words. summary: 100–150 words. "
            "learning_objectives: 3–5 concise bullets (strings)."
        )
        user = json.dumps(payload, ensure_ascii=False, indent=2)
        schema = {
            "type": "object",
            "properties": {
                "introduction": {"type": "string"},
                "learning_objectives": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
            "required": ["introduction", "learning_objectives", "summary"],
        }
        return _call_claude_tool("module_narrative", schema, system, user, max_tokens=2048)

    _emit(
        emit,
        {"type": "progress", "step": "module_text", "message": f"Module narrative: {module_title[:40]}…"},
    )
    return _llm_cached("module_narrative", payload, run)


def _quiz_batch_llm(
    module_title: str,
    pack: List[dict],
    emit: EmitFn,
) -> List[dict]:
    """pack: {video_id, title, shorty_excerpt, questions: []}"""
    payload = {"module_title": module_title, "lessons": pack}

    def run() -> List[dict]:
        system = (
            "Build exactly 5 assessment questions for this module. "
            "Prefer paraphrasing the provided synthetic questions when possible. "
            "Each needs: question, answer, type (multiple_choice|short_answer|essay), "
            "difficulty (easy|medium|hard), source_video_id. "
            "For multiple_choice, embed 4 choices in the question text as A/B/C/D lines."
        )
        user = json.dumps(payload, ensure_ascii=False, indent=2)
        schema = {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "type": {"type": "string"},
                            "difficulty": {"type": "string"},
                            "source_video_id": {"type": "string"},
                        },
                        "required": [
                            "question",
                            "answer",
                            "type",
                            "difficulty",
                            "source_video_id",
                        ],
                    },
                }
            },
            "required": ["questions"],
        }
        out = _call_claude_tool("module_quiz", schema, system, user, max_tokens=2500)
        return out.get("questions") or []

    _emit(emit, {"type": "progress", "step": "quiz", "message": f"Module quiz: {module_title[:36]}…"})
    return _llm_cached("module_quiz", payload, run)


def _course_wrap_llm(
    course_title: str,
    module_summaries: List[str],
    other_course_titles: List[str],
    glossary_seed: List[dict],
    emit: EmitFn,
) -> dict:
    payload = {
        "course_title": course_title,
        "module_summaries": module_summaries,
        "other_courses": other_course_titles,
        "entities": glossary_seed[:120],
    }

    def run() -> dict:
        system = (
            "You finalize the course. "
            "description: 2–3 sentences. level: Beginner|Intermediate|Advanced. "
            "prerequisites: pick zero or more titles from other_courses that should come first (strings). "
            "final_exam: exactly 10 questions synthesizing the whole course; same shape as module quiz. "
            "glossary: for each entity {name,type} produce term (name) and definition (1–2 sentences). "
            "Cap glossary at 40 most important terms if the list is long."
        )
        user = json.dumps(payload, ensure_ascii=False, indent=2)
        schema = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "level": {"type": "string"},
                "prerequisites": {"type": "array", "items": {"type": "string"}},
                "final_exam": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "answer": {"type": "string"},
                            "type": {"type": "string"},
                            "difficulty": {"type": "string"},
                            "source_video_id": {"type": "string"},
                        },
                        "required": [
                            "question",
                            "answer",
                            "type",
                            "difficulty",
                            "source_video_id",
                        ],
                    },
                },
                "glossary": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "term": {"type": "string"},
                            "definition": {"type": "string"},
                        },
                        "required": ["term", "definition"],
                    },
                },
            },
            "required": [
                "description",
                "level",
                "prerequisites",
                "final_exam",
                "glossary",
            ],
        }
        return _call_claude_tool("course_wrap", schema, system, user, max_tokens=6000)

    _emit(emit, {"type": "progress", "step": "wrap", "message": f"Course wrap + final exam: {course_title[:40]}…"})
    return _llm_cached("course_wrap", payload, run)


def _duration_estimate_chars(n: int) -> str:
    if n <= 0:
        return "~5 min"
    minutes = max(3, min(180, n // 1200))
    return f"~{minutes} min"


def _estimate_hours(rows: Dict[str, VideoRow]) -> float:
    total_chars = sum(r.transcript_len for r in rows.values())
    # ~750 chars/min consumption → hours
    hours = total_chars / 45000.0
    return round(max(0.5, hours), 1)


def _lesson_notes_from_shorty(shorty: str) -> str:
    t = (shorty or "").strip()
    if not t:
        return "_No Shorty available — open the video transcript on YouTube._"
    return t


def _discussion_questions(sqs: List[str]) -> List[str]:
    out = [q for q in sqs if q][:3]
    return out[:3] if out else ["What is the main claim or takeaway from this video?"]


def _top_entities(ents: List[dict], limit: int = 8) -> List[str]:
    names = []
    for e in ents:
        n = (e.get("name") or "").strip()
        if n and n not in names:
            names.append(n)
    return names[:limit]


def _make_course_id(cluster_ids: List[int], id_suffix: str = "") -> str:
    m = min(cluster_ids) if cluster_ids else 0
    return f"course_{m:03d}{id_suffix}"


def _build_course_inner(
    cluster_ids: List[int],
    course_title: str,
    color: str,
    file_tag: str,
    video_ids: List[str],
    conn: sqlite3.Connection,
    other_course_titles: List[str],
    emit: EmitFn,
    id_suffix: str = "",
) -> Tuple[dict, Path]:
    rows = fetch_videos(conn, video_ids)
    ordered = sort_video_ids_by_upload(video_ids, rows)
    summaries = [
        {
            "video_id": vid,
            "title": rows[vid].title,
            "channel": rows[vid].channel,
            "upload_date": rows[vid].upload_date,
            "shorty_excerpt": (rows[vid].shorty or "")[:400],
        }
        for vid in ordered
    ]
    raw_modules = _module_plan_llm(course_title, summaries, emit)
    modules_norm = _normalize_modules(raw_modules, ordered)

    modules_out: List[dict] = []
    mod_summaries_for_wrap: List[str] = []

    glossary_entities: List[dict] = []
    seen_e = set()
    for r in rows.values():
        for e in r.entities:
            key = (e.get("name") or "").lower()
            if key and key not in seen_e:
                seen_e.add(key)
                glossary_entities.append({"name": e.get("name"), "type": e.get("type", "concept")})

    for mi, mod in enumerate(modules_norm, start=1):
        mtitle = mod["title"]
        vids = mod["video_ids"]
        lesson_titles = [rows[v].title for v in vids if v in rows]
        shorties = [rows[v].shorty for v in vids if v in rows]
        narrative = _module_narrative_llm(course_title, mtitle, lesson_titles, shorties, emit)
        intro = narrative.get("introduction") or ""
        objectives = narrative.get("learning_objectives") or []
        summary = narrative.get("summary") or ""
        mod_summaries_for_wrap.append(f"{mtitle}: {summary[:300]}")

        quiz_pack = []
        for v in vids:
            if v not in rows:
                continue
            rr = rows[v]
            quiz_pack.append(
                {
                    "video_id": v,
                    "title": rr.title,
                    "shorty_excerpt": (rr.shorty or "")[:600],
                    "questions": rr.synthetic_questions[:12],
                }
            )
        q_raw = _quiz_batch_llm(mtitle, quiz_pack, emit)
        quiz: List[dict] = []
        for q in q_raw[:5]:
            quiz.append(
                {
                    "question": q.get("question", ""),
                    "answer": q.get("answer", ""),
                    "type": q.get("type", "short_answer"),
                    "difficulty": q.get("difficulty", "medium"),
                    "source_video_id": q.get("source_video_id") or (vids[0] if vids else ""),
                }
            )
        while len(quiz) < 5:
            quiz.append(
                {
                    "question": "What is one key idea from this module?",
                    "answer": "Answers will vary; review the lesson notes.",
                    "type": "short_answer",
                    "difficulty": "easy",
                    "source_video_id": vids[0] if vids else "",
                }
            )

        lessons: List[dict] = []
        for li, v in enumerate(vids, start=1):
            if v not in rows:
                continue
            rr = rows[v]
            lessons.append(
                {
                    "number": li,
                    "title": rr.title,
                    "video_id": v,
                    "video_url": f"https://www.youtube.com/watch?v={v}",
                    "channel": rr.channel,
                    "watch_date": rr.watch_date or "",
                    "duration_estimate": _duration_estimate_chars(rr.transcript_len),
                    "key_concepts": _top_entities(rr.entities),
                    "lesson_notes": _lesson_notes_from_shorty(rr.shorty),
                    "discussion_questions": _discussion_questions(rr.synthetic_questions),
                }
            )

        modules_out.append(
            {
                "number": mi,
                "title": mtitle,
                "learning_objectives": objectives[:8],
                "introduction": intro,
                "lessons": lessons,
                "quiz": quiz,
                "summary": summary,
            }
        )

    wrap = _course_wrap_llm(course_title, mod_summaries_for_wrap, other_course_titles, glossary_entities, emit)
    final_exam_raw = wrap.get("final_exam") or []
    final_exam: List[dict] = []
    for q in final_exam_raw[:10]:
        final_exam.append(
            {
                "question": q.get("question", ""),
                "answer": q.get("answer", ""),
                "type": q.get("type", "short_answer"),
                "difficulty": q.get("difficulty", "medium"),
                "source_video_id": q.get("source_video_id") or "",
            }
        )
    while len(final_exam) < 10:
        final_exam.append(
            {
                "question": "Describe how the themes of this course connect across modules.",
                "answer": "Sample: foundational ideas build toward applications discussed in later modules.",
                "type": "essay",
                "difficulty": "medium",
                "source_video_id": "",
            }
        )

    glossary = wrap.get("glossary") or []
    slug = slugify(course_title)
    course_id = _make_course_id(cluster_ids, id_suffix)

    estimated = _estimate_hours(rows)
    chans = Counter((r.channel or "").strip() for r in rows.values() if (r.channel or "").strip())
    top_channels = [n for n, _ in chans.most_common(3)]
    course = {
        "id": course_id,
        "slug": slug,
        "cluster_ids": cluster_ids,
        "color": color,
        "title": course_title,
        "description": wrap.get("description") or "",
        "level": wrap.get("level") or "Intermediate",
        "estimated_hours": estimated,
        "modules": modules_out,
        "glossary": glossary,
        "final_exam": final_exam,
        "prerequisites": wrap.get("prerequisites") or [],
        "top_channels": top_channels,
        "generated_at": datetime.now().isoformat(),
        "source_db": os.environ.get("ASK_SHORTY_FULL_DB", DEFAULT_FULL_DB),
    }

    COURSES_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{file_tag}_{slug}.json"
    out_path = COURSES_DIR / fname
    out_path.write_text(json.dumps(course, ensure_ascii=False, indent=2), encoding="utf-8")
    return course, out_path


def generate_from_specs(
    specs: List[CourseSpec],
    full_db: str,
    emit: EmitFn = None,
) -> List[dict]:
    conn = sqlite3.connect(full_db)
    built: List[dict] = []
    all_titles: List[str] = []
    for sp in specs:
        all_titles.append(sp.title)
    try:
        for spec in specs:
            if spec.split_subtitles:
                rows = fetch_videos(conn, spec.video_ids)
                ordered = sort_video_ids_by_upload(spec.video_ids, rows)
                summaries = [
                    {
                        "video_id": vid,
                        "title": rows[vid].title,
                        "channel": rows[vid].channel,
                        "upload_date": rows[vid].upload_date,
                        "shorty_excerpt": (rows[vid].shorty or "")[:400],
                    }
                    for vid in ordered
                ]
                parts = _split_video_ids_with_llm(spec.title, spec.split_subtitles, summaries, emit)
                for si, (sub_title, vids) in enumerate(parts):
                    if not vids:
                        continue
                    tag = f"{spec.file_tag}_part{si+1}"
                    _emit(
                        emit,
                        {
                            "type": "progress",
                            "step": "course",
                            "message": f"Building split course: {sub_title}",
                        },
                    )
                    course, path = _build_course_inner(
                        spec.cluster_ids,
                        sub_title,
                        spec.color,
                        tag,
                        vids,
                        conn,
                        [t for t in all_titles if t != sub_title],
                        emit,
                        id_suffix=f"_s{si+1}",
                    )
                    built.append({"course": course, "path": str(path)})
            else:
                _emit(
                    emit,
                    {"type": "progress", "step": "course", "message": f"Building: {spec.title}"},
                )
                course, path = _build_course_inner(
                    spec.cluster_ids,
                    spec.title,
                    spec.color,
                    spec.file_tag,
                    spec.video_ids,
                    conn,
                    [t for t in all_titles if t != spec.title],
                    emit,
                    id_suffix="",
                )
                built.append({"course": course, "path": str(path)})
    finally:
        conn.close()
    write_catalog(built)
    return built


def write_catalog(built: List[dict]) -> None:
    courses = []
    total_lessons = 0
    total_hours = 0.0
    for item in built:
        c = item["course"]
        mods = c.get("modules") or []
        nless = sum(len(m.get("lessons") or []) for m in mods)
        total_lessons += nless
        total_hours += float(c.get("estimated_hours") or 0)
        p = Path(item["path"]).resolve()
        try:
            file_rel = p.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            file_rel = p.as_posix()
        courses.append(
            {
                "id": c["id"],
                "cluster_ids": c.get("cluster_ids", []),
                "title": c.get("title", ""),
                "slug": c.get("slug", ""),
                "level": c.get("level", ""),
                "module_count": len(mods),
                "lesson_count": nless,
                "estimated_hours": c.get("estimated_hours", 0),
                "generated_at": c.get("generated_at", ""),
                "file": file_rel,
                "color": c.get("color", "#3b82f6"),
                "top_channels": c.get("top_channels", []),
                "description": (c.get("description") or "")[:220],
            }
        )
    catalog = {
        "generated_at": datetime.now().isoformat(),
        "total_courses": len(courses),
        "total_lessons": total_lessons,
        "total_hours": round(total_hours, 1),
        "courses": sorted(courses, key=lambda x: x.get("title", "")),
    }
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


def load_catalog() -> Optional[dict]:
    if not CATALOG_PATH.exists():
        return None
    try:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def rebuild_catalog_from_disk() -> dict:
    """Scan data/courses/*.json and rebuild course_catalog.json."""
    COURSES_DIR.mkdir(parents=True, exist_ok=True)
    built = []
    for p in sorted(COURSES_DIR.glob("*.json")):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            built.append({"course": c, "path": str(p)})
        except Exception:
            continue
    write_catalog(built)
    return load_catalog() or {}


def _spec_has_existing_output(spec: CourseSpec) -> bool:
    """True if data/courses/{file_tag}_*.json exists (split parts use same prefix + _partN)."""
    COURSES_DIR.mkdir(parents=True, exist_ok=True)
    pat = f"{spec.file_tag}_*.json"
    return any(COURSES_DIR.glob(pat))


def run_generation(
    cluster_ids: Optional[List[int]] = None,
    all_clusters: bool = False,
    full_db: Optional[str] = None,
    emit: EmitFn = None,
    missing_only: bool = False,
) -> List[dict]:
    db_path = full_db or os.environ.get("ASK_SHORTY_FULL_DB") or DEFAULT_FULL_DB
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Full corpus DB not found: {db_path}")

    data = json.loads(CLUSTERS_PATH.read_text(encoding="utf-8"))
    specs = build_course_specs(data)
    if not all_clusters and cluster_ids is not None:
        want = set(cluster_ids)
        specs = [s for s in specs if any(cid in want for cid in s.cluster_ids)]
    if missing_only:
        specs = [s for s in specs if not _spec_has_existing_output(s)]
        _emit(emit, {"type": "progress", "step": "filter", "message": f"Missing-only: {len(specs)} course(s) to build"})
        if not specs:
            _emit(emit, {"type": "progress", "step": "done", "message": "All courses already exist on disk"})
            rebuild_catalog_from_disk()
            return []
    return generate_from_specs(specs, db_path, emit=emit)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build courses from clusters.json")
    ap.add_argument(
        "--clusters",
        type=str,
        default="",
        help="Comma-separated cluster ids (e.g. 2,7,18). Omit with --all for everything.",
    )
    ap.add_argument("--all", action="store_true", help="Generate all courses (merges/splits included)")
    ap.add_argument("--db", type=str, default=None, help="Override full corpus SQLite path")
    ap.add_argument(
        "--rebuild-catalog",
        action="store_true",
        help="Only rescan data/courses/*.json into course_catalog.json",
    )
    ap.add_argument(
        "--missing-only",
        action="store_true",
        help="With --all, skip specs that already have data/courses/{file_tag}_*.json",
    )
    args = ap.parse_args()
    if args.rebuild_catalog:
        cat = rebuild_catalog_from_disk()
        print(json.dumps({"ok": True, "catalog": cat.get("total_courses", 0)}, indent=2))
        return

    if args.all:
        ids = None
        all_c = True
    else:
        raw = (args.clusters or "").strip()
        if not raw:
            print("Specify --clusters 2 or --all")
            return
        ids = [int(x.strip()) for x in raw.split(",") if x.strip()]
        all_c = False

    def emit(e: dict) -> None:
        print(json.dumps(e, ensure_ascii=False))

    out = run_generation(
        cluster_ids=ids,
        all_clusters=all_c,
        full_db=args.db,
        emit=emit,
        missing_only=args.missing_only and all_c,
    )
    print(json.dumps({"built": len(out), "paths": [o["path"] for o in out]}, indent=2))


if __name__ == "__main__":
    main()
