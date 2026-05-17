#!/usr/bin/env python3
"""
Ask Shorty V2 — hierarchical retrieval + answer (additive; does not replace V1).
"""

from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from ask_shorty import _call_answer_text, _extract_json_array

from query_router import QueryRouter
from reranker import RERANK_MODEL_NAME
from v2_bm25 import GenericBM25Index, default_v2_index_dir, load_shared_bm25_index
from v2_memory_maps import V2MemoryMaps
from v2_schema import ensure_v2_tables

# Video BM25: wide recall, then capped merge for segment search — final LLM blocks stay modest.
VIDEO_BM25_TOP_K = 100
VIDEO_CAND_MERGED_CAP = 100
# Global segment BM25: top-N distinct videos to merge into the pool after video-level BM25.
SEGMENT_RESCUE_VIDEO_TOP = 20
# When the merge pool is already at cap, swap out this many lowest video-BM25 slots for rescue hits.
SEGMENT_RESCUE_REPLACE_COUNT = 10
# How many segment hits to scan per query string before collapsing to videos (need enough diversity).
SEGMENT_RESCUE_SEGMENT_PER_QUERY = 300
# Max segment (doc, score) pairs from segment BM25 before pool / rerank (limits downstream work).
SEGMENT_BM25_TOP_K = 150
SEGMENTS_PER_VIDEO = 6
EVENT_TOP = 12
RERANK_POOL = 100
# Cross-encoder is O(n) in batch size; keep n small for sub‑second latency.
CROSS_ENCODER_MAX_CANDIDATES = max(1, int(os.environ.get("V2_CROSS_ENCODER_MAX", "20")))

# Fast search (POST /api/search_fast): no LLM, no CE; tighter caps for interactive latency.
FAST_SEARCH_TOP_K = max(5, int(os.environ.get("V2_FAST_SEARCH_TOP_K", "20")))
FAST_VIDEO_BM25_TOP_K = 50
FAST_VIDEO_CAND_CAP = 50
FAST_SEGMENT_BM25_TOP_K = 80
FAST_SEGMENT_RESCUE_VIDEO_TOP = 10
FAST_SEGMENT_RESCUE_SEGMENT_PER_QUERY = 150
FAST_SEGMENT_RESCUE_REPLACE_COUNT = 5
# Skip CE when top video BM25 clearly dominates mid-ranked scores (saves ~seconds).
CE_SKIP_BM25_DOMINANCE_RATIO = float(os.environ.get("V2_CE_SKIP_BM25_RATIO", "1.42"))
CE_SKIP_BM25_MIN_TOP_SCORE = float(os.environ.get("V2_CE_SKIP_BM25_MIN_TOP", "3.5"))

LLM_VIDEOS = 10

_V2_QUERY_EXPAND_SYSTEM = (
    "You return only a JSON array of strings. No markdown fences, no explanation."
)

_v2_log_lock = threading.Lock()
V2_LOG_MARKER = "<!--V2_LOG_ENTRIES_TOP-->"

# Process-wide RAM maps (full segment_index scan, etc.) — one per resolved DB path.
_v2_maps_lock = threading.Lock()
_v2_maps_singleton: Dict[str, V2MemoryMaps] = {}

# Topic tokens for context filtering (title + Shorty must mention enough of these).
_TOPIC_TAIL = re.compile(
    r"\b(?:about|regarding|on|related\s+to|for)\s+(.+?)(?:\?[.!]*\s*$|[.!]\s*$|$)",
    re.I | re.S,
)
_V2_TOPIC_STOP = frozenset(
    {
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "how",
        "why",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "been",
        "being",
        "i",
        "my",
        "me",
        "we",
        "our",
        "you",
        "your",
        "last",
        "thing",
        "things",
        "video",
        "videos",
        "one",
        "watch",
        "watched",
        "watching",
        "seen",
        "saw",
        "most",
        "recent",
        "latest",
        "first",
        "earliest",
        "please",
        "tell",
        "give",
        "show",
        "find",
        "some",
        "any",
        "all",
        "just",
        "ever",
        "anything",
        "something",
        "stuff",
    }
)


def _extract_topic_match_tokens(question: str) -> List[str]:
    """Lowercase tokens after about/on/… or, if none, longest non-stopwords from the query."""
    low = (question or "").strip().lower()
    if not low:
        return []
    m = _TOPIC_TAIL.search(low)
    if m:
        tail = (m.group(1) or "").strip().strip("'\"")
        toks = [
            t.lower()
            for t in re.findall(r"[a-z0-9]+", tail, re.I)
            if len(t) >= 3 and t.lower() not in _V2_TOPIC_STOP
        ]
        out: List[str] = []
        for t in toks:
            if t not in out:
                out.append(t)
        return out[:16]
    toks = [
        t.lower()
        for t in re.findall(r"[a-z0-9]+", low, re.I)
        if len(t) >= 4 and t.lower() not in _V2_TOPIC_STOP
    ]
    out2: List[str] = []
    for t in toks:
        if t not in out2:
            out2.append(t)
    return out2[:12]


def _token_word_boundary_hit(text: str, tok: str) -> bool:
    """True if tok appears as a whole token (not a substring inside a longer word)."""
    if not tok or not text:
        return False
    return (
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(tok)}(?![A-Za-z0-9])",
            text,
            re.I,
        )
        is not None
    )


def _score_topic_title_shorty(
    title_low: str, shorty_low: str, tokens: List[str]
) -> Optional[Tuple[int, int]]:
    """
    Every token must match (word-boundary) in title and/or Shorty.
    Returns (title_hit_count, shorty_only_hit_count) or None if any token misses.
    Title hits count tokens matched in the title (preferred for ordering).
    """
    if not tokens:
        return (0, 0)
    title_hits = 0
    shorty_only = 0
    for t in tokens:
        in_t = _token_word_boundary_hit(title_low, t)
        in_s = _token_word_boundary_hit(shorty_low, t)
        if not in_t and not in_s:
            return None
        if in_t:
            title_hits += 1
        elif in_s:
            shorty_only += 1
    return (title_hits, shorty_only)


def _select_llm_context_video_ids(
    db_path: str, ranked: List[str], question: str
) -> Tuple[List[str], List[str], int]:
    """
    Prefer videos whose title or Shorty mention every topic token as whole words;
    rank title matches above Shorty-only matches. No padding with unrelated hits.
    Returns (selected_ids, tokens_used, matched_count).
    """
    tokens = _extract_topic_match_tokens(question)
    if not ranked:
        return [], tokens, 0
    if not tokens:
        return ranked[:LLM_VIDEOS], [], 0
    cap = min(len(ranked), max(LLM_VIDEOS * 20, 200))
    scan = ranked[:cap]
    placeholders = ",".join("?" * len(scan))
    title_by: Dict[str, str] = {}
    shorty_by: Dict[str, str] = {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT v.video_id, v.title, COALESCE(t.shorty, '') AS shorty
            FROM videos v
            JOIN transcripts t ON t.video_id = v.video_id
            WHERE v.video_id IN ({placeholders})
            """,
            tuple(scan),
        )
        for row in cur.fetchall():
            vid = str(row["video_id"])
            title_by[vid] = ((row["title"] or "") or "").lower()
            shorty_by[vid] = ((row["shorty"] or "") or "").lower()
    rows: List[Tuple[int, int, int, str]] = []
    for i, vid in enumerate(scan):
        sc = _score_topic_title_shorty(
            title_by.get(vid, ""), shorty_by.get(vid, ""), tokens
        )
        if sc is None:
            continue
        th, sho = sc
        rows.append((-th, -sho, i, vid))
    rows.sort()
    matched = [r[3] for r in rows[:LLM_VIDEOS]]
    if matched:
        return matched, tokens, len(matched)
    return ranked[:LLM_VIDEOS], tokens, 0


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _v2_log_path() -> Path:
    return _project_root() / "data" / "v2_request_log.html"


def v2_log_patch_ref_id(job_id: int, ref_id: str) -> None:
    """Replace pending ref placeholder for a completed V2 job (call from worker after allocate)."""
    path = _v2_log_path()
    with _v2_log_lock:
        if not path.is_file():
            return
        text = path.read_text(encoding="utf-8")
        rid = html.escape(ref_id or "", quote=False)
        pat = rf'(<span id="v2-ref-{int(job_id)}">)pending(</span>)'

        def rep(m: re.Match[str]) -> str:
            return m.group(1) + rid + m.group(2)

        new_t, n = re.subn(pat, rep, text, count=1)
        if n:
            path.write_text(new_t, encoding="utf-8")


def _append_v2_request_html(
    *,
    job_id: Optional[int],
    ref_id_display: str,
    ts_iso: str,
    query: str,
    route_type: str,
    route_reason: str,
    video_bm25_ranked_count: int,
    merged_candidate_count: int,
    segment_bm25_pairs_scored: int,
    segments_kept_total: int,
    context_passage_count: int,
    candidate_video_ids: List[str],
    full_system: str,
    full_user: str,
    model_response: str,
    timing_ms: int,
    corpus_total_videos: int,
    corpus_with_shorty: int,
    extra_debug: Optional[Dict[str, Any]] = None,
) -> None:
    path = _v2_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    def he(s: str) -> str:
        return html.escape(s or "", quote=True)

    jb = str(job_id) if job_id is not None else "—"
    ref_span = (
        f'<span id="v2-ref-{job_id}">pending</span>'
        if job_id is not None
        else he(ref_id_display)
    )

    dbg_pre = ""
    if extra_debug is not None:
        try:
            dbg_pre = f"<pre>{he(json.dumps(extra_debug, indent=2, ensure_ascii=False))}</pre>\n"
        except Exception:
            dbg_pre = ""

    section = f"""<details class="v2q" data-job-id="{he(jb)}" id="v2job-{he(jb)}">
<summary><strong>{he(ts_iso)}</strong> — job <code>{he(jb)}</code> — ref: {ref_span}</summary>
<div class="v2meta">
  route: <code>{he(route_type)}</code> ({he(route_reason)}) ·
  video BM25 ranked: {video_bm25_ranked_count} ·
  merged candidates: {merged_candidate_count} ·
  segment BM25 scores: {segment_bm25_pairs_scored} ·
  segments kept: {segments_kept_total} ·
  context blocks to LLM: {context_passage_count} ·
  time: {timing_ms} ms<br/>
  corpus: {corpus_total_videos} videos indexed · {corpus_with_shorty} with Shorties ·
  {corpus_total_videos - corpus_with_shorty} without Shorty yet
</div>
<p><b>Query</b></p><pre>{he(query)}</pre>
<p><b>Candidate video IDs</b> ({len(candidate_video_ids)})</p>
<pre>{he(", ".join(candidate_video_ids[:200]) + ("…" if len(candidate_video_ids) > 200 else ""))}</pre>
<p><b>System prompt</b></p><pre>{he(full_system)}</pre>
<p><b>User prompt</b></p><pre>{he(full_user)}</pre>
<p><b>Model response</b></p><pre>{he(model_response)}</pre>
{dbg_pre}
</details>
"""

    with _v2_log_lock:
        if path.is_file():
            old = path.read_text(encoding="utf-8")
            if V2_LOG_MARKER in old:
                new = old.replace(V2_LOG_MARKER, section + "\n" + V2_LOG_MARKER, 1)
            else:
                new = section + "\n" + old
        else:
            new = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Ask Shorty V2 — request log</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 24px auto; padding: 0 12px;
            background: #0f172a; color: #e2e8f0; }}
    h1 {{ font-size: 1.25rem; }}
    .sub {{ color: #94a3b8; font-size: 0.9rem; }}
    details.v2q {{ margin: 12px 0; border: 1px solid #334155; border-radius: 8px; padding: 8px 12px;
                   background: #020617; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    .v2meta {{ font-size: 12px; color: #94a3b8; margin-bottom: 8px; line-height: 1.5; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #020617; padding: 8px;
           border-radius: 6px; overflow: auto; max-height: 480px; border: 1px solid #1e293b; }}
    code {{ font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>Ask Shorty V2 — request / response log</h1>
  <p class="sub">Newest entries first. Open a section for full prompts and outputs.</p>
{V2_LOG_MARKER}
</body>
</html>
""".replace(
                V2_LOG_MARKER, section + "\n" + V2_LOG_MARKER, 1
            )
        path.write_text(new, encoding="utf-8")


def _resolve_db_path() -> str:
    return os.environ.get("ASK_SHORTY_DB_PATH") or "data/transcripts.db"


def _corpus_shorty_counts(conn: sqlite3.Connection) -> Tuple[int, int]:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM videos")
    total_v = int(cur.fetchone()[0])
    cur.execute(
        """
        SELECT COUNT(DISTINCT v.video_id) FROM videos v
        INNER JOIN transcripts t ON t.video_id = v.video_id
        WHERE t.shorty IS NOT NULL AND trim(t.shorty) != ''
        """
    )
    with_s = int(cur.fetchone()[0])
    return total_v, with_s


def _watch_date_range_for_videos(
    conn: sqlite3.Connection, video_ids: Sequence[str]
) -> Tuple[Optional[str], Optional[str]]:
    if not video_ids:
        return None, None
    qmarks = ",".join("?" for _ in video_ids)
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT watch_date FROM videos
        WHERE video_id IN ({qmarks}) AND watch_date IS NOT NULL AND trim(watch_date) != ''
        """,
        tuple(video_ids),
    )
    rows = [r[0] for r in cur.fetchall() if r[0]]
    if not rows:
        return None, None
    srt = sorted(str(x)[:10] for x in rows)
    return srt[0], srt[-1]


def _resolved_db_path(db_path: str) -> str:
    return str(Path(db_path).resolve())


def _median_float(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    m = n // 2
    if n % 2:
        return float(s[m])
    return 0.5 * (float(s[m - 1]) + float(s[m]))


def _expand_query(question: str) -> List[str]:
    """
    LLM-generated alternative phrasings for BM25. Original query is always first.
    On any failure, returns a single-element list with the trimmed original.
    """
    q = (question or "").strip()
    if not q:
        return []
    try:
        user_prompt = (
            "Given this search query, generate 2 alternative phrasings using different vocabulary. "
            "Return only a JSON array of strings, no explanation. Original: "
            + q
        )
        raw = _call_answer_text(
            _V2_QUERY_EXPAND_SYSTEM,
            user_prompt,
            max_tokens=100,
            temperature=0.2,
        )
        arr = _extract_json_array(raw) or []
        out: List[str] = [q]
        seen: Set[str] = {q.lower()}
        for s in arr:
            t = (s or "").strip()
            if not t:
                continue
            low = t.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(t)
            if len(out) >= 4:
                break
        return out
    except Exception:
        return [q]


def _merge_video_bm25_scores(
    per_query: Sequence[List[Tuple[str, float]]],
    top_k: int,
) -> List[Tuple[str, float]]:
    best: Dict[str, float] = {}
    for lst in per_query:
        for vid, sc in lst:
            f = float(sc)
            prev = best.get(vid)
            if prev is None or f > prev:
                best[vid] = f
    return sorted(best.items(), key=lambda x: -x[1])[:top_k]


def _merge_segment_bm25_scores(
    per_query: Sequence[List[Tuple[str, float]]],
    top_k: int,
) -> List[Tuple[str, float]]:
    best: Dict[str, float] = {}
    for lst in per_query:
        for sid, sc in lst:
            f = float(sc)
            prev = best.get(sid)
            if prev is None or f > prev:
                best[sid] = f
    return sorted(best.items(), key=lambda x: -x[1])[:top_k]


def _segment_rescue_top_videos(
    seg_bm25: Optional[GenericBM25Index],
    query_strings: Sequence[str],
    *,
    n_videos: int,
    segments_per_query: int,
) -> List[str]:
    """
    Full-index segment BM25 (no video filter); return up to n_videos video_ids
    ranked by best segment BM25 score per video (after merging query variants).
    """
    if seg_bm25 is None:
        return []
    qs = [(s or "").strip() for s in query_strings if (s or "").strip()]
    if not qs:
        return []
    per = [seg_bm25.search(sq, segments_per_query) for sq in qs]
    merged_cap = min(segments_per_query * len(qs), 1200)
    merged = _merge_segment_bm25_scores(per, merged_cap)
    pls = seg_bm25.payload
    s_doc = list(pls.get("doc_ids") or [])
    s_vid = list(pls.get("video_ids") or [])
    sid_to_vid = {
        s_doc[i]: s_vid[i] for i in range(min(len(s_doc), len(s_vid)))
    }
    best_vid: Dict[str, float] = {}
    for sid, sc in merged:
        v = sid_to_vid.get(sid)
        if not v:
            continue
        f = float(sc)
        if f > best_vid.get(v, 0.0):
            best_vid[v] = f
    ranked = sorted(best_vid.items(), key=lambda x: -x[1])[:n_videos]
    return [vid for vid, _ in ranked]


def _v2_rank_score_for_video(
    vid: str,
    *,
    video_bm25_by: Dict[str, float],
    ce_by: Dict[str, float],
    max_seg_score: Callable[[str], float],
    evt_bonus: Dict[str, float],
    ce_fast_path: bool,
) -> float:
    """
    Per-video retrieval score for eval artifacts: CE score when reranked,
    else max(video BM25, segment BM25) with optional event bonus (matches CE path).
    """
    if vid in ce_by:
        return ce_by[vid]
    vb = video_bm25_by.get(vid, 0.0)
    ss = max_seg_score(vid)
    eb = min(2.0, evt_bonus.get(vid, 0.0) * 0.05)
    if ce_fast_path:
        return ss if ss > 0 else vb
    return max(vb, ss) + eb


def _bm25_confidence_skip_cross_encoder(
    vid_ranked: List[Tuple[str, float]],
) -> Tuple[bool, str]:
    """
    If the best video BM25 score is already well above a mid-pack median,
    cross-encoder rerank rarely changes ordering enough to justify latency.
    """
    if len(vid_ranked) < 6:
        return False, "bm25_lt6"
    scores = [float(s) for _, s in vid_ranked[: min(24, len(vid_ranked))]]
    top = scores[0]
    if top <= 0:
        return False, "top_nonpositive"
    body = scores[3:12]
    if len(body) < 3:
        body = scores[1:6]
    pos_body = [x for x in body if x > 0]
    med_body = _median_float(pos_body if pos_body else body)
    med_body = max(med_body, 1e-9)
    ratio = top / med_body
    if ratio >= CE_SKIP_BM25_DOMINANCE_RATIO and top >= CE_SKIP_BM25_MIN_TOP_SCORE:
        return True, f"ratio={ratio:.2f},top={top:.2f}"
    if ratio >= 1.75 and top >= 2.0:
        return True, f"ratio={ratio:.2f},top={top:.2f}"
    return False, f"ratio={ratio:.2f},top={top:.2f}"


def _get_shared_v2_memory_maps(db_path: str) -> V2MemoryMaps:
    """
    One V2MemoryMaps per resolved DB path for the whole process.
    Avoids re-scanning SQLite (especially segment_index) on every new AskShortyV2 instance.
    """
    key = _resolved_db_path(db_path)
    with _v2_maps_lock:
        hit = _v2_maps_singleton.get(key)
        if hit is not None:
            return hit
        with sqlite3.connect(key) as conn:
            ensure_v2_tables(conn)
        mm = V2MemoryMaps.load(key)
        _v2_maps_singleton[key] = mm
        return mm


class AskShortyV2:
    def __init__(self, db_path: Optional[str] = None) -> None:
        raw = db_path or _resolve_db_path()
        self.db_path = _resolved_db_path(raw)
        self.index_dir = Path(
            os.environ.get("ASK_SHORTY_V2_INDEX_DIR") or default_v2_index_dir(self.db_path)
        ).resolve()
        self.router = QueryRouter()
        self._maps: Optional[V2MemoryMaps] = None
        self._video_bm25: Optional[GenericBM25Index] = None
        self._seg_bm25: Optional[GenericBM25Index] = None
        self._evt_bm25: Optional[GenericBM25Index] = None
        self._ce: Any = None

    def _lazy_load(self) -> None:
        if self._maps is not None:
            return
        self._maps = _get_shared_v2_memory_maps(self.db_path)
        vpath = self.index_dir / "video_bm25_index.pkl"
        spath = self.index_dir / "segment_bm25_index.pkl"
        epath = self.index_dir / "event_bm25_index.pkl"
        self._video_bm25 = load_shared_bm25_index(vpath)
        self._seg_bm25 = load_shared_bm25_index(spath)
        self._evt_bm25 = load_shared_bm25_index(epath)

    def _get_cross_encoder(self) -> Any:
        if self._ce is None:
            from sentence_transformers import CrossEncoder

            self._ce = CrossEncoder(RERANK_MODEL_NAME)
        return self._ce

    def retrieve_videos(
        self,
        question: str,
        restrict_videos: Optional[Sequence[str]] = None,
        *,
        fast_mode: bool = False,
        fast_top_k: int = FAST_SEARCH_TOP_K,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Hierarchical retrieval (steps 1–6). Uses LLM query variants unless fast_mode."""
        t_retrieve0 = time.perf_counter()

        def _ms_since(t: float) -> float:
            return round((time.perf_counter() - t) * 1000.0, 2)

        timing: Dict[str, Any] = {}
        if fast_mode:
            v_bm25_top = FAST_VIDEO_BM25_TOP_K
            v_cand_cap = FAST_VIDEO_CAND_CAP
            seg_rescue_n = FAST_SEGMENT_RESCUE_VIDEO_TOP
            seg_rescue_per_q = FAST_SEGMENT_RESCUE_SEGMENT_PER_QUERY
            seg_rescue_replace = FAST_SEGMENT_RESCUE_REPLACE_COUNT
            seg_bm25_top_k = FAST_SEGMENT_BM25_TOP_K
            rerank_pool = max(1, int(fast_top_k))
        else:
            v_bm25_top = VIDEO_BM25_TOP_K
            v_cand_cap = VIDEO_CAND_MERGED_CAP
            seg_rescue_n = SEGMENT_RESCUE_VIDEO_TOP
            seg_rescue_per_q = SEGMENT_RESCUE_SEGMENT_PER_QUERY
            seg_rescue_replace = SEGMENT_RESCUE_REPLACE_COUNT
            seg_bm25_top_k = SEGMENT_BM25_TOP_K
            rerank_pool = RERANK_POOL

        t_lazy = time.perf_counter()
        self._lazy_load()
        timing["lazy_load_ms"] = _ms_since(t_lazy)

        q = (question or "").strip()
        low = q.lower()

        t_cl = time.perf_counter()
        route_res = self.router.classify(q)
        route = route_res.route_type
        timing["classify_ms"] = _ms_since(t_cl)
        if not fast_mode:
            print(
                f"[ask_shorty_v2] V2 route={route!r} reason={route_res.reason!r}",
                flush=True,
            )

        t_exp = time.perf_counter()
        if fast_mode:
            query_variants = [q] if q else []
            timing["query_expand_ms"] = 0.0
        else:
            try:
                query_variants = _expand_query(q) if q else []
            except Exception:
                query_variants = [q] if q else []
            if q and not query_variants:
                query_variants = [q]
            timing["query_expand_ms"] = _ms_since(t_exp)

        instant: List[str] = []
        if self._maps:
            instant = self._maps.instant_candidates(low)

        restrict: Optional[Set[str]] = set(restrict_videos) if restrict_videos else None

        t_vid = time.perf_counter()
        vid_ranked: List[Tuple[str, float]] = []
        if self._video_bm25 and query_variants:
            per_vid = [
                self._video_bm25.search(sq, v_bm25_top) for sq in query_variants
            ]
            vid_ranked = _merge_video_bm25_scores(per_vid, v_bm25_top)
        elif self._video_bm25 and q:
            vid_ranked = self._video_bm25.search(q, v_bm25_top)

        # Video BM25 first — instant RAM hits (topic bigrams like "rather than") can
        # flood the cap and exclude strong BM25 matches (eval showed this for exoplanet).
        ordered: List[str] = []
        seen: Set[str] = set()
        for vid, _ in vid_ranked:
            if len(ordered) >= v_cand_cap:
                break
            if restrict and vid not in restrict:
                continue
            if vid not in seen:
                seen.add(vid)
                ordered.append(vid)
        for vid in instant:
            if len(ordered) >= v_cand_cap:
                break
            if restrict and vid not in restrict:
                continue
            if vid not in seen:
                seen.add(vid)
                ordered.append(vid)
        timing["video_bm25_merge_ms"] = _ms_since(t_vid)

        n_ordered_after_bm25 = len(ordered)
        t_rescue = time.perf_counter()
        rescue_qs = query_variants if query_variants else ([q] if q else [])
        rescue_vids = _segment_rescue_top_videos(
            self._seg_bm25,
            rescue_qs,
            n_videos=seg_rescue_n,
            segments_per_query=seg_rescue_per_q,
        )
        video_bm25_by_pool = {v: float(s) for v, s in vid_ranked}
        rescue_incoming: List[str] = []
        for vid in rescue_vids:
            if restrict and vid not in restrict:
                continue
            if vid not in seen:
                rescue_incoming.append(vid)

        if len(ordered) >= v_cand_cap and rescue_incoming:
            n_replace = min(seg_rescue_replace, len(rescue_incoming))
            victims = sorted(
                ordered, key=lambda v: video_bm25_by_pool.get(v, 0.0)
            )[:n_replace]
            victim_set = set(victims)
            ordered = [v for v in ordered if v not in victim_set]
            for v in victims:
                seen.discard(v)
            for vid in rescue_incoming[:n_replace]:
                seen.add(vid)
                ordered.append(vid)
        else:
            for vid in rescue_incoming:
                if len(ordered) >= v_cand_cap:
                    break
                seen.add(vid)
                ordered.append(vid)
        timing["segment_rescue_ms"] = _ms_since(t_rescue)
        timing["segment_rescue_added"] = len(ordered) - n_ordered_after_bm25

        allowed: Set[str] = set(ordered)
        if not allowed:
            timing["segment_bm25_ms"] = 0.0
            timing["event_bm25_ms"] = 0.0
            timing["pool_build_sql_ms"] = 0.0
            timing["cross_encoder_ms"] = 0.0
            timing["watch_date_sort_ms"] = 0.0
            timing["cross_encoder_skipped"] = False
            timing["retrieve_total_ms"] = _ms_since(t_retrieve0)
            return [], {
                "route_type": route,
                "route_reason": route_res.reason,
                "topic_watch_sort": route_res.topic_watch_sort,
                "video_bm25_ranked_count": len(vid_ranked),
                "merged_candidate_count": 0,
                "instant_ram_candidates": len(instant),
                "bm25_query_variants": list(query_variants) if query_variants else [],
                "segment_rescue_ms": timing.get("segment_rescue_ms", 0.0),
                "segment_rescue_added": timing.get("segment_rescue_added", 0),
                "segment_rescue_top_video_ids": rescue_vids,
                "segment_bm25_pairs_scored": 0,
                "segment_bm25_top_k_cap": seg_bm25_top_k,
                "segments_kept_total": 0,
                "segment_hits": {},
                "pool_size": 0,
                "fast_mode": fast_mode,
                "instant_candidate_ids": list(instant),
                "timing_ms": timing,
                "note": "empty candidate pool",
            }

        seg_hits: Dict[str, List[Tuple[str, float]]] = {v: [] for v in allowed}
        n_seg_pairs = 0
        t_seg = time.perf_counter()
        if self._seg_bm25:
            if query_variants:
                per_seg = [
                    self._seg_bm25.search_in_video_subset(sq, allowed, seg_bm25_top_k)
                    for sq in query_variants
                ]
                seg_pairs = _merge_segment_bm25_scores(per_seg, seg_bm25_top_k)
            else:
                seg_pairs = self._seg_bm25.search_in_video_subset(
                    q, allowed, seg_bm25_top_k
                )
            n_seg_pairs = len(seg_pairs)
            pls = self._seg_bm25.payload
            s_doc = pls.get("doc_ids") or []
            s_vid = pls.get("video_ids") or []
            sid_to_vid = {
                s_doc[i]: s_vid[i]
                for i in range(min(len(s_doc), len(s_vid)))
            }
            for sid, sc in seg_pairs:
                v = sid_to_vid.get(sid)
                if v and v in seg_hits:
                    seg_hits[v].append((sid, sc))
            for v in seg_hits:
                seg_hits[v] = sorted(seg_hits[v], key=lambda x: -x[1])[:SEGMENTS_PER_VIDEO]
        timing["segment_bm25_ms"] = _ms_since(t_seg)

        evt_bonus: Dict[str, float] = {}
        t_evt = time.perf_counter()
        if route == "cause_effect" and self._evt_bm25:
            epairs = self._evt_bm25.search_in_video_subset(q, allowed, EVENT_TOP)
            pl = self._evt_bm25.payload
            doc_ids = pl.get("doc_ids") or []
            vids = pl.get("video_ids") or []
            for eid, sc in epairs:
                if eid in doc_ids:
                    i = doc_ids.index(eid)
                    if i < len(vids):
                        vv = vids[i]
                        evt_bonus[vv] = max(evt_bonus.get(vv, 0.0), float(sc))
        timing["event_bm25_ms"] = _ms_since(t_evt)

        def _max_seg_score(vid: str) -> float:
            hits = seg_hits.get(vid) or []
            return max((sc for _sid, sc in hits), default=0.0)

        bm25_skip, bm25_detail = (
            (True, "fast_mode")
            if fast_mode
            else _bm25_confidence_skip_cross_encoder(vid_ranked)
        )
        skip_watch = (
            not fast_mode
            and route == "topic_lookup"
            and route_res.topic_watch_sort in ("oldest_first", "recent_first")
        )
        skip_ce_fast_path = fast_mode or skip_watch or bm25_skip

        skip_bits: List[str] = []
        if skip_watch:
            skip_bits.append("watch_date_topic_lookup")
        if bm25_skip:
            skip_bits.append("bm25_confidence:" + bm25_detail)

        timing["cross_encoder_skipped"] = skip_ce_fast_path
        timing["cross_encoder_skip_reason"] = (
            "|".join(skip_bits) if skip_ce_fast_path else "ran_ce_limited_pool"
        )
        timing["cross_encoder_max_candidates"] = CROSS_ENCODER_MAX_CANDIDATES

        video_bm25_by: Dict[str, float] = {v: float(s) for v, s in vid_ranked}
        ce_by: Dict[str, float] = {}

        pool: List[Tuple[str, str]] = []
        t_pool = time.perf_counter()
        if skip_ce_fast_path:
            timing["pool_build_sql_ms"] = 0.0
            timing["cross_encoder_ms"] = 0.0
            timing["cross_encoder_input_count"] = 0
            cap = min(rerank_pool, len(ordered))
            base = ordered[:cap]
            merge_pos = {vid: i for i, vid in enumerate(base)}
            out_videos = sorted(
                base,
                key=lambda v: (-_max_seg_score(v), merge_pos[v]),
            )
        else:
            ce_cap = min(CROSS_ENCODER_MAX_CANDIDATES, len(ordered))
            if route == "general":
                ce_slice = sorted(ordered, key=lambda v: -_max_seg_score(v))[:ce_cap]
            else:
                ce_slice = ordered[:ce_cap]
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                for vid in ce_slice:
                    snips: List[str] = []
                    for sid, _sc in seg_hits.get(vid, [])[:3]:
                        cur.execute(
                            "SELECT summary, start_s, end_s FROM segment_index WHERE segment_id = ?",
                            (int(sid),),
                        )
                        rr = cur.fetchone()
                        if rr:
                            t0 = float(rr["start_s"] or 0)
                            t1 = float(rr["end_s"] or 0)
                            snips.append(
                                f"[{t0:.1f}-{t1:.1f}s] {(rr['summary'] or '')[:400]}"
                            )
                    cur.execute(
                        """
                        SELECT t.shorty FROM transcripts t WHERE t.video_id = ? LIMIT 1
                        """,
                        (vid,),
                    )
                    sr = cur.fetchone()
                    shorty = (sr["shorty"] or "") if sr else ""
                    blob = (shorty[:1200] + " " + " ".join(snips)).strip()
                    pool.append((vid, blob))

            if not any((b or "").strip() for _v, b in pool):
                if self._maps:
                    pool = [
                        (v, self._maps.video_routing_text.get(v, "")[:2000])
                        for v in ce_slice
                    ]

            timing["pool_build_sql_ms"] = _ms_since(t_pool)

            t_ce = time.perf_counter()
            ce = self._get_cross_encoder()
            ce_pairs = [(q, (tx or "").strip()) for _vid, tx in pool if (tx or "").strip()]
            timing["cross_encoder_input_count"] = len(ce_pairs)
            if ce_pairs:
                scores = ce.predict(ce_pairs)
                vids_order = [v for v, tx in pool if (tx or "").strip()]
                scored = list(zip(vids_order, scores))
                scored = [
                    (vid, float(s) + min(2.0, evt_bonus.get(vid, 0.0) * 0.05))
                    for (vid, s) in scored
                ]
                scored = sorted(scored, key=lambda x: -x[1])
                for vid, sc in scored:
                    ce_by[vid] = float(sc)
                ranked_ce = [vid for vid, _ in scored]
                seen_ce = set(ranked_ce)
                tail = [v for v in ordered if v not in seen_ce]
                out_videos = (ranked_ce + tail)[:rerank_pool]
            else:
                out_videos = ordered[:rerank_pool]
            timing["cross_encoder_ms"] = _ms_since(t_ce)

        topic_watch_sort = route_res.topic_watch_sort or "recent_first"
        t_watch = time.perf_counter()
        if route == "topic_lookup" and out_videos:
            idx_map = {vid: i for i, vid in enumerate(out_videos)}
            ph = ",".join("?" * len(out_videos))
            watch_by: Dict[str, str] = {}
            with sqlite3.connect(self.db_path) as _wc:
                cw = _wc.cursor()
                cw.execute(
                    f"SELECT video_id, watch_date FROM videos WHERE video_id IN ({ph})",
                    tuple(out_videos),
                )
                for row in cw.fetchall():
                    raw = row[1]
                    s = (str(raw).strip()[:10] if raw else "") or "0000-00-00"
                    if len(s) == 10 and s[4:5] == "-" and s[7:8] == "-":
                        watch_by[str(row[0])] = s
                    else:
                        watch_by[str(row[0])] = "0000-00-00"
                for v in out_videos:
                    watch_by.setdefault(v, "0000-00-00")
            if topic_watch_sort == "oldest_first":
                out_videos = sorted(
                    out_videos,
                    key=lambda v: (
                        "9999-99-99"
                        if watch_by.get(v, "0000-00-00") == "0000-00-00"
                        else watch_by.get(v, "0000-00-00"),
                        idx_map.get(v, 0),
                    ),
                )
            else:
                out_videos = sorted(
                    out_videos,
                    key=lambda v: (watch_by.get(v, "0000-00-00"), -idx_map.get(v, 0)),
                    reverse=True,
                )
            timing["watch_date_sort_ms"] = _ms_since(t_watch)
        else:
            timing["watch_date_sort_ms"] = 0.0

        seg_pick: Dict[str, List[str]] = {
            v: [p[0] for p in (seg_hits.get(v) or [])] for v in allowed
        }
        segments_kept_total = sum(len(seg_hits[v]) for v in allowed)

        rank_scores: Dict[str, float] = {
            vid: _v2_rank_score_for_video(
                vid,
                video_bm25_by=video_bm25_by,
                ce_by=ce_by,
                max_seg_score=_max_seg_score,
                evt_bonus=evt_bonus,
                ce_fast_path=skip_ce_fast_path,
            )
            for vid in out_videos
        }

        timing["retrieve_total_ms"] = _ms_since(t_retrieve0)

        debug: Dict[str, Any] = {
            "route_type": route,
            "route_reason": route_res.reason,
            "topic_watch_sort": route_res.topic_watch_sort,
            "topic_watch_sort_effective": topic_watch_sort
            if route == "topic_lookup"
            else None,
            "instant_ram_candidates": len(instant),
            "bm25_query_variants": list(query_variants) if query_variants else [],
            "segment_rescue_ms": timing.get("segment_rescue_ms", 0.0),
            "segment_rescue_added": timing.get("segment_rescue_added", 0),
            "segment_rescue_top_video_ids": rescue_vids,
            "video_bm25_ranked_count": len(vid_ranked),
            "merged_candidate_count": len(allowed),
            "segment_bm25_pairs_scored": n_seg_pairs,
            "segment_bm25_top_k_cap": seg_bm25_top_k,
            "segments_kept_total": segments_kept_total,
            "video_bm25_top": [v for v, _ in vid_ranked[:8]],
            "video_bm25_scores": video_bm25_by,
            "rank_scores": rank_scores,
            "pool_size": len(out_videos),
            "segment_hits": seg_pick,
            "fast_mode": fast_mode,
            "instant_candidate_ids": list(instant),
            "topic_lookup_sorted_by_watch_date": bool(
                route == "topic_lookup" and len(out_videos) > 0
            ),
            "cross_encoder_skip_reason": timing.get("cross_encoder_skip_reason"),
            "cross_encoder_max_candidates": timing.get("cross_encoder_max_candidates"),
            "cross_encoder_input_count": timing.get("cross_encoder_input_count"),
            "timing_ms": timing,
        }
        return out_videos, debug

    def search_fast(
        self,
        question: str,
        restrict_videos: Optional[Sequence[str]] = None,
        top_k: int = FAST_SEARCH_TOP_K,
    ) -> Dict[str, Any]:
        """
        Local retrieval only: no LLM, no cross-encoder load/use.
        Returns ranked video hits with metadata and best snippet.
        """
        top_k = max(1, min(int(top_k), 50))
        ranked, dbg = self.retrieve_videos(
            question,
            restrict_videos=restrict_videos,
            fast_mode=True,
            fast_top_k=top_k,
        )
        return self._format_fast_search_results(
            question, ranked[:top_k], dbg, top_k=top_k
        )

    def _format_fast_search_results(
        self,
        question: str,
        ranked: List[str],
        dbg: Dict[str, Any],
        *,
        top_k: int,
    ) -> Dict[str, Any]:
        q = (question or "").strip()
        title_toks = {
            t
            for t in re.findall(r"[A-Za-z0-9]+", q.lower())
            if len(t) >= 3
        }
        instant_set = set(dbg.get("instant_candidate_ids") or [])
        rescue_set = set(dbg.get("segment_rescue_top_video_ids") or [])
        video_bm25_by: Dict[str, float] = dict(dbg.get("video_bm25_scores") or {})
        rank_scores: Dict[str, float] = dict(dbg.get("rank_scores") or {})
        seg_hits: Dict[str, List[str]] = dict(dbg.get("segment_hits") or {})
        route = dbg.get("route_type") or "general"

        results: List[Dict[str, Any]] = []
        if not ranked:
            return {
                "query": q,
                "route_type": route,
                "route_reason": dbg.get("route_reason"),
                "results": [],
                "timing_ms": dbg.get("timing_ms") or {},
                "top_k": top_k,
            }

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            ph = ",".join("?" * len(ranked))
            meta_by: Dict[str, sqlite3.Row] = {}
            cur.execute(
                f"""
                SELECT v.video_id, v.title, v.channel, v.watch_date,
                       COALESCE(t.shorty, '') AS shorty
                FROM videos v
                LEFT JOIN transcripts t ON t.video_id = v.video_id
                WHERE v.video_id IN ({ph})
                """,
                tuple(ranked),
            )
            for row in cur.fetchall():
                meta_by[str(row["video_id"])] = row

            seg_detail: Dict[str, Dict[str, Any]] = {}
            for vid in ranked:
                sids = seg_hits.get(vid) or []
                if not sids:
                    continue
                sid = int(sids[0])
                cur.execute(
                    """
                    SELECT segment_id, start_s, end_s, summary
                    FROM segment_index WHERE segment_id = ?
                    """,
                    (sid,),
                )
                sr = cur.fetchone()
                if sr:
                    seg_detail[vid] = dict(sr)

        for vid in ranked:
            row = meta_by.get(vid)
            title = ((row["title"] or "") if row else "") or vid
            channel = (row["channel"] or "") if row else ""
            watched = (row["watch_date"] or "") if row else ""
            shorty = (row["shorty"] or "") if row else ""
            wd = str(watched or "")[:10] if watched else ""

            reasons: List[str] = []
            if vid in instant_set:
                reasons.append("entity_or_topic")
            vb = float(video_bm25_by.get(vid, 0.0))
            if vb > 0:
                reasons.append("video_bm25")
            if vid in rescue_set:
                reasons.append("segment_rescue")
            if seg_hits.get(vid):
                reasons.append("segment_bm25")
            title_low = (title or "").lower()
            if title_toks and any(t in title_low for t in title_toks):
                reasons.append("title_match")

            seg = seg_detail.get(vid)
            snippet = ""
            snippet_type = "shorty"
            timestamp_range = None
            segment_id = None
            if seg and (seg.get("summary") or "").strip():
                snippet = (seg["summary"] or "").strip()[:400]
                snippet_type = "segment"
                segment_id = int(seg["segment_id"])
                t0 = float(seg.get("start_s") or 0)
                t1 = float(seg.get("end_s") or 0)
                timestamp_range = f"{t0:.0f}-{t1:.0f}s"
            elif shorty:
                snippet = shorty.strip()[:400]
                snippet_type = "shorty"

            score = float(rank_scores.get(vid, vb))
            if seg_hits.get(vid) and snippet_type == "segment":
                score_source = "segment_bm25"
            elif vb > 0:
                score_source = "video_bm25"
            elif vid in instant_set:
                score_source = "entity_or_topic"
            else:
                score_source = "merged_rank"

            results.append(
                {
                    "video_id": vid,
                    "title": title,
                    "channel": channel,
                    "watch_date": wd or None,
                    "score": round(score, 4),
                    "score_source": score_source,
                    "match_reasons": reasons or ["merged_rank"],
                    "snippet": snippet,
                    "snippet_type": snippet_type,
                    "timestamp_range": timestamp_range,
                    "segment_id": segment_id,
                }
            )

        return {
            "query": q,
            "route_type": route,
            "route_reason": dbg.get("route_reason"),
            "results": results,
            "timing_ms": dbg.get("timing_ms") or {},
            "top_k": top_k,
        }

    def answer(
        self,
        question: str,
        video_ids: Optional[List[str]] = None,
        emit: Optional[Callable[..., None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        job_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        _t0 = time.time()

        def _elapsed_ms() -> int:
            return int(round((time.time() - _t0) * 1000))

        _debug_events: List[Dict[str, Any]] = []

        def _emit(ev_type: str, **data: Any) -> None:
            ev = {"type": ev_type, "elapsed_ms": _elapsed_ms(), **data}
            _debug_events.append(ev)
            if emit:
                try:
                    emit(ev)
                except Exception:
                    pass

        q = (question or "").strip()
        _emit("step", label="V2 — starting answer (hierarchical RAG)")

        route_res = self.router.classify(q)
        route = route_res.route_type
        _emit(
            "step",
            label=(
                f"V2 — query classified as {route} ({route_res.reason})"
            ),
        )

        t_retrieve = time.perf_counter()
        ranked, dbg = self.retrieve_videos(q, restrict_videos=video_ids)
        dbg = dict(dbg)
        _tim: Dict[str, Any] = dict(dbg.get("timing_ms") or {})
        _tim["answer_retrieve_call_ms"] = round(
            (time.perf_counter() - t_retrieve) * 1000.0, 2
        )
        dbg["timing_ms"] = _tim
        seg_by_vid: Dict[str, List[str]] = (dbg or {}).get("segment_hits") or {}

        _emit(
            "step",
            label=(
                f"V2 — video BM25 ranked {dbg.get('video_bm25_ranked_count', 0)} videos; "
                f"{dbg.get('merged_candidate_count', 0)} candidates after RAM+merge"
            ),
        )
        ce_skipped = bool(dbg.get("timing_ms", {}).get("cross_encoder_skipped"))
        ce_n = dbg.get("cross_encoder_input_count")
        _emit(
            "step",
            label=(
                f"V2 — segment BM25 scored {dbg.get('segment_bm25_pairs_scored', 0)} segment "
                f"hits (cap {dbg.get('segment_bm25_top_k_cap', SEGMENT_BM25_TOP_K)}); "
                f"kept {dbg.get('segments_kept_total', 0)} segment rows (pre-rerank); "
                f"CE skipped={ce_skipped}"
                + (f"; CE inputs={ce_n}" if ce_n is not None else "")
                + f"; CE cap={dbg.get('cross_encoder_max_candidates', CROSS_ENCODER_MAX_CANDIDATES)}"
            ),
        )
        _emit(
            "step",
            label=(
                f"V2 — segment BM25 cap check: requested_top_k="
                f"{dbg.get('segment_bm25_top_k_cap', SEGMENT_BM25_TOP_K)}, "
                f"returned_pairs={dbg.get('segment_bm25_pairs_scored', 0)}"
            ),
        )
        try:
            _emit(
                "step",
                label=(
                    "V2 — retrieve timing (ms): "
                    + json.dumps(dbg.get("timing_ms") or {}, sort_keys=True)
                ),
            )
        except Exception:
            pass

        if not ranked:
            _emit("step", label="V2 — no candidates; skipping LLM")
            _emit(
                "done",
                label="done (no V2 context)",
                total_elapsed_ms=_elapsed_ms(),
            )
            out = {
                "answer": "No indexed videos matched this question in the V2 retrieval path.",
                "used_context": [],
                "sources": [],
                "debug_events": _debug_events,
            }
            _tim["select_llm_context_ms"] = 0.0
            _tim["context_blocks_sql_ms"] = 0.0
            _tim["llm_answer_ms"] = 0.0
            _tim["answer_total_wall_ms"] = float(_elapsed_ms())
            dbg["timing_ms"] = _tim
            try:
                with sqlite3.connect(self.db_path) as _c2:
                    ct, cs = _corpus_shorty_counts(_c2)
                _append_v2_request_html(
                    job_id=job_id,
                    ref_id_display="pending",
                    ts_iso=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    query=q,
                    route_type=str(dbg.get("route_type", route)),
                    route_reason=str(dbg.get("route_reason", route_res.reason)),
                    video_bm25_ranked_count=int(dbg.get("video_bm25_ranked_count", 0) or 0),
                    merged_candidate_count=int(dbg.get("merged_candidate_count", 0) or 0),
                    segment_bm25_pairs_scored=int(dbg.get("segment_bm25_pairs_scored", 0) or 0),
                    segments_kept_total=int(dbg.get("segments_kept_total", 0) or 0),
                    context_passage_count=0,
                    candidate_video_ids=[],
                    full_system="(no LLM call)",
                    full_user="(no context)",
                    model_response=out["answer"],
                    timing_ms=_elapsed_ms(),
                    corpus_total_videos=ct,
                    corpus_with_shorty=cs,
                    extra_debug=dbg,
                )
            except Exception:
                pass
            return out

        t_sel = time.perf_counter()
        top_v, ctx_tokens, ctx_matched = _select_llm_context_video_ids(
            self.db_path, ranked, q
        )
        _tim["select_llm_context_ms"] = round(
            (time.perf_counter() - t_sel) * 1000.0, 2
        )
        dbg["timing_ms"] = _tim
        dbg["llm_topic_tokens"] = ctx_tokens
        dbg["llm_context_title_shorty_matches"] = ctx_matched
        dbg["llm_context_video_count"] = len(top_v)
        _emit(
            "step",
            label=(
                "V2 — LLM context: %d video(s)%s"
                % (
                    len(top_v),
                    (
                        " (tokens %r, %d matched title/Shorty word-boundary)"
                        % (ctx_tokens, ctx_matched)
                        if ctx_tokens
                        else " (no topic-tail tokens; using top reranked)"
                    ),
                )
            ),
        )

        blocks: List[str] = []
        sources: List[Dict[str, Any]] = []

        t_ctx = time.perf_counter()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            corpus_total, corpus_shorty = _corpus_shorty_counts(conn)
            d0, d1 = _watch_date_range_for_videos(conn, list(ranked))

            cur = conn.cursor()
            video_lines: List[str] = []
            for vid in top_v:
                cur.execute(
                    """
                    SELECT v.title, v.channel, v.watch_date, t.shorty
                    FROM videos v
                    JOIN transcripts t ON t.video_id = v.video_id
                    WHERE v.video_id = ?
                    """,
                    (vid,),
                )
                vr = cur.fetchone()
                title = ((vr["title"] or "") if vr else "") or vid
                channel = (vr["channel"] or "") if vr else ""
                watched = (vr["watch_date"] or "") if vr else ""
                shorty = (vr["shorty"] or "") if vr else ""
                wd = str(watched or "")
                video_lines.append(
                    f"- {title or ''} | channel: {channel or ''} | watched: {(wd[:10] if wd else '?')}"
                )

                want = seg_by_vid.get(vid) or []
                if want:
                    placeholders = ",".join("?" for _ in want)
                    cur.execute(
                        f"""
                        SELECT segment_id, start_s, end_s, summary
                        FROM segment_index
                        WHERE video_id = ? AND segment_id IN ({placeholders})
                        ORDER BY COALESCE(start_s, -1)
                        """,
                        (vid, *[int(x) for x in want]),
                    )
                    seg_rows = cur.fetchall()
                else:
                    cur.execute(
                        """
                        SELECT segment_id, start_s, end_s, summary
                        FROM segment_index WHERE video_id = ?
                        ORDER BY COALESCE(start_s, -1) LIMIT 5
                        """,
                        (vid,),
                    )
                    seg_rows = cur.fetchall()
                seg_txt = ""
                t_range = ""
                if seg_rows:
                    first = seg_rows[0]
                    last = seg_rows[-1]
                    t_range = f"{float(first['start_s'] or 0):.0f}-{float(last['end_s'] or 0):.0f}s"
                    seg_txt = "\n".join(
                        (
                            f"[{float(r['start_s'] or 0):.1f}-{float(r['end_s'] or 0):.1f}s] "
                            + (r["summary"] or "")[:500]
                        )
                        for r in seg_rows
                    )

                if route == "cause_effect":
                    cur.execute(
                        """
                        SELECT title, cause, effect FROM event_index
                        WHERE video_id = ? LIMIT 6
                        """,
                        (vid,),
                    )
                    ev_lines = []
                    for er in cur.fetchall():
                        ev_lines.append(
                            " | ".join(
                                x
                                for x in (er["title"], er["cause"], er["effect"])
                                if x
                            )
                        )
                    extra = "\nEvents:\n" + "\n".join(ev_lines) if ev_lines else ""
                else:
                    extra = ""

                block = (
                    f"VIDEO: {title or ''}\nCHANNEL: {channel or ''}\nWATCHED: {watched or ''}\n"
                    f"VIDEO_ID: {vid or ''}\nTIME_RANGE_HINT: {t_range or ''}\n"
                    f"SHORTY:\n{shorty or ''}\nSEGMENTS:\n{seg_txt or ''}{extra or ''}"
                )
                blocks.append(block)
                sources.append(
                    {
                        "video_id": vid,
                        "title": title,
                        "channel": channel,
                        "watch_date": watched,
                        "timestamp_range": t_range,
                        "route_type": route,
                    }
                )

        _tim["context_blocks_sql_ms"] = round(
            (time.perf_counter() - t_ctx) * 1000.0, 2
        )
        dbg["timing_ms"] = _tim

        no_shorty = max(0, corpus_total - corpus_shorty)
        date_range_s = (
            f"{d0 or ''} to {d1 or ''}"
            if d0 and d1
            else ("single: " + str(d0 or d1 or ""))
            if (d0 or d1)
            else "unknown (missing watch_date on matches)"
        )

        stats_block = f"""
RETRIEVAL SUMMARY (for your opening lines):
- Videos matching this query in the V2 ranked pool: {len(ranked)} (after rerank).
- Watch-date span across those ranked matches: {date_range_s}.
- Indexed library: {corpus_total} videos total; {corpus_shorty} have a Shorty; about {no_shorty} may still lack a Shorty (not fully processed for dense Q&A).

Videos included in context below (title | channel | watch date):
{chr(10).join(video_lines)}
"""

        system = (
            "You are Ask Shorty (V2). Answer using ONLY the provided blocks and the retrieval summary. "
            "In the first paragraph or bullet block: (1) state how many videos were found for this topic "
            f"in this retrieval pass ({len(ranked)} in the ranked pool). "
            "(2) Give the watch-date span of those ranked matches when dates exist. "
            "(3) List each context video with title, channel, and watch date. "
            f"(4) If relevant, note that the library has about {no_shorty} indexed videos without a Shorty yet, "
            "so additional relevant videos may exist but are not fully processed for this pipeline. "
            "Then answer the question. Cite video title and channel. "
            "Include watch date as (watched: YYYY-MM-DD) when present. Use timestamp ranges when given."
        )
        user_p = (
            f"Question:\n{q}\n\n{stats_block}\n\nContext:\n" + "\n---\n".join(blocks)
        )

        _emit(
            "context",
            label=f"context assembled — {len(blocks)} block(s) for LLM",
            count=len(blocks),
            blocks=[b.split("\n")[0] for b in blocks],
        )

        if should_cancel and should_cancel():
            return {"answer": "", "used_context": [], "sources": [], "debug_events": _debug_events}

        _emit("step", label="V2 — calling answer model (OpenRouter or Anthropic per ANSWER_MODEL)")

        t_llm = time.perf_counter()
        answer_text = _call_answer_text(
            system,
            user_p,
            max_tokens=2048,
            temperature=0.2,
        )
        _tim["llm_answer_ms"] = round((time.perf_counter() - t_llm) * 1000.0, 2)
        _tim["answer_total_wall_ms"] = float(_elapsed_ms())
        dbg["timing_ms"] = _tim
        if not isinstance(answer_text, str):
            answer_text = str(answer_text) if answer_text is not None else ""

        _emit("step", label="V2 — received model answer")
        try:
            _emit(
                "step",
                label=(
                    "V2 — full timing (ms): "
                    + json.dumps(dbg.get("timing_ms") or {}, sort_keys=True)
                ),
            )
            _emit("timing", timing=(dbg.get("timing_ms") or {}))
        except Exception:
            pass
        _emit("done", label="done", total_elapsed_ms=_elapsed_ms())

        total_ms = _elapsed_ms()

        try:
            _append_v2_request_html(
                job_id=job_id,
                ref_id_display="pending",
                ts_iso=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                query=q,
                route_type=str(dbg.get("route_type", route)),
                route_reason=str(dbg.get("route_reason", route_res.reason)),
                video_bm25_ranked_count=int(dbg.get("video_bm25_ranked_count", 0) or 0),
                merged_candidate_count=int(dbg.get("merged_candidate_count", 0) or 0),
                segment_bm25_pairs_scored=int(dbg.get("segment_bm25_pairs_scored", 0) or 0),
                segments_kept_total=int(dbg.get("segments_kept_total", 0) or 0),
                context_passage_count=len(blocks),
                candidate_video_ids=list(ranked),
                full_system=system,
                full_user=user_p,
                model_response=answer_text,
                timing_ms=total_ms,
                corpus_total_videos=corpus_total,
                corpus_with_shorty=corpus_shorty,
                extra_debug=dbg,
            )
        except Exception as le:
            _emit("error", label="V2 HTML log write failed", message=str(le))

        return {
            "answer": answer_text,
            "used_context": blocks,
            "sources": sources,
            "debug_events": _debug_events,
        }


@lru_cache(maxsize=8)
def _shared_ask_shorty_v2_engine(db_path_resolved: str) -> AskShortyV2:
    """One warm engine per DB path — eval was constructing AskShortyV2 per query (full reload)."""
    return AskShortyV2(db_path=db_path_resolved)


def v2_retrieval_ranked_list(
    db_path: str,
    question: str,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Evaluate_rag helper: live retrieve_videos + real BM25/CE scores in before_hits."""
    key = _resolved_db_path(db_path)
    eng = _shared_ask_shorty_v2_engine(key)
    vids, dbg = eng.retrieve_videos(question)
    scores_by_vid: Dict[str, float] = dict(dbg.get("rank_scores") or {})
    before = [
        {
            "video_id": v,
            "score": round(float(scores_by_vid.get(v, 0.0)), 4),
            "source_type": "v2_hierarchical",
            "text_snippet": "",
        }
        for v in vids[:100]
    ]
    return vids, before
