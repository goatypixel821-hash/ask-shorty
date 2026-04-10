#!/usr/bin/env python3
"""
Build a local evaluation dataset for Ask Shorty — no external API needed.

Sources:
  1. synthetic_questions table  → fact_lookup  (direct question, known source video)
  2. entities table             → entity_topic (who/what queries from named entities)
  3. same-channel video pairs   → summary_comparison
  4. paraphrase variants        → paraphrase   (regex rewrites of synthetic questions)
  5. short/edge transcripts     → tricky       (short or sparse transcripts = harder cases)

Sampling is stratified so the output set spans:
  - multiple channels
  - short / medium / long transcript lengths
  - early / mid / recent watch dates
  - all five query types

Outputs (all under eval_data/candidates/, which is git-ignored):
  candidates.jsonl
  candidates.csv
  build_stats.json

Auto-detection: if no --db-path is given, the script searches known locations for
the largest transcripts.db and uses that.  On this machine the full corpus is at:
  C:/Users/number2/Desktop/youtube-history-viewer-copy/data/transcripts.db

Usage:
  python build_eval_dataset.py                         # auto-detects full corpus
  python build_eval_dataset.py --inventory             # prints corpus stats, saves JSON
  python build_eval_dataset.py --db-path PATH/transcripts.db --target 300
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
EVAL_DATA_DIR = Path(__file__).parent / "eval_data"
CANDIDATES_DIR = EVAL_DATA_DIR / "candidates"

# ---------------------------------------------------------------------------
# Corpus auto-detection
# ---------------------------------------------------------------------------

# Ordered list of candidate DB locations, most-preferred first.
# The script picks the first one that exists, then falls back to the largest.
_KNOWN_DB_CANDIDATES: List[Path] = [
    # Full watching-history corpus (youtube-history-viewer-copy project)
    Path.home() / "Desktop" / "youtube-history-viewer-copy" / "data" / "transcripts.db",
    # Shorty project's own (usually smaller) DB
    Path(__file__).parent / "data" / "transcripts.db",
]

# Companion Chroma paths (paired with the DB candidates above).
_KNOWN_CHROMA_CANDIDATES: List[Optional[Path]] = [
    Path.home() / "Desktop" / "youtube-history-viewer-copy" / "data" / "transcript_chroma",
    Path(__file__).parent / "data" / "transcript_chroma_new",
]


def find_best_db(explicit: Optional[str] = None) -> Path:
    """
    Return the best available transcripts.db path.

    Priority:
    1. An explicit --db-path argument (if given)
    2. The largest DB among known candidate locations
    """
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"--db-path not found: {p}")
        return p

    existing = [(p, p.stat().st_size) for p in _KNOWN_DB_CANDIDATES if p.exists()]
    if not existing:
        # Last resort — default shorty path (may or may not exist)
        return Path(__file__).parent / "data" / "transcripts.db"

    # Prefer the largest DB (the full corpus)
    best = max(existing, key=lambda x: x[1])[0]
    return best


def find_companion_chroma(db_path: Path, explicit: Optional[str] = None) -> Optional[Path]:
    """Return the Chroma path that accompanies db_path, or None."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    for db_cand, chroma_cand in zip(_KNOWN_DB_CANDIDATES, _KNOWN_CHROMA_CANDIDATES):
        if chroma_cand and db_path.resolve() == db_cand.resolve():
            return chroma_cand if chroma_cand.exists() else None

    return None


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def run_inventory(db_path: Path, chroma_path: Optional[Path], out_dir: Path) -> None:
    """
    Print a full corpus inventory and save it to a JSON file.
    """
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    def _count(sql: str) -> int:
        try:
            c.execute(sql)
            return c.fetchone()[0]
        except Exception:
            return -1

    total_videos       = _count("SELECT COUNT(*) FROM videos")
    has_transcript     = _count("SELECT COUNT(*) FROM videos WHERE has_transcript=1")
    transcript_rows    = _count("SELECT COUNT(*) FROM transcripts")
    with_shorty        = _count("SELECT COUNT(*) FROM transcripts WHERE shorty IS NOT NULL AND trim(shorty)!=''")
    with_synq          = _count("SELECT COUNT(DISTINCT video_id) FROM synthetic_questions")
    total_synq         = _count("SELECT COUNT(*) FROM synthetic_questions")
    with_entities      = _count("SELECT COUNT(DISTINCT video_id) FROM entities")
    total_entities     = _count("SELECT COUNT(*) FROM entities")
    unique_channels    = _count("SELECT COUNT(DISTINCT channel) FROM videos")

    c.execute("SELECT MIN(created_at), MAX(created_at) FROM videos")
    date_range = c.fetchone() or ("?", "?")
    c.execute("SELECT MIN(watch_date), MAX(watch_date) FROM videos WHERE watch_date IS NOT NULL")
    watch_range = c.fetchone() or ("?", "?")

    c.execute("SELECT channel, COUNT(*) n FROM videos GROUP BY channel ORDER BY n DESC LIMIT 10")
    top_channels = [(r[0], r[1]) for r in c.fetchall()]

    # Videos that have a transcript but no Shorty
    c.execute("""
        SELECT COUNT(DISTINCT t.video_id)
        FROM transcripts t
        WHERE (t.shorty IS NULL OR trim(t.shorty)='')
    """)
    transcripts_no_shorty = c.fetchone()[0]

    # Videos that have a transcript but no synthetic questions
    c.execute("""
        SELECT COUNT(DISTINCT t.video_id)
        FROM transcripts t
        LEFT JOIN synthetic_questions sq ON sq.video_id = t.video_id
        WHERE sq.video_id IS NULL
    """)
    transcripts_no_synq = c.fetchone()[0]

    conn.close()

    # Chroma stats
    chroma_total   = None
    chroma_by_type: Dict[str, int] = {}
    chroma_has_type_field = None
    if chroma_path and chroma_path.exists():
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_path))
            col = client.get_collection("transcripts")
            chroma_total = col.count()
            # Sample to detect type-field presence
            sample = col.get(limit=50, include=["metadatas"])
            type_counts: Dict[str, int] = {}
            no_type = 0
            for m in sample["metadatas"]:
                t = (m or {}).get("type", "<none>")
                type_counts[t] = type_counts.get(t, 0) + 1
                if "type" not in (m or {}):
                    no_type += 1
            chroma_has_type_field = no_type == 0
            chroma_by_type = type_counts
        except Exception as e:
            chroma_by_type = {"error": str(e)}

    inv = {
        "inventory_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "db_path": str(db_path),
        "db_size_mb": round(db_path.stat().st_size / 1e6, 2),
        "chroma_path": str(chroma_path) if chroma_path else None,
        "corpus": {
            "total_videos":                total_videos,
            "videos_with_transcript":      has_transcript,
            "transcript_rows":             transcript_rows,
            "transcripts_with_shorty":     with_shorty,
            "transcripts_without_shorty":  transcripts_no_shorty,
            "videos_with_synq":            with_synq,
            "total_synq_rows":             total_synq,
            "transcripts_without_synq":    transcripts_no_synq,
            "videos_with_entities":        with_entities,
            "total_entity_rows":           total_entities,
            "unique_channels":             unique_channels,
            "earliest_video_created":      date_range[0],
            "latest_video_created":        date_range[1],
            "earliest_watch_date":         watch_range[0],
            "latest_watch_date":           watch_range[1],
            "top_10_channels":             top_channels,
        },
        "chroma": {
            "total_vectors":     chroma_total,
            "has_type_field":    chroma_has_type_field,
            "type_distribution": chroma_by_type,
            "note": (
                "WARNING: Chroma lacks 'type' metadata — "
                "evaluate_rag.py will use fallback (all-type) queries. "
                "Run reindex_all.py to rebuild with type metadata."
                if chroma_has_type_field is False else
                "OK: Chroma has type metadata, evaluate_rag.py type filters will work."
                if chroma_has_type_field is True else
                "Chroma not found or not checked."
            ),
        },
        "gaps": {
            "videos_with_no_transcript":  total_videos - has_transcript,
            "transcripts_with_no_shorty": transcripts_no_shorty,
            "transcripts_with_no_synq":   transcripts_no_synq,
            "chroma_indexed_count":       chroma_total,
        },
    }

    print("\n" + "=" * 60)
    print("CORPUS INVENTORY")
    print("=" * 60)
    print(f"  DB path        : {db_path}")
    print(f"  DB size        : {inv['db_size_mb']} MB")
    print(f"  Total videos   : {total_videos}")
    print(f"  w/ transcript  : {has_transcript}")
    print(f"  w/ Shorty      : {with_shorty}")
    print(f"  w/ synq        : {with_synq}  ({total_synq} rows)")
    print(f"  w/ entities    : {with_entities}  ({total_entities} rows)")
    print(f"  Unique channels: {unique_channels}")
    print(f"  Date range     : {date_range[0]}  to  {date_range[1]}")
    print(f"\n  GAPS:")
    print(f"    No transcript : {total_videos - has_transcript}")
    print(f"    No Shorty     : {transcripts_no_shorty}")
    print(f"    No synq       : {transcripts_no_synq}")
    print(f"\n  CHROMA ({chroma_path or 'not specified'}):")
    if chroma_total is not None:
        print(f"    Total vectors : {chroma_total}")
        print(f"    Type field    : {'YES' if chroma_has_type_field else 'NO - fallback mode needed'}")
        print(f"    By type       : {chroma_by_type}")
    else:
        print("    Not available / not checked")
    print(f"\n  Top channels   :")
    for ch, n in top_channels[:5]:
        print(f"    {n:4d}  {ch}")

    out_dir.mkdir(parents=True, exist_ok=True)
    inv_path = out_dir / "corpus_inventory.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)
    print(f"\nInventory saved -> {inv_path}")

    if chroma_has_type_field is False:
        print(
            "\n[WARNING] The Chroma index lacks 'type' metadata.\n"
            "  evaluate_rag.py will use a compatibility fallback (all docs treated as chunks).\n"
            "  To get full multi-layer eval, rebuild the index:\n"
            "    python reindex_all.py --db-path " + str(db_path) + "\n"
            "  (point reindex_all.py at the full corpus DB above)"
        )


# ---------------------------------------------------------------------------
# Schema helpers — query dict that satisfies evaluate_rag.py
# ---------------------------------------------------------------------------

def _make_query(
    *,
    query_id: str,
    query: str,
    query_type: str,
    source_video_id: str,
    source_video_title: str,
    expected_video_ids: List[str],
    gold_answer_note: str = "",
    difficulty: str = "medium",
    expected_chunk_ids: Optional[List[str]] = None,
    support_notes: str = "",
    channel: str = "",
    transcript_length_bin: str = "medium",
    watch_date: str = "",
    ground_truth: str = "",
    notes: str = "",
) -> Dict[str, Any]:
    """Canonical eval query record."""
    return {
        # --- identity ---
        "query_id": query_id,
        # evaluate_rag.py looks for "id"; keep both
        "id": query_id,
        # --- content ---
        "query": query,
        "query_type": query_type,
        # evaluate_rag.py looks for "category"; map here
        "category": _type_to_category(query_type),
        # --- labels ---
        "source_video_id": source_video_id,
        "source_video_title": source_video_title,
        "expected_video_ids": expected_video_ids,
        # evaluate_rag.py uses "relevant_video_ids"
        "relevant_video_ids": expected_video_ids,
        "expected_chunk_ids": expected_chunk_ids,
        "gold_answer_note": gold_answer_note,
        "ground_truth": ground_truth or gold_answer_note,
        # --- metadata ---
        "difficulty": difficulty,
        "label_status": "candidate",
        "support_notes": support_notes,
        "channel": channel,
        "transcript_length_bin": transcript_length_bin,
        "watch_date": watch_date,
        "notes": notes,
    }


def _type_to_category(qt: str) -> str:
    """Map local query_type to evaluate_rag.py category."""
    return {
        "fact_lookup": "specific_fact",
        "entity_topic": "thematic",
        "summary_comparison": "cross_video",
        "paraphrase": "specific_fact",
        "tricky": "specific_fact",
    }.get(qt, "thematic")


# ---------------------------------------------------------------------------
# Database queries
# ---------------------------------------------------------------------------

def _load_corpus(db_path: str) -> Dict[str, Any]:
    """
    Load all relevant tables from SQLite into memory dicts.
    Returns keys: videos, transcripts, synthetic_questions, entities.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT v.video_id, v.title, v.channel, v.watch_date,
               LENGTH(t.text) AS transcript_len, t.text AS transcript_text,
               t.shorty
        FROM videos v
        JOIN transcripts t ON t.video_id = v.video_id
        WHERE t.text IS NOT NULL AND TRIM(t.text) != ''
        ORDER BY v.video_id
        """
    )
    videos = [dict(r) for r in cur.fetchall()]

    cur.execute(
        """
        SELECT sq.id, sq.video_id, sq.question
        FROM synthetic_questions sq
        JOIN transcripts t ON t.video_id = sq.video_id
        WHERE t.text IS NOT NULL AND TRIM(t.text) != ''
        ORDER BY sq.video_id, sq.id
        """
    )
    synq = [dict(r) for r in cur.fetchall()]

    cur.execute(
        """
        SELECT e.video_id, e.name, e.type, e.aliases
        FROM entities e
        ORDER BY e.video_id
        """
    )
    entities = [dict(r) for r in cur.fetchall()]

    conn.close()
    return {"videos": videos, "synq": synq, "entities": entities}


def _transcript_length_bin(length: int) -> str:
    if length < 3_000:
        return "short"
    if length < 15_000:
        return "medium"
    return "long"


def _watch_date_bin(watch_date: Optional[str]) -> str:
    if not watch_date:
        return "unknown"
    try:
        dt = datetime.fromisoformat(watch_date[:10])
        year = dt.year
        if year < 2023:
            return "old"
        if year == 2023:
            return "2023"
        if year == 2024:
            return "2024"
        return "recent"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Generator 1: fact_lookup — from synthetic_questions
# ---------------------------------------------------------------------------

def gen_fact_lookup(
    corpus: Dict[str, Any],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Each synthetic question already has a source video_id, so it's a ready-made
    fact_lookup query.  We also record it as a paraphrase variant if we apply a
    simple rewrite.
    """
    vid_map = {v["video_id"]: v for v in corpus["videos"]}
    out: List[Dict[str, Any]] = []

    for sq in corpus["synq"]:
        vid  = sq["video_id"]
        meta = vid_map.get(vid)
        if not meta:
            continue

        tlen  = meta.get("transcript_len") or 0
        tbin  = _transcript_length_bin(tlen)
        wdate = meta.get("watch_date") or ""
        ch    = meta.get("channel") or ""
        title = meta.get("title") or vid

        qid = f"fk_{vid[:8]}_{sq['id']}"
        out.append(_make_query(
            query_id=qid,
            query=sq["question"],
            query_type="fact_lookup",
            source_video_id=vid,
            source_video_title=title,
            expected_video_ids=[vid],
            gold_answer_note="",
            difficulty="easy",
            channel=ch,
            transcript_length_bin=tbin,
            watch_date=wdate,
            notes=f"from synthetic_questions id={sq['id']}",
        ))

    return out


# ---------------------------------------------------------------------------
# Generator 2: entity_topic — from entities table
# ---------------------------------------------------------------------------

_ENTITY_TEMPLATES = [
    "What is discussed about {name} in the videos?",
    "Which video covers {name}?",
    "What role does {name} play according to the indexed videos?",
    "Find information about {name}.",
    "What did the video say about {name}?",
    "How is {name} described in the video content?",
    "What happened with {name}?",
    "What connection does {name} have in the video?",
]

_ENTITY_TYPE_TEMPLATES: Dict[str, List[str]] = {
    "person": [
        "Who is {name} and what did they do according to the videos?",
        "What is {name}'s role described as in the indexed content?",
        "What did {name} say or do according to the videos?",
    ],
    "organization": [
        "What is discussed about {name} in the videos?",
        "What did {name} do according to the indexed content?",
        "How is {name} described in the video transcripts?",
    ],
    "software": [
        "What is {name} according to the videos?",
        "How is {name} used or described in the video content?",
        "What issue or feature of {name} is covered?",
    ],
    "protocol": [
        "What is explained about {name} in the videos?",
        "How does {name} work according to the video content?",
    ],
    "location": [
        "What is discussed about {name} in the video content?",
        "What happened in {name} according to the indexed videos?",
    ],
}


def gen_entity_topic(
    corpus: Dict[str, Any],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    vid_map = {v["video_id"]: v for v in corpus["videos"]}
    out: List[Dict[str, Any]] = []
    seen: set = set()

    for ent in corpus["entities"]:
        vid  = ent["video_id"]
        meta = vid_map.get(vid)
        if not meta:
            continue

        name  = ent.get("name", "").strip()
        etype = ent.get("type", "").lower()
        if not name or len(name) < 2:
            continue

        dedup_key = (vid, name.lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Pick a template
        templates = _ENTITY_TYPE_TEMPLATES.get(etype, _ENTITY_TEMPLATES)
        template  = rng.choice(templates)
        query     = template.format(name=name)

        tlen  = meta.get("transcript_len") or 0
        tbin  = _transcript_length_bin(tlen)
        wdate = meta.get("watch_date") or ""
        ch    = meta.get("channel") or ""
        title = meta.get("title") or vid

        slug = re.sub(r"[^a-z0-9]", "_", name.lower())[:20]
        qid  = f"et_{vid[:8]}_{slug}"

        out.append(_make_query(
            query_id=qid,
            query=query,
            query_type="entity_topic",
            source_video_id=vid,
            source_video_title=title,
            expected_video_ids=[vid],
            gold_answer_note=f"entity={name} type={etype}",
            difficulty="easy",
            channel=ch,
            transcript_length_bin=tbin,
            watch_date=wdate,
            notes=f"entity type={etype}",
        ))

    return out


# ---------------------------------------------------------------------------
# Generator 3: summary_comparison — videos sharing a channel
# ---------------------------------------------------------------------------

_COMPARISON_TEMPLATES = [
    "What topics do the videos from {channel} cover?",
    "Compare what {channel} discussed across different videos.",
    "What common themes appear in {channel}'s content?",
    "Summarise the main points covered by {channel}.",
    "What are the key ideas discussed in {channel}'s videos?",
    "What does {channel} focus on across their indexed videos?",
]


def gen_summary_comparison(
    corpus: Dict[str, Any],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    from collections import defaultdict

    by_channel: Dict[str, List[Dict]] = defaultdict(list)
    for v in corpus["videos"]:
        ch = (v.get("channel") or "").strip()
        if ch:
            by_channel[ch].append(v)

    out: List[Dict[str, Any]] = []
    for ch, vids in by_channel.items():
        if len(vids) < 2:
            continue

        template = rng.choice(_COMPARISON_TEMPLATES)
        query    = template.format(channel=ch)
        vid_ids  = [v["video_id"] for v in vids]
        titles   = [v.get("title") or v["video_id"] for v in vids]

        slug = re.sub(r"[^a-z0-9]", "_", ch.lower())[:24]
        qid  = f"sc_{slug}"

        out.append(_make_query(
            query_id=qid,
            query=query,
            query_type="summary_comparison",
            source_video_id=vid_ids[0],
            source_video_title=titles[0],
            expected_video_ids=vid_ids,
            gold_answer_note=f"channel={ch!r}  videos={len(vid_ids)}",
            difficulty="medium",
            support_notes=(
                "Multiple videos expected. Corroboration across chunks "
                "and Shorties from all videos should count."
            ),
            channel=ch,
            notes=f"{len(vids)} videos from {ch!r}: {', '.join(titles[:3])}{'...' if len(titles) > 3 else ''}",
        ))

    return out


# ---------------------------------------------------------------------------
# Generator 4: paraphrase — syntactic rewrites of fact_lookup queries
# ---------------------------------------------------------------------------

_REWRITES: List[tuple] = [
    (r"^What\b", "Can you explain what"),
    (r"^Who\b", "Which person"),
    (r"^How\b", "In what way"),
    (r"^When\b", "At what point in time"),
    (r"^Where\b", "In which location"),
    (r"^Why\b", "For what reason"),
    (r"\?$", " — please explain."),
    (r"^What is ", "Tell me about "),
    (r"^What are ", "Describe the "),
    (r"^Did\b", "Is it true that"),
]


def _paraphrase(text: str, rng: random.Random) -> str:
    """Apply one random rewrite rule to a question."""
    rng.shuffle(_REWRITES)  # pick first that matches
    for pattern, replacement in _REWRITES:
        new = re.sub(pattern, replacement, text, count=1)
        if new != text:
            return new
    # fallback: prefix
    return "I'd like to know: " + text


def gen_paraphrase(
    fact_lookup: List[Dict[str, Any]],
    rng: random.Random,
    max_count: int = 60,
) -> List[Dict[str, Any]]:
    """
    Take a random sample of fact_lookup queries and create paraphrase variants.
    These test whether retrieval is robust to indirect phrasing.
    """
    sample = rng.sample(fact_lookup, min(max_count, len(fact_lookup)))
    out: List[Dict[str, Any]] = []

    for orig in sample:
        new_q = _paraphrase(orig["query"], rng)
        if new_q == orig["query"]:
            continue

        new = dict(orig)
        new["query_id"] = "pp_" + orig["query_id"]
        new["id"]       = new["query_id"]
        new["query"]    = new_q
        new["query_type"] = "paraphrase"
        new["category"] = "specific_fact"
        new["difficulty"] = "medium"
        new["notes"]    = f"paraphrase of {orig['query_id']}: {orig['query'][:60]}"
        new["label_status"] = "candidate"
        out.append(new)

    return out


# ---------------------------------------------------------------------------
# Generator 5: tricky — short / sparse transcripts and ambiguous questions
# ---------------------------------------------------------------------------

_TRICKY_TEMPLATES = [
    "What is the main point of this video?",
    "What does the speaker conclude?",
    "What problem is being described?",
    "What is the speaker's recommendation?",
    "What outcome is described at the end of the video?",
    "What was the key finding mentioned?",
    "What is surprising about the content of this video?",
    "What context is needed to understand this video?",
]


def gen_tricky(
    corpus: Dict[str, Any],
    rng: random.Random,
    max_count: int = 40,
) -> List[Dict[str, Any]]:
    """
    Tricky queries:
    - Short transcripts  → less evidence available, harder to retrieve
    - Videos without Shorties → chunk-only retrieval must carry all weight
    - Open-ended summary questions → no single clear answer
    """
    out: List[Dict[str, Any]] = []
    vids_with_shorty = {
        v["video_id"] for v in corpus["videos"] if v.get("shorty")
    }

    # Sort by transcript length so we pick the shortest first
    sorted_vids = sorted(corpus["videos"], key=lambda v: v.get("transcript_len") or 0)

    candidates = []
    for v in sorted_vids:
        tlen = v.get("transcript_len") or 0
        if tlen == 0:
            continue
        vid = v["video_id"]
        title = v.get("title") or vid
        ch    = v.get("channel") or ""
        wdate = v.get("watch_date") or ""
        tbin  = _transcript_length_bin(tlen)

        # Label difficulty
        has_shorty = vid in vids_with_shorty
        if tbin == "short" and not has_shorty:
            diff = "hard"
        elif tbin == "short":
            diff = "medium"
        else:
            diff = "medium"

        template = rng.choice(_TRICKY_TEMPLATES)
        # Make it video-specific
        query = f"In the video \"{title}\", {template[0].lower() + template[1:]}"

        slug = re.sub(r"[^a-z0-9]", "_", vid)[:16]
        qid  = f"tr_{slug}"

        candidates.append(_make_query(
            query_id=qid,
            query=query,
            query_type="tricky",
            source_video_id=vid,
            source_video_title=title,
            expected_video_ids=[vid],
            gold_answer_note="",
            difficulty=diff,
            support_notes=(
                "Short transcript or no Shorty — retrieval relies on chunk overlap. "
                "Note whether neighbour expansion helps."
                if diff == "hard" else ""
            ),
            channel=ch,
            transcript_length_bin=tbin,
            watch_date=wdate,
            notes=f"transcript_len={tlen}  has_shorty={has_shorty}",
        ))

    return rng.sample(candidates, min(max_count, len(candidates)))


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def _stratified_sample(
    queries: List[Dict[str, Any]],
    target: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Sample `target` queries while keeping rough balance across:
      - query_type
      - channel
      - transcript_length_bin

    Strategy: reservoir sampling weighted by under-representation.
    """
    if len(queries) <= target:
        return list(queries)

    # Count current representation per type
    from collections import Counter
    type_counts: Counter = Counter()
    result: List[Dict[str, Any]] = []

    # Shuffle to avoid systematic bias
    pool = list(queries)
    rng.shuffle(pool)

    # Pass 1: one of each type, each channel — greedy fill
    seen_types: Dict[str, int] = {}
    seen_channels: Dict[str, int] = {}
    type_target = max(1, target // 5)     # ~5 query types
    channel_target = max(1, target // 8)  # rough per-channel cap

    priority: List[Dict] = []
    remainder: List[Dict] = []
    for q in pool:
        qt = q["query_type"]
        ch = q.get("channel", "")
        if (seen_types.get(qt, 0) < type_target
                and seen_channels.get(ch, 0) < channel_target):
            priority.append(q)
            seen_types[qt] = seen_types.get(qt, 0) + 1
            seen_channels[ch] = seen_channels.get(ch, 0) + 1
        else:
            remainder.append(q)

    # Pass 2: fill remaining slots
    result = priority[:target]
    still_needed = target - len(result)
    if still_needed > 0:
        result += rng.sample(remainder, min(still_needed, len(remainder)))

    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Deduplication by query text
# ---------------------------------------------------------------------------

def _dedup(queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_q: set = set()
    seen_id: set = set()
    out: List[Dict] = []
    for q in queries:
        nq = q["query"].strip().lower()
        qid = q["query_id"]
        if nq in seen_q or qid in seen_id:
            continue
        seen_q.add(nq)
        seen_id.add(qid)
        out.append(q)
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "query_id", "query_type", "difficulty", "label_status",
    "query",
    "source_video_title", "source_video_id",
    "expected_video_ids",
    "gold_answer_note", "ground_truth",
    "channel", "transcript_length_bin", "watch_date",
    "support_notes", "notes",
]


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            # Flatten list fields for CSV readability
            row["expected_video_ids"] = "|".join(row.get("expected_video_ids") or [])
            w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local eval dataset from the Ask Shorty corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_eval_dataset.py                   # auto-detects full corpus\n"
            "  python build_eval_dataset.py --inventory       # print & save corpus stats\n"
            "  python build_eval_dataset.py --db-path C:/path/to/transcripts.db\n"
            "  python build_eval_dataset.py --target 500 --seed 123\n"
        ),
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help=(
            "Path to transcripts.db.  "
            "If omitted the script auto-detects the largest known DB on this machine."
        ),
    )
    parser.add_argument(
        "--chroma-path",
        default=None,
        help="Path to the Chroma vector store (used for inventory only).",
    )
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Print a full corpus inventory and exit (no candidates built).",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=300,
        help="Target number of candidates in the output (default 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--out-dir",
        default=str(CANDIDATES_DIR),
        help="Output directory (default: eval_data/candidates/)",
    )
    args = parser.parse_args()

    # Resolve DB path (auto-detect if not given)
    db_path = find_best_db(args.db_path)
    chroma_path = find_companion_chroma(db_path, args.chroma_path)

    print(f"Using DB       : {db_path}  ({round(db_path.stat().st_size / 1e6, 1)} MB)")
    if chroma_path:
        print(f"Using Chroma   : {chroma_path}")
    else:
        print("Chroma         : not found (inventory will skip Chroma stats)")

    if args.inventory:
        run_inventory(db_path, chroma_path, EVAL_DATA_DIR / "candidates")
        return

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)

    print(f"\nLoading corpus from {db_path} …")
    corpus = _load_corpus(str(db_path))
    print(
        f"  {len(corpus['videos'])} videos with transcripts, "
        f"{len(corpus['synq'])} synthetic questions, "
        f"{len(corpus['entities'])} entities"
    )

    # ---- Generate from each source ----------------------------------------
    print("Generating fact_lookup from synthetic questions …")
    fact_lookup = gen_fact_lookup(corpus, rng)
    print(f"  {len(fact_lookup)} candidates")

    print("Generating entity_topic from entities table …")
    entity_topic = gen_entity_topic(corpus, rng)
    print(f"  {len(entity_topic)} candidates")

    print("Generating summary_comparison from same-channel video pairs …")
    summary_comp = gen_summary_comparison(corpus, rng)
    print(f"  {len(summary_comp)} candidates")

    print("Generating paraphrase variants …")
    paraphrase = gen_paraphrase(fact_lookup, rng, max_count=120)
    print(f"  {len(paraphrase)} candidates")

    print("Generating tricky / edge-case queries …")
    tricky = gen_tricky(corpus, rng, max_count=80)
    print(f"  {len(tricky)} candidates")

    # ---- Combine and sample -----------------------------------------------
    all_candidates = fact_lookup + entity_topic + summary_comp + paraphrase + tricky
    all_candidates = _dedup(all_candidates)
    print(f"\nTotal unique candidates before sampling: {len(all_candidates)}")

    sampled = _stratified_sample(all_candidates, args.target, rng)
    print(f"After stratified sampling: {len(sampled)}")

    # ---- Stats summary -----------------------------------------------------
    from collections import Counter
    type_dist    = Counter(q["query_type"]             for q in sampled)
    tbin_dist    = Counter(q["transcript_length_bin"]  for q in sampled)
    diff_dist    = Counter(q["difficulty"]             for q in sampled)
    channel_dist = Counter(q["channel"] or "unknown"  for q in sampled)

    print("\nQuery type distribution:")
    for k, v in sorted(type_dist.items()):
        print(f"  {k:<22} {v}")
    print("\nTranscript length distribution:")
    for k, v in sorted(tbin_dist.items()):
        print(f"  {k:<12} {v}")
    print("\nDifficulty distribution:")
    for k, v in sorted(diff_dist.items()):
        print(f"  {k:<10} {v}")
    print(f"\nUnique channels represented: {len(channel_dist)}")
    print(f"Top channels: {dict(channel_dist.most_common(5))}")

    # ---- Write outputs -------------------------------------------------------
    jsonl_path = out_dir / "candidates.jsonl"
    csv_path   = out_dir / "candidates.csv"
    stats_path = out_dir / "build_stats.json"

    write_jsonl(jsonl_path, sampled)
    write_csv(csv_path, sampled)

    stats = {
        "built_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "db_path": str(db_path),
        "seed": args.seed,
        "total_candidates": len(sampled),
        "type_distribution": dict(type_dist),
        "transcript_length_distribution": dict(tbin_dist),
        "difficulty_distribution": dict(diff_dist),
        "unique_channels": len(channel_dist),
        "unique_videos": len({q["source_video_id"] for q in sampled}),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nOutputs written to {out_dir}/")
    print(f"  {jsonl_path.name}   ({len(sampled)} records)")
    print(f"  {csv_path.name}")
    print(f"  {stats_path.name}")
    print(
        "\nNext step: review candidates interactively with:\n"
        "  python review_eval_dataset.py\n"
        "or open candidates.csv in a spreadsheet and mark label_status manually."
    )


if __name__ == "__main__":
    main()
