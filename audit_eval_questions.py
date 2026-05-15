#!/usr/bin/env python3
"""
Audit per-query eval artifacts (fk_ / tr_ only) for leakage and weak questions.

Usage:
  python audit_eval_questions.py eval_results/20260514_065902
  python audit_eval_questions.py eval_results/20260514_065902/queries
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# V1 baseline modes to read recall@5 from (first match wins).
_V1_RECALL_MODES = ("full_system__baseline", "chunk_synq__baseline")

# Generic channel/corpus templates that match many videos.
_GENERIC_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bwhat does .+ focus on\b", re.I),
    re.compile(r"\bfocus on across\b", re.I),
    re.compile(r"\bsummarise the main points\b", re.I),
    re.compile(r"\bsummarize the main points\b", re.I),
    re.compile(r"\bwhat common themes\b", re.I),
    re.compile(r"\bcompare what .+ discussed\b", re.I),
    re.compile(r"\bwhat topics do\b", re.I),
    re.compile(r"\bwhat are the key ideas discussed\b", re.I),
]

_WORD_RE = re.compile(r"[a-z0-9]+", re.I)


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall((text or "").lower()))


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _title_words(title: str) -> List[str]:
    return [w for w in _WORD_RE.findall((title or "").lower()) if len(w) >= 3]


def _title_in_query(query: str, title: str) -> bool:
    """True if title appears verbatim/near-verbatim in the query."""
    qn = _normalize(query)
    tn = _normalize(title)
    if not qn or not tn:
        return False
    if tn in qn or qn in tn:
        return True
    # Quoted title (common in tr_ tricky questions).
    raw_title = (title or "").strip()
    if raw_title and raw_title.lower() in (query or "").lower():
        return True
    tw = _title_words(title)
    if not tw:
        return False
    qw = set(_WORD_RE.findall(qn))
    overlap = sum(1 for w in tw if w in qw)
    return overlap / len(tw) >= 0.55


def _is_generic(query: str) -> bool:
    return any(p.search(query) for p in _GENERIC_PATTERNS)


def _suspect_reasons(query: str, title: str) -> List[str]:
    reasons: List[str] = []
    if _title_in_query(query, title):
        reasons.append("title-leak")
    if _word_count(query) < 8:
        reasons.append("short")
    if _is_generic(query):
        reasons.append("generic")
    return reasons


def _v1_recall_at_5(data: Dict[str, Any]) -> Optional[bool]:
    modes = data.get("modes") or {}
    for key in _V1_RECALL_MODES:
        block = modes.get(key)
        if not isinstance(block, dict):
            continue
        metrics = block.get("metrics")
        if isinstance(metrics, dict) and "recall_at_5" in metrics:
            return bool(metrics["recall_at_5"])
        # Fallback: compute from top_video_ids if metrics missing.
        top = block.get("top_video_ids") or []
        rel = data.get("relevant_video_ids") or []
        if top and rel:
            return any(r in top[:5] for r in rel)
    return None


def _resolve_queries_dir(path: Path) -> Path:
    path = path.resolve()
    if path.is_dir() and (path / "queries").is_dir():
        return path / "queries"
    return path


def _load_records(queries_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(queries_dir.glob("*.json")):
        name = p.stem
        if name.startswith("sc_"):
            continue
        if not (name.startswith("fk_") or name.startswith("tr_")):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] skip {p.name}: {exc}", file=sys.stderr)
            continue
        recall = _v1_recall_at_5(data)
        rows.append(
            {
                "filename": p.name,
                "query": (data.get("query") or "").strip(),
                "query_type": data.get("query_type") or "?",
                "source_video_title": (data.get("source_video_title") or "").strip(),
                "relevant_video_ids": list(data.get("relevant_video_ids") or []),
                "recall_at_5": recall,
                "prefix": "fk" if name.startswith("fk_") else "tr",
            }
        )
    return rows


def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ")
    if len(s) > n:
        s = s[: n - 1] + "..."
    return s.encode("ascii", "replace").decode("ascii")


def _print_table(rows: List[Dict[str, Any]]) -> None:
    w_q, w_t, w_r = 52, 40, 6
    header = (
        f"{'SUS':<4} {'R@5':<{w_r}} "
        f"{'Query':<{w_q}} {'Source title':<{w_t}}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        reasons = _suspect_reasons(row["query"], row["source_video_title"])
        flag = ",".join(reasons) if reasons else ""
        r5 = row["recall_at_5"]
        r5s = "?" if r5 is None else ("1" if r5 else "0")
        print(
            f"{flag:<4} {r5s:<{w_r}} "
            f"{_truncate(row['query'], w_q):<{w_q}} "
            f"{_truncate(row['source_video_title'], w_t):<{w_t}}"
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Audit fk_/tr_ eval query JSON files for weak or leaky questions."
    )
    ap.add_argument(
        "eval_dir",
        type=Path,
        help="Eval run directory (e.g. eval_results/20260514_065902) or .../queries",
    )
    args = ap.parse_args()

    queries_dir = _resolve_queries_dir(args.eval_dir)
    if not queries_dir.is_dir():
        print(f"Not a directory: {queries_dir}", file=sys.stderr)
        sys.exit(1)

    rows = _load_records(queries_dir)
    if not rows:
        print(f"No fk_/tr_ JSON files in {queries_dir}", file=sys.stderr)
        sys.exit(1)

    fk_rows = [r for r in rows if r["prefix"] == "fk"]
    tr_rows = [r for r in rows if r["prefix"] == "tr"]

    print(f"Eval queries dir: {queries_dir}\n")

    if fk_rows:
        print(f"=== fact_lookup (fk_) — {len(fk_rows)} questions ===\n")
        _print_table(fk_rows)
        print()

    if tr_rows:
        print(f"=== tricky (tr_) — {len(tr_rows)} questions ===\n")
        _print_table(tr_rows)
        print()

    suspect = sum(
        1 for r in rows if _suspect_reasons(r["query"], r["source_video_title"])
    )
    clean = len(rows) - suspect
    print("=== Summary ===")
    print(f"  Total questions : {len(rows)}  (fk={len(fk_rows)}, tr={len(tr_rows)})")
    print(f"  Suspect         : {suspect}")
    print(f"  Clean           : {clean}")
    print("\nSuspect flags: title-leak | short (<8 words) | generic template")


if __name__ == "__main__":
    main()
