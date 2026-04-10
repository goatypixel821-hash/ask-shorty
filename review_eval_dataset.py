#!/usr/bin/env python3
"""
Interactive review and downselect tool for the eval dataset.

Reads eval_data/candidates/candidates.jsonl and lets you go through each
query one at a time, marking it approved / rejected / edited.

Progress is saved to eval_data/review/review_progress.json after every
decision so you can quit and resume at any time.

Modes:
  (default)     Interactive review, one query at a time
  --finalize    Copy all approved items to eval_data/final/golden.jsonl
                and golden.csv, then print a summary
  --stats       Print review statistics without changing anything
  --export-csv  Write the current approved set to a CSV for spreadsheet editing
  --import-csv  Re-import a CSV (after spreadsheet edits) updating label_status

Usage:
  python review_eval_dataset.py
  python review_eval_dataset.py --finalize
  python review_eval_dataset.py --stats
  python review_eval_dataset.py --export-csv eval_data/review/review_export.csv
  python review_eval_dataset.py --import-csv eval_data/review/review_export.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_DATA = Path(__file__).parent / "eval_data"
CANDIDATES_FILE  = EVAL_DATA / "candidates" / "candidates.jsonl"
APPROVED_FILE    = EVAL_DATA / "review" / "approved.jsonl"
REJECTED_FILE    = EVAL_DATA / "review" / "rejected.jsonl"
PROGRESS_FILE    = EVAL_DATA / "review" / "review_progress.json"
FINAL_JSONL      = EVAL_DATA / "final" / "golden.jsonl"
FINAL_CSV        = EVAL_DATA / "final" / "golden.csv"


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


CSV_FIELDS = [
    "query_id", "query_type", "difficulty", "label_status",
    "query",
    "source_video_title", "source_video_id",
    "expected_video_ids",
    "gold_answer_note", "ground_truth",
    "channel", "transcript_length_bin", "watch_date",
    "support_notes", "notes",
]


def _write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            row["expected_video_ids"] = "|".join(row.get("expected_video_ids") or [])
            w.writerow(row)


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            # Re-expand pipe-joined list
            raw_vids = row.get("expected_video_ids", "")
            row["expected_video_ids"] = [v for v in raw_vids.split("|") if v]
            row["relevant_video_ids"] = row["expected_video_ids"]
            # keep id alias
            row["id"] = row.get("query_id", row.get("id", ""))
            records.append(row)
    return records


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def _load_progress() -> Dict[str, Any]:
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"reviewed_ids": [], "cursor": 0}


def _save_progress(prog: Dict[str, Any]) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(prog, f, indent=2)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _divider(char: str = "─", width: int = 72) -> str:
    return char * width


def _display_query(q: Dict[str, Any], idx: int, total: int) -> None:
    print("\n" + _divider("═"))
    print(f"  [{idx + 1}/{total}]  {q['query_id']}  |  type={q.get('query_type','?')}  "
          f"|  diff={q.get('difficulty','?')}")
    print(_divider())
    print(f"  QUERY    : {q['query']}")
    print(f"  VIDEO    : {q.get('source_video_title', '')} ({q.get('source_video_id', '')})")
    print(f"  CHANNEL  : {q.get('channel', '')}")
    print(f"  ANSWER   : {q.get('gold_answer_note') or q.get('ground_truth') or '(none)'}")
    print(f"  TLEN BIN : {q.get('transcript_length_bin', '?')}")
    if q.get("support_notes"):
        print(f"  SUPPORT  : {q['support_notes'][:100]}")
    if q.get("notes"):
        print(f"  NOTES    : {q['notes'][:100]}")
    expected = q.get("expected_video_ids") or []
    if len(expected) > 1:
        print(f"  EXPECTED : {', '.join(expected)}")
    print(_divider())


def _print_commands() -> None:
    print(
        "  [a]pprove  [r]eject  [e]dit  [s]kip  [?]show context  [q]uit"
    )


# ---------------------------------------------------------------------------
# Edit a query interactively
# ---------------------------------------------------------------------------

_EDITABLE_FIELDS = [
    "query", "query_type", "difficulty", "gold_answer_note",
    "ground_truth", "expected_video_ids", "support_notes", "notes",
]


def _edit_query(q: Dict[str, Any]) -> Dict[str, Any]:
    print("\n  Editable fields:")
    for i, field in enumerate(_EDITABLE_FIELDS):
        current = q.get(field, "")
        if isinstance(current, list):
            current = "|".join(current)
        print(f"  {i + 1}. {field}: {str(current)[:70]}")
    print("  0. Done editing")

    while True:
        raw = input("  Field number to edit (0 to finish): ").strip()
        if raw == "0" or raw == "":
            break
        try:
            n = int(raw) - 1
            if n < 0 or n >= len(_EDITABLE_FIELDS):
                print("  Out of range.")
                continue
        except ValueError:
            print("  Enter a number.")
            continue

        field = _EDITABLE_FIELDS[n]
        current = q.get(field, "")
        if isinstance(current, list):
            current = "|".join(current)
        print(f"  Current: {current}")
        new_val = input(f"  New value for {field}: ").strip()

        if field == "expected_video_ids":
            q[field] = [v.strip() for v in new_val.split("|") if v.strip()]
            q["relevant_video_ids"] = q[field]
        else:
            q[field] = new_val

    return q


# ---------------------------------------------------------------------------
# Core review loop
# ---------------------------------------------------------------------------

def run_review(candidates_file: Path) -> None:
    candidates = _load_jsonl(candidates_file)
    if not candidates:
        print(f"No candidates found at {candidates_file}")
        print("Run: python build_eval_dataset.py")
        return

    prog = _load_progress()
    reviewed_set: set = set(prog.get("reviewed_ids", []))
    approved = _load_jsonl(APPROVED_FILE)
    rejected = _load_jsonl(REJECTED_FILE)
    approved_ids = {q["query_id"] for q in approved}

    # Filter to unreviewed
    pending = [q for q in candidates if q["query_id"] not in reviewed_set]
    total   = len(candidates)

    print(f"\nLoaded {total} candidates. {len(reviewed_set)} already reviewed.")
    print(f"  Approved so far : {len(approved_ids)}")
    print(f"  Remaining       : {len(pending)}")
    if not pending:
        print("\nAll candidates reviewed. Run --finalize to create the golden set.")
        return

    print("\nStarting review. Commands: [a]pprove [r]eject [e]dit [s]kip [?]context [q]uit\n")

    for i, q in enumerate(pending):
        _display_query(q, total - len(pending) + i, total)
        _print_commands()

        while True:
            cmd = input("  > ").strip().lower()
            if cmd == "a":
                q["label_status"] = "approved"
                _append_jsonl(APPROVED_FILE, q)
                approved_ids.add(q["query_id"])
                reviewed_set.add(q["query_id"])
                print("  ✓ Approved")
                break
            elif cmd == "r":
                q["label_status"] = "rejected"
                _append_jsonl(REJECTED_FILE, q)
                reviewed_set.add(q["query_id"])
                print("  ✗ Rejected")
                break
            elif cmd == "e":
                q = _edit_query(q)
                q["label_status"] = "approved"
                _append_jsonl(APPROVED_FILE, q)
                approved_ids.add(q["query_id"])
                reviewed_set.add(q["query_id"])
                print("  ✓ Edited and approved")
                break
            elif cmd == "s":
                reviewed_set.add(q["query_id"])
                print("  → Skipped (you can come back later)")
                break
            elif cmd == "?":
                print(f"\n  Full query record:")
                for k, v in q.items():
                    if k not in ("category", "relevant_video_ids", "id"):
                        print(f"    {k}: {v}")
                _print_commands()
            elif cmd == "q":
                prog["reviewed_ids"] = list(reviewed_set)
                _save_progress(prog)
                print(
                    f"\nProgress saved. Approved: {len(approved_ids)}  "
                    f"Remaining: {len(pending) - i - 1}"
                )
                return
            else:
                print("  Unknown command. [a]pprove [r]eject [e]dit [s]kip [?] [q]uit")

        prog["reviewed_ids"] = list(reviewed_set)
        _save_progress(prog)

    print(
        f"\nAll done! Approved: {len(approved_ids)}  Rejected: {len(rejected) + 1}  "
        "\nRun --finalize to create the golden eval set."
    )


# ---------------------------------------------------------------------------
# Finalize: merge approved into golden set
# ---------------------------------------------------------------------------

def finalize(target: int = 100) -> None:
    approved = _load_jsonl(APPROVED_FILE)
    if not approved:
        print("No approved items yet. Run the interactive review first.")
        return

    # Deduplicate by query_id (keep last version in case of edits)
    seen: Dict[str, Dict] = {}
    for q in approved:
        seen[q["query_id"]] = q
    golden = list(seen.values())

    # Sort for reproducible output
    golden.sort(key=lambda q: (q.get("query_type", ""), q["query_id"]))

    _write_jsonl(FINAL_JSONL, golden)
    _write_csv(FINAL_CSV, golden)

    from collections import Counter
    type_dist = Counter(q.get("query_type", "?") for q in golden)
    diff_dist = Counter(q.get("difficulty", "?") for q in golden)

    print(f"\nFinalised golden set: {len(golden)} queries")
    print(f"  Query type distribution: {dict(type_dist)}")
    print(f"  Difficulty distribution: {dict(diff_dist)}")
    print(f"\nOutputs:")
    print(f"  {FINAL_JSONL}")
    print(f"  {FINAL_CSV}")
    print(f"\nRun evaluation:")
    print(f"  python evaluate_rag.py --queries-file {FINAL_JSONL} --mode all --no-answer")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats() -> None:
    candidates = _load_jsonl(CANDIDATES_FILE)
    approved   = _load_jsonl(APPROVED_FILE)
    rejected   = _load_jsonl(REJECTED_FILE)
    final      = _load_jsonl(FINAL_JSONL)
    prog       = _load_progress()
    reviewed   = set(prog.get("reviewed_ids", []))

    print(f"\nReview stats:")
    print(f"  Candidates   : {len(candidates)}")
    print(f"  Reviewed     : {len(reviewed)}")
    print(f"  Approved     : {len(approved)}")
    print(f"  Rejected     : {len(rejected)}")
    print(f"  Skipped      : {len(reviewed) - len(approved) - len(rejected)}")
    print(f"  Final golden : {len(final)}")

    if approved:
        from collections import Counter
        type_dist = Counter(q.get("query_type", "?") for q in approved)
        diff_dist = Counter(q.get("difficulty", "?") for q in approved)
        ch_dist   = Counter(q.get("channel", "?")    for q in approved)
        print(f"\n  Approved by type : {dict(type_dist)}")
        print(f"  Approved by diff : {dict(diff_dist)}")
        print(f"  Unique channels  : {len(ch_dist)}")


# ---------------------------------------------------------------------------
# Auto-approve: diverse stratified selection for fast golden set
# ---------------------------------------------------------------------------

def auto_approve(n: int, candidates_file: Path) -> None:
    """
    Automatically approve the N most diverse candidates without manual review.

    Selection strategy:
      1. Prefer candidates where retrieval_feasible=True (Chroma can find them).
      2. Enforce hard per-type and per-channel caps so the set is diverse.
      3. Fill remaining slots by cycling through types in priority order.

    Adds approved records to approved.jsonl and marks them in progress.json.
    Does NOT touch already-approved or already-rejected items.
    """
    from collections import Counter

    candidates = _load_jsonl(candidates_file)
    if not candidates:
        print(f"No candidates at {candidates_file}  —  run build_eval_dataset.py first.")
        return

    prog        = _load_progress()
    reviewed    = set(prog.get("reviewed_ids", []))
    existing_approved = {q["query_id"]: q for q in _load_jsonl(APPROVED_FILE)}

    # Only consider items not already reviewed
    pool = [q for q in candidates if q["query_id"] not in reviewed]
    if not pool:
        print("All candidates already reviewed.")
        return

    # Sort: feasible first, then by retrieval score if available, then by id
    def _sort_key(q: Dict) -> Tuple:
        feasible = 0 if q.get("retrieval_feasible") is True else (
            1 if q.get("retrieval_feasible") is None else 2
        )
        rank = q.get("expected_rank_global") or 99
        return (feasible, rank, q.get("query_id", ""))

    pool.sort(key=_sort_key)

    # Caps to enforce diversity
    type_cap    = max(1, n // 4)     # at most n/4 per type (5 types → ~25% each)
    channel_cap = max(1, n // 10)    # at most n/10 per channel

    type_counts: Dict[str, int]    = Counter()
    channel_counts: Dict[str, int] = Counter()
    selected: List[Dict]           = []

    for q in pool:
        if len(selected) >= n:
            break
        qt = q.get("query_type", "?")
        ch = q.get("channel", "?") or "unknown"
        if type_counts[qt] >= type_cap:
            continue
        if channel_counts[ch] >= channel_cap:
            continue
        selected.append(q)
        type_counts[qt]    += 1
        channel_counts[ch] += 1

    # If we still need more (caps were tight), relax channel cap
    if len(selected) < n:
        remainder = [q for q in pool if q["query_id"] not in {s["query_id"] for s in selected}]
        for q in remainder:
            if len(selected) >= n:
                break
            qt = q.get("query_type", "?")
            if type_counts[qt] >= type_cap:
                continue
            selected.append(q)
            type_counts[qt] += 1

    # Final fill with anything remaining
    if len(selected) < n:
        used = {s["query_id"] for s in selected}
        for q in pool:
            if len(selected) >= n:
                break
            if q["query_id"] not in used:
                selected.append(q)

    if not selected:
        print("No candidates available to auto-approve.")
        return

    # Mark as approved
    for q in selected:
        q["label_status"] = "auto_approved"
        existing_approved[q["query_id"]] = q
        reviewed.add(q["query_id"])

    _write_jsonl(APPROVED_FILE, list(existing_approved.values()))
    prog["reviewed_ids"] = list(reviewed)
    _save_progress(prog)

    # Report
    final_type  = Counter(q.get("query_type",  "?") for q in selected)
    final_diff  = Counter(q.get("difficulty",  "?") for q in selected)
    feasible    = sum(1 for q in selected if q.get("retrieval_feasible") is True)
    in_top5     = sum(1 for q in selected if (q.get("expected_rank_global") or 99) <= 5)
    print(f"\nAuto-approved {len(selected)} candidates:")
    print(f"  By type            : {dict(final_type)}")
    print(f"  By difficulty      : {dict(final_diff)}")
    print(f"  Retrieval feasible : {feasible} / {len(selected)}")
    print(f"  Expected in top-5  : {in_top5} / {len(selected)}")
    print(f"\nNext: run --finalize to create golden.jsonl, or --stats to review.")
    print(f"      Run interactively first if you want to spot-check before finalizing.")


# ---------------------------------------------------------------------------
# CSV export / re-import for spreadsheet editing
# ---------------------------------------------------------------------------

def export_csv_for_review(out_path: Path) -> None:
    """Export candidates (or approved set if it exists) to a CSV for editing."""
    approved = _load_jsonl(APPROVED_FILE)
    source   = approved if approved else _load_jsonl(CANDIDATES_FILE)
    if not source:
        print("Nothing to export.")
        return
    _write_csv(out_path, source)
    print(f"Exported {len(source)} rows to {out_path}")
    print(
        "Edit the 'label_status' column (approved / rejected / skip), "
        "then run --import-csv to re-import."
    )


def import_csv(path: Path) -> None:
    """
    Re-import a reviewed CSV.  Records with label_status='approved' go to
    approved.jsonl, 'rejected' go to rejected.jsonl.
    """
    rows = _load_csv(path)
    if not rows:
        print(f"No rows loaded from {path}")
        return

    prog        = _load_progress()
    reviewed    = set(prog.get("reviewed_ids", []))
    n_approved  = 0
    n_rejected  = 0
    n_skipped   = 0

    approved_out: List[Dict] = []
    rejected_out: List[Dict] = []

    for r in rows:
        status = (r.get("label_status") or "").strip().lower()
        qid    = r.get("query_id") or r.get("id", "")
        reviewed.add(qid)
        if status == "approved":
            r["label_status"] = "approved"
            approved_out.append(r)
            n_approved += 1
        elif status == "rejected":
            r["label_status"] = "rejected"
            rejected_out.append(r)
            n_rejected += 1
        else:
            n_skipped += 1

    # Merge with existing (overwrite by query_id)
    existing_approved = {q["query_id"]: q for q in _load_jsonl(APPROVED_FILE)}
    existing_rejected = {q["query_id"]: q for q in _load_jsonl(REJECTED_FILE)}
    for r in approved_out:
        existing_approved[r["query_id"]] = r
    for r in rejected_out:
        existing_rejected[r["query_id"]] = r

    _write_jsonl(APPROVED_FILE, list(existing_approved.values()))
    _write_jsonl(REJECTED_FILE, list(existing_rejected.values()))
    prog["reviewed_ids"] = list(reviewed)
    _save_progress(prog)

    print(f"Imported {len(rows)} rows:")
    print(f"  Approved : {n_approved}")
    print(f"  Rejected : {n_rejected}")
    print(f"  Skipped  : {n_skipped}")
    print("Run --finalize to create the golden set.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review and downselect the eval dataset interactively."
    )
    parser.add_argument(
        "--candidates-file",
        default=str(CANDIDATES_FILE),
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Copy approved items to eval_data/final/golden.jsonl",
    )
    parser.add_argument(
        "--auto-approve",
        type=int,
        metavar="N",
        default=0,
        help=(
            "Auto-approve the N most diverse candidates without manual review. "
            "Picks items with good retrieval feasibility across types and channels. "
            "Example: --auto-approve 100"
        ),
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print review statistics",
    )
    parser.add_argument(
        "--export-csv",
        metavar="PATH",
        help="Export approved/candidates to CSV for spreadsheet editing",
    )
    parser.add_argument(
        "--import-csv",
        metavar="PATH",
        help="Re-import a reviewed CSV (label_status column determines action)",
    )
    args = parser.parse_args()

    # Resolve candidates file: prefer enriched file if it exists
    cand_file = Path(args.candidates_file)
    enriched_file = cand_file.parent / "candidates_enriched.jsonl"
    if enriched_file.exists() and cand_file == CANDIDATES_FILE:
        cand_file = enriched_file
        print(f"Using enriched candidates: {cand_file}")

    if args.stats:
        print_stats()
    elif args.auto_approve:
        auto_approve(args.auto_approve, cand_file)
    elif args.finalize:
        finalize()
    elif args.export_csv:
        export_csv_for_review(Path(args.export_csv))
    elif args.import_csv:
        import_csv(Path(args.import_csv))
    else:
        run_review(cand_file)


if __name__ == "__main__":
    main()
