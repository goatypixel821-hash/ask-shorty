#!/usr/bin/env python3
"""
Evaluate Ask Shorty agent mode on sc_replacements_v2 cross-video queries.

The agent UI is GET /agent; jobs are submitted via POST /api/agent/ask and polled
at GET /api/agent/result/<job_id> (see ask_shorty_app.py).

Usage:
  # Start the app first: python ask_shorty_app.py  (listens on :5001)
  python evaluate_agent_sc.py
  python evaluate_agent_sc.py --base-url http://localhost:5001 --limit 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from evaluate_rag import mrr, ndcg_at_k, normalize_query_record, recall_at_k

ROOT = Path(__file__).resolve().parent
DEFAULT_QUERIES_DIR = ROOT / "eval_results" / "sc_replacements_v2"
DEFAULT_OUT_DIR = ROOT / "eval_results" / "agent_sc_eval"
DEFAULT_BASE_URL = "http://localhost:5001"

VIDEO_ID_IN_CONTEXT_RE = re.compile(r"video_id=(\S+)", re.I)
YOUTUBE_URL_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
    re.I,
)
CITATION_ID_RE = re.compile(
    r"(?:"
    r"(?:video_id[=:\s]*)"
    r"|(?:youtube\.com/watch\?v=)"
    r"|(?:youtu\.be/)"
    r"|(?:\[)"
    r"|(?:\()"
    r")"
    r"\s*([a-zA-Z0-9_-]{11})\b",
    re.I,
)
LOOSE_ID_RE = re.compile(r"\b([a-zA-Z0-9_-]{11})\b")


def _http_json(
    method: str,
    url: str,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code} {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed {url}: {exc}") from exc
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {url}: {raw[:200]!r}") from exc


def _dedupe_preserve_order(ids: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for vid in ids:
        v = (vid or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def extract_ranked_video_ids(
    result: Dict[str, Any],
    *,
    known_ids: Optional[Set[str]] = None,
) -> List[str]:
    """
    Build an ordered list of video IDs from agent output.

    Priority (first occurrence wins):
      1. sources[].video_id  (tool retrieval order)
      2. used_context headers video_id=
      3. cited IDs in answer text (URLs, video_id=, brackets)
      4. other 11-char tokens in answer that exist in known_ids (if provided)
    """
    ordered: List[str] = []

    for src in result.get("sources") or []:
        if isinstance(src, dict):
            vid = str(src.get("video_id") or "").strip()
            if vid:
                ordered.append(vid)

    for block in result.get("used_context") or []:
        if not isinstance(block, str):
            continue
        for m in VIDEO_ID_IN_CONTEXT_RE.finditer(block):
            ordered.append(m.group(1).rstrip("]").rstrip(")"))

    answer = str(result.get("answer") or "")
    for m in YOUTUBE_URL_RE.finditer(answer):
        ordered.append(m.group(1))
    for m in CITATION_ID_RE.finditer(answer):
        ordered.append(m.group(1))

    if known_ids:
        for m in LOOSE_ID_RE.finditer(answer):
            vid = m.group(1)
            if vid in known_ids:
                ordered.append(vid)

    return _dedupe_preserve_order(ordered)


def submit_agent_question(
    base_url: str,
    question: str,
    *,
    timeout: float = 30.0,
) -> int:
    base = base_url.rstrip("/")
    payload = _http_json(
        "POST",
        f"{base}/api/agent/ask",
        {"question": question},
        timeout=timeout,
    )
    if not payload.get("success"):
        raise RuntimeError(f"ask failed: {payload}")
    job_id = payload.get("job_id")
    if job_id is None:
        raise RuntimeError(f"ask returned no job_id: {payload}")
    return int(job_id)


def poll_agent_result(
    base_url: str,
    job_id: int,
    *,
    poll_interval: float = 2.0,
    timeout_sec: float = 300.0,
) -> Dict[str, Any]:
    base = base_url.rstrip("/")
    deadline = time.monotonic() + timeout_sec
    last_status = "unknown"

    while time.monotonic() < deadline:
        payload = _http_json(
            "GET",
            f"{base}/api/agent/result/{job_id}",
            timeout=30.0,
        )
        status = payload.get("status", "unknown")
        last_status = status

        if payload.get("success") and status == "completed":
            return payload

        if status in ("error", "cancelled", "missing"):
            err = payload.get("error") or status
            raise RuntimeError(f"job {job_id} {status}: {err}")

        if status not in ("pending", "running"):
            raise RuntimeError(f"job {job_id} unexpected status: {status!r}")

        time.sleep(poll_interval)

    raise TimeoutError(
        f"job {job_id} did not complete within {timeout_sec:.0f}s (last status={last_status})"
    )


def run_agent_query(
    base_url: str,
    question: str,
    *,
    poll_interval: float,
    timeout_sec: float,
) -> tuple[Dict[str, Any], float, int]:
    """POST ask + poll until completed. Returns (payload, wall_time_sec, job_id)."""
    t0 = time.perf_counter()
    job_id = submit_agent_question(base_url, question)
    print(f"    job_id={job_id} polling...", flush=True)
    payload = poll_agent_result(
        base_url,
        job_id,
        poll_interval=poll_interval,
        timeout_sec=timeout_sec,
    )
    return payload, time.perf_counter() - t0, job_id


def load_known_video_ids(db_path: Path) -> Set[str]:
    if not db_path.is_file():
        return set()
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT video_id FROM videos").fetchall()
        return {str(r[0]) for r in rows}
    finally:
        conn.close()


def _timing_stats(seconds: Sequence[float]) -> Dict[str, float]:
    if not seconds:
        return {}
    s = sorted(seconds)
    n = len(s)
    mid = n // 2
    median = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0
    return {
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) / n,
        "median": median,
    }


def save_query_artifact(
    out_queries_dir: Path,
    qid: str,
    query_text: str,
    relevant_ids: List[str],
    retrieved_ids: List[str],
    metrics: Dict[str, Any],
    agent_payload: Dict[str, Any],
    source_query: Dict[str, Any],
    *,
    wall_time_sec: float,
    job_id: Optional[int] = None,
) -> None:
    artifact = {
        "query_id": qid,
        "id": qid,
        "query": query_text,
        "query_type": source_query.get("query_type", "summary_comparison"),
        "category": source_query.get("category", "cross_video"),
        "relevant_video_ids": relevant_ids,
        "expected_video_ids": relevant_ids,
        "retrieved_video_ids": retrieved_ids[:10],
        "wall_time_sec": round(wall_time_sec, 3),
        "job_id": job_id,
        "metrics": metrics,
        "agent_answer": agent_payload.get("answer", ""),
        "agent_sources": agent_payload.get("sources", []),
        "agent_used_context_count": len(agent_payload.get("used_context") or []),
        "notes": "evaluate_agent_sc.py run",
    }
    out_queries_dir.mkdir(parents=True, exist_ok=True)
    path = out_queries_dir / f"{qid}.json"
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate agent mode on sc_* queries.")
    ap.add_argument(
        "--queries-dir",
        type=Path,
        default=DEFAULT_QUERIES_DIR,
        help=f"Directory of sc_*.json queries (default: {DEFAULT_QUERIES_DIR})",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    ap.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Ask Shorty app base URL (default: http://localhost:5001)",
    )
    ap.add_argument(
        "--db-path",
        type=Path,
        default=ROOT / "data" / "transcripts.db",
        help="Optional DB path to filter loose 11-char IDs in answers",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max queries to run (0 = all)",
    )
    ap.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between result polls",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-query timeout in seconds",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Reuse artifacts that already have wall_time_sec and metrics (no agent call). "
            "Artifacts without timing are re-run."
        ),
    )
    args = ap.parse_args()

    queries_dir = args.queries_dir.resolve()
    if not queries_dir.is_dir():
        print(f"Queries directory not found: {queries_dir}", file=sys.stderr)
        sys.exit(1)

    paths = sorted(queries_dir.glob("sc_*.json"))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        print(f"No sc_*.json files in {queries_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir.resolve()
    out_queries_dir = out_dir / "queries"
    out_dir.mkdir(parents=True, exist_ok=True)

    known_ids = load_known_video_ids(args.db_path)

    # Health check (GET /agent returns HTML; we only verify TCP/HTTP reachability)
    try:
        req = Request(f"{args.base_url.rstrip('/')}/agent", method="GET")
        with urlopen(req, timeout=5.0) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status}")
    except Exception as exc:
        print(
            f"WARNING: could not reach {args.base_url} ({exc}). "
            "Is ask_shorty_app.py running on port 5001?",
            file=sys.stderr,
        )

    acc: List[Dict[str, float]] = []
    timings: List[float] = []
    all_results: List[Dict[str, Any]] = []
    failed = 0
    cached = 0

    print(f"Queries : {queries_dir} ({len(paths)} files)")
    print(f"Agent   : {args.base_url.rstrip('/')}/api/agent/ask")
    print(f"Output  : {out_dir}\n")

    for i, path in enumerate(paths, 1):
        raw = json.loads(path.read_text(encoding="utf-8"))
        q_obj = normalize_query_record(raw)
        qid = str(q_obj.get("id") or q_obj.get("query_id") or path.stem)
        query_text = (q_obj.get("query") or "").strip()
        relevant = list(
            dict.fromkeys(q_obj.get("relevant_video_ids") or [])
        )
        category = q_obj.get("category") or "cross_video"

        if not query_text:
            print(f"[{i}/{len(paths)}] SKIP {qid}: empty query")
            failed += 1
            continue

        artifact_path = out_queries_dir / f"{qid}.json"
        if args.skip_existing and artifact_path.is_file():
            prev = json.loads(artifact_path.read_text(encoding="utf-8"))
            wt = prev.get("wall_time_sec")
            if wt is not None and not prev.get("error"):
                cached += 1
                m = prev.get("metrics") or {}
                wall = float(wt)
                timings.append(wall)
                acc.append(
                    {
                        "r5": float(m.get("recall_at_5", 0)),
                        "r10": float(m.get("recall_at_10", 0)),
                        "mrr": float(m.get("mrr", 0)),
                        "ndcg": float(m.get("ndcg_at_10", 0)),
                    }
                )
                print(
                    f"[{i}/{len(paths)}] cached {qid}  "
                    f"time={wall:.1f}s  R@5={int(m.get('recall_at_5', 0))}"
                )
                all_results.append(prev)
                continue

        print(f"[{i}/{len(paths)}] {qid}")
        print(f"    Q: {query_text[:90]}{'...' if len(query_text) > 90 else ''}")

        try:
            agent_payload, wall_time_sec, job_id = run_agent_query(
                args.base_url,
                query_text,
                poll_interval=args.poll_interval,
                timeout_sec=args.timeout,
            )
            retrieved = extract_ranked_video_ids(agent_payload, known_ids=known_ids)
        except Exception as exc:
            print(f"    FAIL: {exc}")
            failed += 1
            err_row = {
                "query_id": qid,
                "query": query_text,
                "category": category,
                "relevant_video_ids": relevant,
                "error": str(exc),
            }
            all_results.append(err_row)
            artifact_path.write_text(
                json.dumps(err_row, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            continue

        r5 = recall_at_k(retrieved, relevant, 5)
        r10 = recall_at_k(retrieved, relevant, 10)
        mrr_score = mrr(retrieved, relevant)
        ndcg_score = ndcg_at_k(retrieved, relevant, 10)

        metrics = {
            "recall_at_5": int(r5),
            "recall_at_10": int(r10),
            "mrr": round(mrr_score, 4),
            "ndcg_at_10": round(ndcg_score, 4),
            "answer_correctness": 0,
            "top_video_ids": retrieved[:10],
            "wall_time_sec": round(wall_time_sec, 3),
        }
        timings.append(wall_time_sec)
        acc.append(
            {
                "r5": float(r5),
                "r10": float(r10),
                "mrr": mrr_score,
                "ndcg": ndcg_score,
            }
        )

        print(
            f"    time={wall_time_sec:.1f}s  "
            f"R@5={int(r5)} R@10={int(r10)} "
            f"MRR={mrr_score:.3f} NDCG={ndcg_score:.3f} "
            f"retrieved={len(retrieved)}"
        )
        if retrieved[:5]:
            print(f"    top5: {', '.join(retrieved[:5])}")

        save_query_artifact(
            out_queries_dir,
            qid,
            query_text,
            relevant,
            retrieved,
            metrics,
            agent_payload,
            q_obj,
            wall_time_sec=wall_time_sec,
            job_id=job_id,
        )
        all_results.append(
            {
                "query_id": qid,
                "query": query_text,
                "category": category,
                "relevant_video_ids": relevant,
                "metrics": metrics,
                "wall_time_sec": round(wall_time_sec, 3),
            }
        )

    # Summary (evaluate_rag.py style)
    n_ok = len(acc)
    print("\n" + "=" * 70)
    print("SUMMARY  (recall@5, recall@10, MRR, NDCG@10, answer_correctness)")
    print("=" * 70)

    if n_ok:
        r5 = sum(x["r5"] for x in acc) / n_ok
        r10 = sum(x["r10"] for x in acc) / n_ok
        mv = sum(x["mrr"] for x in acc) / n_ok
        ng = sum(x["ndcg"] for x in acc) / n_ok
        print("\n  === Mode: agent ===")
        print("  [agent_endpoint]")
        print(
            f"    cross_video: R@5={r5:.2f} R@10={r10:.2f} "
            f"MRR={mv:.3f} NDCG={ng:.3f} (n={n_ok})"
        )
    else:
        r5 = r10 = mv = ng = 0.0
        print("\n  No successful queries to summarize.")

    if failed:
        print(f"\n  Failed/skipped errors: {failed}")
    if cached:
        print(f"  Cached (skip-existing): {cached}")

    print("\n" + "=" * 70)
    print("TIMING  (wall-clock POST /api/agent/ask -> completed result)")
    print("=" * 70)
    if timings:
        ts = _timing_stats(timings)
        print(f"  n={len(timings)}")
        print(f"  min    = {ts['min']:.1f}s")
        print(f"  max    = {ts['max']:.1f}s")
        print(f"  mean   = {ts['mean']:.1f}s")
        print(f"  median = {ts['median']:.1f}s")
        print(f"  total  = {sum(timings):.1f}s")
    else:
        print("  No timing data recorded.")

    timing_summary = _timing_stats(timings) if timings else {}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows = [
        [
            "mode",
            "config",
            "category",
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "ndcg_at_10",
            "answer_correctness",
            "n_queries",
        ],
        [
            "agent",
            "agent_endpoint",
            "cross_video",
            f"{r5:.4f}",
            f"{r10:.4f}",
            f"{mv:.4f}",
            f"{ng:.4f}",
            "0.0000",
            n_ok,
        ],
    ]

    csv_path = out_dir / "eval_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary_rows)

    json_path = out_dir / "eval_results.json"
    json_path.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "base_url": args.base_url.rstrip("/"),
                "queries_dir": str(queries_dir),
                "n_queries": len(paths),
                "n_success": n_ok,
                "n_failed": failed,
                "n_cached": cached,
                "timing_summary_sec": timing_summary,
                "results_per_query": all_results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved summary CSV   -> {csv_path}")
    print(f"Saved full results  -> {json_path}")
    print(f"Per-query artifacts -> {out_queries_dir}/")

    if failed and n_ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
