#!/usr/bin/env python3
"""
Generate evaluation query candidates for the RAG evaluator.

Reads all videos with Shorties from the database, extracts MICRO-DETAILS facts,
and uses Anthropic to generate specific_fact, thematic, cross_video, and
causal_chain query candidates. Saves to eval_queries_draft.json for human review.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from anthropic_client import get_client


def get_videos_with_shorties(db_path: str) -> List[Dict[str, Any]]:
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT t.video_id, t.shorty, v.title, v.channel
        FROM transcripts t
        JOIN videos v ON v.video_id = t.video_id
        WHERE t.shorty IS NOT NULL AND TRIM(t.shorty) != ''
        ORDER BY t.video_id
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def extract_micro_details(shorty: str) -> str:
    """Extract MICRO-DETAILS section from Shorty text, or return a truncated shorty."""
    if not shorty:
        return ""
    match = re.search(r"MICRO-DETAILS[^\n]*\n(.*?)(?=\n[A-Z][A-Z-]+[^\n]*\n|\n*$)", shorty, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()[:2000]
    return shorty[:2000]


def generate_specific_fact_candidates(video_id: str, title: str, shorty: str, micro: str) -> List[Dict[str, Any]]:
    """Use Claude to generate specific_fact query candidates from MICRO-DETAILS."""
    client = get_client()
    prompt = f"""From this video Shorty excerpt (focus on MICRO-DETAILS), generate 1-3 evaluation queries that ask for a precise fact (version, date, number, name, tool). Each query should have a clear ground truth answer in the text.

TITLE: {title}
VIDEO_ID: {video_id}

SHORTY (excerpt):
{micro[:3000]}

Output a JSON array of objects, each with: "query", "ground_truth", "notes". Example:
[{{"query": "What version was vulnerable?", "ground_truth": "2.3.0", "notes": "version in MICRO-DETAILS"}}]
Output ONLY the JSON array, no other text."""

    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text += block.text
    text = text.strip()
    # Parse JSON array from response
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        arr = json.loads(text[start:end])
        out = []
        for i, obj in enumerate(arr):
            if isinstance(obj, dict) and obj.get("query"):
                out.append({
                    "id": "sf_%s_%d" % (video_id[:8], i),
                    "category": "specific_fact",
                    "query": obj.get("query", ""),
                    "ground_truth": obj.get("ground_truth", ""),
                    "relevant_video_ids": [video_id],
                    "notes": obj.get("notes", ""),
                })
        return out
    return []


def generate_thematic_cross_causal(titles_and_ids: List[tuple], sample_shorties: str) -> List[Dict[str, Any]]:
    """Generate thematic, cross_video, and causal_chain query candidates."""
    client = get_client()
    video_list = "\n".join("%s (%s)" % (t, vid) for t, vid in titles_and_ids[:50])
    prompt = f"""We have videos with these titles and video_ids:
{video_list}

Sample Shorty excerpts (for context):
{sample_shorties[:4000]}

Generate evaluation queries in three categories. Output a JSON array of objects, each with: "category" (one of thematic, cross_video, causal_chain), "query", "relevant_video_ids" (list of video_ids that should be retrieved), "notes".

- thematic: broad topic queries (e.g. "What videos discuss supply chain attacks?")
- cross_video: require synthesis across multiple videos (e.g. "Which videos mention both X and Y?")
- causal_chain: sequences of events (e.g. "What led to the breach in video X?")

Generate 2-3 of each category. Output ONLY the JSON array."""

    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text += block.text
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        arr = json.loads(text[start:end])
        out = []
        for i, obj in enumerate(arr):
            if isinstance(obj, dict) and obj.get("query") and obj.get("category") in ("thematic", "cross_video", "causal_chain"):
                c = obj["category"]
                out.append({
                    "id": "%s_%d" % (c[:2], i),
                    "category": c,
                    "query": obj.get("query", ""),
                    "ground_truth": "",
                    "relevant_video_ids": obj.get("relevant_video_ids", []),
                    "notes": obj.get("notes", ""),
                })
        return out
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval query candidates for human review.")
    parser.add_argument("--db-path", type=str, default=str(Path(__file__).parent / "data" / "transcripts.db"))
    parser.add_argument("--output", type=str, default="eval_queries_draft.json")
    parser.add_argument("--max-videos", type=int, default=20, help="Max videos to generate specific_fact from (to limit API cost)")
    args = parser.parse_args()

    videos = get_videos_with_shorties(args.db_path)
    if not videos:
        print("No videos with Shorties in DB.")
        return

    print("Found %d videos with Shorties. Generating query candidates..." % len(videos))
    all_queries: List[Dict[str, Any]] = []
    seen_ids = set()

    for v in videos[: args.max_videos]:
        video_id = v["video_id"]
        title = v["title"] or video_id
        shorty = v["shorty"] or ""
        micro = extract_micro_details(shorty)
        if not micro:
            continue
        try:
            candidates = generate_specific_fact_candidates(video_id, title, shorty, micro)
            for c in candidates:
                if c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    all_queries.append(c)
        except Exception as e:
            print("  [%s] specific_fact error: %s" % (video_id, e))

    titles_and_ids = [(v["title"] or v["video_id"], v["video_id"]) for v in videos]
    sample_shorties = "\n\n---\n\n".join((v["shorty"] or "")[:500] for v in videos[:15])
    try:
        extra = generate_thematic_cross_causal(titles_and_ids, sample_shorties)
        for idx, q in enumerate(extra):
            q["id"] = "gen_%s_%d" % (q["category"][:2], len(all_queries) + idx)
            all_queries.append(q)
    except Exception as e:
        print("  thematic/cross/causal error: %s" % e)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_queries, f, indent=2)
    print("Saved %d query candidates to %s (review and copy to eval_queries.json)" % (len(all_queries), args.output))


if __name__ == "__main__":
    main()
