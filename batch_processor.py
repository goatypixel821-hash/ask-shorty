#!/usr/bin/env python3
"""
Batch processor for Ask Shorty.

Scans all videos that:
- have at least one transcript row
- do NOT yet have a Shorty

Then, in batches of 10:
- generates Shorty
- generates synthetic questions
- extracts entities
- indexes everything into Chroma

Features:
- Resume-safe (skips videos that already have Shorties)
- --limit N to cap how many videos to process
- --retry-failed to reprocess only failed video_ids listed in failed_videos.txt
- --db-path to point at an external transcripts.db
- 1 second pause between batches to rate-limit Anthropic calls
- Cost estimation using Haiku pricing before full run and per-batch

Usage:
  python batch_processor.py
  python batch_processor.py --limit 100
  python batch_processor.py --retry-failed
"""

import argparse
import time
from typing import List, Dict, Any, Optional, Tuple, Callable

from transcript_database import TranscriptDatabase
from shorty_generator import (
    generate_shorty,
    generate_synthetic_questions,
    SHORTY_SYSTEM_PROMPT,
    SHORTY_USER_PROMPT_TEMPLATE,
    SYNTHETIC_Q_SYSTEM_PROMPT,
    SYNTHETIC_Q_USER_PROMPT_TEMPLATE,
)
from entity_extractor import (
    extract_entities,
    store_entities,
    ENTITY_SYSTEM_PROMPT,
    ENTITY_USER_TEMPLATE,
)
from transcript_rag import TranscriptRAG


BATCH_SIZE = 10

# Haiku pricing (USD per 1M tokens)
INPUT_PRICE_PER_M = 1.00
OUTPUT_PRICE_PER_M = 5.00


def estimate_video_tokens(db: TranscriptDatabase, video_id: str) -> Tuple[int, int]:
    """
    Roughly estimate input and output tokens for a single video.
    - input_tokens ≈ len(transcript) / 4
    - Shorty output ≈ 15% of input
    - synthetic questions output ≈ +500
    - entity extraction output ≈ +300
    """
    transcript = db.get_transcript(video_id)
    if not transcript:
        return 0, 0

    input_tokens = int(len(transcript) / 4)
    shorty_out = int(input_tokens * 0.15)
    synthetic_out = 500
    entity_out = 300
    output_tokens = shorty_out + synthetic_out + entity_out
    return input_tokens, output_tokens


def estimate_batch_cost(
    db: TranscriptDatabase,
    videos: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Estimate total tokens and cost for a list of videos.
    Uses Haiku pricing: $1/M input, $5/M output.
    """
    total_in = 0
    total_out = 0
    for v in videos:
        vid = v["video_id"]
        inp, out = estimate_video_tokens(db, vid)
        total_in += inp
        total_out += out

    cost_input = (total_in / 1_000_000.0) * INPUT_PRICE_PER_M
    cost_output = (total_out / 1_000_000.0) * OUTPUT_PRICE_PER_M
    total_cost = cost_input + cost_output

    return {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "cost_input": cost_input,
        "cost_output": cost_output,
        "total_cost": total_cost,
    }


def format_token_count(n: int) -> str:
    """Format token count with ~ and commas."""
    return f"~{n:,}"


def format_cost(c: float) -> str:
    return f"~${c:0.2f}"


def get_videos_needing_shorties(db: TranscriptDatabase, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Return list of video records that have transcripts but no Shorty yet."""
    import sqlite3

    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
    cursor = conn.cursor()

    sql = """
        SELECT v.video_id, v.title, v.channel
        FROM videos v
        JOIN transcripts t ON t.video_id = v.video_id
        WHERE t.shorty IS NULL
        GROUP BY v.video_id
        ORDER BY v.created_at ASC
    """
    if limit is not None:
        sql += " LIMIT ?"
        cursor.execute(sql, (limit,))
    else:
        cursor.execute(sql)

    rows = cursor.fetchall()
    conn.close()

    videos: List[Dict[str, Any]] = []
    for vid, title, channel in rows:
        videos.append(
            {
                "video_id": vid,
                "title": title,
                "channel": channel,
            }
        )
    return videos


def get_videos_from_failed(db: TranscriptDatabase, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Read failed_videos.txt and return video records for those IDs."""
    path = "failed_videos.txt"
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("No failed_videos.txt found; nothing to retry.")
        return []

    failed_ids: List[str] = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: video_id\tTitle\tError
        parts = line.split("\t")
        if parts:
            failed_ids.append(parts[0])

    if not failed_ids:
        print("failed_videos.txt is empty or has no valid entries.")
        return []

    if limit is not None:
        failed_ids = failed_ids[:limit]

    import sqlite3

    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
    cursor = conn.cursor()
    placeholders = ",".join("?" for _ in failed_ids)
    cursor.execute(
        f"""
        SELECT video_id, title, channel
        FROM videos
        WHERE video_id IN ({placeholders})
        """,
        failed_ids,
    )
    rows = cursor.fetchall()
    conn.close()

    videos: List[Dict[str, Any]] = []
    for vid, title, channel in rows:
        videos.append(
            {
                "video_id": vid,
                "title": title,
                "channel": channel,
            }
        )
    return videos


def write_failed_videos(failures: List[Dict[str, Any]]) -> None:
    """Write failed video IDs and errors to failed_videos.txt."""
    path = "failed_videos.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# video_id\ttitle\terror\n")
        for item in failures:
            vid = item.get("video_id", "")
            title = item.get("title", "").replace("\t", " ")
            err = item.get("error", "").replace("\n", " ").replace("\t", " ")
            f.write(f"{vid}\t{title}\t{err}\n")
    print(f"Failed videos saved to: {path}")


def get_pending_queue_tasks(
    db: TranscriptDatabase,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch pending processing_queue tasks in FIFO order.

    Returns list of dicts: {id, video_id, task}.
    """
    import sqlite3

    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
    cursor = conn.cursor()

    sql = """
        SELECT id, video_id, task
        FROM processing_queue
        WHERE status = 'pending'
        ORDER BY created_at ASC, id ASC
    """
    params: List[Any] = []
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))  # ensure integer for SQLite

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    tasks: List[Dict[str, Any]] = []
    for row in rows:
        tasks.append({"id": row[0], "video_id": row[1], "task": row[2]})
    return tasks


def update_queue_task_status(
    db: TranscriptDatabase,
    task_id: int,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Update a single queue task's status and timestamps."""
    import sqlite3
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
    cursor = conn.cursor()

    if status == "started":
        cursor.execute(
            """
            UPDATE processing_queue
            SET status = ?, started_at = ?
            WHERE id = ?
            """,
            (status, now, task_id),
        )
    elif status in ("completed", "failed"):
        cursor.execute(
            """
            UPDATE processing_queue
            SET status = ?, completed_at = ?, error = ?
            WHERE id = ?
            """,
            (status, now, error, task_id),
        )
    else:
        cursor.execute(
            "UPDATE processing_queue SET status = ? WHERE id = ?",
            (status, task_id),
        )

    conn.commit()
    conn.close()


def process_batch(
    db: TranscriptDatabase,
    rag: TranscriptRAG,
    batch: List[Dict[str, Any]],
    start_index: int,
    total: int,
    totals: Dict[str, Any],
    shorty_fn: Callable[..., str],
    synth_q_fn: Callable[..., List[str]],
    entity_fn: Callable[..., List[Dict[str, Any]]],
) -> None:
    """Process a single batch of videos."""
    import sqlite3

    batch_input_est = 0
    batch_output_est = 0
    batch_success = 0
    batch_failures: List[Dict[str, Any]] = []

    for offset, video in enumerate(batch):
        idx = start_index + offset + 1
        video_id = video["video_id"]
        title = video.get("title") or "Untitled Video"
        channel = video.get("channel") or "Unknown Channel"

        print(f"\n=== Processing video {idx} of {total} ===")
        print(f"ID: {video_id}")
        print(f"Title: {title}")
        print(f"Channel: {channel}")

        # Double-check if Shorty already exists (resume safety)
        info = db.get_transcript_and_shorty(video_id)
        if info and info.get("shorty"):
            print("  → Shorty already present, skipping.")
            continue

        transcript_text = (info or {}).get("text") or db.get_transcript(video_id)
        if not transcript_text:
            print("  → No transcript found, skipping.")
            continue

        # Accumulate estimated cost for this video into batch & totals
        vin, vout = estimate_video_tokens(db, video_id)
        batch_input_est += vin
        batch_output_est += vout

        # Pull metadata for Shorty header
        video_info = db.get_video_info(video_id) or {}
        meta = (video_info.get("metadata") or {}) if isinstance(video_info, dict) else {}
        title_meta = video_info.get("title") if isinstance(video_info, dict) else None
        channel_meta = video_info.get("channel") if isinstance(video_info, dict) else None
        upload_date = meta.get("upload_date") if isinstance(meta, dict) else None

        final_title = title_meta or title
        final_channel = channel_meta or channel

        try:
            print("  → Generating Shorty...")
            shorty_text = shorty_fn(
                transcript_text,
                title=final_title,
                channel=final_channel,
                upload_date=upload_date,
            )
            saved = db.save_shorty(video_id, shorty_text)
            if not saved:
                msg = "Failed to save Shorty"
                print(f"  ! {msg}")
                batch_failures.append({"video_id": video_id, "title": final_title, "error": msg})
                totals["videos_failed"].append({"video_id": video_id, "title": final_title, "error": msg})
                continue
            print("  ✓ Shorty saved.")

            print("  → Generating synthetic questions...")
            questions = synth_q_fn(transcript_text, title=final_title)
            if questions:
                conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
                cursor = conn.cursor()
                for q in questions:
                    cursor.execute(
                        """
                        INSERT INTO synthetic_questions (video_id, question, embedding_id)
                        VALUES (?, ?, NULL)
                        """,
                        (video_id, q),
                    )
                conn.commit()
                conn.close()
                print(f"  ✓ Stored {len(questions)} synthetic questions.")
            else:
                print("  ! No synthetic questions generated.")
                questions = []

            print("  → Extracting entities...")
            entities = entity_fn(transcript_text, title=final_title)
            if entities:
                count = store_entities(video_id, entities)
                print(f"  ✓ Stored {count} entities.")
            else:
                print("  ! No entities extracted.")

            print("  → Indexing into Chroma...")
            rag.index_single_transcript(
                video_id,
                transcript_text,
                shorty=shorty_text,
                synthetic_questions=questions if questions else None,
            )
            print("  ✓ Indexing complete.")

            batch_success += 1
            totals["videos_processed"] += 1

        except Exception as e:
            msg = str(e)
            print(f"  !! Error processing video {video_id}: {msg}")
            batch_failures.append({"video_id": video_id, "title": final_title, "error": msg})
            totals["videos_failed"].append({"video_id": video_id, "title": final_title, "error": msg})

    # Update global token and cost totals using the estimated batch values
    totals["total_input_tokens"] += batch_input_est
    totals["total_output_tokens"] += batch_output_est
    batch_cost_in = (batch_input_est / 1_000_000.0) * INPUT_PRICE_PER_M
    batch_cost_out = (batch_output_est / 1_000_000.0) * OUTPUT_PRICE_PER_M
    batch_cost = batch_cost_in + batch_cost_out
    totals["total_cost"] += batch_cost

    # Batch summary
    print("\nBatch complete.")
    print(f"  ✓ {batch_success} Shorties generated")
    if batch_failures:
        print(f"  ! {len(batch_failures)} failed:")
        for f in batch_failures:
            print(f"    - {f['video_id']} - {f['title']} - {f['error']}")
    else:
        print("  ! 0 failures in this batch")

    print(
        f"Actual tokens used this batch (approx): "
        f"{format_token_count(batch_input_est)} input / {format_token_count(batch_output_est)} output"
    )
    print(f"Running total cost: {format_cost(totals['total_cost'])}")


def process_queue_tasks(
    db: TranscriptDatabase,
    rag: TranscriptRAG,
    shorty_fn: Callable[..., str],
    synth_q_fn: Callable[..., List[str]],
    entity_fn: Callable[..., List[Dict[str, Any]]],
    limit: Optional[int],
) -> None:
    """
    Process pending tasks from processing_queue in FIFO order.

    Each queue row represents exactly one task: shorty, synthetic_questions, or entities.
    Transcript chunks are assumed to be already vectorized on grab.
    """
    import sqlite3

    processed_count = 0
    batch_number = 0

    while True:
        batch_number += 1
        if limit is not None:
            remaining = max(limit - processed_count, 0)
            if remaining == 0:
                print("\n[DEBUG] Loop stopping: reached --limit (processed_count=%d, limit=%d)." % (processed_count, limit))
                break
            batch_limit = min(BATCH_SIZE, remaining)
        else:
            batch_limit = BATCH_SIZE  # fetch in batches when no limit

        tasks = get_pending_queue_tasks(db, batch_limit)
        print("\n[DEBUG] Batch #%d: requested batch_limit=%d, fetched %d tasks (processed_count so far=%d, limit=%s)." % (
            batch_number, batch_limit, len(tasks), processed_count, limit if limit is not None else "None"))

        if not tasks:
            if processed_count == 0:
                print("[DEBUG] Loop stopping: no pending tasks in queue.")
            else:
                print("[DEBUG] Loop stopping: no more pending tasks (processed %d this run)." % processed_count)
            break

        print(f"\n=== Processing {len(tasks)} queued tasks ===")
        for task in tasks:
            task_id = task["id"]
            video_id = task["video_id"]
            kind = task["task"]

            print(f"\nTask #{task_id} · video {video_id} · type={kind}")
            update_queue_task_status(db, task_id, "started")
            print("[DEBUG] Task #%d status -> started" % task_id)

            try:
                info = db.get_transcript_and_shorty(video_id)
                transcript_text = (info or {}).get("text") or db.get_transcript(video_id)
                if not transcript_text:
                    msg = "No transcript found; skipping task."
                    print(f"  ! {msg}")
                    update_queue_task_status(db, task_id, "failed", msg)
                    print("[DEBUG] Task #%d status -> failed" % task_id)
                    continue

                # Metadata for Shorty header or entity context
                video_info = db.get_video_info(video_id) or {}
                meta = (video_info.get("metadata") or {}) if isinstance(video_info, dict) else {}
                title_meta = video_info.get("title") if isinstance(video_info, dict) else None
                channel_meta = video_info.get("channel") if isinstance(video_info, dict) else None
                upload_date = meta.get("upload_date") if isinstance(meta, dict) else None

                final_title = title_meta or (video_info.get("title") if isinstance(video_info, dict) else "Untitled Video")
                final_channel = channel_meta or (video_info.get("channel") if isinstance(video_info, dict) else "Unknown Channel")

                if kind == "shorty":
                    print("  → Generating Shorty...")
                    shorty_text = shorty_fn(
                        transcript_text,
                        title=final_title,
                        channel=final_channel,
                        upload_date=upload_date,
                    )
                    saved = db.save_shorty(video_id, shorty_text)
                    if not saved:
                        msg = "Failed to save Shorty."
                        print(f"  ! {msg}")
                        update_queue_task_status(db, task_id, "failed", msg)
                        print("[DEBUG] Task #%d status -> failed" % task_id)
                        continue
                    print("  ✓ Shorty saved.")

                    # Re-index transcript + Shorty (+ existing questions if any)
                    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT question FROM synthetic_questions WHERE video_id = ? ORDER BY created_at ASC",
                        (video_id,),
                    )
                    q_rows = cursor.fetchall()
                    conn.close()
                    existing_questions = [row[0] for row in q_rows if row and row[0]]

                    print("  → Re-indexing transcript and Shorty in Chroma...")
                    rag.index_single_transcript(
                        video_id,
                        transcript_text,
                        shorty=shorty_text,
                        synthetic_questions=existing_questions or None,
                    )
                    print("  ✓ Indexing complete.")

                elif kind == "synthetic_questions":
                    print("  → Generating synthetic questions...")
                    questions = synth_q_fn(transcript_text, title=final_title)
                    if not questions:
                        msg = "No synthetic questions generated."
                        print(f"  ! {msg}")
                        update_queue_task_status(db, task_id, "failed", msg)
                        print("[DEBUG] Task #%d status -> failed" % task_id)
                        continue

                    conn = sqlite3.connect(db.db_path)  # type: ignore[attr-defined]
                    cursor = conn.cursor()
                    for q in questions:
                        cursor.execute(
                            """
                            INSERT INTO synthetic_questions (video_id, question, embedding_id)
                            VALUES (?, ?, NULL)
                            """,
                            (video_id, q),
                        )
                    conn.commit()
                    conn.close()
                    print(f"  ✓ Stored {len(questions)} synthetic questions.")

                    # Fetch current Shorty (if any) for richer indexing
                    info_latest = db.get_transcript_and_shorty(video_id)
                    shorty_text = (info_latest or {}).get("shorty")

                    print("  → Re-indexing transcript + Shorty + questions in Chroma...")
                    rag.index_single_transcript(
                        video_id,
                        transcript_text,
                        shorty=shorty_text,
                        synthetic_questions=questions,
                    )
                    print("  ✓ Indexing complete.")

                elif kind == "entities":
                    print("  → Extracting entities...")
                    entities = entity_fn(transcript_text, title=final_title)
                    if entities:
                        count = store_entities(video_id, entities)
                        print(f"  ✓ Stored {count} entities.")
                    else:
                        print("  ! No entities extracted.")

                else:
                    msg = f"Unknown task type: {kind}"
                    print(f"  ! {msg}")
                    update_queue_task_status(db, task_id, "failed", msg)
                    print("[DEBUG] Task #%d status -> failed" % task_id)
                    continue

                update_queue_task_status(db, task_id, "completed", None)
                print("[DEBUG] Task #%d status -> completed (processed_count now=%d)" % (task_id, processed_count + 1))
                processed_count += 1
                if limit is not None and processed_count >= limit:
                    print("[DEBUG] Breaking for loop: reached limit (%d)." % limit)
                    break

            except Exception as e:
                msg = str(e)
                print(f"  !! Error processing task {task_id} for video {video_id}: {type(e).__name__}: {msg}")
                update_queue_task_status(db, task_id, "failed", msg)
                print("[DEBUG] Task #%d status -> failed (exception); continuing to next task." % task_id)
            except BaseException:
                # KeyboardInterrupt, SystemExit, etc. - do not swallow
                raise


def main():
    parser = argparse.ArgumentParser(description="Batch-process videos for Ask Shorty Shorties.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of videos to process (for testing).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry only videos listed in failed_videos.txt.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["anthropic", "openai-compatible"],
        default="anthropic",
        help="Which LLM provider to use for generation. Default: anthropic.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for openai-compatible endpoint (e.g. http://host:8000/v1). "
             "Only used when --provider openai-compatible.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for openai-compatible endpoint (e.g. qwen2.5-14b). "
             "Only used when --provider openai-compatible.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to transcripts.db (e.g. C:\\Users\\number2\\Desktop\\youtube-history-viewer-copy\\data\\transcripts.db). "
             "If not provided, uses the default DB for this project.",
    )
    parser.add_argument(
        "--queue",
        action="store_true",
        default=True,
        help="Process tasks from processing_queue (default behavior).",
    )
    args = parser.parse_args()

    # Allow pointing at an external transcripts.db (e.g. youtube-history-viewer-copy)
    if args.db_path:
        db = TranscriptDatabase(args.db_path)
    else:
        db = TranscriptDatabase()

    rag = TranscriptRAG()

    # Select provider-specific generation functions
    provider = args.provider
    if provider == "anthropic":
        shorty_fn = generate_shorty
        synth_q_fn = generate_synthetic_questions
        entity_fn = extract_entities
    else:
        # OpenAI-compatible client setup
        import os
        from openai import OpenAI

        base_url = args.base_url or "http://localhost:8000/v1"
        model = args.model or "gpt-3.5-turbo"
        api_key = os.environ.get("OPENAI_API_KEY") or "no-key"
        oa_client = OpenAI(base_url=base_url, api_key=api_key)

        def _openai_chat(system_prompt: str, user_prompt: str, max_tokens: int = 4096, temperature: float = 0.2) -> str:
            resp = oa_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return (resp.choices[0].message.content or "").strip()

        def shorty_fn(
            transcript_text: str,
            title: Optional[str] = None,
            channel: Optional[str] = None,
            upload_date: Optional[str] = None,
        ) -> str:
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
            body = _openai_chat(SHORTY_SYSTEM_PROMPT, user_prompt, max_tokens=4096, temperature=0.2)
            header = (
                f"SOURCE: {safe_title}\n"
                f"CHANNEL: {safe_channel}\n"
                f"DATE: {safe_date}\n"
                f"CREATOR: {safe_channel}\n\n"
            )
            return header + body.lstrip()

        def synth_q_fn(
            transcript_text: str,
            title: Optional[str] = None,
            n: int = 10,
        ) -> List[str]:
            if not transcript_text or not transcript_text.strip():
                raise ValueError("Transcript text is empty; cannot generate questions.")
            safe_title = title or "Untitled Video"
            user_prompt = SYNTHETIC_Q_USER_PROMPT_TEMPLATE.format(
                title=safe_title,
                transcript=transcript_text.strip(),
            )
            raw = _openai_chat(SYNTHETIC_Q_SYSTEM_PROMPT, user_prompt, max_tokens=2048, temperature=0.2)

            import json

            questions: List[str] = []
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            q = item.strip()
                            if q:
                                questions.append(q)
            except Exception:
                for line in raw.splitlines():
                    line = line.strip().lstrip("-").strip()
                    if not line:
                        continue
                    if not line.endswith("?"):
                        continue
                    questions.append(line)
            if len(questions) > n:
                questions = questions[:n]
            return questions

        def entity_fn(
            transcript_text: str,
            title: Optional[str] = None,
        ) -> List[Dict[str, Any]]:
            if not transcript_text or not transcript_text.strip():
                return []
            safe_title = title or "Untitled Video"
            user_prompt = ENTITY_USER_TEMPLATE.format(title=safe_title, transcript=transcript_text.strip())
            raw = _openai_chat(ENTITY_SYSTEM_PROMPT, user_prompt, max_tokens=2048, temperature=0.1)

            # Reuse the same cleanup logic as entity_extractor: strip fences and preamble
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                raw = raw.rsplit("```", 1)[0].strip()
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                raw = raw[start : end + 1]

            import json

            try:
                data = json.loads(raw)
            except Exception:
                return []

            entities: List[Dict[str, Any]] = []
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name", "")).strip()
                    etype = str(item.get("type", "")).strip()
                    aliases = item.get("aliases") or []
                    if not name:
                        continue
                    if not isinstance(aliases, list):
                        aliases = []
                    aliases = [str(a).strip() for a in aliases if isinstance(a, str) and a.strip()]
                    entities.append(
                        {
                            "name": name,
                            "type": etype,
                            "aliases": aliases,
                        }
                    )
            return entities

    # New default: process from the processing_queue if requested (queue mode).
    if args.queue:
        process_queue_tasks(
            db=db,
            rag=rag,
            shorty_fn=shorty_fn,
            synth_q_fn=synth_q_fn,
            entity_fn=entity_fn,
            limit=args.limit,
        )
        return

    if args.retry_failed:
        videos = get_videos_from_failed(db, args.limit)
        mode_desc = "failed-only"
    else:
        videos = get_videos_needing_shorties(db, args.limit)
        mode_desc = "needing-Shorties"

    total = len(videos)
    if total == 0:
        print(f"No videos to process in {mode_desc} mode. Nothing to do.")
        return

    print(f"Found {total} videos needing Shorties.")

    # Full-run cost estimate
    est = estimate_batch_cost(db, videos)
    batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print("\nCOST ESTIMATE (full run):")
    print(f"  Estimated input tokens:  {format_token_count(est['input_tokens'])}")
    print(f"  Estimated output tokens: {format_token_count(est['output_tokens'])}")
    print(f"  Estimated cost:          {format_cost(est['total_cost'])} (Haiku pricing)")
    print(f"\nProcess in batches of {BATCH_SIZE}.")
    print(f"Total batches: {batches}")

    choice = input("\nProceed with full run? (yes/no/limit): ").strip().lower()
    if choice == "no":
        print("Exiting without processing.")
        return
    elif choice.startswith("limit"):
        # allow "limit 100" or "limit:100"
        parts = choice.replace(":", " ").split()
        if len(parts) >= 2 and parts[1].isdigit():
            limit_val = int(parts[1])
        else:
            raw = input("Enter numeric limit: ").strip()
            if not raw.isdigit():
                print("Invalid limit. Exiting.")
                return
            limit_val = int(raw)
        videos = videos[:limit_val]
        total = len(videos)
        batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nLimiting run to {total} videos ({batches} batches).")
    elif choice.isdigit():
        limit_val = int(choice)
        videos = videos[:limit_val]
        total = len(videos)
        batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nLimiting run to {total} videos ({batches} batches).")
    elif choice != "yes":
        print("Unrecognized option, exiting.")
        return

    # Running totals across session
    totals: Dict[str, Any] = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "videos_processed": 0,
        "videos_failed": [],
    }

    start = 0
    batch_index = 0

    while start < total:
        batch_index += 1
        batch = videos[start : start + BATCH_SIZE]
        batch_start_num = start + 1
        batch_end_num = min(start + len(batch), total)

        # Per-batch estimate (for display)
        est_batch = estimate_batch_cost(db, batch)

        print(
            f"\n=== Batch {batch_index} of {batches} "
            f"(videos {batch_start_num}-{batch_end_num}) ==="
        )
        print("Titles:")
        for v in batch:
            t = v.get("title") or "Untitled Video"
            c = v.get("channel") or "Unknown Channel"
            print(f"  - {t} ({c})")

        print(
            f"Estimated batch cost: {format_cost(est_batch['total_cost'])}\n"
            f"Tokens so far this run: "
            f"{format_token_count(totals['total_input_tokens'])} input / "
            f"{format_token_count(totals['total_output_tokens'])} output\n"
            f"Cost so far this run: {format_cost(totals['total_cost'])}"
        )

        ans = input("\nProceed with this batch? (yes/skip/stop): ").strip().lower()
        if ans == "skip":
            print("Skipping this batch.")
            start += BATCH_SIZE
            continue
        elif ans == "stop":
            print("Stopping before this batch.")
            break
        elif ans != "yes":
            print("Unrecognized option, treating as 'skip'.")
            start += BATCH_SIZE
            continue

        # Process the batch
        process_batch(db, rag, batch, start, total, totals, shorty_fn, synth_q_fn, entity_fn)
        start += BATCH_SIZE

        if start < total:
            print("\nSleeping 1s before next batch to rate-limit...")
            time.sleep(1.0)

    # Session summary
    print("\n=== Session Complete ===")
    print(f"Videos processed: {totals['videos_processed']}")
    print(f"Videos failed: {len(totals['videos_failed'])}")
    print(
        f"Total input tokens:  {format_token_count(totals['total_input_tokens'])}\n"
        f"Total output tokens: {format_token_count(totals['total_output_tokens'])}\n"
        f"Total cost:          {format_cost(totals['total_cost'])}"
    )

    if totals["videos_failed"]:
        write_failed_videos(totals["videos_failed"])


if __name__ == "__main__":
    main()

