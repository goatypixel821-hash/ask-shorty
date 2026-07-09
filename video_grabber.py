#!/usr/bin/env python3
"""
Video Grabber Service - Bookmarklet handler for manual transcript entry.

The bookmarklet passes video URL, title, and channel as URL parameters from the browser.
This service shows a page where you paste the transcript (from YouTube's "Show transcript"),
then Save & Vectorize stores the transcript, queues Shorty/questions/entities, and indexes for search.
"""

import sys
import os
import re
import subprocess
import threading
import logging
from pathlib import Path
from typing import Optional

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUNBUFFERED'] = '1'
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
            sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        except Exception:
            pass

from flask import Flask, jsonify, request, render_template, redirect
import sqlite3

from transcript_database import TranscriptDatabase
from transcript_rag import TranscriptRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

base_dir = Path(__file__).resolve().parent
grab_log_path = base_dir / 'data' / 'grab_log.txt'

_DEFAULT_OPENROUTER_MODEL = "qwen/qwen-2.5-72b-instruct"


def _out(msg: str) -> None:
    line = msg + '\n'
    try:
        sys.stderr.write(line)
        sys.stderr.flush()
    except Exception:
        pass
    try:
        grab_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(grab_log_path, 'a', encoding='utf-8') as f:
            f.write(line)
            f.flush()
    except Exception:
        pass


def _resolve_db_path() -> tuple[Path, str]:
    """
    Pick the SQLite file the grabber uses.

    Priority (after .env is loaded):
    1. ASK_SHORTY_DB_PATH env var (from shell or .env)
    2. <directory containing this video_grabber.py>/data/transcripts.db

    Note: SHORTY_PROJECT_ROOT does not select the DB directly; it only helps load
    shorty's .env and resolve batch_processor.py when the grabber script is run
    from another checkout (e.g. youtube-history-viewer-copy).
    """
    env_raw = (os.environ.get("ASK_SHORTY_DB_PATH") or "").strip()
    if env_raw:
        return Path(env_raw).resolve(), "ASK_SHORTY_DB_PATH"
    default = (base_dir / "data" / "transcripts.db").resolve()
    return default, "base_dir/data/transcripts.db (next to this video_grabber.py)"


def _log_db_path_resolution(db_path: Path, db_path_source: str) -> None:
    sr = (os.environ.get("SHORTY_PROJECT_ROOT") or "").strip() or "(not set)"
    line = (
        "[grabber] db_path_resolved=%s | source=%s | video_grabber.py=%s | "
        "SHORTY_PROJECT_ROOT=%s | cwd=%s"
        % (
            db_path,
            db_path_source,
            base_dir,
            sr,
            Path.cwd().resolve(),
        )
    )
    logger.info(line)
    try:
        print(line, flush=True)
    except Exception:
        pass
    _out(line)


def _load_project_dotenv() -> None:
    """
    Load .env files so OPENROUTER_* matches how you launch the grabber.

    Uses override=False: variables already set in the process environment (e.g. the
    terminal) are never replaced by .env values. Each .env only defines defaults for
    keys that are still unset; among files, earlier paths in the list win for a key.

    Order:
    - .env next to this script (shorty repo)
    - .env in the process current working directory (e.g. youtube-history-viewer-copy)
    - .env under SHORTY_PROJECT_ROOT if set (explicit shorty checkout)
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    roots = [base_dir, Path.cwd().resolve()]
    sr = (os.environ.get("SHORTY_PROJECT_ROOT") or "").strip()
    if sr:
        roots.append(Path(sr).resolve())
    seen = set()
    for root in roots:
        env_path = root / ".env"
        if not env_path.is_file():
            continue
        key = str(env_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        load_dotenv(env_path, override=False)


_load_project_dotenv()

db_path, _db_path_source = _resolve_db_path()
_log_db_path_resolution(db_path, _db_path_source)

try:
    from local_yt_dlp_fetcher import VideoDownloader, YT_DLP_AVAILABLE
    _LOCAL_YT_DLP_AVAILABLE = bool(YT_DLP_AVAILABLE)
except ImportError:
    VideoDownloader = None  # type: ignore[misc, assignment]
    _LOCAL_YT_DLP_AVAILABLE = False

_local_video_downloader = None
if _LOCAL_YT_DLP_AVAILABLE and VideoDownloader is not None:
    _local_video_downloader = VideoDownloader(str(base_dir / 'data' / 'downloads'))


def _openrouter_key_status_line() -> str:
    """Safe one-line status for logs (never print full API key)."""
    raw = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not raw:
        return "OPENROUTER_API_KEY=(not set)"
    return "OPENROUTER_API_KEY=%s… (len=%d)" % (raw[:8], len(raw))


def _batch_processor_script() -> Optional[Path]:
    """
    Resolve batch_processor.py. When this script lives outside shorty (e.g. a copy
    under youtube-history-viewer-copy), set SHORTY_PROJECT_ROOT to the shorty repo path.
    """
    here = base_dir / "batch_processor.py"
    if here.is_file():
        return here.resolve()
    sr = (os.environ.get("SHORTY_PROJECT_ROOT") or "").strip()
    if sr:
        p = (Path(sr).resolve() / "batch_processor.py")
        if p.is_file():
            return p
    cwd_bp = (Path.cwd().resolve() / "batch_processor.py")
    if cwd_bp.is_file():
        return cwd_bp
    return None


def _pipeline_log(msg: str) -> None:
    """Same as _out for files/stderr, plus stdout so the grabber terminal shows pipeline progress."""
    _out(msg)
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _log_startup_openrouter_and_batch() -> None:
    bp = _batch_processor_script()
    bp_s = str(bp) if bp else "(not found — put batch_processor.py next to video_grabber.py or set SHORTY_PROJECT_ROOT)"
    line = (
        "[grabber] %s | batch_processor=%s | grabber_script_dir=%s | cwd=%s"
        % (
            _openrouter_key_status_line(),
            bp_s,
            str(base_dir),
            str(Path.cwd().resolve()),
        )
    )
    logger.info(line)
    try:
        print(line, flush=True)
    except Exception:
        pass
    # Same path as other grab lines (stderr + data/grab_log.txt); print alone is easy to miss if stdout is redirected.
    _out(line)


db = TranscriptDatabase(str(db_path))
rag = TranscriptRAG()

logger.info("Video Grabber Service initialized")
logger.info(f"Database: {db_path}")
_log_startup_openrouter_and_batch()


def _extract_video_id(url: str):
    if not url:
        return None
    match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
    return match.group(1) if match else None


def vectorize_video_in_background(video_id: str):
    def do_vectorize():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT text FROM transcripts WHERE video_id = ? ORDER BY created_at DESC LIMIT 1",
                (video_id,)
            )
            row = cursor.fetchone()
            conn.close()
            if row and row[0]:
                _out(f"  [Vectorize] Indexing transcript for {video_id}...")
                rag.index_single_transcript(video_id, row[0])
                _out(f"  [Vectorize] Complete for {video_id}")
                logger.info(f"Vectorized transcript for {video_id}")
            else:
                _out(f"  [Vectorize] No transcript found for {video_id}")
        except Exception as e:
            logger.error(f"Vectorization error for {video_id}: {e}", exc_info=True)

    thread = threading.Thread(target=do_vectorize, daemon=True)
    thread.start()


def enqueue_llm_tasks_for_video(video_id: str):
    try:
        db.enqueue_processing_tasks(video_id)
        _out(f"  [Queue] Enqueued LLM tasks for {video_id} (shorty, synthetic_questions, entities)")
    except Exception as e:
        logger.error(f"Queue enqueue error for {video_id}: {e}", exc_info=True)


def _openrouter_model_name() -> str:
    v = (os.environ.get("OPENROUTER_MODEL") or "").strip()
    return v or _DEFAULT_OPENROUTER_MODEL


def _append_batch_processor_log(header: str, proc: subprocess.CompletedProcess) -> None:
    try:
        grab_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(grab_log_path, "a", encoding="utf-8") as gf:
            gf.write(header + "\n")
            if proc.stdout:
                gf.write(proc.stdout)
            if proc.stderr:
                gf.write(proc.stderr)
            gf.flush()
    except Exception:
        pass


def _run_batch_processor_subprocess(bp_args: list) -> subprocess.CompletedProcess:
    script = _batch_processor_script()
    if not script:
        return subprocess.CompletedProcess(
            args=[],
            returncode=127,
            stdout="",
            stderr="batch_processor.py not found",
        )
    cmd = [sys.executable, str(script), "--db-path", str(db_path.resolve())] + bp_args
    return subprocess.run(
        cmd,
        cwd=str(script.parent),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=None,
    )


def _openrouter_background_pipeline_worker(video_id: str) -> None:
    model = _openrouter_model_name()
    try:
        _pipeline_log(
            f"[OpenRouter pipeline] START video_id={video_id} model={model} "
            f"(main excludes triples,segments,events)"
        )
        main_args = [
            "--provider",
            "openrouter",
            "--model",
            model,
            "--queue",
            "--video-id",
            video_id,
            "--exclude-tasks",
            "triples,segments,events",
        ]
        r = _run_batch_processor_subprocess(main_args)
        _append_batch_processor_log(
            f"\n--- batch_processor main {' '.join(main_args)} exit={r.returncode} ---\n",
            r,
        )
        if r.returncode != 0:
            _pipeline_log(f"[OpenRouter pipeline] MAIN_FAIL video_id={video_id} exit={r.returncode}")
            return
        _pipeline_log(f"[OpenRouter pipeline] MAIN_DONE video_id={video_id}")

        _pipeline_log(f"[OpenRouter pipeline] TRIPLES_START video_id={video_id}")
        trip_args = [
            "--provider",
            "openrouter",
            "--model",
            model,
            "--queue",
            "--video-id",
            video_id,
            "--only-tasks",
            "triples",
        ]
        r2 = _run_batch_processor_subprocess(trip_args)
        _append_batch_processor_log(
            f"\n--- batch_processor triples {' '.join(trip_args)} exit={r2.returncode} ---\n",
            r2,
        )
        if r2.returncode != 0:
            _pipeline_log(f"[OpenRouter pipeline] TRIPLES_FAIL video_id={video_id} exit={r2.returncode}")
        else:
            _pipeline_log(f"[OpenRouter pipeline] TRIPLES_DONE video_id={video_id}")

            _pipeline_log(f"[OpenRouter pipeline] SEGMENTS_START video_id={video_id}")
            seg_args = [
                "--provider",
                "openrouter",
                "--model",
                model,
                "--queue",
                "--video-id",
                video_id,
                "--only-tasks",
                "segments",
            ]
            r3 = _run_batch_processor_subprocess(seg_args)
            _append_batch_processor_log(
                f"\n--- batch_processor segments {' '.join(seg_args)} exit={r3.returncode} ---\n",
                r3,
            )
            if r3.returncode != 0:
                _pipeline_log(
                    f"[OpenRouter pipeline] SEGMENTS_FAIL video_id={video_id} exit={r3.returncode}"
                )
            else:
                _pipeline_log(f"[OpenRouter pipeline] SEGMENTS_DONE video_id={video_id}")

            _pipeline_log(f"[OpenRouter pipeline] EVENTS_START video_id={video_id}")
            ev_args = [
                "--provider",
                "openrouter",
                "--model",
                model,
                "--queue",
                "--video-id",
                video_id,
                "--only-tasks",
                "events",
            ]
            r4 = _run_batch_processor_subprocess(ev_args)
            _append_batch_processor_log(
                f"\n--- batch_processor events {' '.join(ev_args)} exit={r4.returncode} ---\n",
                r4,
            )
            if r4.returncode != 0:
                _pipeline_log(
                    f"[OpenRouter pipeline] EVENTS_FAIL video_id={video_id} exit={r4.returncode}"
                )
            else:
                _pipeline_log(f"[OpenRouter pipeline] EVENTS_DONE video_id={video_id}")

        _pipeline_log(f"[OpenRouter pipeline] FINISH video_id={video_id}")
    except Exception as e:
        logger.error(f"OpenRouter pipeline error for {video_id}: {e}", exc_info=True)
        try:
            _pipeline_log(f"[OpenRouter pipeline] ERROR video_id={video_id}: {e}")
        except Exception:
            pass


def _maybe_spawn_openrouter_pipeline(video_id: str) -> None:
    if not (os.environ.get("OPENROUTER_API_KEY") or "").strip():
        return
    if not _batch_processor_script():
        _pipeline_log(
            "[OpenRouter pipeline] SKIP: batch_processor.py not found "
            "(set SHORTY_PROJECT_ROOT to your shorty repo, or run grabber from shorty)."
        )
        logger.error("OpenRouter pipeline skipped: batch_processor.py not found")
        return
    _pipeline_log(f"[OpenRouter pipeline] Scheduling background run for video_id={video_id}")
    t = threading.Thread(
        target=_openrouter_background_pipeline_worker,
        args=(video_id,),
        daemon=True,
        name=f"openrouter-pipeline-{video_id}",
    )
    t.start()


def _finalize_saved_transcript(
    video_id: str,
    transcript_chars: int,
) -> None:
    """Vectorize, enqueue LLM tasks, optionally run OpenRouter batch (non-blocking)."""
    _out(f"Transcript saved: {video_id} ({transcript_chars} chars)")
    _out("Vectorizing in background...")
    vectorize_video_in_background(video_id)
    enqueue_llm_tasks_for_video(video_id)
    _maybe_spawn_openrouter_pipeline(video_id)
    _out("Done.")


def _strip_timestamps_from_paste(text: str) -> str:
    if not text:
        return text
    ts_line = re.compile(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*[-–—]?\s*$', re.MULTILINE)
    cleaned = ts_line.sub('', text)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    return '\n\n'.join(lines) if lines else cleaned.strip()


def _fetch_local_metadata(url: str) -> Optional[dict]:
    """Fetch rich metadata via local yt-dlp module (personal use only)."""
    if not _local_video_downloader:
        return None
    try:
        _out("🔍 Fetching metadata...")
        metadata = _local_video_downloader.fetch_metadata(url, quiet=True)
        if metadata:
            desc_len = len(metadata.get('description', '') or '')
            tags_count = len(metadata.get('tags', []) or [])
            _out(f"✅ Metadata saved: {desc_len} chars description, {tags_count} tags")
        return metadata
    except Exception as e:
        _out(f"⚠️ Metadata fetch warning: {e}")
        return None


def _apply_local_metadata(
    video_id: str,
    url: str,
    title: str,
    channel: str,
) -> tuple[Optional[dict], str, str]:
    metadata = _fetch_local_metadata(url)
    if metadata:
        if metadata.get('title'):
            title = metadata['title']
        if metadata.get('channel'):
            channel = metadata['channel']
        try:
            db.save_metadata(video_id, metadata)
        except Exception as e:
            logger.warning(f"fetch-transcript save_metadata: {e}")
    return metadata, title, channel


def _start_bookmarklet_background(
    video_id: str,
    transcript_success: bool,
    metadata: Optional[dict],
) -> None:
    """Verbose background-work logging (matches pre-eea8f02 grabber)."""
    _out("\n🚀 Starting background processing...")
    if transcript_success:
        _out("  → Vectorizing transcript (background)")
        vectorize_video_in_background(video_id)
        description = (metadata or {}).get('description', '') if metadata else ''
        tags = (metadata or {}).get('tags', []) if metadata else []
        if description or tags:
            _out("  → Skipping categorization (not enabled in this grabber)")
        else:
            _out("  → Skipping categorization (no description/tags)")
        _out("  → Enqueuing LLM tasks (Shorty, synthetic questions, entities)")
        enqueue_llm_tasks_for_video(video_id)
        _maybe_spawn_openrouter_pipeline(video_id)
    else:
        _out("  → Skipping vectorization (no transcript)")


def _log_grab_summary(
    video_id: str,
    title: str,
    channel: str,
    transcript_success: bool,
    metadata: Optional[dict],
) -> None:
    _out(f"\n✅ VIDEO GRABBED: {video_id}")
    _out(f"   Title: {title}")
    _out(f"   Channel: {channel or 'Unknown'}")
    _out(f"   Transcript: {'✅' if transcript_success else '❌'}")
    _out(f"   Metadata: {'✅' if metadata else '❌'}")
    _out(f"{'='*60}\n")
    logger.info(f"✅ Video grabbed: {video_id} - {title}")


def _render_grab_page():
    """Manual paste page (public / no-local-module fallback)."""
    url = request.args.get('url', '').strip()
    title = request.args.get('title', '').strip() or 'Untitled'
    channel = request.args.get('channel', '').strip() or 'Unknown channel'

    video_id = _extract_video_id(url)
    if not video_id:
        return render_template(
            'grab.html',
            error='Invalid YouTube URL',
            url=url,
            title=title,
            channel=channel,
            video_id=None,
        )

    return render_template(
        'grab.html',
        url=url,
        title=title,
        channel=channel,
        video_id=video_id,
        error=None,
    )


# --- Routes ---

@app.route('/')
def root():
    return jsonify({
        'service': 'video_grabber',
        'status': 'running',
        'port': int(os.getenv('GRABBER_PORT', 5000)),
        'endpoints': {
            'quick_fetch': '/tools/quick-fetch (GET: bookmarklet popup, auto-fetch when local module present)',
            'grab': '/grab (GET: manual paste fallback)',
            'save': '/api/save-transcript (POST)',
            'fetch': '/api/fetch-transcript (POST)',
            'save_pasted': '/api/save-pasted-transcript (POST)',
            'annotate': '/api/annotate (POST)',
            'tags': '/api/tags (GET)',
            'health': '/health',
            'status': '/api/status'
        }
    })


@app.route('/tools/quick-fetch', methods=['GET'])
def quick_fetch_page():
    """Bookmarklet popup: auto-fetch on load (pre-eea8f02 quick_fetch.html) when local module present."""
    if not _LOCAL_YT_DLP_AVAILABLE:
        return redirect('/grab?' + request.query_string.decode('utf-8'), code=302)
    return render_template('quick_fetch.html')


@app.route('/grab', methods=['GET'])
def grab_page():
    """Manual transcript paste page (no local yt-dlp module, or direct /grab URL)."""
    return _render_grab_page()


@app.route('/api/save-transcript', methods=['POST'])
def save_transcript():
    """
    Save pasted transcript and metadata (from URL params at grab time).
    Queues LLM tasks (Shorty, synthetic questions, entities) and vectorizes in background.
    """
    try:
        data = request.json or {}
        transcript_text = (data.get('transcript_text') or '').strip()
        url = (data.get('url') or '').strip()
        title = (data.get('title') or '').strip() or 'Untitled'
        channel = (data.get('channel') or '').strip() or 'Unknown channel'

        video_id = _extract_video_id(url)
        if not video_id:
            return jsonify({'success': False, 'error': 'Invalid YouTube URL'}), 400
        if not transcript_text:
            return jsonify({'success': False, 'error': 'Transcript is empty. Paste the text first.'}), 400

        transcript_text = _strip_timestamps_from_paste(transcript_text)
        if not transcript_text:
            return jsonify({'success': False, 'error': 'Transcript had only timestamps. Paste the actual text.'}), 400

        canonical_url = url if url else f'https://www.youtube.com/watch?v={video_id}'
        db.add_video(video_id, title, channel, canonical_url)
        success = db.save_transcript(video_id, transcript_text)
        if not success:
            return jsonify({'success': False, 'error': 'Failed to save transcript'}), 500

        try:
            db.set_watch_date(video_id)
        except Exception as e:
            _out(f"Warning: failed to set watch_date for {video_id}: {e}")

        _finalize_saved_transcript(video_id, len(transcript_text))

        return jsonify({
            'success': True,
            'video_id': video_id,
            'message': 'Transcript saved. Shorty and search index will be ready after background processing.'
        })
    except Exception as e:
        logger.error(f"save-transcript: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/fetch-transcript', methods=['POST'])
def api_fetch_transcript():
    """
    Auto-fetch transcript from YouTube (youtube-transcript-api). On success, same follow-up as save-transcript.
    When local_yt_dlp_fetcher.py is present, also fetches rich metadata via yt-dlp.
    If the video has no auto transcript, returns paste_required for the quick-fetch bookmarklet flow.
    """
    try:
        data = request.json or {}
        url = (data.get('url') or '').strip()
        title = (data.get('title') or '').strip() or 'Untitled'
        channel = (data.get('channel') or '').strip() or 'Unknown channel'
        video_id = _extract_video_id(url)
        if not video_id:
            return jsonify({'success': False, 'error': 'Invalid YouTube URL'}), 400

        if not _LOCAL_YT_DLP_AVAILABLE:
            return jsonify({
                'success': True,
                'paste_required': True,
                'video_id': video_id,
                'title': title,
                'channel': channel,
                'url': url if url else f'https://www.youtube.com/watch?v={video_id}',
                'message': 'Auto-fetch is not enabled on this machine. Paste the transcript manually.',
            })

        from simple_transcript_fetcher import SimpleTranscriptFetcher

        _out(f"\n{'='*60}")
        _out("📥 GRABBING VIDEO")
        _out(f"{'='*60}")
        _out(f"Video ID: {video_id}")
        _out(f"Title: {title}")
        _out(f"Channel: {channel}")
        _out(f"URL: {url}")
        _out(f"{'='*60}\n")

        logger.info(f"📥 Grabbing video: {video_id} - {title}")

        fetcher = SimpleTranscriptFetcher(str(db_path))
        _out("🔍 Fetching transcript...")
        result = fetcher.fetch_transcript_from_url(url, title, channel)
        transcript_success = bool(result.get('success'))

        if transcript_success:
            _out("✅ Transcript fetched successfully")
        else:
            _out("⚠️ Transcript fetch failed (video still added to DB)")

        metadata, title, channel = _apply_local_metadata(video_id, url, title, channel)

        try:
            db.set_watch_date(video_id)
        except Exception as e:
            _out(f"⚠️ Failed to set watch_date for {video_id}: {e}")

        if result.get('success'):
            msg = result.get('message') or ''
            if msg == 'Transcript already exists' or result.get('cached'):
                _start_bookmarklet_background(video_id, True, metadata)
                _log_grab_summary(video_id, title, channel, True, metadata)
                return jsonify({
                    'success': True,
                    'video_id': result.get('video_id') or video_id,
                    'title': title,
                    'channel': channel,
                    'message': msg or 'Transcript already exists',
                    'has_metadata': metadata is not None,
                })
            transcript_text = (result.get('transcript') or '').strip()
            if transcript_text:
                _start_bookmarklet_background(video_id, True, metadata)
                _log_grab_summary(video_id, title, channel, True, metadata)
                return jsonify({
                    'success': True,
                    'video_id': video_id,
                    'title': title,
                    'channel': channel,
                    'message': result.get('message') or 'Transcript fetched and saved',
                    'has_metadata': metadata is not None,
                })
            _start_bookmarklet_background(video_id, False, metadata)
            _log_grab_summary(video_id, title, channel, False, metadata)
            return jsonify({
                'success': True,
                'video_id': video_id,
                'warning': True,
                'message': 'Video saved but transcript text was empty.',
                'has_metadata': metadata is not None,
            })

        err_raw = (result.get('error') or 'Unknown error')
        err_l = err_raw.lower()
        if any(
            s in err_l
            for s in (
                'no transcript',
                'disabled',
                'not available',
                'could not retrieve',
                'subtitles are disabled',
            )
        ):
            canonical = url if url else f'https://www.youtube.com/watch?v={video_id}'
            try:
                db.add_video(video_id, title, channel, canonical)
            except Exception as e:
                logger.warning(f"fetch-transcript add_video: {e}")
            _start_bookmarklet_background(video_id, False, metadata)
            _log_grab_summary(video_id, title, channel, False, metadata)
            return jsonify({
                'success': True,
                'paste_required': True,
                'video_id': video_id,
                'title': title,
                'channel': channel,
                'url': canonical,
                'message': err_raw,
                'has_metadata': metadata is not None,
            })

        _log_grab_summary(video_id, title, channel, False, metadata)
        return jsonify({'success': False, 'error': err_raw}), 400
    except Exception as e:
        logger.error(f"fetch-transcript: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/save-pasted-transcript', methods=['POST'])
def api_save_pasted_transcript():
    """Paste fallback for quick-fetch: same persistence and background work as /api/save-transcript."""
    try:
        data = request.json or {}
        transcript_text = (data.get('transcript_text') or '').strip()
        url = (data.get('url') or '').strip()
        title = (data.get('title') or '').strip() or 'Untitled'
        channel = (data.get('channel') or '').strip() or 'Unknown channel'
        video_id = (data.get('video_id') or '').strip()
        if not video_id:
            vid = _extract_video_id(url)
            video_id = vid or ''
        if not video_id:
            return jsonify({'success': False, 'error': 'Missing video_id'}), 400
        if not transcript_text:
            return jsonify({'success': False, 'error': 'Transcript is empty. Paste the text first.'}), 400

        transcript_text = _strip_timestamps_from_paste(transcript_text)
        if not transcript_text:
            return jsonify({'success': False, 'error': 'Transcript had only timestamps. Paste the actual text.'}), 400

        canonical_url = url if url else f'https://www.youtube.com/watch?v={video_id}'
        db.add_video(video_id, title, channel, canonical_url)
        success = db.save_transcript(video_id, transcript_text)
        if not success:
            return jsonify({'success': False, 'error': 'Failed to save transcript'}), 500

        try:
            db.set_watch_date(video_id)
        except Exception as e:
            _out(f"Warning: failed to set watch_date for {video_id}: {e}")

        _finalize_saved_transcript(video_id, len(transcript_text))

        return jsonify({
            'success': True,
            'video_id': video_id,
            'message': 'Transcript saved and vectorizing in background.',
        })
    except Exception as e:
        logger.error(f"save-pasted-transcript: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/annotate', methods=['POST', 'OPTIONS'])
def api_annotate():
    """Watch-time mark: video_id + timestamp + note + tags (no transcript required)."""
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json(force=True, silent=True) or {}
        video_id = (data.get('video_id') or '').strip()
        url = (data.get('url') or '').strip()
        title = (data.get('title') or '').strip() or 'Untitled'
        channel = (data.get('channel') or '').strip() or 'Unknown channel'
        note_text = data.get('note_text')
        tags_raw = data.get('tags')

        if not video_id and url:
            video_id = _extract_video_id(url) or ''
        if not video_id:
            return jsonify({'success': False, 'error': 'video_id is required'}), 400

        ts = data.get('timestamp_seconds')
        if ts is None:
            return jsonify({'success': False, 'error': 'timestamp_seconds is required'}), 400
        try:
            timestamp_seconds = float(ts)
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'timestamp_seconds must be a number'}), 400
        if timestamp_seconds < 0:
            return jsonify({'success': False, 'error': 'timestamp_seconds must be >= 0'}), 400

        tags: list = []
        if isinstance(tags_raw, list):
            tags = [str(t).strip() for t in tags_raw if str(t).strip()]

        canonical_url = url if url else f'https://www.youtube.com/watch?v={video_id}'

        video_created = db.ensure_bare_video(video_id, title, channel, canonical_url)
        if video_created:
            try:
                db.set_watch_date(video_id)
            except Exception as e:
                _out(f"Warning: failed to set watch_date for {video_id}: {e}")

        ann_id = db.insert_annotation(
            video_id,
            timestamp_seconds,
            note_text=note_text if note_text is not None else None,
            tags=tags,
        )
        if ann_id is None:
            return jsonify({'success': False, 'error': 'Failed to save annotation'}), 500

        return jsonify({
            'success': True,
            'annotation_id': ann_id,
            'video_created': video_created,
            'video_id': video_id,
            'timestamp_seconds': timestamp_seconds,
            'note_text': (note_text or '').strip() or None,
            'tags': tags,
        })
    except Exception as e:
        logger.error(f"annotate: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/tags', methods=['GET', 'OPTIONS'])
def api_tags():
    """Distinct tags used in annotations (for extension dropdown)."""
    if request.method == 'OPTIONS':
        return '', 204
    try:
        return jsonify({'tags': db.get_distinct_annotation_tags()})
    except Exception as e:
        logger.error(f"tags: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'video_grabber',
        'database': str(db_path.exists())
    })


@app.route('/api/status', methods=['GET'])
def status():
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM videos")
        video_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM transcripts")
        transcript_count = cursor.fetchone()[0]
        conn.close()
        return jsonify({
            'status': 'running',
            'videos': video_count,
            'transcripts': transcript_count,
            'database': str(db_path)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('GRABBER_PORT', 5000))
    print("\n" + "=" * 60)
    print("Video Grabber Service")
    print("=" * 60)
    print(f"Port: {port}")
    print(f"Database: {db_path.resolve()} ({_db_path_source})")
    if _LOCAL_YT_DLP_AVAILABLE:
        print("Auto-fetch: enabled (local_yt_dlp_fetcher.py)")
        print(f"Bookmarklet: http://localhost:{port}/tools/quick-fetch?url=...")
    else:
        print("Auto-fetch: disabled (manual paste only)")
        print(f"Grab page (bookmarklet): http://localhost:{port}/grab?url=...")
    print(f"Health: http://localhost:{port}/health")
    print("=" * 60)
    print("Use the Library and Ask UIs (ports 5002 / 5001) for browsing and search.")
    print("=" * 60 + "\n")
    logger.info(f"Starting Video Grabber Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
