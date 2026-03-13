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
import threading
import logging
from pathlib import Path

if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUNBUFFERED'] = '1'
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
            sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        except Exception:
            pass

from flask import Flask, jsonify, request, render_template
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

base_dir = Path(__file__).parent
db_path = base_dir / 'data' / 'transcripts.db'
grab_log_path = base_dir / 'data' / 'grab_log.txt'


def _out(msg: str):
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

db = TranscriptDatabase(str(db_path))
rag = TranscriptRAG()

logger.info("Video Grabber Service initialized")
logger.info(f"Database: {db_path}")


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


def _strip_timestamps_from_paste(text: str) -> str:
    if not text:
        return text
    ts_line = re.compile(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*[-–—]?\s*$', re.MULTILINE)
    cleaned = ts_line.sub('', text)
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    return '\n\n'.join(lines) if lines else cleaned.strip()


# --- Routes ---

@app.route('/')
def root():
    return jsonify({
        'service': 'video_grabber',
        'status': 'running',
        'port': int(os.getenv('GRABBER_PORT', 5000)),
        'endpoints': {
            'grab': '/grab (GET: url, title, channel as query params)',
            'save': '/api/save-transcript (POST)',
            'health': '/health',
            'status': '/api/status'
        }
    })


@app.route('/grab', methods=['GET'])
def grab_page():
    """
    Bookmarklet target: receives url, title, channel as URL parameters from the browser.
    Renders a page with video info and a textarea to paste the transcript.
    """
    url = request.args.get('url', '').strip()
    title = request.args.get('title', '').strip() or 'Untitled'
    channel = request.args.get('channel', '').strip() or 'Unknown channel'

    video_id = _extract_video_id(url)
    if not video_id:
        return render_template('grab.html', error='Invalid YouTube URL', url=url, title=title, channel=channel, video_id=None)

    return render_template(
        'grab.html',
        url=url,
        title=title,
        channel=channel,
        video_id=video_id,
        error=None
    )


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

        _out(f"Transcript saved: {video_id} ({len(transcript_text)} chars)")
        _out("Vectorizing in background...")
        vectorize_video_in_background(video_id)
        enqueue_llm_tasks_for_video(video_id)
        _out("Done.")

        return jsonify({
            'success': True,
            'video_id': video_id,
            'message': 'Transcript saved. Shorty and search index will be ready after background processing.'
        })
    except Exception as e:
        logger.error(f"save-transcript: {e}", exc_info=True)
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
    print(f"Grab page (bookmarklet): http://localhost:{port}/grab?url=...&title=...&channel=...")
    print(f"Health: http://localhost:{port}/health")
    print("=" * 60)
    print("Use the Library and Ask UIs (ports 5002 / 5001) for browsing and search.")
    print("=" * 60 + "\n")
    logger.info(f"Starting Video Grabber Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
