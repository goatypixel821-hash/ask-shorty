#!/usr/bin/env python3
"""
Video Grabber Service - Lightweight Bookmarklet Handler
========================================================

Minimal Flask service that only handles video grabbing via bookmarklet.
Does NOT load all videos into memory - only processes new videos.

Features:
- /api/fetch-transcript endpoint (bookmarklet)
- Background vectorization
- Background sorting/categorization
- Minimal memory footprint
- Can run 24/7 without heavy resource usage
"""

import sys
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUNBUFFERED'] = '1'
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
            sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
        except:
            pass

from flask import Flask, jsonify, request, render_template
import json
import sqlite3
import re
import threading
import datetime
import logging
from pathlib import Path

# Import only what we need
from simple_transcript_fetcher import SimpleTranscriptFetcher
from video_downloader import VideoDownloader
from transcript_database import TranscriptDatabase
from transcript_rag import TranscriptRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Add CORS headers for all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Get base directory
base_dir = Path(__file__).parent
db_path = base_dir / 'data' / 'transcripts.db'
download_dir = base_dir / 'downloads'
grab_log_path = base_dir / 'data' / 'grab_log.txt'


def _out(msg: str):
    """Write to stderr and to grab log file so you always see activity (terminal often doesn't show on Windows)."""
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

# Initialize only what we need
transcript_fetcher = SimpleTranscriptFetcher(str(db_path))
video_downloader = VideoDownloader(str(download_dir))
db = TranscriptDatabase(str(db_path))
rag = TranscriptRAG()

logger.info("Video Grabber Service initialized")
logger.info(f"Database: {db_path}")
logger.info(f"Download dir: {download_dir}")


def recalculate_tier(video):
    """
    Recalculate video tier based on available data.
    Simplified version - no need for full VIDEO_DATA structure.
    """
    tier = 3  # Default tier
    
    # Check for transcript
    if video.get('has_transcript'):
        tier = 2
    
    # Check for metadata
    if video.get('description') and len(video.get('description', '')) > 100:
        tier = 1
    
    # Check for AI categorization
    if video.get('ai_category') and video.get('ai_category') != 'Unknown':
        tier = 1
    
    video['tier'] = tier
    return tier


def vectorize_video_in_background(video_id):
    """Vectorize a single video transcript in background"""
    def do_vectorize():
        try:
            # Get transcript from DB
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT text FROM transcripts WHERE video_id = ? ORDER BY created_at DESC LIMIT 1",
                (video_id,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0]:
                # Has transcript - index it
                _out(f"  [Vectorize] Indexing transcript for {video_id}...")
                rag.index_single_transcript(video_id, row[0])
                _out(f"  [Vectorize] ✅ Complete for {video_id}")
                logger.info(f"✅ Vectorized transcript for {video_id}")
            else:
                _out(f"  [Vectorize] ⚠️ No transcript found for {video_id}")
                logger.info(f"⚠️ No transcript to vectorize for {video_id}")
        except Exception as e:
            logger.error(f"❌ Vectorization error for {video_id}: {e}", exc_info=True)
    
    # Run in background thread
    thread = threading.Thread(target=do_vectorize, daemon=True)
    thread.start()


def enqueue_llm_tasks_for_video(video_id: str):
    """
    Enqueue heavy LLM work (Shorty, synthetic questions, entities) for later batch processing.

    This keeps the grabber lightweight: we only fetch/save transcripts and vectorize
    raw chunks immediately.
    """
    try:
        db.enqueue_processing_tasks(video_id)
        _out(f"  [Queue] ✅ Enqueued LLM tasks for {video_id} (shorty, synthetic_questions, entities)")
    except Exception as e:
        logger.error(f"❌ Queue enqueue error for {video_id}: {e}", exc_info=True)


def categorize_video_in_background(video_id, title, description, tags):
    """Categorize video using AI in background"""
    def do_categorize():
        try:
            # Import categorization module if available
            categorize_func = None
            try:
                from categorize_new_videos_laptop_local import categorize_video
                categorize_func = categorize_video
            except ImportError:
                try:
                    from categorize_new_videos import categorize_video
                    categorize_func = categorize_video
                except ImportError:
                    logger.debug("Categorization module not available, skipping")
                    return
            
            if not categorize_func:
                return
            
            # Prepare video data
            video_data = {
                'id': video_id,
                'title': title,
                'description': description or '',
                'ai_tags': tags or []
            }
            
            # Categorize
            try:
                result = categorize_func(video_data)
                
                if result:
                    # Update database with category
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    
                    # Check if videos table exists and has ai_category column
                    cursor.execute("PRAGMA table_info(videos)")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    if 'ai_category' in columns:
                        category = result.get('category') or result.get('ai_category') or 'Unknown'
                        cursor.execute(
                            "UPDATE videos SET ai_category = ? WHERE video_id = ?",
                            (category, video_id)
                        )
                        conn.commit()
                        _out(f"  [Categorize] ✅ {video_id}: {category}")
                        logger.info(f"✅ Categorized {video_id}: {category}")
                    
                    conn.close()
            except Exception as e:
                logger.warning(f"⚠️ Categorization error for {video_id}: {e}")
        except Exception as e:
            logger.error(f"❌ Categorization error for {video_id}: {e}", exc_info=True)
    
    # Run in background thread
    thread = threading.Thread(target=do_categorize, daemon=True)
    thread.start()


@app.route('/')
def root():
    """Root endpoint - shows service info"""
    return jsonify({
        'service': 'video_grabber',
        'status': 'running',
        'port': int(os.getenv('GRABBER_PORT', 5000)),
        'endpoints': {
            'bookmarklet': '/api/fetch-transcript',
            'health': '/health',
            'status': '/api/status'
        }
    })


@app.route('/tools/quick-fetch')
def quick_fetch_page():
    """Quick fetch popup page - renders the bookmarklet UI"""
    return render_template('quick_fetch.html')


@app.route('/api/fetch-transcript', methods=['POST'])
def fetch_transcript_endpoint():
    """
    Fetch transcript from URL - Main bookmarklet endpoint
    
    This is the ONLY endpoint needed for the grabber service.
    It handles:
    1. Fetching transcript
    2. Fetching metadata
    3. Saving to database
    4. Background vectorization
    5. Background categorization
    """
    # Diagnostic: write immediately so we know THIS grabber received the request
    try:
        hit_file = base_dir / 'data' / 'grabber_hit.txt'
        hit_file.parent.mkdir(parents=True, exist_ok=True)
        with open(hit_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.datetime.now().isoformat()} REQUEST RECEIVED (Desktop video_grabber) path={base_dir}\n")
            f.flush()
    except Exception:
        pass
    try:
        data = request.json
        url = data.get('url')
        title = data.get('title')
        channel = data.get('channel')
        
        if not url:
            return jsonify({'success': False, 'error': 'No URL provided'}), 400
        
        # Extract video ID
        video_id = None
        match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
        if match:
            video_id = match.group(1)
        else:
            return jsonify({'success': False, 'error': 'Invalid YouTube URL'}), 400
        
        # Ensure we have a title
        if not title:
            title = 'New Scraped Video'
        
        # Visual output for monitoring (stderr + flush so it shows in terminal on Windows)
        _out(f"\n{'='*60}")
        _out(f"📥 GRABBING VIDEO")
        _out(f"{'='*60}")
        _out(f"Video ID: {video_id}")
        _out(f"Title: {title}")
        _out(f"Channel: {channel or 'Unknown'}")
        _out(f"URL: {url}")
        _out(f"{'='*60}\n")
        
        logger.info(f"📥 Grabbing video: {video_id} - {title}")
        
        # Fetch transcript (this also adds video to DB)
        _out("🔍 Fetching transcript...")
        result = transcript_fetcher.fetch_transcript_from_url(url, title=title, channel=channel)
        transcript_success = result.get('success', False)
        if transcript_success:
            _out("✅ Transcript fetched successfully")
        else:
            _out("⚠️ Transcript fetch failed (video still added to DB)")
        # Fetch rich metadata using yt-dlp
        metadata = None
        try:
            _out("🔍 Fetching metadata...")
            metadata = video_downloader.fetch_metadata(url)
            if metadata:
                if metadata.get('title'):
                    title = metadata['title']
                if metadata.get('channel'):
                    channel = metadata['channel']
                transcript_fetcher.db.save_metadata(video_id, metadata)
                desc_len = len(metadata.get('description', ''))
                tags_count = len(metadata.get('tags', []))
                _out(f"✅ Metadata saved: {desc_len} chars description, {tags_count} tags")
        except Exception as e:
            _out(f"⚠️ Metadata fetch warning: {e}")
            metadata = None
        response_data = {
            'success': True,
            'video_id': video_id,
            'title': title,
            'channel': channel or 'Unknown Channel',
            'transcript_success': transcript_success,
            'has_metadata': metadata is not None
        }
        _out("\n🚀 Starting background processing...")
        if transcript_success:
            _out("  → Vectorizing transcript (background)")
            vectorize_video_in_background(video_id)
            description = metadata.get('description', '') if metadata else ''
            tags = metadata.get('tags', []) if metadata else []
            if description or tags:
                _out("  → Categorizing video (background)")
                categorize_video_in_background(video_id, title, description, tags)
            else:
                _out("  → Skipping categorization (no description/tags)")
            # Enqueue heavy LLM work for later batch processing
            _out("  → Enqueuing LLM tasks (Shorty, synthetic questions, entities)")
            enqueue_llm_tasks_for_video(video_id)
        else:
            _out("  → Skipping vectorization (no transcript)")
        _out(f"\n✅ VIDEO GRABBED: {video_id}")
        _out(f"   Title: {title}")
        _out(f"   Channel: {channel or 'Unknown'}")
        _out(f"   Transcript: {'✅' if transcript_success else '❌'}")
        _out(f"   Metadata: {'✅' if metadata else '❌'}")
        _out(f"{'='*60}\n")
        logger.info(f"✅ Video grabbed: {video_id} - {title}")
        return jsonify(response_data)

        # --- PASTE MODE (if IP blocked again): uncomment block below and comment out the block above ---
        # transcript_fetcher.db.add_video(video_id, title or 'Unknown', channel or 'Unknown Channel', url)
        # _out("⏭️ Skipping transcript fetch (paste mode - IP block workaround)")
        # _out("⏭️ Skipping metadata fetch (yt-dlp commented out)")
        # response_data = {'success': True, 'paste_required': True, 'video_id': video_id, 'title': title, 'channel': channel or 'Unknown Channel', 'url': url}
        # _out(f"\n📋 Open the paste window, paste transcript, then click Save.\n{'='*60}\n")
        # return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Error in fetch-transcript: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _strip_timestamps_from_paste(text: str) -> str:
    """Remove only YouTube-style timestamp-only lines (e.g. 0:00, 1:23, 12:34:56). Keeps all real text."""
    if not text:
        return text
    # Match a line that is ONLY: optional space, time (0:00 or 1:23:45), optional " - " or " – ", then end.
    # Does NOT match lines like "See you at 12:00" or "At 0:00 we started" (those keep the whole line).
    ts_line = re.compile(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*[-–—]?\s*$', re.MULTILINE)
    cleaned = ts_line.sub('', text)
    # Trim each line, drop blanks, join with double newline so paragraphs stay readable
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    return '\n\n'.join(lines) if lines else cleaned.strip()


@app.route('/api/save-pasted-transcript', methods=['POST'])
def save_pasted_transcript():
    """Save transcript text that user pasted (IP block workaround). Strips timestamps, then vectorize."""
    try:
        data = request.json
        video_id = data.get('video_id')
        transcript_text = (data.get('transcript_text') or '').strip()
        title = data.get('title') or 'Unknown'
        channel = data.get('channel') or 'Unknown Channel'
        url = data.get('url') or f'https://www.youtube.com/watch?v={video_id}'
        if not video_id:
            return jsonify({'success': False, 'error': 'No video_id'}), 400
        if not transcript_text:
            return jsonify({'success': False, 'error': 'Transcript is empty. Paste the text first.'}), 400
        transcript_text = _strip_timestamps_from_paste(transcript_text)
        if not transcript_text:
            return jsonify({'success': False, 'error': 'Transcript had only timestamps. Paste the actual text.'}), 400
        transcript_fetcher.db.add_video(video_id, title, channel, url)
        success = transcript_fetcher.db.save_transcript(video_id, transcript_text)
        if not success:
            return jsonify({'success': False, 'error': 'Failed to save transcript'}), 500
        # Log confirms: we save only the cleaned text (timestamp-only lines removed)
        _out(f"\n✅ Pasted transcript saved: {video_id} ({len(transcript_text)} chars saved, timestamps removed)")
        _out("🚀 Vectorizing (background)...")
        vectorize_video_in_background(video_id)
        _out("  → Enqueuing LLM tasks (Shorty, synthetic questions, entities)")
        enqueue_llm_tasks_for_video(video_id)
        _out(f"{'='*60}\n")
        return jsonify({'success': True, 'video_id': video_id, 'message': 'Transcript saved and vectorizing.'})
    except Exception as e:
        logger.error(f"save-pasted-transcript: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'video_grabber',
        'database': str(db_path.exists())
    })


@app.route('/api/status', methods=['GET'])
def status():
    """Get service status"""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Count videos
        cursor.execute("SELECT COUNT(*) FROM videos")
        video_count = cursor.fetchone()[0]
        
        # Count transcripts
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
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/debug/routes', methods=['GET'])
def debug_routes():
    """Debug endpoint to list all registered routes"""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
    return jsonify({
        'routes': routes,
        'total': len(routes)
    })


if __name__ == '__main__':
    port = int(os.getenv('GRABBER_PORT', 5000))
    
    print("\n" + "="*60)
    print("🔷 VIDEO GRABBER SERVICE")
    print("="*60)
    print(f"Port: {port}")
    print(f"Bookmarklet endpoint: http://localhost:{port}/api/fetch-transcript")
    print(f"Health check: http://localhost:{port}/health")
    print(f"Status: http://localhost:{port}/api/status")
    print("="*60)
    print("💡 This service only handles video grabbing")
    print("💡 Use main app.py (port 5001) for viewing/searching")
    print("="*60)
    print("\n📊 Monitoring: Watch this window for grab activity")
    print(f"   If nothing appears here, watch the log file: data\\grab_log.txt")
    print("   (In another terminal: Get-Content data\\grab_log.txt -Wait -Tail 25)\n")
    
    logger.info(f"🚀 Starting Video Grabber Service on port {port}")
    logger.info(f"📌 Bookmarklet endpoint: http://localhost:{port}/api/fetch-transcript")
    logger.info("💡 This service only handles video grabbing - use main app.py for viewing")
    
    app.run(host='0.0.0.0', port=port, debug=False)

