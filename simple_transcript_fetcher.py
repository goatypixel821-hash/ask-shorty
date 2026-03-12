#!/usr/bin/env python3
"""
Simple Transcript Fetcher
Fetches YouTube transcripts and stores them in SQLite database
"""

import re
from typing import Optional, Dict, Any
from transcript_database import TranscriptDatabase

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
    TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    print("❌ youtube-transcript-api not installed. Run: pip install youtube-transcript-api")
    YouTubeTranscriptApi = None
    TextFormatter = None
    TRANSCRIPT_API_AVAILABLE = False

class SimpleTranscriptFetcher:
    def __init__(self, db_path: str = "data/transcripts.db"):
        """Initialize transcript fetcher with database"""
        self.db = TranscriptDatabase(db_path)
        self.formatter = TextFormatter() if TextFormatter else None
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def fetch_transcript(self, video_id: str, url: str = None) -> Dict[str, Any]:
        """Fetch transcript for a single video"""
        if not TRANSCRIPT_API_AVAILABLE:
            return {
                'success': False,
                'error': 'youtube-transcript-api not installed',
                'video_id': video_id
            }
        
        # Check if we already have this transcript
        if self.db.has_transcript(video_id):
            transcript_text = self.db.get_transcript(video_id)
            return {
                'success': True,
                'video_id': video_id,
                'transcript': transcript_text,
                'cached': True,
                'message': 'Transcript already exists'
            }
        
        try:
            # Fetch transcript from YouTube
            from youtube_transcript_api import YouTubeTranscriptApi
            api = YouTubeTranscriptApi()
            transcript_list = api.fetch(video_id)
            
            if not transcript_list:
                return {
                    'success': False,
                    'error': 'No transcript available',
                    'video_id': video_id
                }
            
            # Format transcript text
            if self.formatter:
                transcript_text = self.formatter.format_transcript(transcript_list)
            else:
                # Simple formatting if formatter not available
                transcript_text = ' '.join([item['text'] for item in transcript_list])
            
            # Save to database
            success = self.db.save_transcript(video_id, transcript_text)
            
            if success:
                return {
                    'success': True,
                    'video_id': video_id,
                    'transcript': transcript_text,
                    'cached': False,
                    'message': 'Transcript fetched and saved'
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to save transcript',
                    'video_id': video_id
                }
                
        except Exception as e:
            error_msg = str(e)
            if 'No transcript' in error_msg:
                error_msg = 'No transcript available for this video'
            elif 'Video unavailable' in error_msg:
                error_msg = 'Video is unavailable or private'
            
            return {
                'success': False,
                'error': error_msg,
                'video_id': video_id
            }
    
    def fetch_transcript_from_url(self, url: str, title: str = None, channel: str = None) -> Dict[str, Any]:
        """Fetch transcript from YouTube URL"""
        video_id = self.extract_video_id(url)
        
        if not video_id:
            return {
                'success': False,
                'error': 'Invalid YouTube URL',
                'url': url
            }
        
        # Add video to database if not exists
        # Even if we don't have a title/channel, we should save the ID so the transcript isn't orphaned
        save_title = title if title else 'Unknown Title'
        save_channel = channel if channel else 'Unknown Channel'
        
        self.db.add_video(video_id, save_title, save_channel, url)
        
        return self.fetch_transcript(video_id, url)
    
    def get_transcript_status(self, video_id: str) -> Dict[str, Any]:
        """Get transcript status for a video"""
        video_info = self.db.get_video_info(video_id)
        
        if not video_info:
            return {
                'video_id': video_id,
                'exists': False,
                'has_transcript': False
            }
        
        return {
            'video_id': video_id,
            'exists': True,
            'has_transcript': video_info['has_transcript'],
            'title': video_info['title'],
            'channel': video_info['channel'],
            'transcript_fetched_at': video_info['transcript_fetched_at']
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.db.get_stats()

def main():
    """Test the transcript fetcher"""
    print("🚀 Testing Simple Transcript Fetcher")
    print("=" * 50)
    
    fetcher = SimpleTranscriptFetcher()
    
    # Test with a known video (this might fail if no transcript available)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - has transcript
    test_video_id = "dQw4w9WgXcQ"
    
    print(f"📹 Testing with video: {test_video_id}")
    
    # Check status
    status = fetcher.get_transcript_status(test_video_id)
    print(f"📊 Video status: {status}")
    
    # Try to fetch transcript
    result = fetcher.fetch_transcript(test_video_id)
    print(f"📝 Fetch result: {result['success']}")
    
    if result['success']:
        print(f"✅ Transcript length: {len(result['transcript'])} characters")
        print(f"📄 First 100 chars: {result['transcript'][:100]}...")
    else:
        print(f"❌ Error: {result['error']}")
    
    # Get database stats
    stats = fetcher.get_database_stats()
    print(f"📊 Database stats: {stats}")
    
    print("🎉 Transcript fetcher test completed!")

if __name__ == "__main__":
    main()

