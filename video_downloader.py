import os
import glob

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    print("❌ yt-dlp not installed. Video downloading and rich metadata will be disabled.")
    print("   Run: pip install yt-dlp")
    YT_DLP_AVAILABLE = False

class VideoDownloader:
    def __init__(self, download_dir="youtube-history-viewer/downloads"):
        self.download_dir = download_dir
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

    def fetch_metadata(self, video_url, quiet=False):
        """
        Fetches video metadata without downloading the video.
        Returns a dictionary with metadata or None if failed.
        quiet: if True, don't print progress (for batch scripts).
        """
        if not YT_DLP_AVAILABLE:
            return None

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True, # Critical: Don't download the video
        }
        
        if not quiet:
            print(f"ℹ️ Fetching metadata for {video_url}...")
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return {
                    'description': info.get('description', ''),
                    'upload_date': info.get('upload_date', ''),
                    'tags': info.get('tags', []),
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'duration': info.get('duration', 0),
                    'thumbnail': info.get('thumbnail', ''),
                    'channel': info.get('uploader', ''),
                    'title': info.get('title', ''),
                    'chapters': info.get('chapters', [])
                }
        except Exception as e:
            print(f"❌ Metadata fetch failed: {e}")
            return None

    def download_video(self, video_url, video_id):
        """
        Downloads a video using yt-dlp.
        Returns the path to the downloaded file or None if failed.
        """
        if not YT_DLP_AVAILABLE:
            print("❌ Cannot download: yt-dlp not installed")
            return None

        # Output template: downloads/video_id.ext
        output_template = os.path.join(self.download_dir, f"{video_id}.%(ext)s")
        
        ydl_opts = {
            'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]', # Limit to 1080p to save space
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
            'merge_output_format': 'mp4', # Ensure we get mp4
        }

        print(f"⬇️ Starting download for {video_id}...")
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Find the downloaded file (it might have different extensions before merge, but we asked for mp4)
            # Check specifically for the expected filename
            expected_file = os.path.join(self.download_dir, f"{video_id}.mp4")
            if os.path.exists(expected_file):
                print(f"✅ Download completed: {expected_file}")
                return expected_file
            
            # Fallback: look for any file with that ID
            files = glob.glob(os.path.join(self.download_dir, f"{video_id}.*"))
            if files:
                print(f"✅ Download completed (alt format): {files[0]}")
                return files[0]
                
            print(f"❌ Download finished but file not found for {video_id}")
            return None

        except Exception as e:
            print(f"❌ Download failed: {e}")
            return None

