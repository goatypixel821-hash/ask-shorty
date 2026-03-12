#!/usr/bin/env python3
"""
Start Video Grabber Service
============================

Simple script to start the lightweight video grabber service.
"""

import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    grabber_script = Path(__file__).parent / 'video_grabber.py'
    
    print("🔷 Starting Video Grabber Service...")
    print("=" * 50)
    print("This service handles video grabbing via bookmarklet")
    print("It does NOT load all videos into memory")
    print("=" * 50)
    print()
    
    try:
        subprocess.run([sys.executable, str(grabber_script)], check=True)
    except KeyboardInterrupt:
        print("\n\nShutting down grabber service...")
        sys.exit(0)

