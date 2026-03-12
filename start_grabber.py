#!/usr/bin/env python3
"""
Start Video Grabber Service
============================

Simple script to start the lightweight video grabber service.
"""

import subprocess
import sys
import time
from pathlib import Path


if __name__ == '__main__':
    grabber_script = Path(__file__).parent / 'video_grabber.py'

    print("🔷 Starting Video Grabber Service supervisor...")
    print("=" * 50)
    print("This service handles video grabbing via bookmarklet")
    print("It does NOT load all videos into memory")
    print("If the grabber crashes, it will be restarted automatically.")
    print("=" * 50)
    print()

    try:
        while True:
            print("🚀 Launching video_grabber.py...")
            proc = subprocess.run([sys.executable, str(grabber_script)])
            code = proc.returncode

            if code == 0:
                print("\nGrabber exited cleanly. Shutting down supervisor.")
                sys.exit(0)

            # Non-zero exit code: log and restart after short delay
            print(f"\n⚠️ Grabber crashed (exit code {code}) — restarting in 2 seconds...")
            time.sleep(2)
            print("🔄 Restarting grabber...\n")
    except KeyboardInterrupt:
        print("\n\nShutting down grabber service supervisor...")
        sys.exit(0)

