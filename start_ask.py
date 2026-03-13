#!/usr/bin/env python3
"""
Start Ask Shorty UI Service
===========================

Simple supervisor script to start the Ask Shorty UI (`ask_shorty_app.py`)
and automatically restart it if it crashes (for example, due to Windows /
PyTorch / GPU memory issues after the first request).
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def _log(msg: str) -> None:
    """Print a log line with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


if __name__ == "__main__":
    ask_script = Path(__file__).parent / "ask_shorty_app.py"

    print("🔷 Starting Ask Shorty UI supervisor...")
    print("=" * 50)
    print("This service exposes the Ask Shorty UI and API (port 5001 by default).")
    print("If the app crashes after handling a request, it will be restarted automatically.")
    print("=" * 50)
    print()

    try:
        while True:
            _log("Launching ask_shorty_app.py...")
            proc = subprocess.run([sys.executable, str(ask_script)])
            code = proc.returncode

            if code == 0:
                _log("Ask Shorty UI exited cleanly. Shutting down supervisor.")
                sys.exit(0)

            # Non-zero exit code: log and restart after short delay
            _log(f"Ask Shorty UI crashed (exit code {code}) — restarting in 2 seconds...")
            time.sleep(2)
            _log("Restarting ask_shorty_app.py...\n")
    except KeyboardInterrupt:
        print("\n\nShutting down Ask Shorty UI supervisor...")
        sys.exit(0)

