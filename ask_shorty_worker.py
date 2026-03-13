#!/usr/bin/env python3
"""
Single-request worker for Ask Shorty.

This script is intended to be run as a subprocess from ask_shorty_app.py.
It initializes the RAG stack (SentenceTransformer / Chroma) in THIS process,
handles one question, prints a JSON result to stdout, then exits.

Input:  JSON object on stdin: {"question": "...", "video_ids": [... or null]}
Output: JSON object on stdout:
    {"success": true, "answer": "...", "used_context": [...]}
or {"success": false, "error": "..."} on failure.
"""

import json
import sys

from ask_shorty import AskShorty


def main() -> int:
    try:
        raw = sys.stdin.read()
        data = json.loads(raw or "{}")
    except Exception as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON input: {e}"}))
        return 1

    question = (data.get("question") or "").strip()
    video_ids = data.get("video_ids")

    if not question:
        print(json.dumps({"success": False, "error": "Question is required"}))
        return 0

    try:
        engine = AskShorty()
        result = engine.answer_question(question, video_ids=video_ids)
        out = {
            "success": True,
            "answer": result.get("answer", ""),
            "used_context": result.get("used_context", []),
        }
        print(json.dumps(out))
        return 0
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())

