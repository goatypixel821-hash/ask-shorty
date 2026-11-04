#!/usr/bin/env python3
"""
Serve ask_shorty_app (:5001) and library_app (:5002) on the synthetic fixture corpus
with the fake LLM and stand-in models. Nothing leaves the machine.

    python tests/ui/serve_fixture_apps.py /path/to/empty/workdir
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    work = Path(sys.argv[1]).resolve()
    work.mkdir(parents=True, exist_ok=True)

    from tests.conftest import StandinCrossEncoder, StandinEmbedder, build_corpus

    env = build_corpus(work, StandinEmbedder())
    os.environ["ASK_SHORTY_DB_PATH"] = env["db_path"]
    os.environ["ASK_SHORTY_CHROMA_PATH"] = env["chroma_dir"]
    os.environ["ASK_SHORTY_V2"] = "1"

    import pytest

    from tests.fakes import install_fake_llm

    mp = pytest.MonkeyPatch()
    fake = install_fake_llm(mp)

    import ask_shorty
    import ask_shorty_v2
    import reranker

    ce = StandinCrossEncoder()
    mp.setattr(reranker, "_cross_encoder", ce)
    mp.setattr(ask_shorty_v2.AskShortyV2, "_get_cross_encoder", lambda self: ce)
    mp.setattr(ask_shorty_v2, "_v2_log_path", lambda: work / "v2_request_log.html")

    import ask_shorty_app
    import library_app

    eng = ask_shorty.AskShorty()
    eng._rag = env["rag"]  # V1 uses the fixture Chroma store + stand-in embedder
    ask_shorty_app._engine = eng

    threading.Thread(
        target=lambda: library_app.app.run(host="127.0.0.1", port=5002, use_reloader=False), daemon=True
    ).start()
    print(f"[fixture-apps] db={env['db_path']} fake_llm_calls={len(fake.calls)}", flush=True)
    ask_shorty_app.init_app()  # startup work no longer happens at import
    ask_shorty_app.app.run(host="127.0.0.1", port=5001, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
