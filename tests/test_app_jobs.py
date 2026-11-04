"""ask_shorty_app job lifecycle (Flask test client) and job-state races; ask_shorty_worker contract."""

from __future__ import annotations

import importlib
import io
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from tests.conftest import db_rows
from transcript_database import TranscriptDatabase


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    boot = tmp_path_factory.mktemp("appboot") / "data" / "transcripts.db"
    mp = pytest.MonkeyPatch()
    mp.setenv("ASK_SHORTY_DB_PATH", str(boot))
    mp.setenv("ASK_SHORTY_V2", "1")
    mod = sys.modules.get("ask_shorty_app")
    mod = importlib.reload(mod) if mod else importlib.import_module("ask_shorty_app")
    yield mod
    mp.undo()


@pytest.fixture()
def app(app_module, corpus_copy, monkeypatch, tmp_path):
    monkeypatch.setenv("ASK_SHORTY_DB_PATH", corpus_copy["db_path"])
    monkeypatch.setattr(app_module, "db", TranscriptDatabase(corpus_copy["db_path"]))
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr(app_module, "_jobs_dir", jobs)
    monkeypatch.setattr(app_module, "_engine_v2", None)
    monkeypatch.setattr(app_module, "_engine", None)
    app_module._ensure_jobs_table()
    app_module._ensure_ref_id_column()
    return app_module


def _new_job(app, status="running", updated_at=None):
    with sqlite3.connect(app.db.db_path) as c:
        cur = c.execute(
            "INSERT INTO ask_jobs (question, status, created_at, updated_at) VALUES ('q', ?, datetime('now'), ?)",
            (status, updated_at or time.strftime("%Y-%m-%d %H:%M:%S")),
        )
        return cur.lastrowid


def _status(app, job_id):
    return db_rows(app.db.db_path, "SELECT status, error FROM ask_jobs WHERE id=?", (job_id,))[0]


class StubV2:
    def __init__(self, fn):
        self.fn = fn

    def answer(self, question, **kw):
        return self.fn(question, **kw)


def _ok(question, **kw):
    return {"answer": "ok", "used_context": [], "sources": [{"video_id": "KSTRL42rel1"}], "debug_events": []}


# --------------------------------------------------------------------------- lifecycle


def test_ask_v2_end_to_end_via_http(app, fake_llm):
    client = app.app.test_client()
    r = client.post("/api/ask_v2", json={"question": "How many images are on the Voyager Golden Record?"})
    assert r.status_code == 200
    job_id = r.get_json()["job_id"]
    for _ in range(200):
        res = client.get(f"/api/ask/result/{job_id}").get_json()
        if res["status"] not in ("pending", "running"):
            break
        time.sleep(0.05)
    assert res["status"] == "completed", res
    assert res["sources"][0]["video_id"] == "VGRrec00004"
    assert "115 images" in res["answer"] and res["ref_id"].startswith("ASK-")
    by_ref = client.get(f"/api/ask/ref/{res['ref_id']}").get_json()
    assert by_ref["answer"] == res["answer"]


def test_v1_job_runner_with_llm_error_marks_error(app, monkeypatch):
    class Boom:
        def answer_question(self, *a, **k):
            raise RuntimeError("upstream 500")

    monkeypatch.setattr(app, "get_engine", lambda: Boom())
    jid = _new_job(app, "pending")
    app._run_job(jid, "q", None)
    assert _status(app, jid) == ("error", "upstream 500")


# --------------------------------------------------------------------------- races


def test_timeout_landing_mid_completion_is_not_overwritten(app, monkeypatch):
    """
    Regression: the worker checked 'was I timed out?' and then wrote 'completed' in a
    separate statement. A poll marking timeout in between got overwritten.
    """
    jid = _new_job(app)
    monkeypatch.setattr(app, "get_engine_v2", lambda: StubV2(_ok))
    real_alloc = app._allocate_ref_id

    def poll_lands_now():  # runs between the skip-check and the completion write
        with sqlite3.connect(app.db.db_path) as c:
            c.execute("UPDATE ask_jobs SET status='error', error='timeout' WHERE id=?", (jid,))
        return real_alloc()

    monkeypatch.setattr(app, "_allocate_ref_id", poll_lands_now)
    app._run_job_v2(jid, "q", None)
    assert _status(app, jid) == ("error", "timeout")


def test_poll_timeout_does_not_clobber_fresh_completion(app, monkeypatch):
    """Regression: poll read 'running' (stale updated_at), worker completed, poll wrote error/timeout."""
    jid = _new_job(app, updated_at="2000-01-01 00:00:00")
    answer = json.dumps({"answer": "done", "used_context": [], "sources": []})

    def worker_finishes_now(ts):
        with sqlite3.connect(app.db.db_path) as c:
            c.execute("UPDATE ask_jobs SET status='completed', answer=?, ref_id='ASK-X' WHERE id=?", (answer, jid))
        return 10_000.0

    monkeypatch.setattr(app, "_seconds_since_job_timestamp", worker_finishes_now)
    res = app.app.test_client().get(f"/api/ask/result/{jid}").get_json()
    assert res["status"] == "completed" and res["answer"] == "done"
    assert _status(app, jid)[0] == "completed"


def test_worker_exception_does_not_overwrite_cancel(app, monkeypatch):
    jid = _new_job(app)

    def cancelled_then_crash(question, **kw):
        with sqlite3.connect(app.db.db_path) as c:
            c.execute("UPDATE ask_jobs SET status='cancelled', error='cancelled' WHERE id=?", (jid,))
        raise RuntimeError("connection reset")

    monkeypatch.setattr(app, "get_engine_v2", lambda: StubV2(cancelled_then_crash))
    app._run_job_v2(jid, "q", None)
    assert _status(app, jid) == ("cancelled", "cancelled")


def test_cancel_of_finished_job_is_rejected(app):
    jid = _new_job(app, "completed")
    r = app.app.test_client().post(f"/api/agent/cancel/{jid}")
    assert r.status_code in (400, 409)
    assert _status(app, jid)[0] == "completed"


def test_stale_job_file_is_not_served_as_this_jobs_answer(app, monkeypatch):
    """
    Regression: data/jobs/<id>.json is only a crash-recovery copy, but a file left by an
    earlier DB/run with the same id was returned as this job's answer when it errored.
    """
    jid = _new_job(app, "pending")
    (app._jobs_dir / f"{jid}.json").write_text(json.dumps({"answer": "SOMEONE ELSE'S ANSWER"}))

    def boom(question, **kw):
        raise RuntimeError("LLM timeout")

    monkeypatch.setattr(app, "get_engine_v2", lambda: StubV2(boom))
    app._run_job_v2(jid, "q", None)
    res = app.app.test_client().get(f"/api/ask/result/{jid}").get_json()
    assert res["status"] == "error" and "SOMEONE ELSE" not in json.dumps(res)


def test_crash_recovery_from_job_file_still_works(app):
    """The intended recovery path: worker wrote the file, crashed before the DB write."""
    jid = _new_job(app, "error")
    (app._jobs_dir / f"{jid}.json").write_text(json.dumps({"answer": "recovered", "sources": []}))
    res = app.app.test_client().get(f"/api/ask/result/{jid}").get_json()
    assert res["status"] == "completed" and res["answer"] == "recovered"


def test_jobs_dir_lives_beside_the_db(app_module):
    assert Path(app_module._jobs_dir).parent == Path(app_module.db.db_path).resolve().parent


# --------------------------------------------------------------------------- worker script


def test_worker_stdout_is_a_single_json_object(monkeypatch, capsys):
    """Regression: engine progress prints went to stdout, so the JSON contract was broken."""
    import ask_shorty_worker as w

    class Noisy:
        def __init__(self):
            print("[ask] [0ms] starting answer_question")

        def answer_question(self, q, video_ids=None):
            print("[ask] [5ms] RAG search start")
            return {"answer": "A", "used_context": ["c"]}

    monkeypatch.setattr(w, "AskShorty", Noisy)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"question": "hi"})))
    assert w.main() == 0
    out = capsys.readouterr()
    assert json.loads(out.out) == {"success": True, "answer": "A", "used_context": ["c"]}
    assert "[ask]" in out.err


def test_worker_bad_input(monkeypatch, capsys):
    import ask_shorty_worker as w

    monkeypatch.setattr(sys, "stdin", io.StringIO("{not json"))
    assert w.main() == 1
    assert json.loads(capsys.readouterr().out)["success"] is False
