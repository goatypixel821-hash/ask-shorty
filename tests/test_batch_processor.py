"""batch_processor: provider-mismatch guard, queue processing, and LLM failure handling."""

from __future__ import annotations

import sys

import pytest

import batch_processor as bp
import shorty_generator as sg
from tests.conftest import db_rows
from tests.fakes import FakeLLM, install_fake_llm
from transcript_database import TranscriptDatabase
from triple_extractor import extract_triples_openai

VID = "SRDGH780003"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(bp, "_wait_for_internet", lambda *a, **k: None)
    monkeypatch.setattr(bp, "NETWORK_RETRY_BACKOFF_SEC", (0, 0))
    monkeypatch.setattr(bp.time, "sleep", lambda *_: None)
    monkeypatch.setattr(bp, "TranscriptRAG", lambda *a, **k: None)  # queue mode never touches Chroma


def _run_queue(db, fake: FakeLLM, tasks, monkeypatch, **fns):
    """Enqueue ``tasks`` for VID and run the real queue loop with real generators over ``fake``."""
    # Dummy value only so the provider's 'key is set' check passes; the client is faked.
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-for-fake-client")
    install_fake_llm(monkeypatch, fake)
    db.enqueue_processing_tasks(VID, tasks)
    chat = lambda s, u, *a, **k: fake.respond(s, u)  # noqa: E731
    bp.process_queue_tasks(
        db,
        None,
        shorty_fn=fns.get("shorty_fn", sg.generate_shorty),
        synth_q_fn=fns.get("synth_q_fn", sg.generate_synthetic_questions),
        entity_fn=fns.get("entity_fn", lambda t, title=None: [{"name": "Tomasz Wierzbicki", "type": "person", "aliases": []}]),
        triples_fn=fns.get("triples_fn", lambda shorty, title: extract_triples_openai(shorty, title, chat)),
        limit=None,
        video_id=VID,
        hsc_chat_fn=lambda s, u, mt, t: fake.respond(s, u),
    )
    return {r[0]: (r[1], r[2]) for r in db_rows(db.db_path, "SELECT task, status, error FROM processing_queue WHERE video_id=?", (VID,))}


def _shorty(db):
    return db.get_transcript_and_shorty(VID)["shorty"]


# --------------------------------------------------------------------------- provider guard


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["batch_processor.py", *argv])
    return bp.main()


def test_guard_refuses_anthropic_flag_when_shorty_provider_is_openrouter(corpus_copy, monkeypatch, capsys):
    monkeypatch.setattr(sg, "SHORTY_PROVIDER", "openrouter")
    db = TranscriptDatabase(corpus_copy["db_path"])
    db.enqueue_processing_tasks(VID, ["shorty"])
    with pytest.raises(SystemExit) as exc:
        _main(monkeypatch, "--queue", "--provider", "anthropic", "--db-path", corpus_copy["db_path"])
    assert exc.value.code == 2  # regression: used to `return` (exit status 0)
    assert "Refusing to run" in capsys.readouterr().out
    assert db_rows(db.db_path, "SELECT status FROM processing_queue WHERE video_id=?", (VID,)) == [("pending",)]


def test_guard_allows_matching_provider(corpus_copy, monkeypatch, capsys):
    monkeypatch.setattr(sg, "SHORTY_PROVIDER", "anthropic")
    called = {}
    monkeypatch.setattr(bp, "process_queue_tasks", lambda *a, **k: called.setdefault("yes", True))
    _main(monkeypatch, "--queue", "--provider", "anthropic", "--db-path", corpus_copy["db_path"])
    assert called == {"yes": True}
    assert "Refusing" not in capsys.readouterr().out


def test_openrouter_provider_without_key_stops(corpus_copy, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(bp, "process_queue_tasks", lambda *a, **k: pytest.fail("must not run"))
    _main(monkeypatch, "--queue", "--provider", "openrouter", "--db-path", corpus_copy["db_path"])
    assert "OPENROUTER_API_KEY is not set" in capsys.readouterr().out


# --------------------------------------------------------------------------- happy path


def test_queue_happy_path(corpus_copy, monkeypatch):
    monkeypatch.setattr(sg, "SHORTY_PROVIDER", "openrouter")
    db = TranscriptDatabase(corpus_copy["db_path"])
    fake = FakeLLM(responder=_hsc_aware_responder)
    status = _run_queue(db, fake, ["shorty", "synthetic_questions", "entities", "triples"], monkeypatch)
    assert {k: v[0] for k, v in status.items()} == dict.fromkeys(status, "completed")
    sh = _shorty(db)
    assert sh.startswith("SOURCE: The Science of 78% Hydration Sourdough\nCHANNEL: Crumb Lab")
    assert "TOPICS" in sh
    assert db_rows(db.db_path, "SELECT COUNT(*) FROM entities WHERE video_id=?", (VID,))[0][0] == 1
    assert db.count_facts_for_video(VID) == 2


def test_rerunning_entities_does_not_duplicate(corpus_copy, monkeypatch):
    """Regression: store_entities only appended, so each re-run doubled the video's entities."""
    db = TranscriptDatabase(corpus_copy["db_path"])
    for _ in range(2):
        _run_queue(db, FakeLLM(), ["entities"], monkeypatch)
    assert db_rows(db.db_path, "SELECT name FROM entities WHERE video_id=?", (VID,)) == [("Tomasz Wierzbicki",)]


def test_synthetic_questions_follow_openrouter_provider(corpus_copy, monkeypatch):
    """Regression: generate_synthetic_questions always used the Anthropic SDK, even with SHORTY_PROVIDER=openrouter."""
    monkeypatch.setattr(sg, "SHORTY_PROVIDER", "openrouter")
    import anthropic_client

    fake = FakeLLM()
    install_fake_llm(monkeypatch, fake)
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-for-fake-client")
    monkeypatch.setattr(anthropic_client, "_client", None)  # Anthropic path would now raise (no key)
    assert sg.generate_synthetic_questions("some transcript", title="t") == ["What is the test topic?", "Who is involved?"]


# --------------------------------------------------------------------------- failure handling


@pytest.mark.parametrize("mode", ["empty", "garbage"])
def test_bad_shorty_output_is_not_saved_and_task_not_completed(corpus_copy, monkeypatch, mode):
    """
    Regression: an empty completion produced a header-only Shorty ('SOURCE: ...') which was
    saved and the task marked completed. Garbage text short of a real body is rejected too.
    """
    monkeypatch.setattr(sg, "SHORTY_PROVIDER", "openrouter")
    db = TranscriptDatabase(corpus_copy["db_path"])
    before = _shorty(db)
    status = _run_queue(db, FakeLLM(mode=mode), ["shorty"], monkeypatch)
    assert status["shorty"][0] in ("failed", "permanently_failed")
    assert "not saving" in status["shorty"][1]
    assert _shorty(db) == before


@pytest.mark.parametrize("mode", ["timeout", "error"])
def test_llm_exception_marks_failed_and_keeps_shorty(corpus_copy, monkeypatch, mode):
    monkeypatch.setattr(sg, "SHORTY_PROVIDER", "openrouter")
    db = TranscriptDatabase(corpus_copy["db_path"])
    before = _shorty(db)
    fake = FakeLLM(mode=mode)
    status = _run_queue(db, fake, ["shorty"], monkeypatch)
    assert status["shorty"][0] in ("failed", "permanently_failed")
    assert _shorty(db) == before
    if mode == "timeout":  # transient: retried (backoff patched to 0) before failing
        assert len(fake.calls) >= 3


def test_failed_tasks_end_permanently_failed_not_loop_forever(corpus_copy, monkeypatch):
    monkeypatch.setattr(sg, "SHORTY_PROVIDER", "openrouter")
    db = TranscriptDatabase(corpus_copy["db_path"])
    status = _run_queue(db, FakeLLM(mode="error"), ["shorty"], monkeypatch)
    assert status["shorty"][0] == "permanently_failed"
    attempts = db_rows(db.db_path, "SELECT attempts FROM processing_queue WHERE video_id=?", (VID,))[0][0]
    assert attempts == 6


def test_garbage_triples_keep_existing_facts(corpus_copy, monkeypatch):
    """Regression: unparseable triple JSON -> [] -> replace_facts_for_video wiped the facts, task 'completed'."""
    db = TranscriptDatabase(corpus_copy["db_path"])
    assert db.count_facts_for_video(VID) == 2
    status = _run_queue(db, FakeLLM(mode="garbage"), ["triples"], monkeypatch)
    assert status["triples"][0] in ("failed", "permanently_failed")
    assert db.count_facts_for_video(VID) == 2


def test_garbage_segments_keep_existing_segments(corpus_copy, monkeypatch):
    db = TranscriptDatabase(corpus_copy["db_path"])
    assert db.count_segments_for_video(VID) == 3
    status = _run_queue(db, FakeLLM(mode="garbage"), ["segments"], monkeypatch)
    assert status["segments"][0] in ("failed", "permanently_failed")
    assert db.count_segments_for_video(VID) == 3


def test_garbage_events_keep_existing_events(corpus_copy, monkeypatch):
    db = TranscriptDatabase(corpus_copy["db_path"])
    vid = "CRRNGTN1859"
    assert db.count_events_for_video(vid) == 1
    monkeypatch.setattr(sys.modules[__name__], "VID", vid)
    status = _run_queue(db, FakeLLM(mode="garbage"), ["events"], monkeypatch)
    assert status["events"][0] in ("failed", "permanently_failed")
    assert db.count_events_for_video(vid) == 1


def test_empty_triples_are_not_cached(tmp_path, monkeypatch):
    """Regression: extract_triples cached [] from a garbage response, so retries never called the model."""
    import triple_extractor as te

    calls = []
    monkeypatch.setattr(te, "extract_triples_anthropic", lambda s, t: calls.append(1) or [])
    te.extract_triples("shorty text", "title", project_root=tmp_path)
    te.extract_triples("shorty text", "title", project_root=tmp_path)
    assert len(calls) == 2


def test_validate_shorty_body():
    with pytest.raises(ValueError):
        sg.validate_shorty_body("")
    with pytest.raises(ValueError):
        sg.validate_shorty_body('{"error": {"message": "rate limited", "code": 429}, "padding": "xxxxxxxxxx"}')
    body = "CONTEXT\nA real Shorty body with enough content to be useful.\n"
    assert sg.validate_shorty_body(body) == body.strip()


def _hsc_aware_responder(system: str, user: str) -> str:
    from tests.fakes import default_responder

    from triple_extractor import TRIPLE_SYSTEM_PROMPT

    if system.strip() == TRIPLE_SYSTEM_PROMPT.strip():
        return '[{"subject": "high hydration", "relation": "increases", "object": "enzyme activity"},' \
               ' {"subject": "levain", "relation": "kept at", "object": "24 C"}]'
    return default_responder(system, user)
