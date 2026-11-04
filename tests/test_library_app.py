"""library_app (Flask, port 5002) against the fixture DB."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture()
def lib(corpus_copy, monkeypatch):
    monkeypatch.setenv("ASK_SHORTY_DB_PATH", corpus_copy["db_path"])
    mod = sys.modules.get("library_app")
    mod = importlib.reload(mod) if mod else importlib.import_module("library_app")
    return mod


def test_db_path_honours_env(lib, corpus_copy):
    """Regression: DB_PATH was hard-coded to <repo>/data/transcripts.db."""
    assert lib.DB_PATH == corpus_copy["db_path"]


def test_index_lists_all_fixture_videos(lib):
    html = lib.app.test_client().get("/").get_data(as_text=True)
    for title in ("Kestrel DB 4.2 Release Deep Dive", "Decoding the Honeybee Waggle Dance"):
        assert title in html


def test_channel_and_status_filters(lib):
    assert lib.get_video_counts({"channel": "Deep Orbit"}) == 2
    assert lib.get_video_counts({"status": "has_shorty"}) == 7
    assert lib.get_video_counts({"status": "missing_shorty"}) == 0
    assert lib.get_channels() == ["Byte Harbor", "Crumb Lab", "Deep Orbit", "Field Notes", "Ops Diaries"]


def test_video_detail_shows_shorty_questions_entities(lib):
    html = lib.app.test_client().get("/video/KSTRLpm0002").get_data(as_text=True)
    assert "Marcus Oyelaran" in html and "Tidewater" in html
    assert "Who fixed the Tidewater page-split race?" in html


def test_unknown_video_404(lib):
    assert lib.app.test_client().get("/video/NOPE0000000").status_code == 404


def test_edit_shorty_round_trip(lib):
    c = lib.app.test_client()
    r = c.post("/video/WGGLdnc0007/save/shorty", data={"shorty": "CONTEXT\nEdited Shorty text for the bee video."})
    assert r.status_code in (200, 302)
    assert "Edited Shorty text" in c.get("/video/WGGLdnc0007").get_data(as_text=True)
