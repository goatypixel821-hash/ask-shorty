"""config.py: precedence, missing-DB failure, opt-in creation, host binding, no import side effects."""

from __future__ import annotations

import importlib
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import config
from transcript_database import TranscriptDatabase

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    """Clean slate: no path env vars, an (initially empty) local config file, HOME in tmp."""
    for k in [v for vars_, _, _ in config.SETTINGS.values() for v in vars_]:
        monkeypatch.delenv(k, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    toml = tmp_path / "ask_shorty.local.toml"
    toml.write_text("", encoding="utf-8")
    monkeypatch.setenv("ASK_SHORTY_CONFIG", str(toml))

    def write(text: str) -> None:
        toml.write_text(text, encoding="utf-8")
        # mtime resolution can be coarse; make sure the cache sees the change
        st = toml.stat()
        os.utime(toml, (st.st_atime, st.st_mtime + 1 + len(config._file_cache)))

    return type("Cfg", (), {"home": home, "toml": toml, "write": staticmethod(write), "tmp": tmp_path})


# --------------------------------------------------------------------------- precedence


def test_defaults_anchor_to_repo_not_cwd(cfg, monkeypatch):
    monkeypatch.chdir(cfg.tmp)
    assert config.data_dir() == ROOT / "data"
    assert config.db_path() == ROOT / "data" / "transcripts.db"
    assert config.chroma_path() == ROOT / "data" / "transcript_chroma_finetuned"
    assert config.provider() == "openrouter"
    assert config.shorty_model() == "anthropic/claude-3-haiku"
    assert config.answer_model() == "claude-sonnet-4-20250514"
    assert config.openrouter_model() == "qwen/qwen-2.5-72b-instruct"
    assert config.host() == "127.0.0.1"
    assert config.create_db_allowed() is False


def test_env_beats_file_beats_default(cfg, monkeypatch):
    assert config.resolve("db_path")[1].startswith("default")
    cfg.write(f"db_path = '{cfg.tmp / 'from_file.db'}'\nprovider = \"anthropic\"\n")
    assert config.db_path() == cfg.tmp / "from_file.db"
    assert config.resolve("db_path")[1] == "ask_shorty.local.toml db_path"
    assert config.provider() == "anthropic"
    assert config.shorty_model() == "claude-3-haiku-20240307"  # default follows provider

    monkeypatch.setenv("ASK_SHORTY_DB_PATH", str(cfg.tmp / "from_env.db"))
    monkeypatch.setenv("SHORTY_PROVIDER", "OpenRouter")
    assert config.db_path() == cfg.tmp / "from_env.db"
    assert config.resolve("db_path")[1] == "env ASK_SHORTY_DB_PATH"
    assert config.provider() == "openrouter"


def test_data_dir_drives_derived_defaults(cfg):
    cfg.write(f"data_dir = '{cfg.tmp / 'd'}'\n")
    assert config.db_path() == cfg.tmp / "d" / "transcripts.db"
    assert config.chroma_path() == cfg.tmp / "d" / "transcript_chroma_finetuned"


def test_ask_shorty_table_and_windows_literal_paths(cfg):
    cfg.write("[ask_shorty]\ndb_path = 'C:\\Users\\someone\\data\\transcripts.db'\n")
    assert str(config.db_path()).endswith("transcripts.db")
    assert "someone" in str(config.db_path())


def test_full_db_precedence(cfg, monkeypatch):
    # nothing configured, no viewer DB on this machine -> same as db_path
    assert config.full_db_path() == config.db_path()
    # Shaun's machine: the viewer DB exists under ~/Desktop -> that's the default
    viewer = cfg.home / "Desktop" / "youtube-history-viewer-copy" / "data" / "transcripts.db"
    viewer.parent.mkdir(parents=True)
    viewer.write_bytes(b"")
    assert config.full_db_path() == viewer
    assert config.viewer_db_default() == viewer
    cfg.write(f"full_db_path = '{cfg.tmp / 'file_full.db'}'\n")
    assert config.full_db_path() == cfg.tmp / "file_full.db"
    # ASK_SHORTY_DB_PATH applies to the full DB too (as ask_shorty.py always did) ...
    monkeypatch.setenv("ASK_SHORTY_DB_PATH", str(cfg.tmp / "db.db"))
    assert config.full_db_path() == cfg.tmp / "db.db"
    # ... the legacy build_courses name beats it ...
    monkeypatch.setenv("ASK_SHORTY_FULL_DB", str(cfg.tmp / "legacy.db"))
    assert config.full_db_path() == cfg.tmp / "legacy.db"
    # ... and the explicit one wins.
    monkeypatch.setenv("ASK_SHORTY_FULL_DB_PATH", str(cfg.tmp / "full.db"))
    assert config.full_db_path() == cfg.tmp / "full.db"
    # The merge tool's fixed defaults ignore all of that.
    assert config.repo_db_default() == ROOT / "data" / "transcripts.db"
    assert config.viewer_db_default() == viewer


def test_full_chroma_precedence(cfg, monkeypatch):
    assert config.full_chroma_path() == config.chroma_path()
    viewer = cfg.home / "Desktop" / "youtube-history-viewer-copy" / "data" / "transcript_chroma_new"
    viewer.mkdir(parents=True)
    assert config.full_chroma_path() == viewer
    monkeypatch.setenv("ASK_SHORTY_CHROMA_PATH", str(cfg.tmp / "c"))
    assert config.full_chroma_path() == cfg.tmp / "c"


def test_bad_settings_fail_loudly(cfg, monkeypatch):
    cfg.write('db_pth = "typo"\n')
    with pytest.raises(config.ConfigError, match="Unknown setting.*db_pth"):
        config.db_path()
    cfg.write('provider = "gemini"\n')
    with pytest.raises(config.ConfigError, match="Invalid provider"):
        config.provider()
    cfg.write("db_path = \n")
    with pytest.raises(config.ConfigError, match="Can't parse"):
        config.db_path()
    monkeypatch.setenv("ASK_SHORTY_CONFIG", str(cfg.tmp / "nope.toml"))
    with pytest.raises(config.ConfigError, match="missing file"):
        config.db_path()


def test_missing_local_file_is_fine(cfg, monkeypatch):
    monkeypatch.delenv("ASK_SHORTY_CONFIG")
    monkeypatch.setattr(config, "REPO_ROOT", cfg.tmp / "norepo")
    assert config.file_settings() == {}


def test_shorty_generator_reads_provider_from_local_file(cfg):
    import shorty_generator

    cfg.write('provider = "anthropic"\nshorty_model = "claude-x"\n')
    try:
        mod = importlib.reload(shorty_generator)
        assert (mod.SHORTY_PROVIDER, mod.SHORTY_MODEL) == ("anthropic", "claude-x")
    finally:
        cfg.write("")
        importlib.reload(shorty_generator)


def test_show_command_prints_sources(cfg, capsys):
    cfg.write('answer_model = "m1"\n')
    assert config.main([]) == 0
    out = capsys.readouterr().out
    assert re.search(r"answer_model\s+m1\s+\[ask_shorty.local.toml answer_model\]", out)
    assert "db_path MISSING" in out


# --------------------------------------------------------------------------- missing DB


def test_missing_db_raises_and_creates_nothing(cfg):
    missing = cfg.tmp / "nowhere" / "transcripts.db"
    with pytest.raises(config.DatabaseNotFoundError) as ei:
        TranscriptDatabase(str(missing))
    msg = str(ei.value)
    assert str(missing) in msg and "ASK_SHORTY_DB_PATH" in msg and "init-db" in msg
    assert not missing.exists() and not missing.parent.exists()
    assert isinstance(ei.value, FileNotFoundError)


def test_default_db_missing_names_its_source(cfg, monkeypatch):
    monkeypatch.setenv("ASK_SHORTY_DB_PATH", str(cfg.tmp / "typo.db"))
    with pytest.raises(config.DatabaseNotFoundError, match=r"from db_path <- env ASK_SHORTY_DB_PATH"):
        TranscriptDatabase()
    assert not (cfg.tmp / "typo.db").exists()


def test_v2_engine_and_library_refuse_missing_db(cfg, monkeypatch):
    missing = cfg.tmp / "v2" / "transcripts.db"
    from ask_shorty_v2 import AskShortyV2

    with pytest.raises(config.DatabaseNotFoundError):
        AskShortyV2(db_path=str(missing))
    monkeypatch.setenv("ASK_SHORTY_DB_PATH", str(missing))
    lib = importlib.reload(importlib.import_module("library_app"))
    with pytest.raises(config.DatabaseNotFoundError):
        lib.get_db_connection()
    assert not missing.exists()


def test_create_opt_in(cfg, monkeypatch):
    a = cfg.tmp / "a" / "t.db"
    TranscriptDatabase(str(a), create=True)
    assert a.is_file()

    b = cfg.tmp / "b" / "t.db"
    monkeypatch.setenv("ASK_SHORTY_CREATE_DB", "1")
    TranscriptDatabase(str(b))
    assert b.is_file()
    monkeypatch.delenv("ASK_SHORTY_CREATE_DB")

    c = cfg.tmp / "c" / "t.db"
    cfg.write("create_db = true\n")
    TranscriptDatabase(str(c))
    assert c.is_file()
    cfg.write("")

    # explicit create=False wins over the setting
    monkeypatch.setenv("ASK_SHORTY_CREATE_DB", "1")
    with pytest.raises(config.DatabaseNotFoundError):
        TranscriptDatabase(str(cfg.tmp / "d" / "t.db"), create=False)


def test_init_db_command(cfg, monkeypatch, capsys):
    target = cfg.tmp / "new" / "transcripts.db"
    assert config.main(["init-db", str(target)]) == 0
    assert "Created" in capsys.readouterr().out
    with sqlite3.connect(target) as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"videos", "transcripts", "processing_queue"} <= tables
    # idempotent on an existing DB, and defaults to db_path
    monkeypatch.setenv("ASK_SHORTY_DB_PATH", str(target))
    assert config.main(["init-db"]) == 0
    assert "Schema checked" in capsys.readouterr().out


def test_existing_db_opens_without_opt_in(corpus_copy):
    assert TranscriptDatabase(corpus_copy["db_path"]).db_path == corpus_copy["db_path"]


# --------------------------------------------------------------------------- apps: host + startup


@pytest.fixture()
def app_mod(cfg, monkeypatch):
    monkeypatch.setenv("ASK_SHORTY_DB_PATH", str(cfg.tmp / "missing.db"))
    mod = sys.modules.get("ask_shorty_app")
    mod = importlib.reload(mod) if mod else importlib.import_module("ask_shorty_app")
    return mod


def test_ask_app_binds_localhost_by_default(app_mod, monkeypatch):
    calls = {}
    monkeypatch.setattr(app_mod, "init_app", lambda **k: None)
    monkeypatch.setattr(app_mod.app, "run", lambda **kw: calls.update(kw))
    app_mod.main()
    assert calls["host"] == "127.0.0.1" and calls["port"] == 5001
    monkeypatch.setenv("ASK_SHORTY_HOST", "0.0.0.0")
    app_mod.main()
    assert calls["host"] == "0.0.0.0"


def test_ask_app_missing_db_exits_78_without_creating_it(app_mod, cfg, monkeypatch):
    monkeypatch.setattr(app_mod.app, "run", lambda **kw: pytest.fail("must not serve"))
    with pytest.raises(SystemExit) as ei:
        app_mod.main()
    assert ei.value.code == config.EXIT_CONFIG == 78
    assert not (cfg.tmp / "missing.db").exists()
    assert not (cfg.tmp / "jobs").exists()


def test_grabber_main_binds_localhost_and_fails_on_missing_db(cfg, monkeypatch):
    import video_grabber as vg

    vg = importlib.reload(vg)
    monkeypatch.setenv("ASK_SHORTY_DATA_DIR", str(cfg.tmp / "data"))
    monkeypatch.setenv("ASK_SHORTY_DB_PATH", str(cfg.tmp / "missing.db"))
    monkeypatch.setattr(vg.app, "run", lambda **kw: pytest.fail("must not serve"))
    with pytest.raises(SystemExit) as ei:
        vg.main()
    assert ei.value.code == 78 and not (cfg.tmp / "missing.db").exists()

    db = cfg.tmp / "ok.db"
    TranscriptDatabase(str(db), create=True)
    vg = importlib.reload(vg)
    monkeypatch.setenv("ASK_SHORTY_DB_PATH", str(db))
    monkeypatch.setattr(vg, "rag", config.Lazy(lambda: object()))  # no model load
    calls = {}
    monkeypatch.setattr(vg.app, "run", lambda **kw: calls.update(kw))
    vg.main()
    assert calls["host"] == "127.0.0.1" and calls["port"] == 5000
    assert vg.db.db_path == str(db)


def test_library_main_binds_localhost(cfg, monkeypatch, corpus_copy):
    monkeypatch.setenv("ASK_SHORTY_DB_PATH", corpus_copy["db_path"])
    lib = importlib.reload(importlib.import_module("library_app"))
    calls = {}
    monkeypatch.setattr(lib.app, "run", lambda **kw: calls.update(kw))
    lib.main()
    assert calls["host"] == "127.0.0.1" and calls["port"] == 5002
    monkeypatch.setattr(lib, "DB_PATH", str(cfg.tmp / "missing.db"))
    with pytest.raises(SystemExit) as ei:
        lib.main()
    assert ei.value.code == 78 and not (cfg.tmp / "missing.db").exists()


# --------------------------------------------------------------------------- import side effects


def test_importing_apps_has_no_side_effects(tmp_path):
    """
    Importing the apps used to create the DB, data/grab_log.txt, a jobs dir and load
    the embedding model. Run in a fresh interpreter, pointed at paths that don't exist.
    """
    work = tmp_path / "cwd"
    work.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    data = tmp_path / "data"
    repo_data_before = (ROOT / "data").exists()
    env = {
        **os.environ,
        "ASK_SHORTY_DB_PATH": str(tmp_path / "db" / "transcripts.db"),
        "ASK_SHORTY_DATA_DIR": str(data),
        "ASK_SHORTY_V2": "1",
        "ASK_SHORTY_NO_DOTENV": "1",
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHONPATH": str(ROOT),
    }
    code = r"""
import sys
import video_grabber, ask_shorty_app, library_app, ask_shorty, ask_shorty_v2, batch_processor, config
assert repr(video_grabber.db) == "<Lazy unresolved>", video_grabber.db
assert repr(video_grabber.rag) == "<Lazy unresolved>", video_grabber.rag
assert video_grabber.db_path is None
assert repr(ask_shorty_app.db) == "<Lazy unresolved>"
assert ask_shorty_app._jobs_dir is None and not ask_shorty_app._initialized
assert ask_shorty_app._engine is None and ask_shorty_app._engine_v2 is None
import threading
assert not [t for t in threading.enumerate() if t.name == "v2-warmup"]
print("IMPORT-OK")
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=work, env=env, capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "IMPORT-OK" in r.stdout
    assert list(work.iterdir()) == []
    assert not (tmp_path / "db").exists()
    assert not data.exists()
    assert list(home.iterdir()) == []
    assert (ROOT / "data").exists() == repo_data_before


def test_lazy_proxy_resolves_once_and_can_be_replaced():
    made = []
    lz = config.Lazy(lambda: made.append(1) or type("O", (), {"x": 5})())
    assert repr(lz) == "<Lazy unresolved>" and made == []
    assert lz.x == 5 and lz.x == 5 and made == [1]
    lz.x = 7
    assert lz.x == 7


def test_no_hard_coded_user_paths_left():
    """Regression guard for the C:\\Users\\<name> defaults this branch removed."""
    pat = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+(?!<|\\?\{)", re.I)
    offenders = []
    for p in ROOT.rglob("*.py"):
        if any(part in (".git", ".venv", "node_modules") for part in p.parts):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    # tests/test_config.py writes a fake Windows path into a temp TOML on purpose
    offenders = [o for o in offenders if not o.startswith("tests/test_config.py")]
    assert offenders == []
