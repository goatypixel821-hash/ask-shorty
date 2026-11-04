#!/usr/bin/env python3
"""
Ask Shorty configuration: one place that decides where the data lives and which
LLM provider/models are used.

Every setting is resolved, at call time, in this order:

  1. environment variable (the shell, or ``.env`` next to this file)
  2. ``ask_shorty.local.toml`` next to this file (gitignored; copy
     ``ask_shorty.local.toml.example``), or the file named by ``ASK_SHORTY_CONFIG``
  3. the built-in default

Settings
--------
data_dir          ASK_SHORTY_DATA_DIR       <repo>/data
db_path           ASK_SHORTY_DB_PATH        <data_dir>/transcripts.db
full_db_path      ASK_SHORTY_FULL_DB_PATH,  ~/Desktop/youtube-history-viewer-copy/data/transcripts.db
                  ASK_SHORTY_FULL_DB,       if that file exists, otherwise db_path
                  then ASK_SHORTY_DB_PATH
chroma_path       ASK_SHORTY_CHROMA_PATH    <data_dir>/transcript_chroma_finetuned
full_chroma_path  ASK_SHORTY_FULL_CHROMA_PATH, ~/Desktop/youtube-history-viewer-copy/data/transcript_chroma_new
                  then ASK_SHORTY_CHROMA_PATH  if that dir exists, otherwise chroma_path
provider          SHORTY_PROVIDER           openrouter            (anthropic | openrouter)
shorty_model      SHORTY_MODEL              per provider (see DEFAULT_SHORTY_MODELS)
answer_model      ANSWER_MODEL              claude-sonnet-4-20250514
openrouter_model  OPENROUTER_MODEL          qwen/qwen-2.5-72b-instruct
host              ASK_SHORTY_HOST           127.0.0.1   (0.0.0.0 = reachable from the LAN)
create_db         ASK_SHORTY_CREATE_DB      false       (allow creating a missing DB)

``db_path`` is the working DB (grabber, batch processor, V2, library, ask jobs).
``full_db_path`` is the full-library DB used by V1 Ask, the observatory, clusters,
courses and most check_*.py scripts. With nothing configured on a machine that
has no youtube-history-viewer-copy, both are the same file.

A database that doesn't exist is an error (``DatabaseNotFoundError``), not a
silently created empty file. To create a new one on purpose::

    python config.py init-db            # creates db_path
    python config.py init-db PATH

``python config.py`` prints every effective setting and where it came from.

Importing this module has no side effects: nothing is read until a setting is asked for.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_CONFIG_NAME = "ask_shorty.local.toml"

DEFAULT_SHORTY_MODELS = {
    "anthropic": "claude-3-haiku-20240307",
    "openrouter": "anthropic/claude-3-haiku",
}
DEFAULT_ANSWER_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen-2.5-72b-instruct"
DEFAULT_HOST = "127.0.0.1"

# Exit status for "configuration problem, don't restart me" (sysexits.h EX_CONFIG).
# start_ask.py / start_grabber.py stop instead of restart-looping on it.
EXIT_CONFIG = 78

_TRUE = ("1", "true", "yes", "on")


class ConfigError(RuntimeError):
    pass


class DatabaseNotFoundError(ConfigError, FileNotFoundError):
    """The configured transcripts DB doesn't exist (and creating one wasn't asked for)."""

    def __init__(self, path: Any, source: str = ""):
        self.path = str(path)
        self.source = source
        src = f" (from {source})" if source else ""
        super().__init__(
            f"Ask Shorty database not found: {self.path}{src}\n"
            f"  Point ASK_SHORTY_DB_PATH (working DB) or ASK_SHORTY_FULL_DB_PATH (full library DB)\n"
            f"  at your existing transcripts.db, in the environment, .env or {LOCAL_CONFIG_NAME}.\n"
            f"  To create a NEW empty database there instead, run:  python config.py init-db \"{self.path}\"\n"
            f"  (or set ASK_SHORTY_CREATE_DB=1 for one run). Run `python config.py` to see all settings."
        )


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

_dotenv_lock = threading.Lock()
_dotenv_done = False


def _load_dotenv_once() -> None:
    """``.env`` values count as environment variables; the real environment wins."""
    global _dotenv_done
    if _dotenv_done:
        return
    with _dotenv_lock:
        if _dotenv_done:
            return
        _dotenv_done = True
        load_project_dotenv(override=False)


def load_project_dotenv(override: bool = False) -> None:
    """
    Load ``<repo>/.env`` (no-op without python-dotenv or the file, or when
    ASK_SHORTY_NO_DOTENV=1, which the test suite sets). Modules that read feature
    flags into constants call this with override=True, as they always have.
    """
    if (os.environ.get("ASK_SHORTY_NO_DOTENV") or "").strip().lower() in _TRUE:
        return
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=override)


def local_config_path() -> Path:
    raw = (os.environ.get("ASK_SHORTY_CONFIG") or "").strip()
    return Path(raw).expanduser() if raw else REPO_ROOT / LOCAL_CONFIG_NAME


_file_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _read_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover - 3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as e:
            raise ConfigError(
                f"{path} exists but this Python can't read TOML: use Python 3.11+ or `pip install tomli`."
            ) from e
    with open(path, "rb") as f:
        try:
            return tomllib.load(f)
        except Exception as e:
            raise ConfigError(f"Can't parse {path}: {e}") from e


def file_settings() -> Dict[str, Any]:
    """Settings from the local TOML file ({} if there is none). Re-read when it changes."""
    path = local_config_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        if (os.environ.get("ASK_SHORTY_CONFIG") or "").strip():
            raise ConfigError(f"ASK_SHORTY_CONFIG points at a missing file: {path}")
        return {}
    key = str(path)
    hit = _file_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    data = _read_toml(path)
    # Accept a flat file or an [ask_shorty] table.
    if isinstance(data.get("ask_shorty"), dict):
        data = {**{k: v for k, v in data.items() if k != "ask_shorty"}, **data["ask_shorty"]}
    unknown = sorted(set(data) - set(SETTINGS))
    if unknown:
        raise ConfigError(f"Unknown setting(s) in {path}: {', '.join(unknown)}. Known: {', '.join(SETTINGS)}")
    _file_cache[key] = (mtime, data)
    return data


# ---------------------------------------------------------------------------
# Settings table
# ---------------------------------------------------------------------------

def _home_viewer(*parts: str) -> Path:
    return Path.home() / "Desktop" / "youtube-history-viewer-copy" / "data" / Path(*parts)


def _default_full_db() -> Tuple[Any, str]:
    viewer = _home_viewer("transcripts.db")
    if viewer.is_file():
        return viewer, f"default ({viewer} exists)"
    return resolve("db_path")[0], "default (same as db_path)"


def _default_full_chroma() -> Tuple[Any, str]:
    viewer = _home_viewer("transcript_chroma_new")
    if viewer.is_dir():
        return viewer, f"default ({viewer} exists)"
    return resolve("chroma_path")[0], "default (same as chroma_path)"


def _default_shorty_model() -> Tuple[Any, str]:
    return DEFAULT_SHORTY_MODELS.get(resolve("provider")[0], ""), "default for provider"


_Default = Callable[[], Tuple[Any, str]]


def _const(v: Any) -> _Default:
    return lambda: (v, "default")


# name -> (env vars in priority order, kind, default factory)
SETTINGS: Dict[str, Tuple[List[str], str, _Default]] = {
    "data_dir": (["ASK_SHORTY_DATA_DIR"], "path", lambda: (REPO_ROOT / "data", "default (<repo>/data)")),
    "db_path": (["ASK_SHORTY_DB_PATH"], "path", lambda: (resolve("data_dir")[0] / "transcripts.db", "default (<data_dir>/transcripts.db)")),
    # ASK_SHORTY_FULL_DB: the name build_courses.py already used.
    "full_db_path": (["ASK_SHORTY_FULL_DB_PATH", "ASK_SHORTY_FULL_DB", "ASK_SHORTY_DB_PATH"], "path", _default_full_db),
    "chroma_path": (["ASK_SHORTY_CHROMA_PATH"], "path", lambda: (resolve("data_dir")[0] / "transcript_chroma_finetuned", "default (<data_dir>/transcript_chroma_finetuned)")),
    "full_chroma_path": (["ASK_SHORTY_FULL_CHROMA_PATH", "ASK_SHORTY_CHROMA_PATH"], "path", _default_full_chroma),
    "provider": (["SHORTY_PROVIDER"], "provider", _const("openrouter")),
    "shorty_model": (["SHORTY_MODEL"], "str", _default_shorty_model),
    "answer_model": (["ANSWER_MODEL"], "str", _const(DEFAULT_ANSWER_MODEL)),
    "openrouter_model": (["OPENROUTER_MODEL"], "str", _const(DEFAULT_OPENROUTER_MODEL)),
    "host": (["ASK_SHORTY_HOST"], "str", _const(DEFAULT_HOST)),
    "create_db": (["ASK_SHORTY_CREATE_DB"], "bool", _const(False)),
}


def _coerce(name: str, kind: str, value: Any, source: str) -> Any:
    if kind == "path":
        return Path(os.path.expandvars(str(value))).expanduser()
    if kind == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in _TRUE
    s = str(value).strip()
    if kind == "provider":
        s = s.lower()
        if s not in DEFAULT_SHORTY_MODELS:
            raise ConfigError(f"Invalid provider {s!r} (from {source}). Use one of: {sorted(DEFAULT_SHORTY_MODELS)}")
    return s


def resolve(name: str) -> Tuple[Any, str]:
    """Return ``(value, source)`` for a setting; ``source`` says where it came from."""
    if name not in SETTINGS:
        raise KeyError(name)
    _load_dotenv_once()
    env_vars, kind, default = SETTINGS[name]
    for var in env_vars:
        raw = (os.environ.get(var) or "").strip()
        if raw:
            return _coerce(name, kind, raw, f"env {var}"), f"env {var}"
    fs = file_settings()
    if name in fs and str(fs[name]).strip() != "":
        src = f"{local_config_path().name} {name}"
        return _coerce(name, kind, fs[name], src), src
    value, src = default()
    return _coerce(name, kind, value, src), src


def get(name: str) -> Any:
    return resolve(name)[0]


# Convenience accessors -------------------------------------------------------

def data_dir() -> Path:
    return get("data_dir")


def db_path() -> Path:
    return get("db_path")


def full_db_path() -> Path:
    return get("full_db_path")


def chroma_path() -> Path:
    return get("chroma_path")


def full_chroma_path() -> Path:
    return get("full_chroma_path")


def provider() -> str:
    return get("provider")


def shorty_model() -> str:
    return get("shorty_model")


def answer_model() -> str:
    return get("answer_model")


def openrouter_model() -> str:
    return get("openrouter_model")


def host() -> str:
    return get("host")


def create_db_allowed() -> bool:
    return bool(get("create_db"))


# Fixed defaults that deliberately ignore ASK_SHORTY_DB_PATH (for tools that compare
# the two DBs, e.g. merge_transcript_dbs.py / find_video.py).
def repo_db_default() -> Path:
    return data_dir() / "transcripts.db"


def viewer_db_default() -> Path:
    return _home_viewer("transcripts.db")


# ---------------------------------------------------------------------------
# Database existence
# ---------------------------------------------------------------------------

def _source_for(path: Any) -> str:
    try:
        target = Path(str(path)).expanduser().resolve()
    except OSError:
        return ""
    for name in ("db_path", "full_db_path"):
        try:
            value, src = resolve(name)
            if Path(value).resolve() == target:
                return f"{name} <- {src}"
        except Exception:
            pass
    return ""


def require_db(path: Any = None, *, create: Optional[bool] = None) -> str:
    """
    Return ``path`` (default: db_path) as a string if the DB file exists.

    If it doesn't: create its directory and return it when ``create`` is true (or,
    when ``create`` is None, when ASK_SHORTY_CREATE_DB / create_db allows it);
    otherwise raise ``DatabaseNotFoundError`` naming the path and the settings.
    """
    p = Path(str(path)).expanduser() if path is not None else db_path()
    if p.is_file():
        return str(p)
    if p.exists():
        raise ConfigError(f"Ask Shorty database path is not a file: {p}")
    if create is None:
        create = create_db_allowed()
    if not create:
        raise DatabaseNotFoundError(p, _source_for(p))
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def init_db(path: Any = None) -> Path:
    """Explicitly create (or migrate) a transcripts DB with the full schema."""
    from transcript_database import TranscriptDatabase

    p = Path(str(path)).expanduser() if path is not None else db_path()
    TranscriptDatabase(str(p), create=True)
    return p


# ---------------------------------------------------------------------------
# Small helpers for modules that used to do work at import time
# ---------------------------------------------------------------------------

class Lazy:
    """
    Proxy that builds its target on first attribute access, so a module can keep
    ``db = Lazy(lambda: TranscriptDatabase(...))`` without opening anything at import.
    Tests (and callers) can still replace the module attribute with a real object.
    """

    def __init__(self, factory: Callable[[], Any]):
        object.__setattr__(self, "_factory", factory)
        object.__setattr__(self, "_obj", None)
        object.__setattr__(self, "_lock", threading.Lock())

    def _target(self) -> Any:
        obj = object.__getattribute__(self, "_obj")
        if obj is None:
            with object.__getattribute__(self, "_lock"):
                obj = object.__getattribute__(self, "_obj")
                if obj is None:
                    obj = object.__getattribute__(self, "_factory")()
                    object.__setattr__(self, "_obj", obj)
        return obj

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target(), name, value)

    def __repr__(self) -> str:
        obj = object.__getattribute__(self, "_obj")
        return f"<Lazy {'unresolved' if obj is None else repr(obj)}>"


def describe() -> List[Tuple[str, str, str]]:
    rows = []
    for name in SETTINGS:
        try:
            value, src = resolve(name)
            rows.append((name, str(value), src))
        except Exception as e:  # show the problem instead of hiding the rest
            rows.append((name, f"ERROR: {e}", ""))
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Show Ask Shorty settings, or create a new database.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("show", help="print effective settings (default)")
    p_init = sub.add_parser("init-db", help="create a NEW transcripts DB (default: db_path)")
    p_init.add_argument("path", nargs="?", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "init-db":
        target = Path(args.path).expanduser() if args.path else db_path()
        existed = target.is_file()
        init_db(target)
        print(f"{'Schema checked' if existed else 'Created'}: {target.resolve()}")
        return 0

    cfg = local_config_path()
    print(f"Ask Shorty settings (local config file: {cfg}{'' if cfg.is_file() else ' - not present'})")
    rows = describe()
    w = max(len(r[0]) for r in rows)
    for name, value, src in rows:
        print(f"  {name:<{w}}  {value}" + (f"    [{src}]" if src else ""))
    for name in ("db_path", "full_db_path"):
        try:
            p = get(name)
            print(f"  {name} {'exists' if Path(p).is_file() else 'MISSING'}: {p}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
