"""
Normalize entity strings for global graph linking (lowercase, aliases, compact key).
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, Optional

# Built-in aliases (alias key -> canonical normalized form)
DEFAULT_ALIASES: Dict[str, str] = {
    "dcs-3000": "dcs3000",
    "dcs3000": "dcs3000",
    "red hook": "dcs3000",
    "redhook": "dcs3000",
    "cuda": "cuda",
    "pytorch": "pytorch",
    "py torch": "pytorch",
    "py-torch": "pytorch",
}


def _compact_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _strip_punct_keep_spaces(s: str) -> str:
    s = re.sub(r"[^\w\s]", " ", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def load_aliases_from_db(db_path: str) -> Dict[str, str]:
    """Load entity_alias table (alias -> canonical). Canonical is normalized separately."""
    out: Dict[str, str] = {}
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT alias, canonical FROM entity_alias")
            for row in cur.fetchall():
                a, c = row[0], row[1]
                if a and c:
                    ka = _compact_alnum(str(a))
                    if ka:
                        out[ka] = _compact_alnum(str(c)) or str(c).lower().strip()
    except sqlite3.OperationalError:
        pass
    return out


def normalize_entity(
    text: str,
    *,
    extra_aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Normalize a surface entity string.

    Returns:
        {"normalized": str, "original": str}
    """
    original = (text or "").strip()
    if not original:
        return {"normalized": "", "original": original}

    spaced = _strip_punct_keep_spaces(original)
    compact = _compact_alnum(spaced)

    aliases: Dict[str, str] = dict(DEFAULT_ALIASES)
    if extra_aliases:
        aliases.update(extra_aliases)

    # Try spaced lower, compact, and alias keys
    candidates = [
        spaced.lower(),
        compact,
        original.lower().strip(),
    ]
    normalized = compact
    for cand in candidates:
        if not cand:
            continue
        if cand in aliases:
            normalized = aliases[cand]
            break
        cc = _compact_alnum(cand)
        if cc in aliases:
            normalized = aliases[cc]
            break
    else:
        normalized = compact

    return {"normalized": normalized, "original": original}
