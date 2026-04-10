"""Quick probe: which DB has facts / global_facts for cross-video tests."""
import sqlite3
from pathlib import Path

CANDIDATES = [
    Path(__file__).resolve().parent.parent / "data" / "transcripts.db",
    Path.home() / "Desktop" / "youtube-history-viewer-copy" / "data" / "transcripts.db",
]


def main() -> None:
    for dbpath in CANDIDATES:
        if not dbpath.exists():
            print("missing", dbpath)
            continue
        conn = sqlite3.connect(str(dbpath))
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = {r[0] for r in cur.fetchall()}
        print("---", dbpath, "---")
        for t in ("facts", "global_facts", "segments", "events"):
            print(f"  {t}:", "yes" if t in tables else "no")
        if "facts" in tables:
            cur.execute("SELECT COUNT(*) FROM facts")
            print("  facts count:", cur.fetchone()[0])
            cur.execute("SELECT COUNT(DISTINCT video_id) FROM facts")
            print("  distinct videos:", cur.fetchone()[0])
        if "global_facts" in tables:
            cur.execute("SELECT COUNT(*) FROM global_facts")
            print("  global_facts count:", cur.fetchone()[0])
        conn.close()


if __name__ == "__main__":
    main()
