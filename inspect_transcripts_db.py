import sqlite3

import config

DB_PATH = str(config.full_db_path())


def main() -> None:
    con = sqlite3.connect(config.require_db(DB_PATH))
    cur = con.cursor()
    tables = [row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
    print("tables:", tables)
    for table in tables:
        cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
        print(f"\n[{table}] columns:")
        for col in cols:
            print(f"  - {col[1]} ({col[2]})")
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  rows: {count}")
    con.close()


if __name__ == "__main__":
    main()
