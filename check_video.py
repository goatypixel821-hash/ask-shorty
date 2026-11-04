import sqlite3
import config
db = config.require_db(config.db_path())
conn = sqlite3.connect(db)
r = conn.execute("SELECT shorty FROM transcripts WHERE video_id='nHDnyNzvF50'").fetchone()
print('has shorty:', bool(r and r[0]))
if r and r[0]:
    print(r[0][:200])
