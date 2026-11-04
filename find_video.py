import sqlite3
import config
for db in [str(config.repo_db_default()), str(config.viewer_db_default())]:
    conn = sqlite3.connect(config.require_db(db))
    r = conn.execute("SELECT title FROM videos WHERE video_id='nHDnyNzvF50'").fetchone()
    r2 = conn.execute("SELECT shorty FROM transcripts WHERE video_id='nHDnyNzvF50'").fetchone()
    print(db[-30:], 'has video:', bool(r), 'has shorty:', bool(r2 and r2[0]))
