import sqlite3, json
import config
db = config.require_db(config.full_db_path())
conn = sqlite3.connect(db)
rows = conn.execute("SELECT video_id, json_metadata FROM videos WHERE json_metadata IS NOT NULL LIMIT 3").fetchall()
for vid, meta in rows:
    print('video_id:', vid)
    try:
        d = json.loads(meta)
        print(json.dumps(d, indent=2)[:800])
    except:
        print(meta[:400])
    print()
