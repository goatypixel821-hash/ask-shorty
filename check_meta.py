import sqlite3, json
db = r'C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db'
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
