import sqlite3
db = r'C:\Users\number2\Desktop\shorty\data\transcripts.db'
conn = sqlite3.connect(db)
for vid in ['LwcJM51BDUM', 'cgp-gfSbmNE']:
    r = conn.execute("SELECT shorty FROM transcripts WHERE video_id=?", (vid,)).fetchone()
    print(vid, 'has shorty:', bool(r and r[0]))
    if r and r[0]:
        print(r[0][:150])
    print()
