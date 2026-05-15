import sqlite3
for db in [
    r'C:\Users\number2\Desktop\shorty\data\transcripts.db',
    r'C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db'
]:
    conn = sqlite3.connect(db)
    r = conn.execute("SELECT title FROM videos WHERE video_id='nHDnyNzvF50'").fetchone()
    r2 = conn.execute("SELECT shorty FROM transcripts WHERE video_id='nHDnyNzvF50'").fetchone()
    print(db[-30:], 'has video:', bool(r), 'has shorty:', bool(r2 and r2[0]))
