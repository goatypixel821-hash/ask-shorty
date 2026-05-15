import sqlite3
db = r'C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db'
conn = sqlite3.connect(db)
r = conn.execute("SELECT COUNT(*), COUNT(watch_date), SUM(CASE WHEN watch_date IS NOT NULL AND watch_date != '' THEN 1 ELSE 0 END), MIN(watch_date), MAX(watch_date) FROM videos").fetchone()
print('total:', r[0], 'has_watch_date:', r[1], 'nonempty:', r[2])
print('earliest:', r[3], 'latest:', r[4])
