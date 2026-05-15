import sqlite3
from collections import Counter
db = r'C:\Users\number2\Desktop\shorty\data\transcripts.db'
conn = sqlite3.connect(db)
rows = conn.execute("""
    SELECT channel, COUNT(*) as cnt 
    FROM videos 
    WHERE channel IS NOT NULL AND channel != ''
    AND video_id IN (SELECT video_id FROM transcripts WHERE shorty IS NOT NULL AND shorty != '')
    GROUP BY channel 
    ORDER BY cnt DESC 
    LIMIT 50
""").fetchall()
print('Top 50 channels with Shorties:')
for channel, cnt in rows:
    print(f'  {cnt:4d}  {channel}')
