import sqlite3
db = r'C:\Users\number2\Desktop\shorty\data\transcripts.db'
conn = sqlite3.connect(db)
rows = conn.execute("""
    SELECT v.video_id, v.title, v.watch_date, 
           CASE WHEN t.shorty IS NOT NULL AND t.shorty != '' THEN 'has shorty' ELSE 'no shorty' END as status
    FROM videos v
    LEFT JOIN transcripts t ON t.video_id = v.video_id
    WHERE lower(v.channel) LIKE '%sanborn%'
    ORDER BY v.watch_date DESC
""").fetchall()
print(f'Found {len(rows)} Sanborn videos:')
for r in rows:
    print(f'  {r[2] or "no date"}  {r[3]}  {r[1][:60]}')
    print(f'    https://youtube.com/watch?v={r[0]}')
