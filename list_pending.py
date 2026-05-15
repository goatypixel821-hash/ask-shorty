import sqlite3
db = r'C:\Users\number2\Desktop\shorty\data\transcripts.db'
conn = sqlite3.connect(db)
rows = conn.execute("""
    SELECT DISTINCT pq.video_id, v.title, v.channel
    FROM processing_queue pq
    LEFT JOIN videos v ON v.video_id = pq.video_id
    WHERE pq.task = 'shorty' AND pq.status = 'pending'
    ORDER BY v.title
""").fetchall()
print(f'Total: {len(rows)}')
for r in rows[:20]:
    print(r)
