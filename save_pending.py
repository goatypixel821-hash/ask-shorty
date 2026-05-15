import sqlite3, csv
db = r'C:\Users\number2\Desktop\shorty\data\transcripts.db'
conn = sqlite3.connect(db)
rows = conn.execute("""
    SELECT DISTINCT pq.video_id, v.title, v.channel
    FROM processing_queue pq
    LEFT JOIN videos v ON v.video_id = pq.video_id
    WHERE pq.task = 'shorty' AND pq.status = 'pending'
    ORDER BY v.channel, v.title
""").fetchall()
with open(r'data\pending_shorty_review.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['video_id', 'title', 'channel'])
    w.writerows(rows)
print(f'Written {len(rows)} rows to data\pending_shorty_review.csv')
