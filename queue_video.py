import sqlite3
db = r'C:\Users\number2\Desktop\shorty\data\transcripts.db'
conn = sqlite3.connect(db)
for task in ['shorty', 'synthetic_questions', 'entities', 'triples']:
    conn.execute("INSERT INTO processing_queue (video_id, task, status) VALUES ('ltB-4QDunls', '" + task + "', 'pending')")
conn.commit()
print('queued in shorty DB')
