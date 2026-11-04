import sqlite3
import config
db = config.require_db(config.db_path())
conn = sqlite3.connect(db)
for task in ['shorty', 'synthetic_questions', 'entities', 'triples']:
    conn.execute("INSERT INTO processing_queue (video_id, task, status) VALUES ('ltB-4QDunls', '" + task + "', 'pending')")
conn.commit()
print('queued in shorty DB')
