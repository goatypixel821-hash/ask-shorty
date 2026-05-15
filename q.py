import sqlite3
conn = sqlite3.connect(r'data\transcripts.db')
for row in conn.execute("SELECT task, status, COUNT(*) FROM processing_queue GROUP BY task, status ORDER BY task, status"):
    print(row)
