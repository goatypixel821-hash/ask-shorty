import sqlite3
from datetime import datetime, timedelta
db = r'C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db'
conn = sqlite3.connect(db)
cutoff = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
r = conn.execute("SELECT COUNT(*) FROM transcripts WHERE created_at > ?", (cutoff,)).fetchone()
print('transcripts added in last hour:', r[0])
r2 = conn.execute("SELECT COUNT(*) FROM processing_queue WHERE created_at > ?", (cutoff,)).fetchone()
print('queue tasks added in last hour:', r2[0])
