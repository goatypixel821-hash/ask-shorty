import sqlite3
import config
db = config.require_db(config.db_path())
conn = sqlite3.connect(db)

# Check events table schema
print('=== EVENTS SCHEMA ===')
for row in conn.execute('PRAGMA table_info(events)').fetchall():
    print(row)

print('\n=== EVENTS SAMPLE ===')
rows = conn.execute('SELECT * FROM events LIMIT 3').fetchall()
for row in rows:
    print(row)

print('\n=== SEGMENTS SCHEMA ===')
for row in conn.execute('PRAGMA table_info(segments)').fetchall():
    print(row)
