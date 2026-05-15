import sqlite3
db = r'C:\Users\number2\Desktop\shorty\data\transcripts.db'
conn = sqlite3.connect(db)

# Show segment count and a sample
print('=== SEGMENTS ===')
r = conn.execute('SELECT COUNT(*) FROM segments').fetchone()
print(f'Total segments: {r[0]}')

rows = conn.execute('''
    SELECT s.video_id, v.title, s.start_time, s.end_time, s.summary
    FROM segments s
    JOIN videos v ON v.video_id = s.video_id
    LIMIT 5
''').fetchall()
for row in rows:
    print(f'\nVideo: {row[1][:60]}')
    print(f'Time: {row[2]} - {row[3]}')
    print(f'Summary: {row[4][:200]}')

print('\n=== EVENTS ===')
r = conn.execute('SELECT COUNT(*) FROM events').fetchone()
print(f'Total events: {r[0]}')

rows = conn.execute('''
    SELECT e.video_id, v.title, e.label, e.summary
    FROM events e
    JOIN videos v ON v.video_id = e.video_id
    LIMIT 5
''').fetchall()
for row in rows:
    print(f'\nVideo: {row[1][:60]}')
    print(f'Event: {row[2]}')
    print(f'Summary: {row[3][:200]}')
