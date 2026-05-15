"""
Resets processing_queue rows to 'pending' for videos that show 'completed'
in the queue but have no actual data saved (Shorty, synthetic_questions, triples).

Also resets any 'started' stale rows.

Run this before batch_processor.py to make sure all missing data gets regenerated.
"""
import sqlite3
from pathlib import Path

DB = r'C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db'

conn = sqlite3.connect(DB)
c = conn.cursor()

print("Scanning for queue rows to reset...")

# 1. Reset stale 'started' rows (orphaned from crashed runs)
c.execute("UPDATE processing_queue SET status = 'pending' WHERE status = 'started'")
stale = c.rowcount
print(f"  Stale 'started' -> pending: {stale}")

# 2. Reset 'completed' shorty rows for videos that have no shorty text
c.execute("""
    UPDATE processing_queue SET status = 'pending'
    WHERE task = 'shorty' AND status IN ('completed', 'permanently_failed')
    AND video_id IN (
        SELECT t.video_id FROM transcripts t
        WHERE t.shorty IS NULL OR t.shorty = ''
    )
    AND video_id != 'test123'
""")
no_shorty = c.rowcount
print(f"  Shorty 'completed' but empty -> pending: {no_shorty}")

# 3. Reset 'completed' entities rows for videos that have no entities
c.execute("""
    UPDATE processing_queue SET status = 'pending'
    WHERE task = 'entities' AND status IN ('completed', 'permanently_failed')
    AND video_id IN (
        SELECT t.video_id FROM transcripts t
        WHERE t.shorty IS NOT NULL AND t.shorty != ''
        AND NOT EXISTS (SELECT 1 FROM entities e WHERE e.video_id = t.video_id)
    )
    AND video_id != 'test123'
""")
no_entities = c.rowcount
print(f"  Entities 'completed' but empty -> pending: {no_entities}")

# 4. Reset 'completed' synthetic_questions rows for videos with no synq data
c.execute("""
    UPDATE processing_queue SET status = 'pending'
    WHERE task = 'synthetic_questions' AND status IN ('completed', 'permanently_failed')
    AND video_id IN (
        SELECT t.video_id FROM transcripts t
        WHERE t.shorty IS NOT NULL AND t.shorty != ''
        AND NOT EXISTS (
            SELECT 1 FROM synthetic_questions sq
            WHERE sq.video_id = t.video_id
            AND sq.question NOT LIKE 'What is the title%'
            AND sq.question NOT LIKE 'What is the topic%'
        )
    )
    AND video_id != 'test123'
""")
no_synq = c.rowcount
print(f"  Synthetic_questions 'completed' but empty -> pending: {no_synq}")

# 5. Reset 'completed' triples rows for videos with shorty but no triples
c.execute("""
    UPDATE processing_queue SET status = 'pending'
    WHERE task = 'triples' AND status IN ('completed', 'permanently_failed')
    AND video_id IN (
        SELECT t.video_id FROM transcripts t
        WHERE t.shorty IS NOT NULL AND t.shorty != ''
        AND NOT EXISTS (SELECT 1 FROM facts f WHERE f.video_id = t.video_id)
    )
    AND video_id != 'test123'
""")
no_triples = c.rowcount
print(f"  Triples 'completed' but empty -> pending: {no_triples}")

conn.commit()

# Show final queue state
print()
c.execute("""
    SELECT task, status, COUNT(*) FROM processing_queue
    GROUP BY task, status ORDER BY task, status
""")
print("Queue after reset:")
for r in c.fetchall():
    if r[2] > 0:
        print(f"  {r[0]:30s} {r[1]:20s} {r[2]}")

conn.close()
print()
print(f"Total rows reset: {stale + no_shorty + no_entities + no_synq + no_triples}")
print("Now run batch_processor.py to process all pending tasks.")
