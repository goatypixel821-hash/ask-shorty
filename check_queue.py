import sqlite3
db = r'C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db'
conn = sqlite3.connect(db)
c = conn.cursor()

# Check videos table columns
c.execute("PRAGMA table_info(videos)")
print("videos columns:", [r[1] for r in c.fetchall()])
c.execute("PRAGMA table_info(transcripts)")
print("transcripts columns:", [r[1] for r in c.fetchall()])

print()
print("=== PROCESSING QUEUE STATUS ===")
c.execute("SELECT task, status, COUNT(*) FROM processing_queue GROUP BY task, status ORDER BY task, status")
for r in c.fetchall():
    print(f"  {r[0]:30s} {r[1]:15s} {r[2]}")

print()
print("=== VIDEOS MISSING SYNTHETIC QUESTIONS (sample 5) ===")
c.execute("""
    SELECT t.video_id,
           CASE WHEN t.shorty IS NOT NULL AND t.shorty != '' THEN 'has_shorty' ELSE 'no_shorty' END
    FROM transcripts t
    WHERE NOT EXISTS (SELECT 1 FROM synthetic_questions sq WHERE sq.video_id = t.video_id)
    AND t.video_id != 'test123'
    LIMIT 5
""")
for r in c.fetchall():
    print(f"  {r[0]}  [{r[1]}]")

print()
print("=== VIDEOS MISSING TRIPLES ONLY (sample 5) ===")
c.execute("""
    SELECT t.video_id, LENGTH(t.shorty)
    FROM transcripts t
    WHERE t.shorty IS NOT NULL AND t.shorty != ''
    AND NOT EXISTS (SELECT 1 FROM facts f WHERE f.video_id = t.video_id)
    AND t.video_id != 'test123'
    ORDER BY LENGTH(t.shorty) ASC
    LIMIT 5
""")
for r in c.fetchall():
    print(f"  {r[0]}  shorty_len={r[1]}")

print()
print("=== HOW EVAL QUESTIONS ARE TAGGED ===")
# Check if there's a category or source column in synthetic_questions
c.execute("PRAGMA table_info(synthetic_questions)")
sq_cols = [r[1] for r in c.fetchall()]
print("synthetic_questions columns:", sq_cols)

print()
print("=== SAMPLE EVAL QUERY ARTIFACT ===")
import json
from pathlib import Path
eval_dirs = sorted(Path(r'C:\Users\number2\Desktop\shorty\eval_results').iterdir(), reverse=True)
for d in eval_dirs[:3]:
    qdir = d / 'queries'
    if qdir.exists():
        files = list(qdir.glob('*.json'))
        if files:
            s = json.loads(files[0].read_text(encoding='utf-8'))
            print(f"From {d.name}:")
            print(f"  question: {s.get('question','')[:80]}")
            print(f"  question_type: {s.get('question_type','?')}")
            print(f"  relevant_ids: {s.get('relevant_ids', s.get('ground_truth', '?'))[:3]}")
            break

conn.close()
