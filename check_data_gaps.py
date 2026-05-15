"""
Diagnostic: check eval contamination, missing triples, processing gaps.
"""
import sqlite3, json
from pathlib import Path

DB = r'C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db'
conn = sqlite3.connect(DB)
c = conn.cursor()

print("=" * 60)
print("TABLES")
print("=" * 60)
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in c.fetchall()]
print(tables)

print()
print("=" * 60)
print("COUNTS PER TABLE")
print("=" * 60)
for t in tables:
    try:
        c.execute(f"SELECT COUNT(*) FROM {t}")
        print(f"  {t}: {c.fetchone()[0]}")
    except Exception as e:
        print(f"  {t}: ERROR - {e}")

print()
print("=" * 60)
print("EVAL DATASET SOURCE")
print("=" * 60)
# Check where eval questions come from
eval_files = list(Path(r'C:\Users\number2\Desktop\shorty\eval_dataset').rglob('*.json')) + \
             list(Path(r'C:\Users\number2\Desktop\shorty\data').glob('eval_*.json')) + \
             list(Path(r'C:\Users\number2\Desktop\shorty\data').glob('eval_*.jsonl'))
print("Eval dataset files found:", eval_files[:5])

# Check synthetic_questions overlap with eval
c.execute("SELECT question, video_id FROM synthetic_questions LIMIT 5")
rows = c.fetchall()
print("Sample synthetic_questions:", [(r[0][:60], r[1]) for r in rows])

print()
print("=" * 60)
print("PROCESSING GAPS")
print("=" * 60)
# Videos with Shorty but no triples
c.execute("""
    SELECT COUNT(*) FROM transcripts t
    WHERE t.shorty IS NOT NULL AND t.shorty != ''
    AND NOT EXISTS (SELECT 1 FROM facts f WHERE f.video_id = t.video_id)
""")
no_triples = c.fetchone()[0]
print(f"Videos WITH Shorty but NO triples: {no_triples}")

# Videos with no Shorty at all
c.execute("""
    SELECT COUNT(*) FROM transcripts t
    WHERE (t.shorty IS NULL OR t.shorty = '')
""")
no_shorty = c.fetchone()[0]
print(f"Videos with NO Shorty: {no_shorty}")

# Videos with no synthetic questions
c.execute("""
    SELECT COUNT(*) FROM transcripts t
    WHERE NOT EXISTS (SELECT 1 FROM synthetic_questions sq WHERE sq.video_id = t.video_id)
""")
no_synq = c.fetchone()[0]
print(f"Videos with NO synthetic questions: {no_synq}")

# Videos fully processed (Shorty + triples + synq)
c.execute("""
    SELECT COUNT(*) FROM transcripts t
    WHERE t.shorty IS NOT NULL AND t.shorty != ''
    AND EXISTS (SELECT 1 FROM facts f WHERE f.video_id = t.video_id)
    AND EXISTS (SELECT 1 FROM synthetic_questions sq WHERE sq.video_id = t.video_id)
""")
fully_done = c.fetchone()[0]
print(f"Videos FULLY processed (Shorty + triples + synq): {fully_done}")

c.execute("SELECT COUNT(*) FROM transcripts")
total = c.fetchone()[0]
print(f"Total videos: {total}")

print()
print("=" * 60)
print("EVAL QUESTION SOURCE CHECK")
print("=" * 60)
# Check build_eval_dataset to understand where eval questions come from
eval_ds_path = Path(r'C:\Users\number2\Desktop\shorty\data\eval_dataset.json')
if eval_ds_path.exists():
    with open(eval_ds_path) as f:
        data = json.load(f)
    print(f"eval_dataset.json: {len(data)} questions")
    # Check how many questions appear in synthetic_questions
    c.execute("SELECT question FROM synthetic_questions")
    all_synq = set(r[0].strip() for r in c.fetchall())
    overlap = sum(1 for q in data if q.get('question','').strip() in all_synq)
    print(f"Overlap with synthetic_questions: {overlap} / {len(data)}")
else:
    print("No data/eval_dataset.json found")

# Check the eval_results queries folder for a sample
eval_dirs = sorted(Path(r'C:\Users\number2\Desktop\shorty\eval_results').iterdir(), reverse=True)
if eval_dirs:
    latest = eval_dirs[0]
    q_files = list((latest / 'queries').glob('*.json')) if (latest / 'queries').exists() else []
    if q_files:
        sample = json.loads(q_files[0].read_text(encoding='utf-8'))
        q = sample.get('question','')
        qtype = sample.get('question_type','')
        print(f"\nSample eval question: [{qtype}] {q[:80]}")
        in_synq = q.strip() in all_synq if 'all_synq' in dir() else '?'
        print(f"  In synthetic_questions? {in_synq}")

conn.close()
print("\nDone.")
