"""
Creates data/reserved_eval_ids.json — a list of synthetic_question IDs
and question texts that are permanently held out from training data.

Run this once after each eval dataset is finalized. After that,
finetune_embeddings.py and finetune_crossencoder.py will skip questions
whose text OR ID appears in this file.

Captures all four eval question types:
  fk_   = fact_lookup  -> has numeric sq_id in filename
  pp_fk_ = paraphrase  -> references original fk sq_id
  tr_   = tricky       -> question text match
  sc_   = summary_comparison (cross-video, channel-level) -> question text match
"""
import json
import sqlite3
import re
from pathlib import Path

DB = r'C:\Users\number2\Desktop\youtube-history-viewer-copy\data\transcripts.db'
EVAL_RESULTS_DIR = Path(__file__).parent / 'eval_results'
OUT = Path(__file__).parent / 'data' / 'reserved_eval_ids.json'

conn = sqlite3.connect(DB)
c = conn.cursor()

reserved_sq_ids = set()
reserved_question_texts = set()
reserved_questions = []

eval_dirs = sorted(EVAL_RESULTS_DIR.iterdir(), reverse=True)
for d in eval_dirs[:3]:
    qdir = d / 'queries'
    if not qdir.exists():
        continue
    for jf in qdir.glob('*.json'):
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
            q = data.get('query', '').strip()
            if q:
                reserved_question_texts.add(q)
        except Exception:
            pass

        # fk_ and pp_fk_ -> extract numeric sq_id
        m = re.search(r'_(\d+)\.json$', jf.name)
        if m:
            sq_id = int(m.group(1))
            reserved_sq_ids.add(sq_id)

print(f"Reserved sq_ids (fk/pp_fk): {len(reserved_sq_ids)}")
print(f"Reserved question texts (all types): {len(reserved_question_texts)}")

# Look up in DB to verify
if reserved_sq_ids:
    placeholders = ','.join('?' * len(reserved_sq_ids))
    c.execute(f"SELECT id FROM synthetic_questions WHERE id IN ({placeholders})",
              list(reserved_sq_ids))
    found_ids = set(r[0] for r in c.fetchall())
    print(f"Verified {len(found_ids)} sq_ids exist in DB")

# Also find sq_ids for questions matched by text
if reserved_question_texts:
    c.execute("SELECT id, question FROM synthetic_questions WHERE video_id != 'test123'")
    for row_id, qtext in c.fetchall():
        if qtext.strip() in reserved_question_texts:
            reserved_sq_ids.add(row_id)

print(f"Total reserved sq_ids (including text matches): {len(reserved_sq_ids)}")

conn.close()

out_data = {
    "description": "Questions reserved for eval. Do NOT use in training.",
    "sq_ids": sorted(reserved_sq_ids),
    "question_texts": sorted(reserved_question_texts)[:200],  # cap at 200 for readability
    "count_sq_ids": len(reserved_sq_ids),
    "count_questions": len(reserved_question_texts),
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out_data, f, indent=2, ensure_ascii=False)

print(f"Written -> {OUT}")
print(f"Future training will exclude these {len(reserved_sq_ids)} sq_ids / {len(reserved_question_texts)} question texts.")
