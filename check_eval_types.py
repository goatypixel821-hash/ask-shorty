import json
from pathlib import Path

qdir = Path('eval_results/20260412_041405/queries')
prefixes = {}
for f in sorted(qdir.glob('*.json')):
    prefix = f.name.split('_')[0]
    prefixes[prefix] = prefixes.get(prefix, 0) + 1
print('Query file prefixes:', prefixes)

seen = set()
for prefix in sorted(prefixes):
    files = list(qdir.glob(f'{prefix}_*.json'))
    if files and prefix not in seen:
        seen.add(prefix)
        d = json.loads(files[0].read_text(encoding='utf-8'))
        print(f'\n[{prefix}] example:')
        print(f'  query: {d.get("query","")[:80]}')
        print(f'  type: {d.get("query_type","?")}')
        print(f'  relevant: {d.get("relevant_video_ids",[])[:2]}')
