import json
with open(r'data\clusters.json', encoding='utf-8') as f:
    data = json.load(f)
clusters = data.get('clusters', [])
for c in sorted(clusters, key=lambda x: -x.get('size', 0)):
    print(f"{c.get('size',0):4d}  {c.get('label', 'unlabeled')}")
