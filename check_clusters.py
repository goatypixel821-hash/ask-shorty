import json
with open(r'data\clusters.json', encoding='utf-8') as f:
    data = json.load(f)
print(list(data.keys()))
clusters = data.get('clusters', [])
if clusters:
    print(clusters[0])
