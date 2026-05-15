import json
with open(r'C:\Users\number2\Desktop\shorty\data\model_comparison_20260427.json', encoding='utf-8') as f:
    data = json.load(f)
print(list(data.keys()))
# find the videos list
for k, v in data.items():
    print(k, type(v), str(v)[:100])
