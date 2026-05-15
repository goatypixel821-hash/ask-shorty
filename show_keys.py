import json
with open(r'C:\Users\number2\Desktop\shorty\data\model_comparison_20260427.json', encoding='utf-8') as f:
    data = json.load(f)
print(list(data['videos'][0].keys()))
