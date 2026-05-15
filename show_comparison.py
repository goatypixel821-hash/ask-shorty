import json
with open(r'C:\Users\number2\Desktop\shorty\data\model_comparison_20260427.json', encoding='utf-8') as f:
    data = json.load(f)

for item in data['videos'][:2]:
    print('='*60)
    print(item['title'])
    print()
    print('--- ORIGINAL (existing) ---')
    print(item.get('original_shorty','')[:800])
    print()
    print('--- QWEN 72B ---')
    print(item.get('qwen_shorty','')[:800])
    print()
    print('--- DEEPSEEK V4 FLASH ---')
    print(item.get('deepseek_shorty','')[:800])
    print()
