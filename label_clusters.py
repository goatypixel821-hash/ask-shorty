import json, os, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path('.') / '.env', override=False)
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url='https://openrouter.ai/api/v1')

def get_label(titles):
    prompt = "These YouTube videos are grouped together. Give a short 3-5 word topic label.\n\nVideos:\n" + "\n".join(f"- {t}" for t in titles) + "\n\nRespond with only the label, nothing else."
    for attempt in range(3):
        resp = client.chat.completions.create(
            model='deepseek/deepseek-v4-flash',
            max_tokens=20,
            temperature=0.3,
            messages=[{'role':'user','content':prompt}]
        )
        label = (resp.choices[0].message.content or '').strip()
        if label:
            return label
        time.sleep(1)
    return 'Unlabeled'

with open(r'data\clusters.json', encoding='utf-8') as f:
    data = json.load(f)

clusters = data['clusters']
print(f'Labeling {len(clusters)} clusters...')

for c in clusters:
    titles = [v['title'] for v in c['videos'][:10]]
    label = get_label(titles)
    c['label'] = label
    print(f"  Cluster {c['id']} ({c['count']} videos): {label}")
    time.sleep(0.3)

with open(r'data\clusters.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)
print('Done - restart Flask app to see labels')
