#!/usr/bin/env python3
"""
Relabel existing clusters.json using the working Sonnet model.
Much faster than rebuild — skips embedding, UMAP, HDBSCAN entirely.

    python relabel_clusters.py
"""

import json
import sqlite3
from pathlib import Path

CLUSTERS_PATH = Path("data/clusters.json")
FULL_DB       = "C:/Users/number2/Desktop/youtube-history-viewer-copy/data/transcripts.db"
LABEL_MODEL   = "claude-sonnet-4-20250514"   # same model as Ask Shorty


def get_shorty(video_id: str, conn) -> str:
    c = conn.cursor()
    c.execute("SELECT shorty FROM transcripts WHERE video_id=? AND shorty IS NOT NULL LIMIT 1", (video_id,))
    row = c.fetchone()
    return row[0] if row else ""


def label_cluster(videos, conn):
    from anthropic_client import get_client
    client = get_client()

    snippets = []
    for v in videos[:5]:
        shorty = get_shorty(v["video_id"], conn)
        if shorty:
            snippets.append(f"Title: {v['title']}\nChannel: {v['channel']}\n{shorty[:400]}")

    if not snippets:
        # Fall back to just titles
        snippets = [f"Title: {v['title']}\nChannel: {v['channel']}" for v in videos[:5]]

    prompt = (
        "Below are summaries of videos that cluster together by topic.\n"
        "Give this cluster a short label (3-6 words) capturing the main theme.\n"
        "Output ONLY the label, no explanation.\n\n"
        + "\n\n---\n\n".join(snippets)
    )

    resp = client.messages.create(
        model=LABEL_MODEL,
        max_tokens=32,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip().strip('"').strip("'")


def main():
    if not CLUSTERS_PATH.exists():
        print("data/clusters.json not found. Run build_clusters.py first.")
        return

    data = json.loads(CLUSTERS_PATH.read_text(encoding="utf-8"))
    clusters = data.get("clusters", [])
    print(f"Relabeling {len(clusters)} clusters using {LABEL_MODEL}...\n")

    conn = sqlite3.connect(FULL_DB)

    for i, cluster in enumerate(clusters):
        old_label = cluster.get("label", f"Cluster {cluster['id']}")
        print(f"  [{i+1}/{len(clusters)}] {len(cluster['videos'])} videos — labeling...", end=" ", flush=True)
        try:
            label = label_cluster(cluster["videos"], conn)
            cluster["label"] = label
            print(label)
        except Exception as e:
            print(f"ERROR: {e}")
            cluster["label"] = old_label

    conn.close()

    CLUSTERS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nSaved updated labels to {CLUSTERS_PATH}")
    print("Restart the Flask app and visit /knowledge")


if __name__ == "__main__":
    main()
