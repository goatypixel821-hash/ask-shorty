#!/usr/bin/env python3
"""
Build semantic clusters from Shorties and save to data/clusters.json.

Run once (or to refresh):
    python build_clusters.py

Pipeline:
  1. Load all 950 Shorties from the full corpus SQLite DB
  2. Embed with SentenceTransformer('all-MiniLM-L6-v2')
  3. UMAP -> 2D coordinates for scatter plot
  4. HDBSCAN -> cluster assignments
  5. Claude Haiku labels each cluster (cheap: ~$0.02 total)
  6. Load ALL 5979 videos for timeline data
  7. Save everything to data/clusters.json

Output is cached — delete the file to rebuild.
"""

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

FULL_DB   = "C:/Users/number2/Desktop/youtube-history-viewer-copy/data/transcripts.db"
LOCAL_DB  = "data/transcripts.db"
OUT_PATH  = Path("data/clusters.json")
LABEL_MODEL = "claude-sonnet-4-20250514"   # same model as Ask Shorty

CLUSTER_COLORS = [
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
    "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
    "#14b8a6", "#e11d48", "#d97706", "#7c3aed", "#059669",
    "#dc2626", "#2563eb", "#16a34a", "#ca8a04", "#9333ea",
    "#0891b2", "#be185d", "#65a30d", "#7e22ce", "#047857",
]
NOISE_COLOR = "#374151"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_shorties():
    """Return list of dicts with video_id, title, channel, watch_date, shorty."""
    conn = sqlite3.connect(FULL_DB)
    c = conn.cursor()
    c.execute("""
        SELECT v.video_id, v.title, v.channel, v.watch_date, t.shorty
        FROM transcripts t
        JOIN videos v ON v.video_id = t.video_id
        WHERE t.shorty IS NOT NULL AND length(trim(t.shorty)) > 0
        ORDER BY v.watch_date
    """)
    rows = c.fetchall()
    conn.close()
    return [
        {
            "video_id":   r[0],
            "title":      r[1] or r[0],
            "channel":    r[2] or "",
            "watch_date": (r[3] or "")[:10],
            "shorty":     r[4],
        }
        for r in rows
    ]


def load_all_videos():
    """Return all videos (with and without Shorties) for timeline data."""
    conn = sqlite3.connect(FULL_DB)
    c = conn.cursor()
    c.execute("""
        SELECT v.video_id, v.title, v.channel, v.watch_date,
               CASE WHEN t.shorty IS NOT NULL AND length(trim(t.shorty))>0
                    THEN 1 ELSE 0 END as has_shorty
        FROM videos v
        LEFT JOIN transcripts t ON t.video_id = v.video_id
        ORDER BY v.watch_date
    """)
    rows = c.fetchall()
    conn.close()
    return [
        {
            "video_id":   r[0],
            "title":      r[1] or r[0],
            "channel":    r[2] or "",
            "watch_date": (r[3] or "")[:10],
            "has_shorty": bool(r[4]),
            "cluster_id": -2,   # filled in later
        }
        for r in rows
    ]


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts):
    print(f"  Embedding {len(texts)} Shorties with SentenceTransformer...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return embeddings.astype(np.float32)


# ── UMAP ──────────────────────────────────────────────────────────────────────

def run_umap(embeddings):
    print("  Running UMAP (2D)...")
    import umap
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        verbose=False,
    )
    coords = reducer.fit_transform(embeddings)
    return coords.astype(float)


# ── HDBSCAN ───────────────────────────────────────────────────────────────────

def run_hdbscan(coords):
    print("  Running HDBSCAN...")
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=8,
        min_samples=3,
        cluster_selection_epsilon=0.05,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(coords)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"  Found {n_clusters} clusters, {n_noise} noise points")
    return labels


# ── Haiku labeling ────────────────────────────────────────────────────────────

def label_cluster(videos):
    """Ask Claude Haiku for a short cluster label given up to 5 Shorty snippets."""
    from anthropic_client import get_client
    client = get_client()

    snippets = "\n\n---\n\n".join(
        f"Title: {v['title']}\nChannel: {v['channel']}\n{v['shorty'][:400]}"
        for v in videos[:5]
    )
    prompt = (
        "Below are summaries of videos that cluster together by topic.\n"
        "Give this cluster a short label (3-6 words) capturing the main theme.\n"
        "Output ONLY the label, no explanation.\n\n"
        f"{snippets}"
    )

    resp = client.messages.create(
        model=LABEL_MODEL,
        max_tokens=32,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip().strip('"').strip("'")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists():
        ans = input(f"{OUT_PATH} already exists. Rebuild? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    print("\n=== KNOWLEDGE OBSERVATORY CLUSTER BUILD ===\n")

    # 1. Load Shorties
    print("[1/6] Loading Shorties from corpus DB...")
    shorties = load_shorties()
    print(f"  Loaded {len(shorties)} Shorties")

    # 2. Embed
    print("[2/6] Embedding Shorties...")
    texts = [v["shorty"] for v in shorties]
    embeddings = embed_texts(texts)

    # 3. UMAP
    print("[3/6] Reducing dimensions...")
    coords_2d = run_umap(embeddings)

    # 4. HDBSCAN
    print("[4/6] Clustering...")
    labels = run_hdbscan(coords_2d)

    # 5. Label each cluster with Haiku
    print("[5/6] Labeling clusters with Claude Haiku...")
    cluster_ids = sorted(set(labels) - {-1})
    cluster_map = {}  # cluster_id -> {label, color, videos[]}

    for i, cid in enumerate(cluster_ids):
        idxs = [j for j, l in enumerate(labels) if l == cid]
        cluster_videos = [shorties[j] for j in idxs]
        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

        print(f"  Cluster {cid} ({len(idxs)} videos) — labeling...", end=" ", flush=True)
        try:
            label = label_cluster(cluster_videos)
        except Exception as e:
            label = f"Cluster {cid}"
            print(f"(Haiku error: {e})", end=" ")
        print(label)

        # Sort by watch_date
        cluster_videos.sort(key=lambda v: v["watch_date"] or "")

        # Top channels
        chan_counts: dict = defaultdict(int)
        for v in cluster_videos:
            if v["channel"]:
                chan_counts[v["channel"]] += 1
        top_channels = sorted(chan_counts, key=lambda k: -chan_counts[k])[:5]

        cluster_map[cid] = {
            "id":           cid,
            "label":        label,
            "color":        color,
            "count":        len(idxs),
            "top_channels": top_channels,
            "videos": [
                {
                    "video_id":   shorties[j]["video_id"],
                    "title":      shorties[j]["title"],
                    "channel":    shorties[j]["channel"],
                    "watch_date": shorties[j]["watch_date"],
                    "x":          round(float(coords_2d[j][0]), 4),
                    "y":          round(float(coords_2d[j][1]), 4),
                }
                for j in sorted(idxs, key=lambda j: shorties[j]["watch_date"] or "")
            ],
        }

    # Noise (unclustered) points
    noise_idxs = [j for j, l in enumerate(labels) if l == -1]
    noise_videos = [
        {
            "video_id":   shorties[j]["video_id"],
            "title":      shorties[j]["title"],
            "channel":    shorties[j]["channel"],
            "watch_date": shorties[j]["watch_date"],
            "x":          round(float(coords_2d[j][0]), 4),
            "y":          round(float(coords_2d[j][1]), 4),
        }
        for j in noise_idxs
    ]

    # Build lookup: video_id -> cluster_id for the scatter points
    vid_to_cluster = {}
    for cid, data in cluster_map.items():
        for v in data["videos"]:
            vid_to_cluster[v["video_id"]] = cid
    for v in noise_videos:
        vid_to_cluster[v["video_id"]] = -1

    # 6. Load all videos for timeline
    print("[6/6] Loading all videos for timeline...")
    all_videos = load_all_videos()
    for v in all_videos:
        v["cluster_id"] = vid_to_cluster.get(v["video_id"], -2)
    print(f"  Loaded {len(all_videos)} videos total")

    # Build final JSON
    output = {
        "generated_at":  datetime.now().isoformat(),
        "stats": {
            "total_videos":  len(all_videos),
            "clustered":     len(shorties),
            "cluster_count": len(cluster_ids),
            "noise_count":   len(noise_videos),
            "date_from":     min((v["watch_date"] for v in all_videos if v["watch_date"]), default=""),
            "date_to":       max((v["watch_date"] for v in all_videos if v["watch_date"]), default=""),
        },
        "clusters":     list(cluster_map.values()),
        "noise_videos": noise_videos,
        "all_videos":   all_videos,
    }

    def _json_default(obj):
        """Convert numpy scalar types to plain Python types."""
        import numpy as _np
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    OUT_PATH.write_text(json.dumps(output, indent=2, default=_json_default), encoding="utf-8")
    size_kb = OUT_PATH.stat().st_size // 1024
    print(f"\nSaved to {OUT_PATH} ({size_kb} KB)")
    print(f"  {len(cluster_ids)} clusters  |  {len(noise_videos)} noise  |  {len(all_videos)} total videos")
    print("\nDone. Restart the Flask app and visit /knowledge")


if __name__ == "__main__":
    main()
