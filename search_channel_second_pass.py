#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


VIDEO_LINE_RE = re.compile(
    r"^(?P<video_id>[^|]+)\s+\|\s+(?P<title>[^|]+)\s+\|\s+(?P<channel>[^|]+)\s+\|\s+(?P<watch_date>[^|]+)\s+\|\s+matched=(?P<matched>.+)$"
)
SNIPPET_RE = re.compile(r"^\s*snippet:\s*(.+)$")

DATE_MIN = datetime.strptime("2024-09-01", "%Y-%m-%d").date()
DATE_MAX = datetime.strptime("2025-03-31", "%Y-%m-%d").date()

RELEVANT_SNIPPET_TERMS = [
    "floor",
    "property condition",
    "consulting",
    "back on their feet",
    "back on your feet",
    "buyers agent",
    "buyer's agent",
    "walkthrough",
    "walk through",
]

# Lightweight denylist to strip obvious non-real-estate channels.
NON_RE_CHANNEL_HINTS = [
    "cnn",
    "msnbc",
    "news",
    "pakman",
    "meidastouch",
    "history of the universe",
    "astrum",
    "veritasium",
    "computer",
    "rossmann",
    "infographics",
    "science",
    "chemistry",
    "ai ",
    "economics",
    "legal",
    "politics",
]


def in_date_range(s: str) -> bool:
    s = (s or "").strip()[:10]
    if not s or s == "?":
        return False
    try:
        d = datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return False
    return DATE_MIN <= d <= DATE_MAX


def looks_non_real_estate(channel: str) -> bool:
    low = (channel or "").lower()
    return any(h in low for h in NON_RE_CHANNEL_HINTS)


def snippet_score(text: str) -> int:
    low = (text or "").lower()
    return sum(1 for t in RELEVANT_SNIPPET_TERMS if t in low)


def parse_results(path: Path):
    rows = []
    pending = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = VIDEO_LINE_RE.match(raw.strip())
        if m:
            pending = {
                "video_id": m.group("video_id").strip(),
                "title": m.group("title").strip(),
                "channel": m.group("channel").strip(),
                "watch_date": m.group("watch_date").strip(),
                "matched": m.group("matched").strip(),
                "snippet": "",
            }
            rows.append(pending)
            continue
        sm = SNIPPET_RE.match(raw)
        if sm and pending is not None:
            pending["snippet"] = sm.group(1).strip()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-path",
        default=r"data\channel_search_results.txt",
    )
    ap.add_argument(
        "--out-path",
        default=r"data\channel_search_filtered.txt",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    if not in_path.is_absolute():
        in_path = root / in_path
    if not out_path.is_absolute():
        out_path = root / out_path

    rows = parse_results(in_path)

    # 1) date window
    rows = [r for r in rows if in_date_range(r["watch_date"])]

    by_channel = defaultdict(list)
    seen_video = defaultdict(set)
    for r in rows:
        ch = r["channel"]
        vid = r["video_id"]
        if vid in seen_video[ch]:
            continue
        seen_video[ch].add(vid)
        by_channel[ch].append(r)

    # 2) 3-8 appearances
    channels_3_8 = {ch: vids for ch, vids in by_channel.items() if 3 <= len(vids) <= 8}

    # 3) exclude obvious non-real-estate channels
    excluded_non_re = {ch: vids for ch, vids in channels_3_8.items() if looks_non_real_estate(ch)}
    channels = {ch: vids for ch, vids in channels_3_8.items() if not looks_non_real_estate(ch)}

    # 4) sort by best signal then count
    def channel_key(item):
        ch, vids = item
        best = max((snippet_score(v.get("snippet", "")) for v in vids), default=0)
        return (-best, -len(vids), ch.lower())

    ordered = sorted(channels.items(), key=channel_key)

    lines = []
    lines.append("Channel Search Filtered (2nd pass)")
    lines.append("Window: 2024-09-01 .. 2025-03-31")
    lines.append("Rules: 3-8 videos/channel, obvious non-real-estate channels excluded")
    lines.append("")
    lines.append(f"Channels in date window (any count): {len(by_channel)}")
    lines.append(f"Channels with 3-8 matches: {len(channels_3_8)}")
    lines.append(f"Excluded as obvious non-real-estate: {len(excluded_non_re)}")
    lines.append(f"Final channels kept: {len(ordered)}")
    if excluded_non_re:
        lines.append("Excluded channels: " + ", ".join(sorted(excluded_non_re.keys())))
    lines.append("")

    if not ordered:
        lines.append("No channels matched all filters.")
    else:
        for ch, vids in ordered:
            vids = sorted(vids, key=lambda x: x["watch_date"])
            best = max(vids, key=lambda v: snippet_score(v.get("snippet", "")))
            lines.append(f"- {ch} ({len(vids)} videos)")
            for v in vids:
                lines.append(f"  - {v['watch_date']} | {v['title']}")
            lines.append(
                f"  best snippet: {best.get('snippet','') or '[no snippet]'}"
            )
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"Channels kept: {len(ordered)}")


if __name__ == "__main__":
    main()
