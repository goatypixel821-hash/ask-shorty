#!/usr/bin/env python3
"""
Rebuild fact_nodes and fact_edges from the facts table.

Usage:
  python rebuild_fact_frequency.py --db-path data/transcripts.db
"""

from __future__ import annotations

import argparse
import os
import sys

from transcript_database import TranscriptDatabase


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild graph salience tables (fact_nodes, fact_edges) from facts."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/transcripts.db",
        help="Path to transcripts.db",
    )
    args = parser.parse_args()

    db_path = os.path.abspath(args.db_path)
    if not os.path.isfile(db_path):
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    TranscriptDatabase(db_path=db_path)

    from hsc.fact_frequency import rebuild_fact_frequency

    print(f"Rebuilding fact frequency from: {db_path}")
    rebuild_fact_frequency(db_path)
    print("Done.")


if __name__ == "__main__":
    main()
