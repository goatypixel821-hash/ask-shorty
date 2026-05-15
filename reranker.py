#!/usr/bin/env python3
"""
Second-stage reranker for Ask Shorty.

Design goals:
- Normalize hits from all retrieval layers into a single format.
- Group hits by video + chunk neighbourhood so corroboration is preserved,
  not deduped away.
- Expand each chunk group with its neighbouring chunks for richer context.
- Score groups with a CrossEncoder model.
- Blend the final ranking from rerank score + retrieval score + support signal.

Intended usage (ask_shorty.py):

    from reranker import Reranker

    reranker = Reranker()           # lazy-loads CrossEncoder on first use
    hits = (
        reranker.normalize_chroma_result(chunk_res,  "chunk",              query, title_map, collection)
      + reranker.normalize_chroma_result(shorty_res, "shorty",             query, title_map, collection)
      + reranker.normalize_chroma_result(synq_res,   "synthetic_question", query, title_map, collection)
    )
    groups  = reranker.group_hits(hits)
    ranked  = reranker.rerank_and_blend(query, groups)
    context = reranker.groups_to_context_blocks(ranked)
"""

from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------

def _resolve_rerank_model() -> str:
    """Return the fine-tuned cross-encoder path if it exists, else the default HF model."""
    from pathlib import Path
    local = Path(__file__).parent / "data" / "shorty_crossencoder_model"
    if local.is_dir():
        return str(local)
    return "cross-encoder/ms-marco-MiniLM-L-6-v2"

RERANK_MODEL_NAME = _resolve_rerank_model()

# Chunks whose indices differ by at most this value are put in the same group.
CHUNK_NEIGHBORHOOD_RADIUS = 3

# How many neighbouring chunks to fetch when expanding context.
CONTEXT_NEIGHBORS_BEFORE = 1
CONTEXT_NEIGHBORS_AFTER = 1

# Weights for the blended final score (must sum to 1.0).
WEIGHT_RERANK      = 0.50  # CrossEncoder output (normalised 0-1)
WEIGHT_RETRIEVAL   = 0.25  # 1 - best cosine distance (higher = closer)
WEIGHT_SUPPORT     = 0.15  # support count bonus (log-scaled)
WEIGHT_DIVERSITY   = 0.10  # bonus for hitting multiple source types

# Cap how many groups are sent to the (relatively slow) CrossEncoder.
MAX_GROUPS_TO_RERANK = 40


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievalHit:
    """One normalised result from any Chroma retrieval layer."""
    source_id: str                     # Chroma document id  e.g. "abcVID:chunk:3"
    video_id: str
    video_title: str
    source_type: str                   # "chunk" | "shorty" | "synthetic_question"
    chunk_index: Optional[int]         # set for chunks, None otherwise
    retrieval_score: float             # cosine distance — lower is better
    query_variant: str                 # which rewritten query produced this hit
    text: str                          # original document text from Chroma


@dataclass
class EvidenceGroup:
    """
    A cluster of RetrievalHits from the same video + chunk neighbourhood.

    Multiple hits (same chunk via different query variants, or chunk + synq
    from the same video) fold into one group.  Each original hit is preserved
    in source_hits so the LLM sees the full corroboration signal.
    """
    group_key: str                              # unique identifier for this group
    video_id: str
    video_title: str
    center_chunk_index: Optional[int]           # None for shorty-only groups
    source_hits: List[RetrievalHit] = field(default_factory=list)
    support_count: int = 0
    support_types: List[str] = field(default_factory=list)  # unique source types
    best_retrieval_score: float = 1.0           # lower is better (cosine distance)
    chunk_ids: List[str] = field(default_factory=list)

    # Text fields populated after neighbour expansion
    compact_text: str = ""       # best single text for the reranker
    context_text: str = ""       # expanded window for the final LLM prompt

    # Scores
    rerank_score: float = 0.0    # CrossEncoder output (higher = more relevant)
    final_score: float = 0.0     # blended  (higher = better)


# ---------------------------------------------------------------------------
# Internal model singleton
# ---------------------------------------------------------------------------

_cross_encoder: Any = None
_ce_lock = threading.Lock()


def _get_cross_encoder():
    global _cross_encoder
    with _ce_lock:
        if _cross_encoder is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading CrossEncoder model %s …", RERANK_MODEL_NAME)
            _cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
            logger.info("CrossEncoder loaded.")
    return _cross_encoder


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class Reranker:
    """
    Stateless helper that wraps CrossEncoder scoring.

    All methods are safe to call from multiple threads as long as each call
    is self-contained (no shared mutable state on the instance).
    """

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize_chroma_result(
        self,
        chroma_res: Dict[str, Any],
        source_type: str,
        query_variant: str,
        video_title_map: Dict[str, str],
        *,
        query_index: int = 0,
    ) -> List[RetrievalHit]:
        """
        Convert a raw Chroma ``collection.query()`` result into RetrievalHit objects.

        ``chroma_res`` is the dict returned by Chroma — its lists are indexed
        first by query-batch position (query_index) then by result position.

        ``video_title_map`` maps video_id -> title (can be empty; id is used
        as fallback).
        """
        ids    = (chroma_res.get("ids")       or [[]])[query_index]
        docs   = (chroma_res.get("documents") or [[]])[query_index]
        metas  = (chroma_res.get("metadatas") or [[]])[query_index]
        scores = (chroma_res.get("distances") or [[]])[query_index]

        hits: List[RetrievalHit] = []
        for i, doc in enumerate(docs):
            meta  = metas[i]  if i < len(metas)  else {}
            score = scores[i] if i < len(scores) else 1.0
            src_id = ids[i]   if i < len(ids)    else f"{source_type}:{i}"
            meta = meta or {}

            vid   = meta.get("video_id", "unknown")
            cidx  = meta.get("chunk_index")
            if cidx is not None:
                try:
                    cidx = int(cidx)
                except (ValueError, TypeError):
                    cidx = None

            hits.append(RetrievalHit(
                source_id=src_id,
                video_id=vid,
                video_title=video_title_map.get(vid, vid),
                source_type=source_type,
                chunk_index=cidx,
                retrieval_score=float(score),
                query_variant=query_variant,
                text=doc or "",
            ))
        return hits

    def normalize_flat_results(
        self,
        results: List[Dict[str, Any]],
        video_title_map: Dict[str, str],
    ) -> List[RetrievalHit]:
        """
        Convert the flat list format produced by ``AskShorty._search_layer``
        (dicts with keys id, text, score, metadata, query) into RetrievalHit.
        """
        hits: List[RetrievalHit] = []
        for r in results:
            meta = r.get("metadata") or {}
            vid  = meta.get("video_id", "unknown")
            cidx = meta.get("chunk_index")
            if cidx is not None:
                try:
                    cidx = int(cidx)
                except (ValueError, TypeError):
                    cidx = None
            stype = meta.get("type", r.get("source_type", "chunk"))
            hits.append(RetrievalHit(
                source_id=r.get("id", ""),
                video_id=vid,
                video_title=video_title_map.get(vid, vid),
                source_type=stype,
                chunk_index=cidx,
                retrieval_score=float(r.get("score", 1.0)),
                query_variant=r.get("query", ""),
                text=r.get("text", ""),
            ))
        return hits

    # ------------------------------------------------------------------
    # Neighbour expansion
    # ------------------------------------------------------------------

    def expand_chunk_neighbors(
        self,
        hit: RetrievalHit,
        collection,
        n_before: int = CONTEXT_NEIGHBORS_BEFORE,
        n_after:  int = CONTEXT_NEIGHBORS_AFTER,
    ) -> str:
        """
        Fetch the chunks adjacent to ``hit`` from Chroma by constructing IDs
        directly (no extra embedding call needed).

        Returns the concatenated text of [prev…chunk…next] chunks.
        Falls back to hit.text if the fetch fails.
        """
        if hit.chunk_index is None:
            return hit.text

        video_id   = hit.video_id
        center_idx = hit.chunk_index

        # Support both Chroma ID formats:
        #   new project:  {video_id}:chunk:{N}
        #   old project:  {video_id}_chunk_{N}
        # We detect the format from the hit's own source_id.
        if re.search(r"_chunk_\d+$", hit.source_id):
            fmt = "{vid}_chunk_{n}"
        else:
            fmt = "{vid}:chunk:{n}"

        def _chunk_id(n: int) -> str:
            if fmt == "{vid}_chunk_{n}":
                return f"{video_id}_chunk_{n}"
            return f"{video_id}:chunk:{n}"

        ids_to_fetch = [
            _chunk_id(center_idx + offset)
            for offset in range(-n_before, n_after + 1)
            if center_idx + offset >= 0
        ]
        try:
            res = collection.get(ids=ids_to_fetch)
            fetched_ids   = res.get("ids", [])
            fetched_docs  = res.get("documents", [])
            # Sort by chunk index extracted from id (handle both formats)
            pairs = []
            for fid, fdoc in zip(fetched_ids, fetched_docs):
                m = re.search(r"[_:]chunk[_:](\d+)$", fid)
                idx = int(m.group(1)) if m else 9999
                pairs.append((idx, fdoc or ""))
            pairs.sort(key=lambda p: p[0])
            combined = "\n".join(text for _, text in pairs).strip()
            return combined if combined else hit.text
        except Exception as exc:
            logger.debug("Neighbour expansion failed for %s: %s", hit.source_id, exc)
            return hit.text

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def group_hits(
        self,
        hits: List[RetrievalHit],
        collection=None,
        expand_neighbors: bool = True,
    ) -> List[EvidenceGroup]:
        """
        Group RetrievalHits into EvidenceGroups.

        Rules:
        - Exact duplicate source_ids from different query variants collapse
          into the same group (counted as one hit but supports corroboration).
        - Chunk hits whose chunk_index values are within
          CHUNK_NEIGHBORHOOD_RADIUS of each other (same video) go in one group.
        - Shorty hits are always their own group per video.
        - Synthetic-question hits for the same video are treated as a separate
          group from chunks.

        Each original hit is preserved in group.source_hits so corroboration
        evidence is never lost.
        """
        groups: Dict[str, EvidenceGroup] = {}

        for hit in hits:
            key = self._group_key(hit)
            if key not in groups:
                center = hit.chunk_index
                groups[key] = EvidenceGroup(
                    group_key=key,
                    video_id=hit.video_id,
                    video_title=hit.video_title,
                    center_chunk_index=center,
                )

            g = groups[key]
            g.source_hits.append(hit)

            # Update best (lowest) retrieval score
            if hit.retrieval_score < g.best_retrieval_score:
                g.best_retrieval_score = hit.retrieval_score
                g.compact_text = hit.text

            if hit.source_id not in g.chunk_ids:
                g.chunk_ids.append(hit.source_id)

        # Compute support metadata and expand neighbours
        for g in groups.values():
            g.support_count = len(g.source_hits)
            seen_types = []
            for h in g.source_hits:
                if h.source_type not in seen_types:
                    seen_types.append(h.source_type)
            g.support_types = seen_types

            if not g.compact_text and g.source_hits:
                g.compact_text = g.source_hits[0].text

            # Expand neighbours for the best chunk hit
            if expand_neighbors and collection is not None:
                best_chunk_hit = min(
                    (h for h in g.source_hits if h.chunk_index is not None),
                    key=lambda h: h.retrieval_score,
                    default=None,
                )
                if best_chunk_hit is not None:
                    g.context_text = self.expand_chunk_neighbors(
                        best_chunk_hit, collection
                    )
                else:
                    g.context_text = g.compact_text
            else:
                g.context_text = g.compact_text

        return list(groups.values())

    def _group_key(self, hit: RetrievalHit) -> str:
        """
        Compute a stable group key for a hit.

        - Shorties  →  "<video_id>|shorty"
        - Synq      →  "<video_id>|synq"
        - Chunks    →  "<video_id>|chunk|<neighbourhood_bucket>"
        """
        if hit.source_type == "shorty":
            return f"{hit.video_id}|shorty"
        if hit.source_type == "synthetic_question":
            return f"{hit.video_id}|synq"
        # chunk
        if hit.chunk_index is not None:
            bucket = hit.chunk_index // CHUNK_NEIGHBORHOOD_RADIUS
            return f"{hit.video_id}|chunk|{bucket}"
        return f"{hit.video_id}|chunk|0"

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank_and_blend(
        self,
        query: str,
        groups: List[EvidenceGroup],
        verbose: bool = False,
    ) -> List[EvidenceGroup]:
        """
        Score groups with CrossEncoder, then blend with retrieval + support signals.

        Returns the same list sorted by ``final_score`` descending (best first).
        """
        if not groups:
            return groups

        ce = _get_cross_encoder()

        # Build CrossEncoder inputs — cap to MAX_GROUPS_TO_RERANK cheapest groups
        sorted_by_retrieval = sorted(
            groups, key=lambda g: g.best_retrieval_score
        )[:MAX_GROUPS_TO_RERANK]

        ce_pairs = [
            (query, self._build_rerank_text(g))
            for g in sorted_by_retrieval
        ]

        raw_scores = ce.predict(ce_pairs)

        # Assign rerank scores; groups outside the cap keep a default low score
        remaining = {g.group_key: g for g in groups}
        for g, raw in zip(sorted_by_retrieval, raw_scores):
            g.rerank_score = float(raw)
            remaining.pop(g.group_key, None)
        for g in remaining.values():
            # Assign the worst observed score so they stay at the bottom
            if sorted_by_retrieval:
                g.rerank_score = min(g.rerank_score for g in sorted_by_retrieval) - 1.0
            else:
                g.rerank_score = -10.0

        # Normalise rerank scores to [0, 1] via sigmoid
        all_groups = groups  # the full list
        for g in all_groups:
            g.rerank_score = self._sigmoid(g.rerank_score)

        # Blend final scores
        for g in all_groups:
            g.final_score = self._blend(g)

        all_groups.sort(key=lambda g: g.final_score, reverse=True)

        if verbose:
            for rank, g in enumerate(all_groups[:10], 1):
                print(
                    f"  [{rank:2d}] {g.video_id} | {g.group_key} "
                    f"| rerank={g.rerank_score:.3f} retrieval={g.best_retrieval_score:.3f} "
                    f"support={g.support_count}({','.join(g.support_types)}) "
                    f"final={g.final_score:.3f}"
                )
        return all_groups

    def _build_rerank_text(self, g: EvidenceGroup) -> str:
        """Build the text the CrossEncoder sees for one group."""
        type_label = "+".join(g.support_types) if g.support_types else "chunk"
        support_note = (
            f"[{g.support_count} supporting hits: {type_label}] "
            if g.support_count > 1
            else ""
        )
        title_prefix = f"Video: {g.video_title}\n" if g.video_title else ""
        return f"{title_prefix}{support_note}{g.compact_text}"

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Map any real number to (0, 1)."""
        return 1.0 / (1.0 + math.exp(-x))

    def _blend(self, g: EvidenceGroup) -> float:
        """
        Combine rerank score, retrieval score, support count, and source diversity
        into a single final score.  All inputs are normalised to [0, 1].
        """
        rerank_contrib = WEIGHT_RERANK * g.rerank_score

        # Cosine distance is 0 (perfect) to 2 (worst); normalise to [0,1].
        retrieval_normalised = max(0.0, 1.0 - g.best_retrieval_score / 2.0)
        retrieval_contrib = WEIGHT_RETRIEVAL * retrieval_normalised

        # Log-scale support count: log2(count+1) / log2(MAX) capped at 1.
        max_expected_support = 8.0
        support_normalised = min(
            math.log2(g.support_count + 1) / math.log2(max_expected_support + 1),
            1.0,
        )
        support_contrib = WEIGHT_SUPPORT * support_normalised

        # Diversity: fraction of the three source types present.
        diversity = len(set(g.support_types)) / 3.0
        diversity_contrib = WEIGHT_DIVERSITY * diversity

        return rerank_contrib + retrieval_contrib + support_contrib + diversity_contrib

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def groups_to_context_blocks(
        self,
        groups: List[EvidenceGroup],
        top_n: int = 20,
        include_support_metadata: bool = True,
    ) -> List[str]:
        """
        Convert a ranked list of EvidenceGroups into context strings for Claude.

        When ``include_support_metadata`` is True, each block is prefixed with
        a one-line support summary (e.g. "3 hits via chunk+shorty") so the LLM
        can weight corroborated evidence more heavily.
        """
        blocks: List[str] = []
        for g in groups[:top_n]:
            type_label = "+".join(g.support_types) if g.support_types else "?"
            chunk_note = (
                f" chunk={g.center_chunk_index}" if g.center_chunk_index is not None else ""
            )
            meta_line = (
                f"[EVIDENCE] video_id={g.video_id}{chunk_note}"
                f" support={g.support_count} via={type_label}"
                f" score={g.final_score:.3f}"
            )
            text = g.context_text or g.compact_text
            if include_support_metadata:
                blocks.append(f"{meta_line}\n{text}")
            else:
                blocks.append(text)
        return blocks

    def groups_to_debug_dict(self, groups: List[EvidenceGroup]) -> List[Dict[str, Any]]:
        """Serialisable summary of ranked groups for logging / experiment output."""
        out = []
        for g in groups:
            out.append({
                "group_key": g.group_key,
                "video_id": g.video_id,
                "video_title": g.video_title,
                "center_chunk_index": g.center_chunk_index,
                "support_count": g.support_count,
                "support_types": g.support_types,
                "best_retrieval_score": round(g.best_retrieval_score, 4),
                "rerank_score": round(g.rerank_score, 4),
                "final_score": round(g.final_score, 4),
                "chunk_ids": g.chunk_ids,
                "source_hit_ids": [h.source_id for h in g.source_hits],
                "source_hit_queries": list({h.query_variant for h in g.source_hits}),
            })
        return out
