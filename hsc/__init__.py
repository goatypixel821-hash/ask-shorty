"""Hierarchical Semantic Compression (HSC) — experimental retrieval layers."""

from hsc.query_router import route_query
from hsc.hsc_search import hsc_retrieve
from hsc.graph_reasoner import graph_reason, load_triples, path_to_sentence, load_node_frequency
from hsc.fact_frequency import rebuild_fact_frequency
from hsc.global_graph import global_graph_reason, global_path_to_sentence, load_global_graph
from hsc.global_graph_builder import build_global_graph, ensure_global_graph_fresh
from hsc.entity_normalizer import normalize_entity

__all__ = [
    "route_query",
    "hsc_retrieve",
    "graph_reason",
    "load_triples",
    "path_to_sentence",
    "load_node_frequency",
    "rebuild_fact_frequency",
    "global_graph_reason",
    "global_path_to_sentence",
    "load_global_graph",
    "build_global_graph",
    "ensure_global_graph_fresh",
    "normalize_entity",
]
