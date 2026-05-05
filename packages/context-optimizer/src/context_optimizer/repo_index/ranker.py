"""Owns: PageRank-based file ranking — builds a symbol import graph and sorts files by centrality.

Port of aider's get_ranked_tags() algorithm (aider/aider/repomap.py:365-574) with chat-file
personalization removed. Our prefix is stable (not session-specific), so all files start with
equal personalization weight; the graph topology alone determines rank.

Does NOT own: tag extraction, file rendering, or embedding.
Called by: repo_index/index.py.
Calls: networkx (nx.pagerank), stdlib math/collections.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import networkx as nx
from loguru import logger

from ._types import FileRank, Tag


def rank_files(
    tags_by_file: dict[str, list[Tag]],
    *,
    personalization: dict[str, float] | None = None,
) -> list[FileRank]:
    """Rank files by PageRank on a symbol import graph.

    Graph edges: (referencer_file → definer_file), weighted by sqrt(ref_count) * multipliers.
    Multipliers (from aider):
      - 10x  for long idents (≥8 chars, snake/camel/kebab-case) — avoids short noise tokens
      - 0.1x for _private idents — internal symbols rarely drive architecture
      - 0.1x for idents with >5 definers — stdlib/framework noise like 'open', 'len'

    Returns list[FileRank] sorted descending by pagerank_score.
    Returns [] only if graph has no nodes (no supported files in repo).
    """
    defines: dict[str, set[str]] = defaultdict(set)
    references: dict[str, list[str]] = defaultdict(list)

    for rel_path, tags in tags_by_file.items():
        for tag in tags:
            if tag.kind == "def":
                defines[tag.name].add(rel_path)
            elif tag.kind == "ref":
                references[tag.name].append(rel_path)

    # Header-only or def-only files: treat their defs as implicit self-references so they
    # still participate in the graph rather than becoming dangling nodes.
    if not references:
        references = {k: list(v) for k, v in defines.items()}

    idents = set(defines.keys()) & set(references.keys())

    G: nx.MultiDiGraph = nx.MultiDiGraph()

    # Self-edges for defs that have no matching refs keep those nodes in the graph.
    # Without this, isolated files (pure libraries with no callers yet indexed) would
    # have zero PageRank and fall out of top-N unfairly.
    for ident, definers in defines.items():
        if ident not in references:
            for definer in definers:
                G.add_edge(definer, definer, weight=0.1, ident=ident)

    for ident in idents:
        definers = defines[ident]
        mul = _edge_multiplier(ident, len(definers))
        for referencer, num_refs in Counter(references[ident]).items():
            for definer in definers:
                G.add_edge(referencer, definer, weight=mul * math.sqrt(num_refs), ident=ident)

    if G.number_of_nodes() == 0:
        return []

    pers_args: dict = {}
    if personalization:
        pers_args = {"personalization": personalization, "dangling": personalization}

    try:
        ranked = nx.pagerank(G, weight="weight", **pers_args)
    except ZeroDivisionError:
        try:
            ranked = nx.pagerank(G, weight="weight")
        except ZeroDivisionError:
            logger.warning("REPO_INDEX: ranker pagerank_zero_division files={}", len(tags_by_file))
            return []

    results = [
        FileRank(
            rel_path=node,
            pagerank_score=score,
            tags_count=len(tags_by_file.get(node, [])),
        )
        for node, score in ranked.items()
    ]
    results.sort(key=lambda r: r.pagerank_score, reverse=True)
    logger.info(
        "REPO_INDEX: ranker ranked files={} nodes={} edges={}",
        len(results),
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return results


def _edge_multiplier(ident: str, definer_count: int) -> float:
    """Compute the weight multiplier for edges involving this identifier."""
    is_snake = "_" in ident and any(c.isalpha() for c in ident)
    is_kebab = "-" in ident and any(c.isalpha() for c in ident)
    is_camel = any(c.isupper() for c in ident) and any(c.islower() for c in ident)
    mul = 1.0
    if (is_snake or is_kebab or is_camel) and len(ident) >= 8:
        mul *= 10
    if ident.startswith("_"):
        mul *= 0.1
    if definer_count > 5:
        mul *= 0.1
    return mul


def get_top_n_files(ranked: list[FileRank], n: int) -> list[str]:
    """Return rel_paths of the top-n files, alphabetically sorted for deterministic Repomix output.

    WHY alphabetical sort: Repomix may re-order its --include list; sorting ensures the
    rendered text is byte-for-byte identical for the same file set across runs, which is
    what makes the stable prefix actually stable.
    """
    return sorted(r.rel_path for r in ranked[:n])
