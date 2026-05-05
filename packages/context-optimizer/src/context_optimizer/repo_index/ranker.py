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
                # math.isqrt over math.sqrt: deterministic integer floor sqrt avoids
                # platform libm drift that can flip rank order on borderline ties.
                # num_refs >= 1 (it comes from Counter), so isqrt is always >= 1.
                G.add_edge(referencer, definer, weight=mul * math.isqrt(num_refs), ident=ident)

    if G.number_of_nodes() == 0:
        return []

    pers_args: dict = {}
    if personalization:
        pers_args = {"personalization": personalization, "dangling": personalization}

    try:
        ranked = nx.pagerank(G, weight="weight", **pers_args)
    except ZeroDivisionError:
        # NetworkX's iterative PageRank can divide-by-zero when the
        # personalization/dangling vector sums to zero on some weakly
        # connected component (rare: happens when the personalization mass
        # falls entirely on nodes that don't appear in the graph). Falling
        # back to no personalization uses uniform initial probability,
        # which always sums to 1 across all nodes. Second failure means
        # the graph itself is degenerate; we log and surrender to caller.
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


def select_by_mass(
    ranked: list[FileRank],
    mass_target: float,
    *,
    max_files: int | None = None,
) -> list[str]:
    """Select files covering mass_target fraction of total PageRank score mass.

    WHY mass-based selection: the PageRank score distribution varies by repo shape.
    A well-modularised repo has a smooth power-law decay; a monolith has a cliff after
    3-4 hub files. A fixed top-N overshoots flat repos and undershoots hub repos.
    Covering a fixed fraction of total architectural signal adapts to both shapes.

    max_files: hard upper bound — the selection stops even if mass_target not yet reached.
    This prevents pathologically flat repos (500 equally-ranked files) from bloating the prefix.

    Returns rel_paths alphabetically sorted for deterministic Repomix output.
    """
    if not ranked:
        return []
    total = sum(r.pagerank_score for r in ranked)
    if total == 0:
        candidates = ranked if max_files is None else ranked[:max_files]
        return sorted(r.rel_path for r in candidates)

    cutoff = mass_target * total
    selected: list[str] = []
    cumulative = 0.0
    for r in ranked:
        selected.append(r.rel_path)
        cumulative += r.pagerank_score
        if cumulative >= cutoff:
            break
        if max_files is not None and len(selected) >= max_files:
            break

    return sorted(selected)
