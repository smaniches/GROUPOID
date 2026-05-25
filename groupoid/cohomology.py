"""First cohomology of transport groupoids."""

from __future__ import annotations

import networkx as nx
import numpy as np
from loguru import logger


def compute_h1(
    graph: nx.DiGraph,
    transport_maps: dict[tuple[str, str], np.ndarray],
) -> float:
    """Compute the H^1 cohomology norm of a transport cocycle.

    Returns 0.0 (up to numerical tolerance) when the cocycle is a
    coboundary, meaning the local data is globally consistent.

    Parameters
    ----------
    graph : nx.DiGraph
        The groupoid nerve (directed graph of nodes/clients).
    transport_maps : dict
        Maps (source, target) edge tuples to transport matrices.

    Returns
    -------
    float
        The cohomology norm. Zero indicates global consistency.
    """
    undirected = graph.to_undirected()
    cycles = nx.cycle_basis(undirected)

    if not cycles:
        logger.debug("No cycles in graph, H^1 = 0 trivially")
        return 0.0

    max_holonomy_norm = 0.0
    dim = None

    for cycle in cycles:
        n = len(cycle)
        holonomy = None

        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]

            if (u, v) in transport_maps:
                T = transport_maps[(u, v)]
            elif (v, u) in transport_maps:
                T = np.linalg.inv(transport_maps[(v, u)])
            else:
                logger.warning("Missing transport map for edge ({}, {})", u, v)
                continue

            if holonomy is None:
                holonomy = T
                dim = T.shape[0]
            else:
                holonomy = T @ holonomy

        if holonomy is not None and dim is not None:
            deviation = np.linalg.norm(holonomy - np.eye(dim), ord="fro")
            max_holonomy_norm = max(max_holonomy_norm, deviation)

    logger.debug("H^1 norm = {:.6e}", max_holonomy_norm)
    return max_holonomy_norm
