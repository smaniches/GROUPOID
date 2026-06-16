"""First cohomology of transport groupoids."""

from __future__ import annotations

import networkx as nx
import numpy as np
import numpy.typing as npt
from loguru import logger


class IncompleteCocycleError(Exception):
    """Raised when a cocycle is incompletely specified.

    The holonomy around a cycle is the ordered product of the transport
    maps on *every* edge of that cycle. If any edge on a cycle has no
    transport map (in either direction), the holonomy is undefined: a
    product over a strict subset of the cycle's edges is mathematically
    meaningless and must not be reported as an H^1 (in)consistency. This
    error names the missing edge so the caller can supply it.
    """


def compute_h1(
    graph: nx.DiGraph,
    transport_maps: dict[tuple[str, str], npt.NDArray[np.float64]],
) -> float:
    """Compute the H^1 cohomology norm of a transport cocycle.

    Returns 0.0 (up to numerical tolerance) when the cocycle is a
    coboundary, meaning the local data is globally consistent.

    The H^1 obstruction on a cycle is the deviation from the identity of
    the holonomy, i.e. the ordered product of the restriction/transport
    maps around the *entire* cycle. The cocycle must therefore be fully
    specified: every edge of every basis cycle needs a transport map in
    one direction or the other (the reverse direction is inverted). A
    missing edge map leaves the holonomy undefined, so this function
    raises :class:`IncompleteCocycleError` rather than silently forming a
    partial product.

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

    Raises
    ------
    IncompleteCocycleError
        If any edge on a basis cycle has no transport map in either
        direction, so the cycle's holonomy is undefined.
    """
    undirected = graph.to_undirected()
    cycles = nx.cycle_basis(undirected)

    if not cycles:
        logger.debug("No cycles in graph, H^1 = 0 trivially")
        return 0.0

    max_holonomy_norm = 0.0

    for cycle in cycles:
        n = len(cycle)
        # Resolve every edge's transport map first, raising on the first
        # missing one so a partial product is never formed. The reverse
        # orientation is inverted. Every cycle-basis cycle has length >= 3, so
        # the resulting product is always fully formed.
        edge_maps = []
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]

            if (u, v) in transport_maps:
                edge_maps.append(transport_maps[(u, v)])
            elif (v, u) in transport_maps:
                edge_maps.append(np.linalg.inv(transport_maps[(v, u)]))
            else:
                raise IncompleteCocycleError(
                    f"Incomplete cocycle: no transport map for edge ({u}, {v}) "
                    f"on cycle {cycle}; holonomy is undefined. Supply the edge "
                    f"map in either direction before computing H^1."
                )

        holonomy = edge_maps[0]
        for T in edge_maps[1:]:
            holonomy = T @ holonomy

        dim = holonomy.shape[0]
        deviation = float(np.linalg.norm(holonomy - np.eye(dim), ord="fro"))
        max_holonomy_norm = max(max_holonomy_norm, deviation)

    logger.debug("H^1 norm = {:.6e}", max_holonomy_norm)
    return max_holonomy_norm
