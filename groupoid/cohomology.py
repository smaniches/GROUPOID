"""Cycle-basis holonomy defect diagnostics for transport maps.

The primary scalar in this module is not a cohomology norm.  It is the
maximum Frobenius deviation from identity over the particular cycle basis
returned by :func:`networkx.cycle_basis` on the undirected graph.
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import numpy.typing as npt
from loguru import logger

from groupoid.groupoid import validate_reciprocal_transports


class IncompleteCocycleError(Exception):
    """Raised when a basis-cycle holonomy cannot be evaluated completely.

    A cycle holonomy is the ordered product of the transport maps on every
    edge of that cycle.  If a basis-cycle edge has no map in either
    direction, the product is undefined and this error names the missing
    edge.

    This check is deliberately cycle-local.  A bridge lies on no cycle, so
    a zero cycle-basis defect does not certify that every graph edge has a
    registered transport map.
    """


def cycle_basis_holonomy_defect(
    graph: nx.DiGraph,
    transport_maps: dict[tuple[str, str], npt.NDArray[np.float64]],
) -> float:
    """Return the maximum cycle-basis holonomy Frobenius defect.

    For the cycle basis ``B = nx.cycle_basis(graph.to_undirected())``, this
    function computes

        D_B(T) = max_{gamma in B} ||Hol_T(gamma) - I||_F.

    This is a representation-dependent diagnostic, not a canonical norm on
    first cohomology.  Its magnitude depends in general on the selected cycle
    basis and is not invariant under arbitrary invertible changes of frame.
    Orthogonal conjugation preserves the Frobenius magnitude; cycle
    reversal also preserves it when the holonomy itself is orthogonal.

    Exact zero has a stronger meaning than the magnitude. On a connected
    graph, when every underlying undirected edge carries one invertible
    connection map (represented by one orientation or by a reciprocal pair)
    and every cycle emitted by :func:`networkx.cycle_basis` is evaluated
    completely, ``D_B(T) == 0`` is equivalent to flat transport: every
    closed-loop holonomy is identity.

    That justification is specific to the NetworkX Paton implementation this
    module exercises. The emitted list is not in general the fundamental-cycle
    basis of one fixed spanning tree; the argument instead uses the emission
    order, in which every emitted cycle contributes exactly one chord not
    present in any earlier emitted cycle, so the induced constraint system is
    triangular. It is not claimed for arbitrary graph-theoretic cycle bases or
    for future NetworkX implementations whose emission order may differ.

    The equivalence does not certify graph connectedness, bridge completeness,
    point-action validity, or any finite numerical threshold.

    Parameters
    ----------
    graph
        Directed client/transport graph.  Cycle selection is performed on its
        undirected projection.
    transport_maps
        Maps ``(source, target)`` edge tuples to square transport matrices.
        Reverse traversal uses the matrix inverse. If both orientations of an
        underlying edge are supplied, they must be mutual numerical inverses.

    Returns
    -------
    float
        Maximum Frobenius distance ``||Hol(gamma) - I||_F`` over the selected
        basis cycles.  An acyclic graph returns ``0.0`` because it has no cycle
        holonomy to test.

    Raises
    ------
    IncompleteCocycleError
        If a selected basis cycle contains an edge with no transport map in
        either direction.
    NonReciprocalTransportError
        If both orientations of an underlying edge are supplied but do not
        satisfy the groupoid inverse law.
    numpy.linalg.LinAlgError
        If a reverse-oriented edge must be traversed but its registered matrix
        is singular.
    """
    undirected = graph.to_undirected()
    for u, v in undirected.edges():
        if u == v:
            # A self-loop is one directed transport, not two opposite arrows.
            # ``(u, v)`` and ``(v, u)`` are the same key here, so applying the
            # reciprocal-pair test would spuriously demand an involution.
            continue
        if (u, v) in transport_maps and (v, u) in transport_maps:
            validate_reciprocal_transports(
                transport_maps[(u, v)],
                transport_maps[(v, u)],
                source=u,
                target=v,
            )

    cycles = nx.cycle_basis(undirected)

    if not cycles:
        logger.debug("No cycles in graph; cycle-basis holonomy defect = 0 trivially")
        return 0.0

    max_holonomy_defect = 0.0

    for cycle in cycles:
        n = len(cycle)
        edge_maps: list[npt.NDArray[np.float64]] = []
        for i in range(n):
            u = cycle[i]
            v = cycle[(i + 1) % n]

            if (u, v) in transport_maps:
                edge_maps.append(transport_maps[(u, v)])
            elif (v, u) in transport_maps:
                edge_maps.append(np.linalg.inv(transport_maps[(v, u)]))
            else:
                raise IncompleteCocycleError(
                    f"Incomplete cycle transport: no transport map for edge ({u}, {v}) "
                    f"on cycle {cycle}; holonomy is undefined. Supply the edge "
                    "map in either direction before computing the cycle-basis defect."
                )

        holonomy = edge_maps[0]
        for transport in edge_maps[1:]:
            holonomy = transport @ holonomy

        dim = holonomy.shape[0]
        deviation = float(np.linalg.norm(holonomy - np.eye(dim), ord="fro"))
        max_holonomy_defect = max(max_holonomy_defect, deviation)

    logger.debug("Cycle-basis holonomy defect = {:.6e}", max_holonomy_defect)
    return max_holonomy_defect


def compute_h1(
    graph: nx.DiGraph,
    transport_maps: dict[tuple[str, str], npt.NDArray[np.float64]],
) -> float:
    """Deprecated compatibility alias for :func:`cycle_basis_holonomy_defect`.

    Earlier GROUPOID releases called the returned scalar an ``H^1`` or
    ``H^1 norm``.  The numerical value is preserved for compatibility, but
    that mathematical interpretation is superseded: the value is the
    basis-dependent cycle-holonomy defect defined by
    :func:`cycle_basis_holonomy_defect`.
    """
    warnings.warn(
        "compute_h1() is deprecated: it returns a cycle-basis holonomy "
        "Frobenius defect, not a canonical H^1 norm. Use "
        "cycle_basis_holonomy_defect().",
        DeprecationWarning,
        stacklevel=2,
    )
    return cycle_basis_holonomy_defect(graph, transport_maps)
