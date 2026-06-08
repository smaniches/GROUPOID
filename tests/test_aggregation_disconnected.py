"""Disconnected client-graph handling in the aggregator (FIX #12).

``_get_transport_to_base`` transports a client's parameters to the base node
along a path in the client graph. If the client is in a different connected
component from the base, networkx's ``shortest_path`` raises the low-level
``NetworkXNoPath``. The aggregator must catch that and reraise a clear domain
error naming the unreachable node and the base, rather than leaking a networkx
exception to the caller.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from geomstats.geometry.hypersphere import Hypersphere

from groupoid.aggregation import (
    DisconnectedClientGraphError,
    TransportGroupoidAggregator,
)


def _aggregator_with_isolated_node() -> TransportGroupoidAggregator:
    manifold = Hypersphere(dim=2)
    graph = nx.DiGraph()
    graph.add_edge("A", "B")
    # 'C' is in the graph but in its own component: present (so we get
    # NetworkXNoPath, not NodeNotFound) yet unreachable from the base.
    graph.add_node("C")
    agg = TransportGroupoidAggregator(manifold=manifold, graph=graph, base_node="A")
    agg.register_transport("A", "B", np.eye(3))
    return agg


class TestDisconnectedClientGraph:
    def test_get_transport_to_base_raises_domain_error(self):
        agg = _aggregator_with_isolated_node()
        with pytest.raises(DisconnectedClientGraphError, match="disconnected"):
            agg._get_transport_to_base("C")

    def test_error_names_node_and_base(self):
        agg = _aggregator_with_isolated_node()
        with pytest.raises(DisconnectedClientGraphError) as exc:
            agg._get_transport_to_base("C")
        msg = str(exc.value)
        assert "C" in msg and "A" in msg

    def test_chains_from_networkx_no_path(self):
        # The domain error must preserve the underlying networkx cause so the
        # original failure remains diagnosable.
        agg = _aggregator_with_isolated_node()
        with pytest.raises(DisconnectedClientGraphError) as exc:
            agg._get_transport_to_base("C")
        assert isinstance(exc.value.__cause__, nx.NetworkXNoPath)

    def test_connected_node_still_resolves(self):
        # Sanity: a reachable node is unaffected by the new guard.
        agg = _aggregator_with_isolated_node()
        T = agg._get_transport_to_base("B")
        np.testing.assert_allclose(T, np.eye(3), atol=1e-12)
