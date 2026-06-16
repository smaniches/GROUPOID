"""Sheaf-theoretic data structures for federated learning."""

from __future__ import annotations

import networkx as nx
import numpy as np
import numpy.typing as npt
from loguru import logger


class Sheaf:
    """A cellular sheaf on a graph.

    Assigns vector spaces to nodes and linear restriction maps to edges.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph
        self._restriction_maps: dict[tuple[str, str], npt.NDArray[np.float64]] = {}
        self._sections: dict[str, npt.NDArray[np.float64]] = {}

    def set_restriction_map(
        self, source: str, target: str, matrix: npt.NDArray[np.float64]
    ) -> None:
        """Set the restriction map for an edge."""
        self._restriction_maps[(source, target)] = matrix
        logger.debug("Set restriction map {} -> {}", source, target)

    def get_restriction_map(self, source: str, target: str) -> npt.NDArray[np.float64]:
        """Get the restriction map for an edge."""
        return self._restriction_maps[(source, target)]

    def set_section(self, node: str, value: npt.NDArray[np.float64]) -> None:
        """Set a section value at a node."""
        self._sections[node] = value

    def get_section(self, node: str) -> npt.NDArray[np.float64]:
        """Get the section value at a node."""
        return self._sections[node]

    def restrict(
        self, section: npt.NDArray[np.float64], source: str, target: str
    ) -> npt.NDArray[np.float64]:
        """Apply the restriction map to a section.

        Parameters
        ----------
        section : np.ndarray
            The section value at the source node.
        source : str
            Source node identifier.
        target : str
            Target node identifier.

        Returns
        -------
        np.ndarray
            The restricted section at the target node.
        """
        R = self._restriction_maps[(source, target)]
        result: npt.NDArray[np.float64] = R @ section
        return result

    def restrict_along_path(
        self, section: npt.NDArray[np.float64], path: list[str]
    ) -> npt.NDArray[np.float64]:
        """Restrict a section along a path of nodes.

        Parameters
        ----------
        section : np.ndarray
            The section value at path[0].
        path : list[str]
            Ordered list of nodes forming a path in the graph.

        Returns
        -------
        np.ndarray
            The section restricted to path[-1].
        """
        result = section
        for i in range(len(path) - 1):
            result = self.restrict(result, path[i], path[i + 1])
        return result
