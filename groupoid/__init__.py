"""GROUPOID: Groupoid-based federated learning with Riemannian geometry."""

__version__ = "0.1.0.dev0"

from groupoid.aggregation import (
    DisconnectedClientGraphError,
    FederatedRound,
    TransportGroupoidAggregator,
)
from groupoid.cohomology import IncompleteCocycleError, compute_h1
from groupoid.groupoid import CompositionError, Morphism, compose, inverse
from groupoid.laplacian import SpectralSummary, spectral_analysis
from groupoid.manifold import karcher_mean
from groupoid.sheaf import Sheaf

__all__ = [
    "__version__",
    "CompositionError",
    "DisconnectedClientGraphError",
    "FederatedRound",
    "IncompleteCocycleError",
    "Morphism",
    "Sheaf",
    "SpectralSummary",
    "TransportGroupoidAggregator",
    "compose",
    "compute_h1",
    "inverse",
    "karcher_mean",
    "spectral_analysis",
]
