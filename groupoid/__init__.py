"""GROUPOID: Groupoid-based federated learning with Riemannian geometry."""

__version__ = "0.1.0.dev5"

from groupoid.aggregation import (
    DisconnectedClientGraphError,
    FederatedRound,
    InvalidPointTransportError,
    TransportGroupoidAggregator,
    UnsupportedTransportRepresentationError,
)
from groupoid.cohomology import (
    IncompleteCocycleError,
    compute_h1,
    cycle_basis_holonomy_defect,
)
from groupoid.groupoid import (
    CompositionError,
    Morphism,
    NonReciprocalTransportError,
    compose,
    inverse,
)
from groupoid.laplacian import SpectralSummary, spectral_analysis
from groupoid.manifold import karcher_mean
from groupoid.sheaf import Sheaf

__all__ = [
    "__version__",
    "CompositionError",
    "DisconnectedClientGraphError",
    "FederatedRound",
    "IncompleteCocycleError",
    "InvalidPointTransportError",
    "Morphism",
    "NonReciprocalTransportError",
    "Sheaf",
    "SpectralSummary",
    "TransportGroupoidAggregator",
    "UnsupportedTransportRepresentationError",
    "compose",
    "compute_h1",
    "cycle_basis_holonomy_defect",
    "inverse",
    "karcher_mean",
    "spectral_analysis",
]
