"""Verify public modules and primary correction APIs import without error."""

from __future__ import annotations


def test_import_groupoid():
    import groupoid

    assert hasattr(groupoid, "__version__")
    assert hasattr(groupoid, "cycle_basis_holonomy_defect")
    assert hasattr(groupoid, "compute_h1")
    assert hasattr(groupoid, "NonReciprocalTransportError")


def test_import_manifold():
    from groupoid import manifold

    assert hasattr(manifold, "karcher_mean")


def test_import_groupoid_module():
    from groupoid import groupoid

    assert hasattr(groupoid, "Morphism")
    assert hasattr(groupoid, "compose")
    assert hasattr(groupoid, "inverse")


def test_import_cohomology():
    from groupoid import cohomology

    assert hasattr(cohomology, "cycle_basis_holonomy_defect")
    assert hasattr(cohomology, "compute_h1")


def test_import_sheaf():
    from groupoid import sheaf

    assert hasattr(sheaf, "Sheaf")


def test_import_aggregation():
    from groupoid import aggregation

    assert hasattr(aggregation, "TransportGroupoidAggregator")
    assert hasattr(aggregation, "InvalidPointTransportError")
    assert hasattr(aggregation, "UnsupportedTransportRepresentationError")


def test_import_laplacian():
    from groupoid import laplacian

    assert hasattr(laplacian, "build_sheaf_laplacian")
    assert hasattr(laplacian, "spectral_analysis")


def test_import_transport():
    from groupoid import transport

    assert hasattr(transport, "schild_ladder")
    assert hasattr(transport, "pole_ladder")
    assert hasattr(transport, "compute_tangent_transport_matrix")
    assert hasattr(transport, "compute_transport_matrix")


def test_import_optimizer():
    from groupoid import optimizer

    assert hasattr(optimizer, "RiemannianSGD")
    assert hasattr(optimizer, "RiemannianAdam")
