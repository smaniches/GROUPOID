# Quickstart

## Installation

### From source (development)

Requires Python 3.10, 3.11, or 3.12 (not 3.13+; see the note below).

```bash
git clone https://github.com/smaniches/GROUPOID.git
cd GROUPOID
pip install -e ".[dev]"
pre-commit install
```

!!! note

    GROUPOID is not published on PyPI. Install from source as shown above.

!!! note "Python version"

    The `numpy<2.0` / `scipy<1.14` pins (required for geomstats compatibility)
    have no wheels for Python 3.13+, so `pip` will refuse to install there with
    a `Requires-Python` message. Use Python 3.10-3.12.

## Basic Usage

### Computing the Karcher Mean

```python
import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from groupoid.manifold import karcher_mean

# Create a 2-sphere
manifold = Hypersphere(dim=2)

# Generate 5 random points on S^2
points = manifold.random_point(n_samples=5)

# Compute the Karcher mean
mean = karcher_mean(manifold, points)
print(f"Mean point: {mean}")
print(f"On manifold: {manifold.belongs(mean)}")
```

### Composing Groupoid Morphisms

```python
import numpy as np
from groupoid.groupoid import Morphism, compose, inverse

# Create transport maps between three clients
T_AB = np.array([[0.0, -1.0], [1.0, 0.0]])  # 90-degree rotation
T_BC = np.array([[0.0, 1.0], [-1.0, 0.0]])  # -90-degree rotation

f = Morphism(source="A", target="B", transport_map=T_AB)
g = Morphism(source="B", target="C", transport_map=T_BC)

# Compose: A -> B -> C
h = compose(f, g)
print(f"Composed: {h}")
print(f"Transport map:\n{h.transport_map}")

# Invert: C -> A
h_inv = inverse(h)
print(f"Inverse: {h_inv}")
```

### Checking Cohomological Consistency

```python
import numpy as np
import networkx as nx
from groupoid.cohomology import compute_h1

# Build a triangle graph
graph = nx.DiGraph()
graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])

# Consistent transport maps (coboundary)
g_A = np.eye(3)
g_B = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
g_C = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)

transport_maps = {
    ("A", "B"): g_B @ np.linalg.inv(g_A),
    ("B", "C"): g_C @ np.linalg.inv(g_B),
    ("A", "C"): g_C @ np.linalg.inv(g_A),
}

h1 = compute_h1(graph, transport_maps)
print(f"H^1 norm: {h1:.2e}")  # Should be ~0
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with hypothesis verbose output
pytest tests/theory/ -v --hypothesis-show-statistics

# Run benchmarks
pytest benchmarks/ -v --benchmark-only
```

## Building Documentation

```bash
mkdocs serve
# Open http://127.0.0.1:8000 in your browser
```
