# Examples

Illustrative scripts. The aggregation example executes numerical calculations
on constructed inputs; the curve example generates fabricated placeholders.
Neither is a real-world federated-learning benchmark.


## `transport_aggregation_demo.py` (NUMERICAL AGGREGATION)

Unlike the synthetic curve generator above, this example invokes GROUPOID's
actual Karcher mean, SO(3) point-action aggregation, and cycle-holonomy
diagnostic on five deterministic spherical client points.

```bash
python -m examples.transport_aggregation_demo
python -m examples.transport_aggregation_demo --json
```

The latent points are arranged symmetrically around the north pole, which
provides an independent analytic intrinsic-mean target. They are then expressed
in five known, different coordinate frames. The example reports geodesic error
for an untransported Karcher mean, and for transport aggregation under either
a flat connection or a deliberately corrupted rotation on the closing cycle.
It also checks whether the algorithm's returned local points agree with the
declared frame maps. The JSON form is intended for reproducible machine reads.

This demonstrates the existing implementation on a controlled analytic case,
not real federated model training, a wall-clock benchmark, or external
scientific validation. The frame maps are given, not estimated. The corrupted
case remains a legitimate SO(3) point action but is not cycle-flat.

## `synthetic_curve_demo.py` (SYNTHETIC -- NO TRAINING)

> **Warning: this script fabricates its output by design.**
> It does **not** train any model, does **not** load MNIST, and does **not**
> run a federated learning loop (no such loop exists in this repository yet --
> see [STATUS.md](../STATUS.md) and [LIMITATIONS.md](../LIMITATIONS.md)).

It generates synthetic loss/accuracy curves from a closed-form exponential
decay plus Gaussian noise, purely to illustrate the experiment-tracking
pattern (metric names, logging shape) that a real federated benchmark would
use. The numbers are placeholders and must never be cited, plotted, or
compared as measurements.

```bash
python -m examples.synthetic_curve_demo
```

Weights & Biases logging is **off by default**. The opt-in `--wandb` flag
logs only to a clearly-named scratch project
(`groupoid-synthetic-demo-scratch`), never to a results project, so a
fabricated curve can never be mistaken for a real benchmark run:

```bash
python -m examples.synthetic_curve_demo --wandb
```

For the one real, measured benchmark in this repository, see
[`benchmarks/test_karcher_benchmark.py`](../benchmarks/test_karcher_benchmark.py),
which times the Karcher-mean computation with `pytest-benchmark`.
