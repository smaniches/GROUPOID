# Examples

Illustrative scripts. These are **not** benchmarks and produce **no real
results**.

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
