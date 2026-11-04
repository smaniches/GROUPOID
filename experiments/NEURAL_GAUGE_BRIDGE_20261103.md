# Real-data federated neural gauge-equivalence experiment (2026-11-03)

**Research scope:** Six paired CPU experiments on the 1,797-sample scikit-learn handwritten-digits dataset. This experiment tests whether exact, externally supplied hidden-unit permutations can be reversed during federated aggregation. **It does not demonstrate inference of unknown gauge maps, an advantage over canonical FedAvg, or any curved-manifold neural training.**

## Frozen computational protocol

- Classifier: two-layer tanh MLP, 64 inputs, 6 hidden units, 10 logits; 460 flattened parameters.
- Train/held-out split: stratified 70%/30%, seeded independently per run.
- Four non-IID clients: classwise Dirichlet(0.4) allocation of the training split.
- 18 synchronous rounds; each client performs four full-client-batch SGD steps per round, fixed learning rate 0.4.
- Three otherwise paired arms: canonical FedAvg; raw averaging after hidden-neuron permutation; exact inverse permutation applied before averaging.
- Supplied hidden-unit gauges use the exact symmetry that simultaneously reorders first-layer rows/biases and second-layer columns. Client aggregation is weighted by local training sample count.
- No hidden tuning against held-out outcomes. Per-seed model initialization and non-IID allocation are generated reproducibly from the recorded seeds.

## Held-out results

Accuracy fractions at communication round 18, 540 held-out images per seed.

| Seed | Canonical FedAvg | Raw misaligned average | Exact transported average | Max across-round state L2 difference: canonical vs transported |
|---:|---:|---:|---:|---:|
| 20261103 | 0.6852 | 0.2926 | 0.6852 | 2.029e-15 |
| 11 | 0.8000 | 0.2463 | 0.8000 | 1.835e-15 |
| 23 | 0.7778 | 0.1019 | 0.7778 | 2.085e-15 |
| 37 | 0.6852 | 0.1019 | 0.6852 | 1.214e-15 |
| 41 | 0.7222 | 0.2870 | 0.7222 | 1.433e-15 |
| 59 | 0.6815 | 0.2130 | 0.6815 | 1.295e-15 |

Across the six runs, the transported and canonical parameter trajectories agreed to at most `2.0847e-15` in Euclidean L2. The raw, misaligned average had worse accuracy in every recorded run. The canonical and transported arms are identical up to roundoff by **construction** of this permutation symmetry and the known inverse maps, so these experiments do not establish a new improvement over a correctly aligned baseline.

## Code and evidence custody

Executable reference script: [`real_neural/groupoid_real_neural.py.txt`](real_neural/groupoid_real_neural.py.txt) (preserved original Python source; SHA-256 `4e852ddc63c0443079464cab853efc276e916e6412ca22436aef0eda73401492`). All six complete JSON traces are now committed under [`real_neural/archives/`](real_neural/archives/) at their original hashes. The optional `--verify-groupoid` package bridge was **not** run in the historical six CPU trials because `geomstats` was absent in that environment.

| Archived evidence file (under `real_neural/archives/`) | SHA-256 |
|---|---|
| `groupoid_run_seed1.json` | `20df0a3cdd442f76ba8277b687ee04ae78b41a57e44a3806d8660d663f48f1cc` |
| `groupoid_seed_11.json` | `e9d73f63d6babdf1ef496530b108ac2e7e2969bea87ea4de605a3c63c1cf09ee` |
| `groupoid_seed_23.json` | `5a198e11fd1b41e739eb1341fc53a0953138507851fa93f0af36ecd16038b1e4` |
| `groupoid_seed_37.json` | `e61cfe34536739a798ba7fa3f7ec7cf1fe4775ab136f39330e923f9d9ec45fb4` |
| `groupoid_seed_41.json` | `e64987fcd0e92338f099898c687435e8f552535d1116afcdb96d56be02fe017b` |
| `groupoid_seed_59.json` | `557921a4e1cc51418f397df826c27e3dee34dfa475c347faa8197415db3f9864` |

Repository integration test `tests/test_neural_gauge_bridge.py` directly exercises `TransportGroupoidAggregator` using the same hidden-unit point-action semantics on a small Euclidean MLP. It is an independent implementation check, **not** a substitute for having run the full six-seed digit experiment through the package API.

## Next falsifier

With supplied inverse maps removed, attempt blind matching under the same sample, training-token and communication budgets. Compare against canonical FedAvg and ordinary matching baselines. Score inferred-permutation correctness and held-out task accuracy separately. A failed matching experiment would show that exact transport is mathematically coherent but not yet an operational aggregation method.

## Public source and trace archive (2026-11-03 addendum)

The original CPU script and six JSON result traces described above are now
preserved byte-for-byte under [`real_neural/`](real_neural/README.md). The
original SHA-256 values above remain the source-of-truth custody identifiers.
The archive includes a light, executable integrity gate and a per-round replay
comparison tool. This publication does not change the frozen protocol or
turn the reported equality under known permutations into evidence for
unknown-gauge inference. The six historical runs did not invoke the optional
project-API bridge; that interface has separate integration tests.
