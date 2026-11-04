# Client-scaling extension for archived real-digits neural experiment

Status: **exploratory extension**, not part of the preregistered `S^2`
benchmark. Existing six-seed, four-client archives remain unchanged. The
extension calls the original SHA-256-pinned 460-parameter MLP experiment,
rather than duplicating its numerical machinery.

Run on a host with optional PyTorch and scikit-learn installed:

```bash
python experiments/real_neural/client_scaling.py --clients 7 --out /tmp/clients-7.json
python experiments/real_neural/client_scaling.py --clients 10 --out /tmp/clients-10.json
```

Both commands require exact, previously recorded training conditions
(18 rounds, 4 full-client SGD steps per round, learning rate 0.4,
Dirichlet(0.4) non-IID partitions), except the declared number of clients.
The runner uses five seeds and writes create-only JSON to a caller-chosen
local path. Any seed violating the original minimum 32 examples per client
is explicitly `INVALID_PARTITION`, never silently replaced or reported as
a success. The aggregate process exits nonzero if any case is invalid.

The primary validity check is that inverse-permutation alignment matches
canonical FedAvg. Raw averaging in mismatched parameter coordinates is an
intentionally invalid comparator, not a competitive baseline. The known
permutation inverses are supplied by construction, not learned; this experiment
cannot establish unknown gauge recovery, a non-Euclidean benefit, privacy,
or superiority over properly aligned federated averaging.

An independent local CPU reimplementation, not identical to this source,
was executed at five seeds for 7 and 10 clients (PyTorch 2.10.0+cpu,
scikit-learn 1.8.0, NumPy 2.3.5). Seven clients: 5/5 valid cases,
mean canonical accuracy 0.65148, mean unaligned accuracy 0.19185,
maximum paired parameter difference 3.805e-15. Ten clients: 4/5 valid,
mean canonical accuracy across valid cases 0.68704, mean unaligned
accuracy 0.20231, maximum difference 2.301e-15. Seed 23 failed the
ten-client minimum-sample guard (one client received 23 samples).
The independent script SHA-256 is
`f2b2fd2f2cbb0a1b858ef18b0a312dfbcbd2c934c262aeeb3a63c1f4a151148a`
and independent results SHA-256
`4ec9d54afd32bcf45a4c3a24b7ca84e0b1fd362849048206a0f57f5c9e93f8d1`.
These independent numerical observations **must not** be relabeled as
results produced by the in-repository wrapper before its own execution.
