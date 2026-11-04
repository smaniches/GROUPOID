# Real-data neural gauge-equivalence archive

This directory publishes the original six-seed CPU reference script and all six
complete result traces supporting [the 2026-11-03 experiment](../NEURAL_GAUGE_BRIDGE_20261103.md).
It makes the recorded evidence inspectable and re-runnable; it is not a new
experiment or an independent scientific validation.

## Evidence boundary

The experiment trains a 460-parameter MLP on the real, bundled scikit-learn
handwritten-digits dataset (1,797 samples), with four simulated non-IID client
partitions over 18 federated rounds. Client hidden-unit permutations and their
inverses are deliberately supplied. The transported and canonical arms agree
to floating-point precision **by construction**. The unaligned arm is an
intentionally broken baseline, not evidence of superiority over correctly
aligned FedAvg. No unknown alignment discovery, privacy guarantee, curved
neural parameter space, or fully distributed training is demonstrated.

## Byte-exact provenance

- `groupoid_real_neural.py.txt` is the original executable Python source,
  preserved without reformatting. Its SHA-256 is
  `4e852ddc63c0443079464cab853efc276e916e6412ca22436aef0eda73401492`.
  CPython accepts the `.py.txt` extension when the file is passed explicitly;
  the archival suffix keeps historical source formatting out of current
  source-formatting tools without modifying its original bytes.
- `archives/` contains the six complete JSON traces at their original hashes.
  `tests/test_neural_archive.py` checks all seven hashes and the recorded
  shapes, seeds, round sequence, and scope invariants.
- `verify_replay.py` compares a freshly generated record with an archived
  record, checking the frozen protocol and every round of held-out loss and
  accuracy. Elapsed wall-clock time is correctly excluded from the comparison.

## Reproduce

Use the supported GROUPOID environment (Python 3.11 or 3.12) when running
the optional project-API integration. PyTorch and scikit-learn are
**optional experiment dependencies**, not runtime requirements of GROUPOID.

```bash
python -m pip install -e ".[dev]"
python -m pip install torch scikit-learn
python -m pytest -q tests/test_neural_archive.py
python experiments/real_neural/groupoid_real_neural.py.txt \
  --seed 20261103 --out /tmp/gauge-seed-20261103.json
python experiments/real_neural/verify_replay.py \
  --archive experiments/real_neural/archives/groupoid_run_seed1.json \
  --fresh /tmp/gauge-seed-20261103.json
```

Run the remaining seeds `11`, `23`, `37`, `41`, and `59` and compare against
their corresponding `archives/groupoid_seed_<seed>.json` files.
The original six-run record did **not** execute the optional
`--verify-groupoid` API bridge. That bridge is tested independently in
`tests/test_neural_gauge_bridge.py`; reruns may enable `--verify-groupoid`.

Comparisons are numeric, not bytewise: software, BLAS, and hardware can
change floating-point residuals, and the recorded elapsed time is inherently
nondeterministic. The source and archived trace hashes certify the original
files, not reproducibility on an arbitrary future environment.
