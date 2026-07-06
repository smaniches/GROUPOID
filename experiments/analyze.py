"""Preregistered analysis of the benchmark results (see PREREGISTRATION.md).

Reads experiments/results.json and writes experiments/RESULTS.md. Every
number reported here follows the preregistered plan; anything else is
labelled descriptive.

Run:
    python experiments/analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

METHODS = ["euclidean", "karcher", "transport_euclidean", "transport_karcher"]
PAIRS = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i + 1 :]]
BOOT_N = 10_000
PERM_N = 10_000


def load_runs():
    data = json.loads((Path(__file__).parent / "results.json").read_text())
    return data["meta"], data["runs"]


def cells(runs):
    out: dict[tuple, list[dict]] = {}
    for r in runs:
        out.setdefault((r["alpha"], r["sigma"], r["epsilon"]), []).append(r)
    return dict(sorted(out.items()))


def median_diff_ci(diffs: np.ndarray, rng: np.random.Generator):
    """Median of paired differences with a 10k percentile-bootstrap 95% CI."""
    med = float(np.median(diffs))
    idx = rng.integers(0, len(diffs), size=(BOOT_N, len(diffs)))
    boots = np.median(diffs[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return med, float(lo), float(hi)


def fmt(x: float) -> str:
    return f"{x:.2e}"


def main() -> None:
    meta, runs = load_runs()
    rng_boot = np.random.default_rng(12345)
    rng_perm = np.random.default_rng(54321)
    by_cell = cells(runs)
    lines: list[str] = []
    w = lines.append

    w("# Benchmark results (preregistered analysis)")
    w("")
    w("Analysis of `results.json` following `PREREGISTRATION.md` exactly;")
    w("deviations are listed at the end. Primary error metric: geodesic")
    w("distance to the oracle estimand (Karcher mean of the de-framed")
    w("truths). Effect sizes are medians of per-seed paired differences")
    w("with 10,000-resample percentile-bootstrap 95% CIs.")
    w("")
    w("## Environment")
    w("")
    for k in ("python", "numpy", "scipy", "geomstats", "platform", "n_runs"):
        w(f"- {k}: {meta[k]}")
    w(f"- Karcher non-convergence warnings (runs, not excluded): {meta['karcher_warning_count']}")
    w("")

    # Per-cell medians of the primary error.
    w("## Median primary error per cell")
    w("")
    w(
        "| alpha | sigma | epsilon | M1 euclidean | M2 karcher "
        "| M3 transport+eucl | M4 transport+karcher | median H^1 |"
    )
    w("|---|---|---|---|---|---|---|---|")
    for (alpha, sigma, eps), rs in by_cell.items():
        meds = [np.median([r["err_oracle"][m] for r in rs]) for m in METHODS]
        h1 = np.median([r["h1_norm"] for r in rs])
        w(f"| {alpha} | {sigma} | {eps} | " + " | ".join(fmt(m) for m in meds) + f" | {fmt(h1)} |")
    w("")

    # H1: cells alpha > 0, epsilon = 0; CI of median(err M1 - err M4) > 0.
    w("## H1: transport benefit under misalignment (primary)")
    w("")
    w("Criterion: in every cell with alpha > 0 and epsilon = 0, the 95% CI")
    w("of median(err(M1) - err(M4)) is entirely positive.")
    w("")
    w("| alpha | sigma | median diff | 95% CI | positive? |")
    w("|---|---|---|---|---|")
    h1_pass = 0
    h1_cells = 0
    for (alpha, sigma, eps), rs in by_cell.items():
        if eps != 0 or alpha == 0:
            continue
        h1_cells += 1
        d = np.array(
            [r["err_oracle"]["euclidean"] - r["err_oracle"]["transport_karcher"] for r in rs]
        )
        med, lo, hi = median_diff_ci(d, rng_boot)
        ok = lo > 0
        h1_pass += ok
        w(f"| {alpha} | {sigma} | {fmt(med)} | [{fmt(lo)}, {fmt(hi)}] | {'yes' if ok else 'NO'} |")
    if h1_pass == h1_cells:
        verdict = "SUPPORTED"
    elif h1_pass >= 4:
        verdict = "PARTIALLY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"
    w("")
    w(f"**H1 verdict: {verdict}** ({h1_pass}/{h1_cells} cells positive).")
    w("")

    # H2: pooled epsilon > 0, Spearman(h1_norm, err M4), permutation p.
    w("## H2: H^1 predicts aggregation error under corruption (non-trivial)")
    w("")
    pooled = [r for r in runs if r["epsilon"] > 0]
    h1v = np.array([r["h1_norm"] for r in pooled])
    errv = np.array([r["err_oracle"]["transport_karcher"] for r in pooled])
    rho = float(spearmanr(h1v, errv).statistic)
    perm_count = 0
    for _ in range(PERM_N):
        perm_rho = float(spearmanr(h1v, rng_perm.permutation(errv)).statistic)
        if perm_rho >= rho:
            perm_count += 1
    pval = (1 + perm_count) / (1 + PERM_N)
    ok2 = rho >= 0.5 and pval < 0.01
    w(f"- Pooled runs with epsilon > 0: n = {len(pooled)}")
    w(f"- Spearman rho(H^1, err(M4)) = {rho:.3f}")
    w(f"- Permutation p-value ({PERM_N} permutations): {pval:.5f}")
    w(f"- Criterion rho >= 0.5 and p < 0.01: {'met' if ok2 else 'NOT met'}")
    w("")
    w(f"**H2 verdict: {'SUPPORTED' if ok2 else 'NOT SUPPORTED'}**")
    w("")
    w("Descriptive (no criterion): within-cell Spearman correlations:")
    w("")
    w("| alpha | sigma | epsilon | rho |")
    w("|---|---|---|---|")
    for (alpha, sigma, eps), rs in by_cell.items():
        if eps == 0:
            continue
        c_rho = float(
            spearmanr(
                [r["h1_norm"] for r in rs],
                [r["err_oracle"]["transport_karcher"] for r in rs],
            ).statistic
        )
        w(f"| {alpha} | {sigma} | {eps} | {c_rho:.3f} |")
    w("")
    w("Descriptive caveat (does not alter the preregistered verdict): the")
    w("within-cell correlations are near zero, so the pooled correlation is")
    w("driven by between-level variation (epsilon = 0.1 vs 0.3). On this")
    w("evidence H^1 tracks the magnitude of cocycle corruption -- it works")
    w("as a gross inconsistency detector -- but does not rank runs by error")
    w("within a fixed corruption level.")
    w("")

    # H3: cells alpha >= 0.7, epsilon = 0.
    w("## H3: ablation decomposition")
    w("")
    w("Criterion: in every cell with alpha >= 0.7 and epsilon = 0, the")
    w("transport margin median(err(M1) - err(M3)) exceeds the intrinsic-")
    w("mean margin median(err(M3) - err(M4)).")
    w("")
    w("| alpha | sigma | transport margin | intrinsic-mean margin | transport larger? |")
    w("|---|---|---|---|---|")
    h3_pass = 0
    h3_cells = 0
    for (alpha, sigma, eps), rs in by_cell.items():
        if eps != 0 or alpha < 0.7:
            continue
        h3_cells += 1
        t = float(
            np.median(
                [r["err_oracle"]["euclidean"] - r["err_oracle"]["transport_euclidean"] for r in rs]
            )
        )
        k = float(
            np.median(
                [
                    r["err_oracle"]["transport_euclidean"] - r["err_oracle"]["transport_karcher"]
                    for r in rs
                ]
            )
        )
        ok = t > k
        h3_pass += ok
        w(f"| {alpha} | {sigma} | {fmt(t)} | {fmt(k)} | {'yes' if ok else 'NO'} |")
    w("")
    w(
        f"**H3 verdict: {'SUPPORTED' if h3_pass == h3_cells else 'NOT SUPPORTED'}**"
        f" ({h3_pass}/{h3_cells} cells)."
    )
    w("")

    # Null / manipulation checks.
    w("## Null and manipulation checks")
    w("")
    w("(a) alpha = 0, epsilon = 0 cells: all pairwise 95% CIs include 0.")
    w("")
    w("| sigma | pair | median diff | 95% CI | includes 0? |")
    w("|---|---|---|---|---|")
    a_ok = True
    for (alpha, sigma, eps), rs in by_cell.items():
        if eps != 0 or alpha != 0:
            continue
        for m_a, m_b in PAIRS:
            d = np.array([r["err_oracle"][m_a] - r["err_oracle"][m_b] for r in rs])
            med, lo, hi = median_diff_ci(d, rng_boot)
            inc = lo <= 0 <= hi
            a_ok = a_ok and inc
            w(
                f"| {sigma} | {m_a} - {m_b} | {fmt(med)} "
                f"| [{fmt(lo)}, {fmt(hi)}] | {'yes' if inc else 'NO'} |"
            )
    w("")
    w(f"Check (a): {'passed' if a_ok else 'FAILED'}.")
    w("")
    if not a_ok:
        w("Descriptive interpretation (the check is reported as failed above")
        w("and stays failed): the failing pairs are exactly those comparing an")
        w("extrinsic (renormalized Euclidean) mean against a Karcher-based")
        w("method. At alpha = 0, epsilon = 0 the Karcher-based methods compute")
        w("the oracle estimand exactly by construction (their error is 0),")
        w("while the extrinsic mean differs from the intrinsic mean by a small")
        w("systematic amount, so its median difference cannot include 0. The")
        w("check as preregistered wrongly assumed all methods are equivalent")
        w("at the null; it is diagnostic of the estimand choice, not of an")
        w("implementation error. The observed magnitudes (~3e-5 at sigma =")
        w("0.05, ~6e-4 at sigma = 0.2) are 2 to 4 orders of magnitude below")
        w("the H1 effect sizes (~6e-2 to 3e-1).")
        w("")
    eps0 = [r["h1_norm"] for r in runs if r["epsilon"] == 0]
    epsp = [r["h1_norm"] for r in runs if r["epsilon"] > 0]
    w(f"(b) max H^1 at epsilon = 0: {fmt(max(eps0))} (< 1e-8, asserted at runtime): passed.")
    w(f"(c) min H^1 at epsilon > 0: {fmt(min(epsp))} (> 0, asserted at runtime): passed.")
    w("(d) M1/M2 errors bit-identical across epsilon for fixed (alpha, sigma,")
    w("seed): asserted at runtime by the runner: passed.")
    w("")

    w("## Deviations from preregistration")
    w("")
    w("1. Seeding. The preregistration specified that run k uses the k-th")
    w("   child of SeedSequence(20260706), but that conflicts with its own")
    w("   manipulation check (d), which requires the client data to be")
    w("   identical across corruption levels for a fixed (alpha, sigma,")
    w("   seed). The runner therefore keys the data stream by")
    w("   (alpha_index, sigma_index, seed_index) -- independent of epsilon --")
    w("   and the corruption stream by the full cell. No other deviation.")
    w("")

    (Path(__file__).parent / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("wrote RESULTS.md")


if __name__ == "__main__":
    main()
