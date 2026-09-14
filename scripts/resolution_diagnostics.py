"""Resolution diagnostics for this harness (§9.1 T8 companion).

Implements the three quantities from the paired-evaluation resolution literature
(arXiv:2605.30315) for the n=38 attack harness:

  1. Resolution ratio  = observed run-to-run SD / SD an i.i.d. binomial harness
                         would produce at the same n and p.
  2. MDE               = smallest marginal difference detectable by exact McNemar
                         at n=38, alpha=0.05, power=0.80.
  3. Required N        = paired sample size needed to detect a target effect.

It also puts a confidence interval on the SD estimate itself, because the headline
12.1 pp comes from k=3 replicates and a variance estimated from 3 points is not a
precise number.

Exactness note: conditional on m discordant pairs, McNemar's exact test is a
two-sided binomial test of B ~ Binom(m, 1/2). Because that null is symmetric the
rejection region is {b <= c_m} U {b >= m - c_m} with
c_m = max{c : BinomCDF(c, m, 1/2) <= alpha/2}, so power needs one CDF call per m
rather than a scan over every outcome.

Run: uv run python scripts/resolution_diagnostics.py
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from scipy import stats

OUT_PATH = Path(__file__).resolve().parents[1] / "results" / "resolution_diagnostics.json"

# ── Measured inputs (final_report.md §2.2, §4.9 T12) ──────────────────
N_ATTACKS = 38
REPLICATE_ASR = [0.184, 0.421, 0.342]  # naive, k=3, pinned snapshot, temp=0
ALPHA = 0.05
POWER = 0.80

# Observed discordance rates (b+c)/n from the paired tests actually run
OBSERVED_DISCORDANCE = {
    "loose vs strict verifier (all 38)": (3 + 4) / 38,
    "combined-loose vs verifier-loose": (8 + 4) / 38,
    "naive vs hardened": (12 + 0) / 38,
    "classifier vs undefended": (3 + 12) / 38,
    "SFT vs GRPO refusal": (16 + 2) / 38,
}


@lru_cache(maxsize=None)
def _crit(m: int, alpha: float) -> int:
    """Largest c with BinomCDF(c, m, 1/2) <= alpha/2; -1 if no rejection possible."""
    target = alpha / 2
    for c in range(m // 2 + 1):
        if stats.binom.cdf(c, m, 0.5) > target:
            return c - 1
    return m // 2


def mcnemar_exact_power(n: int, psi: float, p: float, alpha: float = ALPHA) -> float:
    """Exact power of two-sided McNemar.

    psi : P(pair is discordant);  p : P(discordant pair favours arm A), 0.5 = null.
    """
    if psi <= 0:
        return 0.0
    power = 0.0
    for m in range(1, n + 1):
        pm = stats.binom.pmf(m, n, psi)
        if pm < 1e-12:
            continue
        c = _crit(m, alpha)
        if c < 0:
            continue
        lower = stats.binom.cdf(c, m, p)
        upper = stats.binom.sf(m - c - 1, m, p)
        power += pm * min(1.0, lower + upper)
    return power


def mde(n: int, psi: float, alpha: float = ALPHA, target: float = POWER) -> float | None:
    """Smallest marginal difference delta = psi*(2p-1) reaching target power."""
    if mcnemar_exact_power(n, psi, 1.0, alpha) < target:
        return None
    lo, hi = 0.5, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if mcnemar_exact_power(n, psi, mid, alpha) < target:
            lo = mid
        else:
            hi = mid
    return psi * (2 * hi - 1)


def required_n(psi: float, delta: float, alpha: float = ALPHA,
               target: float = POWER, cap: int = 4000) -> int | None:
    p = 0.5 + delta / (2 * psi)
    if not (0.5 < p <= 1.0):
        return None
    n = 20
    while n <= cap and mcnemar_exact_power(n, psi, p, alpha) < target:
        n *= 2
    if n > cap:
        return None
    lo, hi = n // 2, n
    while lo < hi:
        mid = (lo + hi) // 2
        if mcnemar_exact_power(mid, psi, p, alpha) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> None:
    k = len(REPLICATE_ASR)
    mean_asr = sum(REPLICATE_ASR) / k
    sd_obs = stats.tstd(REPLICATE_ASR)
    sd_binom = math.sqrt(mean_asr * (1 - mean_asr) / N_ATTACKS)
    ratio = sd_obs / sd_binom

    print("=" * 74)
    print("1. RESOLUTION RATIO")
    print("=" * 74)
    print(f"  replicates (k={k})         : {[f'{x:.1%}' for x in REPLICATE_ASR]}")
    print(f"  mean ASR                  : {mean_asr:.1%}")
    print(f"  observed run-to-run SD    : {sd_obs*100:.1f} pp")
    print(f"  i.i.d. binomial SD @ n=38 : {sd_binom*100:.1f} pp")
    print(f"  RESOLUTION RATIO          : {ratio:.2f}x")

    df = k - 1
    sd_lo = sd_obs * math.sqrt(df / stats.chi2.ppf(1 - ALPHA / 2, df))
    sd_hi = sd_obs * math.sqrt(df / stats.chi2.ppf(ALPHA / 2, df))
    covers = sd_lo <= sd_binom <= sd_hi
    print()
    print(f"  95% CI on that SD         : [{sd_lo*100:.1f} pp, {sd_hi*100:.1f} pp]")
    print(f"  95% CI on the ratio       : [{sd_lo/sd_binom:.2f}x, {sd_hi/sd_binom:.2f}x]")
    print(f"  CI covers i.i.d. binomial : {'YES' if covers else 'NO'}")
    if covers:
        print("    -> k=3 CANNOT establish excess variance. The 12.1 pp is a point")
        print("       estimate whose CI contains the plain-sampling value.")

    print()
    print("=" * 74)
    print("2. MINIMUM DETECTABLE EFFECT (exact McNemar, n=38, a=0.05, power=0.80)")
    print("=" * 74)
    print(f"  {'paired comparison':36} {'discordance':>12} {'MDE':>12}")
    for name, psi in OBSERVED_DISCORDANCE.items():
        d = mde(N_ATTACKS, psi)
        shown = f"{d*100:.1f} pp" if d is not None else "unreachable"
        print(f"  {name:36} {psi:>11.1%} {shown:>12}")

    print()
    print("=" * 74)
    print("3. REQUIRED PAIRED N (discordance 30%, a=0.05, power=0.80)")
    print("=" * 74)
    for t in (0.05, 0.10, 0.15, 0.20):
        n = required_n(0.30, t)
        print(f"  to detect {t*100:>4.0f} pp : n = {n if n else '>4000'}")

    print()
    print("=" * 74)
    print("4. WHAT AgentDojo (n=629) WOULD BUY")
    print("=" * 74)
    def fmt(d):
        return f"{d*100:5.1f} pp" if d is not None else "unreachable"

    for psi in (0.20, 0.30, 0.40):
        print(f"  discordance {psi:.0%}: MDE {fmt(mde(38, psi))} @ n=38"
              f"   ->  {fmt(mde(629, psi))} @ n=629")

    # ── persist, so audit_report_numbers.py can check the report against it ──
    def pp(d):
        return None if d is None else round(d * 100, 1)

    out = {
        "n_attacks": N_ATTACKS,
        "replicate_asr": REPLICATE_ASR,
        "mean_asr": round(mean_asr, 4),
        "sd_observed_pp": round(sd_obs * 100, 1),
        "sd_binomial_pp": round(sd_binom * 100, 1),
        "resolution_ratio": round(ratio, 2),
        "sd_ci95_pp": [round(sd_lo * 100, 1), round(sd_hi * 100, 1)],
        "ratio_ci95": [round(sd_lo / sd_binom, 2), round(sd_hi / sd_binom, 2)],
        "ci_covers_binomial": bool(covers),
        "alpha": ALPHA,
        "power": POWER,
        "mde_pp_by_comparison": {k: pp(mde(N_ATTACKS, v))
                                 for k, v in OBSERVED_DISCORDANCE.items()},
        "discordance_by_comparison": {k: round(v, 4)
                                      for k, v in OBSERVED_DISCORDANCE.items()},
        "required_n_at_psi_0.30": {f"{int(t*100)}pp": required_n(0.30, t)
                                   for t in (0.05, 0.10, 0.15, 0.20)},
        "mde_pp_n38_vs_n629": {f"psi_{int(p*100)}": [pp(mde(38, p)), pp(mde(629, p))]
                               for p in (0.20, 0.30, 0.40)},
    }
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH.relative_to(OUT_PATH.parents[1])}")


if __name__ == "__main__":
    main()
