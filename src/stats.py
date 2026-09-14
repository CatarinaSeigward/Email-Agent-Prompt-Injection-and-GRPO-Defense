"""Confidence intervals for the small-n proportions this project reports.

Why not the textbook normal approximation (`p ± 1.96·sqrt(p(1-p)/n)`):
at n=38 with p near 0 or 1 it produces intervals that run past [0,1] and
badly undercover. Every ASR in this repo is a proportion out of n≤38, and
several are 0/12 or 1/12, where the Wald interval is degenerate (width 0
at p=0). Wilson's score interval stays inside [0,1] and keeps nominal
coverage down to small n, which is why Brown, Cai & DasGupta (2001)
recommend it as the default for n < 40.

Two distinct uncertainty sources are reported separately throughout,
because they answer different questions and must not be pooled:

  * **Attack-distribution uncertainty** — "if I had drawn a different 38
    attacks from the same generator, how different would ASR be?"
    Estimated by `wilson_ci` on the n of a single run. This is threat T1.

  * **Run-to-run nondeterminism** — "if I replay the *same* 38 attacks
    again, how different would ASR be?" Estimated by the SD across k
    replicates. This is threat T8.

Pooling k replicates of the same 38 rows into one n=114 Wilson interval
would be wrong: the rows are not independent draws, so it would shrink the
interval by ~sqrt(3) while measuring nothing new about the attack
distribution. `mean_sd` and `wilson_ci` are kept separate for this reason.

References:
  Wilson (1927), J. Am. Stat. Assoc. 22(158):209-212.
  Newcombe (1998), Statistics in Medicine 17:873-890 — method 10 for the
    difference of two independent proportions, used by `diff_ci`.
  Brown, Cai & DasGupta (2001), Statistical Science 16(2):101-133.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z_95 = 1.959963984540054  # two-sided 95%


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def pct(self, digits: int = 1) -> str:
        """Render as `36.8% [22.7, 53.6]` for direct paste into a report table."""
        return (f"{self.point * 100:.{digits}f}% "
                f"[{self.low * 100:.{digits}f}, {self.high * 100:.{digits}f}]")

    def pp_width(self) -> float:
        """Interval width in percentage points."""
        return (self.high - self.low) * 100


def wilson_ci(successes: int, n: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a single binomial proportion.

    Degenerate but well-defined at successes=0 or successes=n, unlike Wald.
    """
    if n <= 0:
        return Interval(0.0, 0.0, 1.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


def diff_ci(s1: int, n1: int, s2: int, n2: int, z: float = Z_95) -> Interval:
    """Newcombe method-10 interval for p1 - p2 (two independent proportions).

    Built from the two Wilson intervals rather than from a pooled SE, so it
    inherits Wilson's small-n behaviour. The decision rule is the usual one:
    if the interval excludes 0, the difference is significant at the stated
    level.

    Note the independence assumption. Comparing two *defence configurations*
    replayed over the same attack rows violates it — the rows are shared, so
    the true interval is narrower than this (paired designs have less
    variance). Treat this as the conservative bound; `mcnemar_b_c` is the
    correct paired test when per-row outcomes are available.
    """
    p1, p2 = s1 / n1 if n1 else 0.0, s2 / n2 if n2 else 0.0
    w1, w2 = wilson_ci(s1, n1, z), wilson_ci(s2, n2, z)
    d = p1 - p2
    low = d - math.sqrt((p1 - w1.low) ** 2 + (w2.high - p2) ** 2)
    high = d + math.sqrt((w1.high - p1) ** 2 + (p2 - w2.low) ** 2)
    return Interval(d, max(-1.0, low), min(1.0, high))


def mcnemar_b_c(pairs: list[tuple[bool, bool]]) -> dict:
    """Exact McNemar test on paired binary outcomes (same rows, two configs).

    `pairs` is [(config_a_won, config_b_won), ...] row-aligned. Returns the
    discordant counts and the exact two-sided binomial p-value. This is the
    right test for "did the hardened prompt change the outcome on these
    specific attacks", because both configs see identical inputs — the
    concordant rows carry no information about the difference and pooling
    them (as `diff_ci` does) throws away power.
    """
    b = sum(1 for a, bb in pairs if a and not bb)  # A won, B didn't
    c = sum(1 for a, bb in pairs if bb and not a)  # B won, A didn't
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0}
    # Exact two-sided binomial test against p=0.5 on the discordant pairs.
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return {"b": b, "c": c, "n_discordant": n, "p_value": min(1.0, 2 * tail)}


def mean_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample SD (ddof=1). SD is 0.0 for a single value."""
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var)


if __name__ == "__main__":
    # Sanity check against the numbers final_report.md currently reports,
    # to make T1's "±25pp" claim concrete rather than hand-waved.
    print("Current headline numbers with Wilson 95% CIs:\n")
    for name, s, n in [
        ("baseline overall   14/38", 14, 38),
        ("classifier overall   5/38", 5, 38),
        ("verifier-loose       5/38", 5, 38),
        ("combined-loose       9/38", 9, 38),
        ("A1 baseline          6/12", 6, 12),
        ("A3 verifier strict   4/12", 4, 12),
    ]:
        ci = wilson_ci(s, n)
        print(f"  {name:28} {ci.pct():>24}   width={ci.pp_width():.1f}pp")

    print("\nThe §7.7 headline comparison (combined-loose vs verifier-loose):")
    d = diff_ci(9, 38, 5, 38)
    print(f"  unpaired difference {d.pct()}  -> "
          f"{'excludes' if d.low > 0 or d.high < 0 else 'INCLUDES'} zero")
