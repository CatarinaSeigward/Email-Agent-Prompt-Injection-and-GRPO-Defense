"""Best-of-n over a fixed sample, exactly, with ties.

The judges score on four levels (0/2/3/4), so almost every best-of-n draw is a
tie at the top. Two shortcuts that work for continuous rewards are wrong here:

- grouping the N samples into disjoint blocks of n wastes nearly all of them;
- the closed-form KL of best-of-n, log n - (n-1)/n, assumes no ties. With a
  coarse score, selection pressure stops growing once the top level is almost
  always present in the draw, and the closed form keeps climbing anyway.

This module computes the selection distribution exactly instead. Draw an
n-subset uniformly without replacement from N samples, take the highest proxy
score, break ties uniformly at random. Then

    P(max level = v) = [C(#{proxy <= v}, n) - C(#{proxy < v}, n)] / C(N, n)

and, given the maximum is v, every sample at level v is equally likely to be
the one kept (exchangeability), so each gets P(max = v) / #{proxy = v}.

Everything downstream -- the gold value of best-of-n, its effective KL, the
selection efficiency -- is a sum over those probabilities. No Monte Carlo, so
no sampling noise inside the estimator; `test_bon.py` checks it against Monte
Carlo anyway.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Sequence


def selection_probs(proxy: Sequence[float], n: int) -> list[float]:
    """P(sample i is the one best-of-n keeps), for every i. Sums to 1."""
    N = len(proxy)
    if not 1 <= n <= N:
        raise ValueError(f"need 1 <= n <= N, got n={n}, N={N}")
    counts = Counter(proxy)
    levels = sorted(counts)
    total = math.comb(N, n)
    p_level, below = {}, 0
    for v in levels:
        upto = below + counts[v]
        # Exact integer arithmetic until the final division.
        p_level[v] = (math.comb(upto, n) - math.comb(below, n)) / total
        below = upto
    return [p_level[x] / counts[x] for x in proxy]


def bon_value(proxy: Sequence[float], gold: Sequence[float], n: int) -> float:
    """Expected gold of the sample best-of-n (by proxy) keeps."""
    if len(proxy) != len(gold):
        raise ValueError("proxy and gold must be the same length")
    return sum(p * g for p, g in zip(selection_probs(proxy, n), gold))


def kl_effective(proxy: Sequence[float], n: int) -> float:
    """KL(best-of-n selection || uniform over the N samples), in nats.

    This is the optimisation pressure actually applied. It equals 0 when every
    proxy score is tied, and approaches the closed form when none are.
    """
    N = len(proxy)
    return sum(p * math.log(N * p) for p in selection_probs(proxy, n) if p > 0)


def kl_closed_form(n: int) -> float:
    """Best-of-n KL for a continuous reward: log n - (n-1)/n."""
    return math.log(n) - (n - 1) / n


def efficiency(judge: float, floor: float, ceiling: float) -> float:
    """How far a selector moved gold from random (0) towards the oracle (1)."""
    span = ceiling - floor
    if span <= 0:
        raise ValueError(f"ceiling {ceiling} must exceed floor {floor}")
    return (judge - floor) / span


def bootstrap(stat: Callable[[list[int]], float], N: int, reps: int = 2000,
              seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of stat(indices) over resampled sample indices.

    `stat` receives a list of N indices drawn with replacement and must compute
    everything it needs -- floor, ceiling and judge value -- from that same
    resample, so their dependence is carried into the interval.
    """
    rng = random.Random(seed)
    vals = sorted(stat([rng.randrange(N) for _ in range(N)]) for _ in range(reps))
    return vals[int(0.025 * reps)], vals[int(0.975 * reps) - 1]
