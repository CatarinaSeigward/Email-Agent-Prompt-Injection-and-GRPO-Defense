"""Put an interval on the "25%" and say what p = 0.0013 actually tests.

final_report.md used to print "three judges agree on only 25% of the
strategies any of them gives partial credit to (p = 0.0013)". Those are two
different measurements, and the p-value belongs to neither the 25% nor any
test of it:

  25%        three-way agreement over strategies: 9 shared / 36 in the union.
             Descriptive, on small sets, with no interval until this script.
  p = 0.0013 exact McNemar on single rollouts, gpt-4o-mini vs claude-sonnet-5,
             "did this judge give any credit": (b, c) = (16, 2). It shows the
             two judges' credit rates differ systematically. It says nothing
             about how much the credited sets overlap.

This script recomputes both from results/dual_scoring.json with the same
strategy grouping as scripts/rank_divergence.py, adds a percentile bootstrap
over strategies for the agreement and the pairwise Jaccards, and records how
many of the 447 rollouts could carry a disagreement at all: 339 are
hardened-prompt runs with no destructive call, where every judge and the
oracle agree trivially.

No API calls. Output: results/partial_credit_ci.json
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from rank_divergence import fitness  # noqa: E402  same f_peak as the headline
from src.stats import mcnemar_b_c  # noqa: E402

RESULTS = REPO / "results"
B = 10_000
SEED = 0


def main() -> None:
    data = json.loads((RESULTS / "dual_scoring.json").read_text(encoding="utf-8"))
    models = data["judges"]
    rows = [r for r in data["rows"]
            if all(r["judges"][m]["level"] is not None for m in models)]

    trials: dict[tuple, dict[str, list[int]]] = collections.defaultdict(
        lambda: {m: [] for m in models})
    for r in rows:
        for m in models:
            trials[(r["source"], r["seed_id"])][m].append(r["judges"][m]["level"])
    strategies = list(trials)
    credited = {m: {k for k in strategies if fitness(trials[k][m])[1] > 0} for m in models}

    def overlap(sample: list[tuple]) -> dict[str, float]:
        """Agreement over a (multi)set of strategies, counting repeats."""
        out = {}
        union = [k for k in sample if any(k in credited[m] for m in models)]
        shared = [k for k in union if all(k in credited[m] for m in models)]
        out["all_three"] = len(shared) / len(union) if union else float("nan")
        for i, x in enumerate(models):
            for y in models[i + 1:]:
                u = [k for k in sample if k in credited[x] or k in credited[y]]
                s = [k for k in u if k in credited[x] and k in credited[y]]
                out[f"{x}|{y}"] = len(s) / len(u) if u else float("nan")
        return out

    point = overlap(strategies)
    rng = random.Random(SEED)
    draws = collections.defaultdict(list)
    for _ in range(B):
        for k, v in overlap(rng.choices(strategies, k=len(strategies))).items():
            draws[k].append(v)

    def pct(xs: list[float], q: float) -> float:
        xs = sorted(x for x in xs if x == x)
        return xs[min(len(xs) - 1, int(q * len(xs)))]

    ci = {k: [round(pct(v, 0.025), 3), round(pct(v, 0.975), 3)] for k, v in draws.items()}

    union = set().union(*credited.values())
    shared = set.intersection(*credited.values())
    a, b = models[0], models[1]
    mc = mcnemar_b_c([(r["judges"][a]["level"] > 0, r["judges"][b]["level"] > 0)
                      for r in rows])
    by_defense = collections.Counter(r["defense"] for r in rows)

    out = {
        "n_rollouts": len(rows),
        "rollouts_by_defense": dict(by_defense),
        "n_strategies": len(strategies),
        "credited_per_judge": {m: len(credited[m]) for m in models},
        "union": len(union),
        "shared_by_all_three": len(shared),
        "agreement_all_three": round(point["all_three"], 4),
        "pairwise_jaccard": {k: round(v, 4) for k, v in point.items() if k != "all_three"},
        "bootstrap": {"B": B, "seed": SEED, "unit": "strategy (source, seed_id)",
                      "ci95": ci},
        "mcnemar_rollout_any_credit": {
            "pair": f"{a} vs {b}", "question": "did the judge give any credit (level > 0)",
            "b": mc["b"], "c": mc["c"], "p_exact": round(mc["p_value"], 4),
            "note": "tests whether the two judges' credit RATES differ on the same "
                    "rollouts; it is not a test of the 25% overlap"},
    }
    (RESULTS / "partial_credit_ci.json").write_text(json.dumps(out, indent=2),
                                                    encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
