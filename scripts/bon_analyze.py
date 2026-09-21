"""BoN Stage 5: selection efficiency per judge, and every pre-registered test.

Reads the frozen design from results/bon_preregistration.json and does exactly
what it says -- primary endpoint at n=32, robustness at n=16, H-A/B/C/D and the
saturation check -- with the frozen estimator in src/bon.py. Refuses to run if
that estimator, the judge path or the oracle changed since registration.

    gold      mean over trials 1-2 of oracle level / 4   (held out)
    floor     uniform random selection  = mean gold
    ceiling   best-of-n by the oracle's own trial-0 level
    judge J   best-of-n by J's trial-0 level
    efficiency(J, n) = (gold_J - floor) / (ceiling_n - floor)

Every bootstrap interval resamples the N samples and recomputes floor, ceiling
and judge value together, so their dependence is carried into the interval.

Run: uv run python scripts/bon_analyze.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bon import bon_value, bootstrap, kl_closed_form, kl_effective  # noqa: E402

PREREG = ROOT / "results" / "bon_preregistration.json"
SAMPLES = ROOT / "results" / "bon_samples.jsonl"
SCORES = ROOT / "results" / "bon_scores.json"
OUT = ROOT / "results" / "bon_curves.json"

NS = (1, 2, 4, 8, 16, 32)
HD_NS = (4, 8, 16, 32)


def load_prereg() -> dict:
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    stale = [f for f, h in pre["frozen_code_sha256"].items()
             if hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    if stale:
        raise SystemExit(f"REFUSING TO RUN: frozen code changed since pre-registration: {stale}")
    return pre


def ranks(xs: list[float]) -> list[float]:
    """Average ranks, 1-based (ties share their mean rank)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return cov / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def spearman_exact_p_negative(x: list[float], y: list[float]) -> float:
    """One-sided exact permutation p for rho <= observed (predicted direction)."""
    obs = spearman(x, y)
    perms = list(itertools.permutations(y))
    return sum(1 for p in perms if spearman(x, list(p)) <= obs + 1e-12) / len(perms)


def main() -> None:
    pre = load_prereg()
    judges = pre["judges"]
    samples = {s["id"]: s for s in (json.loads(l) for l in SAMPLES.open(encoding="utf-8")
                                    if l.strip())}
    scores = json.loads(SCORES.read_text(encoding="utf-8"))
    ids = sorted(samples)
    N = len(ids)
    if N != pre["design"]["N"]:
        raise SystemExit(f"expected {pre['design']['N']} samples, found {N}")

    lv = {s: [t["oracle_level"] for t in sorted(samples[s]["trials"], key=lambda t: t["trial"])]
          for s in ids}
    gold = [(lv[s][1] + lv[s][2]) / 8 for s in ids]          # trials 1-2, / 4
    oracle0 = [lv[s][0] for s in ids]
    # Pre-registered: a failed judge call (level None) counts as 0.
    proxy = {m: [scores["levels"][m][s] if scores["levels"][m][s] is not None else 0
                 for s in ids] for m in judges}
    proxy_no_l3 = {m: [0 if v == 3 else v for v in proxy[m]] for m in judges}

    def curves(p, g):
        return {n: bon_value(p, g, n) for n in NS}

    floor = sum(gold) / N
    ceiling = curves(oracle0, gold)
    judge_curve = {m: curves(proxy[m], gold) for m in judges}

    def eff(value, n):
        return (value - floor) / (ceiling[n] - floor)

    efficiency = {m: {n: eff(judge_curve[m][n], n) for n in NS if n > 1} for m in judges}
    efficiency_no_l3 = {m: eff(bon_value(proxy_no_l3[m], gold, 32), 32) for m in judges}

    # ---- bootstrap: floor, ceiling and judge recomputed on every resample ----
    reps, seed = 2000, pre["seeds"]["bootstrap"]

    def eff_stat(m, n):
        def stat(ix):
            g = [gold[i] for i in ix]
            f = sum(g) / len(g)
            c = bon_value([oracle0[i] for i in ix], g, n)
            j = bon_value([proxy[m][i] for i in ix], g, n)
            return (j - f) / (c - f) if c > f else float("nan")
        return stat

    ci = {m: {n: bootstrap(eff_stat(m, n), N, reps, seed) for n in (16, 32)} for m in judges}

    # ---- H-A: at least one pair with non-overlapping 95% intervals at n=32 ----
    separated = [(a, b) for a, b in itertools.combinations(judges, 2)
                 if ci[a][32][1] < ci[b][32][0] or ci[b][32][1] < ci[a][32][0]]

    # ---- H-B: static calibration (over-credit at n=1) predicts efficiency ----
    over_credit = {m: sum(1 for v in proxy[m] if v > 0) / N - sum(1 for v in oracle0 if v > 0) / N
                   for m in judges}
    xs = [over_credit[m] for m in judges]
    ys = [efficiency[m][32] for m in judges]
    rho = spearman(xs, ys)
    p_hb = spearman_exact_p_negative(xs, ys)

    # ---- H-D: does the least efficient judge lose efficiency as n grows? ----
    worst = min(judges, key=lambda m: efficiency[m][32])

    def slope(ns, vals):
        xs_ = [math.log2(n) for n in ns]
        mx, my = sum(xs_) / len(xs_), sum(vals) / len(vals)
        return (sum((a - mx) * (b - my) for a, b in zip(xs_, vals))
                / sum((a - mx) ** 2 for a in xs_))

    def slope_stat(ix):
        g = [gold[i] for i in ix]
        f = sum(g) / len(g)
        vals = []
        for n in HD_NS:
            c = bon_value([oracle0[i] for i in ix], g, n)
            j = bon_value([proxy[worst][i] for i in ix], g, n)
            vals.append((j - f) / (c - f) if c > f else float("nan"))
        return slope(HD_NS, vals)

    hd_slope = slope(HD_NS, [efficiency[worst][n] for n in HD_NS])
    hd_ci = bootstrap(slope_stat, N, reps, seed)

    # ---- saturation: the four-level score caps pressure at log(N / #top) ----
    saturation = {}
    for m in judges:
        top = max(proxy[m])
        n_top = sum(1 for v in proxy[m] if v == top)
        saturation[m] = {
            "top_level": top, "top_share": round(n_top / N, 4),
            "kl_ceiling_log_inv_top_share": round(math.log(N / n_top), 4),
            "kl_effective": {n: round(kl_effective(proxy[m], n), 4) for n in NS},
            "kl_closed_form": {n: round(kl_closed_form(n), 4) for n in NS},
        }

    # ---- self-noise: agreement over CLEAN triples; rate-limit Nones reported ----
    # The two priciest OpenAI judges kept hitting TPM limits on the cache-bypassing
    # self-noise calls, leaving some triples with a None. That is missing data, not
    # disagreement, so it is excluded from the agreement rate and counted on its
    # own. The main proxy scoring these judges' efficiency rests on has 0 failures.
    noise = {}
    for m in judges:
        reps_ = list(scores["self_noise"][m].values())
        clean = [r for r in reps_ if None not in r]
        agree = sum(1 for r in clean if len(set(r)) == 1)
        noise[m] = {"n": len(reps_), "clean": len(clean),
                    "rate_limited": len(reps_) - len(clean),
                    "self_agreement_clean": round(agree / max(len(clean), 1), 3)}

    r4 = lambda x: round(x, 4)  # noqa: E731
    out = {
        "meta": {"N": N, "judges": judges, "ns": list(NS), "bootstrap_reps": reps,
                 "bootstrap_seed": seed, "estimator": "src/bon.py (exact, tie-aware)",
                 "deviations": "results/bon_deviations.json"},
        "floor": r4(floor),
        "ceiling": {n: r4(v) for n, v in ceiling.items()},
        "gold_by_judge": {m: {n: r4(v) for n, v in judge_curve[m].items()} for m in judges},
        "efficiency": {m: {n: r4(v) for n, v in efficiency[m].items()} for m in judges},
        "efficiency_ci95": {m: {n: [r4(lo), r4(hi)] for n, (lo, hi) in ci[m].items()}
                            for m in judges},
        "over_credit_n1": {m: r4(v) for m, v in over_credit.items()},
        "H-A": {"separated_pairs_n32": separated, "supported": bool(separated)},
        "H-B": {"spearman_rho": r4(rho), "p_one_sided_exact": r4(p_hb),
                "supported": p_hb < 0.05, "direction_predicted": "negative"},
        "H-C": {m: {"over_credit": r4(over_credit[m]), "efficiency_n32": r4(efficiency[m][32])}
                for m in judges},
        "H-D": {"judge": worst, "slope_eff_per_doubling": r4(hd_slope),
                "ci95": [r4(hd_ci[0]), r4(hd_ci[1])],
                "goodhart_signature": hd_ci[1] < 0},
        "saturation": saturation,
        "robustness_proxy_without_L3": {m: r4(v) for m, v in efficiency_no_l3.items()},
        "self_noise": noise,
        "judge_failures": scores["failures"],
    }

    # Self-checks before writing, in the atlas's style: refuse on broken arithmetic.
    problems = []
    if not ceiling[32] > floor:
        problems.append(f"ceiling {ceiling[32]:.3f} does not beat floor {floor:.3f}")
    for m in judges:
        for n in NS:
            if not -1e-9 <= judge_curve[m][n] <= 1 + 1e-9:
                problems.append(f"{m} gold at n={n} outside [0,1]")
        lo, hi = ci[m][32]
        if not lo <= efficiency[m][32] <= hi:
            problems.append(f"{m}: point estimate outside its own interval")
    if problems:
        print("REFUSING TO WRITE -- self-checks failed:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"floor {floor:.3f} · ceiling@32 {ceiling[32]:.3f}\n")
    print(f"{'judge':18s} {'over-credit':>11} {'eff@16':>7} {'eff@32':>7}  {'95% CI @32':>15} "
          f"{'no-L3':>6} {'KL@32':>6} {'cap':>5} {'self':>5}")
    for m in sorted(judges, key=lambda m: efficiency[m][32]):
        lo, hi = ci[m][32]
        s = saturation[m]
        print(f"{m:18s} {over_credit[m]:>+11.3f} {efficiency[m][16]:>7.3f} "
              f"{efficiency[m][32]:>7.3f}  [{lo:>6.3f},{hi:>6.3f}] "
              f"{efficiency_no_l3[m]:>6.3f} {s['kl_effective'][32]:>6.2f} "
              f"{s['kl_ceiling_log_inv_top_share']:>5.2f} {noise[m]['self_agreement_clean']:>5.0%}"
              + (f" (RL {noise[m]['rate_limited']})" if noise[m]['rate_limited'] else ""))
    print(f"\nH-A  separated pairs at n=32: {len(separated)}  -> "
          f"{'SUPPORTED' if separated else 'not supported'}")
    print(f"H-B  Spearman rho(over-credit, efficiency) = {rho:+.3f}, exact one-sided "
          f"p = {p_hb:.4f}  -> {'SUPPORTED' if p_hb < 0.05 else 'not supported'}")
    print(f"H-D  {worst}: efficiency slope per doubling of n = {hd_slope:+.3f} "
          f"[{hd_ci[0]:+.3f}, {hd_ci[1]:+.3f}]  -> "
          f"{'Goodhart signature' if hd_ci[1] < 0 else 'no clear decline'}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
