"""E2 analysis -- applies the PRE-REGISTERED decision rule, nothing else.

Reads results/e2_preregistration.json and refuses to deviate from it. The rule
was fixed before the data existed; this script only executes it.

Single source of truth
----------------------
All counts come from results/e2_rescore.json, which scores with
src/eval.py::is_attack_success -- the same action-grounded criterion §4.9 and
§1.2 use. This script deliberately does NOT read the per-arm summary files'
judge counts. An earlier version did, which would have reported 6/30 and 2/30
(fabricated successes, §4.11) instead of 0/30 and 0/30.

Run: uv run python scripts/e2_rescore.py && uv run python scripts/e2_analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def exact_mcnemar(b: int, c: int) -> float:
    from scipy import stats

    m = b + c
    return 1.0 if m == 0 else float(stats.binomtest(b, m, 0.5).pvalue)


def rule_of_three_upper(n: int) -> float:
    """Exact one-sided 95% upper bound on p when 0 of n trials succeeded."""
    return 1 - 0.05 ** (1 / n)


def main() -> None:
    prereg_p = RESULTS / "e2_preregistration.json"
    rescore_p = RESULTS / "e2_rescore.json"
    for p in (prereg_p, rescore_p):
        if not p.exists():
            sys.exit(f"missing {p.relative_to(ROOT)} -- run scripts/e2_rescore.py first")

    pr = json.loads(prereg_p.read_text(encoding="utf-8"))
    rs = json.loads(rescore_p.read_text(encoding="utf-8"))
    thr = pr["significance_threshold_c"]
    arms = rs["arms"]

    print("=" * 74)
    print("E2 ANALYSIS -- pre-registered rule only")
    print("=" * 74)
    print(f"  scorer               : {rs['criterion']}")
    print(f"  registered threshold : c >= {thr} of {pr['n_paired_seeds']} seeds")
    print(f"  alpha                : {pr['alpha']}")
    print()

    if "B" not in arms:
        sys.exit("arm B has not been run.")
    b = arms["B"]
    a1 = arms.get("A1")
    n = b["n_seeds"]

    # ── headline: B vs A0 (§4.9 measured 0 successes) ────────────────
    print("-" * 74)
    print("COMPARISON 1 (headline, pre-registered):  B  vs  A0 (§4.9, 0 successes)")
    print("-" * 74)
    c = b["seeds_cracked_action"]
    p = exact_mcnemar(0, c)
    print(f"  arm B cracked      : {c}/{n} ({c/n:.1%})")
    print(f"  discordant (b, c)  : (0, {c})")
    print(f"  exact McNemar p    : {p:.4f}")
    print()
    if c >= thr:
        print("  VERDICT: SIGNIFICANT -- §4.9's 0.0% does not survive a "
              "defense-aware attacker.")
    else:
        ub = rule_of_three_upper(n)
        print(f"  VERDICT: NOT DECIDABLE. c={c} is below the registered threshold {thr}.")
        print( "           Per the reporting rule, no p-value direction is claimed.")
        print(f"           Exact 95% upper bound on the true rate: {ub:.1%}")
        print( "           This does NOT establish robustness -- only that THIS")
        print(f"           attacker at THIS budget ({b['n_rollouts']} rollouts) did not win.")

    # ── mechanism: B vs A1 ───────────────────────────────────────────
    print()
    print("-" * 74)
    print("COMPARISON 2 (mechanism, pre-registered):  B (aware)  vs  A1 (not aware)")
    print("-" * 74)
    if a1 is None:
        print("  arm A1 not run -- B alone confounds awareness with iteration.")
    else:
        pb, pa = b["per_seed_action"], a1["per_seed_action"]
        shared = sorted(set(pb) & set(pa))
        bb = sum(1 for s in shared if pa[s] and not pb[s])
        cc = sum(1 for s in shared if pb[s] and not pa[s])
        print(f"  paired seeds       : {len(shared)}")
        print(f"  A1 cracked         : {a1['seeds_cracked_action']}/{a1['n_seeds']}")
        print(f"  B  cracked         : {c}/{n}")
        print(f"  discordant (b, c)  : ({bb}, {cc})")
        print(f"  exact McNemar p    : {exact_mcnemar(bb, cc):.4f}")
        if bb == cc == 0:
            print("  -> both arms null; the mechanism comparison was never exercised.")

    # ── P3, qualitative ──────────────────────────────────────────────
    print()
    print("-" * 74)
    print("P3 (pre-registered): which rule did successful attacks break?")
    print("-" * 74)
    total_wins = sum(a["seeds_cracked_action"] for a in arms.values())
    if total_wins == 0:
        print("  No action-grounded successes in either arm. P3 is unevaluable.")
    else:
        for arm in arms:
            log = RESULTS / f"e2_{arm}.jsonl"
            rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
            for r in rows:
                from e2_rescore import action_grounded_success  # noqa: E402
                ok, hits = action_grounded_success(r["agent_actions"])
                if ok:
                    print(f"  [{arm}] {r['seed_id']} r{r['round']}: {hits}")

    out = {
        "scorer": rs["criterion"], "threshold_c": thr, "n_seeds": n,
        "arm_B_cracked": c, "arm_A1_cracked": None if a1 is None else a1["seeds_cracked_action"],
        "headline_p": round(p, 6), "headline_significant": bool(c >= thr),
        "upper_bound_95": round(rule_of_three_upper(n), 4) if c == 0 else None,
        "p3_evaluable": total_wins > 0,
    }
    (RESULTS / "e2_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print()
    print("  wrote results/e2_analysis.json")


if __name__ == "__main__":
    main()
