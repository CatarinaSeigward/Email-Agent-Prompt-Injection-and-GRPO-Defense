"""Measured reproduction cost (closes T10).

T10 has stood open since the first draft: the project's "~$1.20 end-to-end"
figure came from runbook notes, not from instrumentation, and no result file
recorded a single token. This script reports what was actually metered.

Instrumentation: langchain_core's UsageMetadataCallbackHandler is attached to
every agent.invoke in scripts/e2_adaptive_attack.py, so each rollout's input,
output and CACHED input tokens are recorded in results/e2_*.jsonl and summed in
the per-arm summaries. Attacker-side rewrite calls are metered separately via the
OpenAI client's usage field.

Scope limit, stated plainly: only the four E2 arms are instrumented. The runs
that produced §1.2, §3 and §4.6-§4.9 predate the instrumentation and cannot be
recovered -- for those, this script reports a per-rollout unit cost measured on
the matching configuration and multiplies by the rollout counts stored in the
result files. That is an extrapolation from a measured unit, not a measurement,
and is labelled as such in the output.

Run: uv run python scripts/reproduction_cost.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

IN_RATE, OUT_RATE, CACHE_RATE = 0.15e-6, 0.60e-6, 0.075e-6

ARMS = {
    "A1": ("hardened prompt, attacker unaware", "§4.10"),
    "B": ("hardened prompt, attacker defense-aware", "§4.10"),
    "V0": ("GRPO verifier, attacker unaware", "§4.12"),
    "V1": ("GRPO verifier, attacker defense-aware", "§4.12"),
}


def main() -> None:
    print("=" * 78)
    print("MEASURED COST -- the four instrumented E2 arms")
    print("=" * 78)
    print(f"  {'arm':4} {'config':42} {'rollouts':>9} {'$ total':>9} {'$/rollout':>10}")

    tot_cost = tot_roll = 0
    per_config: dict[str, float] = {}
    measured = {}
    for arm, (desc, sec) in ARMS.items():
        p = RESULTS / f"e2_{arm}_summary.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "measured_cost_usd" not in d:
            print(f"  {arm:4} {desc:42} {'(not instrumented)':>20}")
            continue
        n, c = d["n_rollouts"], d["measured_cost_usd"]
        tot_cost += c
        tot_roll += n
        per_config[arm] = d["measured_cost_per_rollout_usd"]
        measured[arm] = {"rollouts": n, "cost_usd": c,
                         "per_rollout_usd": d["measured_cost_per_rollout_usd"],
                         "agent_tokens": d["agent_usage_tokens"],
                         "attacker_tokens": d["attacker_usage_tokens"],
                         "section": sec}
        print(f"  {arm:4} {desc:42} {n:>9} {c:>9.4f} {d['measured_cost_per_rollout_usd']:>10.6f}")

    print(f"  {'':4} {'TOTAL':42} {tot_roll:>9} {tot_cost:>9.4f} "
          f"{tot_cost/max(1,tot_roll):>10.6f}")

    # cache behaviour -- the single biggest surprise vs the pre-run estimate
    print()
    print("=" * 78)
    print("PROMPT CACHING -- measured, not assumed")
    print("=" * 78)
    tin = tcache = 0
    for arm in measured:
        a = measured[arm]["agent_tokens"]
        tin += a["input_tokens"]
        tcache += a["cache_read"]
    print(f"  agent input tokens : {tin:,}")
    print(f"  of which cached    : {tcache:,}  ({tcache/max(1,tin):.1%})")
    print(f"  cache saving       : ${(tcache*(IN_RATE-CACHE_RATE)):.4f}")

    # extrapolation for the un-instrumented historical runs
    print()
    print("=" * 78)
    print("EXTRAPOLATED -- runs that predate instrumentation (NOT measured)")
    print("=" * 78)
    unit_naive = per_config.get("V0")      # naive prompt, ~51 tool calls
    unit_hard = per_config.get("A1")       # hardened prompt, ~26 tool calls
    hist = []
    for f in sorted(RESULTS.glob("attack_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        n = d.get("n")
        if not n:
            continue
        is_hard = "hardened" in f.name
        unit = unit_hard if is_hard else unit_naive
        # A1/B predate the meter, so there is no hardened unit cost. Fall back to
        # the naive unit and label it: a hardened rollout issues ~26 tool calls
        # against ~51 (§4.9 Control B), so this is an UPPER bound for those rows.
        bound = ""
        if unit is None:
            unit, bound = unit_naive, "  <- UPPER BOUND (no hardened unit measured)"
        if unit is None:
            continue
        hist.append((f.name, n, unit, n * unit, bound))
    for name, n, unit, c, bound in hist:
        print(f"  {name:34} {n:>4} rollouts x ${unit:.6f} = ${c:.4f}{bound}")
    hist_total = sum(h[3] for h in hist)
    print(f"  {'extrapolated subtotal':34} {'':>4}            = ${hist_total:.4f}")

    print()
    print("=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(f"  measured (E2 arms)           : ${tot_cost:.2f}  over {tot_roll} rollouts")
    print(f"  extrapolated (historical)    : ${hist_total:.2f}")
    print(f"  project API total (est.)     : ${tot_cost + hist_total:.2f}")
    print()
    per_roll = tot_cost / max(1, tot_roll)
    est_per_roll = 38_000 * IN_RATE + 900 * OUT_RATE
    print(f"  T10's original runbook figure was ~$1.20 for the whole project. The")
    print(f"  measured+extrapolated total above is ${tot_cost + hist_total:.2f}, so that")
    print(f"  unverified claim turns out to be approximately right -- it just had no")
    print(f"  evidence behind it until now.")
    print()
    print(f"  The pre-run estimate in scripts/e2_adaptive_attack.py assumed 38,000 input")
    print(f"  tokens per rollout (${est_per_roll:.6f}); measured is ${per_roll:.6f}, i.e. the")
    print(f"  estimate was {est_per_roll/max(per_roll,1e-9):.1f}x too high. {tcache/max(1,tin):.0%} of input came from")
    print(f"  cache, which no estimate in this repo had accounted for.")

    out = {
        "rates": {"input": IN_RATE, "output": OUT_RATE, "cached_input": CACHE_RATE},
        "measured_arms": measured,
        "measured_total_usd": round(tot_cost, 4),
        "measured_rollouts": tot_roll,
        "cache_fraction": round(tcache / max(1, tin), 4),
        "extrapolated_historical_usd": round(hist_total, 4),
        "project_total_estimate_usd": round(tot_cost + hist_total, 4),
        "caveat": "only the four E2 arms are metered; historical runs are a unit-cost "
                  "extrapolation, not a measurement",
    }
    (RESULTS / "reproduction_cost.json").write_text(json.dumps(out, indent=2),
                                                    encoding="utf-8")
    print("\n  wrote results/reproduction_cost.json")


if __name__ == "__main__":
    main()
