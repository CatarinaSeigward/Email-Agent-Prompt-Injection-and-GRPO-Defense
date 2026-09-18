"""Does the judge divergence compound under search budget?

Reads results/qd_pilot_summary.json (archive overlap logged every checkpoint)
and reports elite-archive Jaccard as a function of candidates evaluated -- the
one number the pilot exists to produce.

The static measurement (rank_divergence.py, ~38 PAIR strategies) showed judges
disagree about partial credit. It could not show whether selection amplifies
that or absorbs it, because its population was not produced by selection. This
does, because here the archive IS the product of selection under each judge.

Caveat carried at every step: the pilot ran trials=1, so f_asr and f_peak are
monotone in a single severity level and the Pareto test is degenerate. What
survives that is the *mechanic* -- whether archives built from one candidate
stream drift apart as it lengthens -- not a calibrated effect size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _trend(series: list[float]) -> str:
    """Cheap monotone read: net change and whether it moves one direction."""
    if len(series) < 2:
        return "n/a"
    net = series[-1] - series[0]
    downs = sum(1 for a, b in zip(series, series[1:]) if b < a - 1e-9)
    ups = sum(1 for a, b in zip(series, series[1:]) if b > a + 1e-9)
    dirn = ("falling" if downs > ups else "rising" if ups > downs else "flat")
    return f"{series[0]:.2f} -> {series[-1]:.2f} (net {net:+.2f}, {dirn})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pilot")
    args = ap.parse_args()
    s = json.loads((RESULTS / f"qd_{args.tag}_summary.json").read_text(encoding="utf-8"))
    models = s["judges"]
    cps = s["checkpoints"]
    budgets = [c["n"] for c in cps]
    jj = [f"{a}|{b}" for i, a in enumerate(models) for b in models[i + 1:]]
    jo = [f"{m}|ORACLE" for m in models]

    def col(pair: str, key: str = "elite_jaccard") -> list[float]:
        return [c["overlap"][pair][key] for c in cps]

    print(f"pilot: {s['candidates']} candidates, trials={s['trials']}, "
          f"{s['errors']} errors, {round(s['minutes'])} min\n")
    if s.get("pareto_degenerate"):
        print("!! trials=1: Pareto degenerate. Mechanic only, not an effect size.\n")

    print("ELITE-ARCHIVE JACCARD vs BUDGET")
    print("budget " + " ".join(f"{b:>5}" for b in budgets))
    print("-" * (7 + 6 * len(budgets)))
    for p in jj + jo:
        tag = p.replace("gpt-4o-mini", "4om").replace("claude-sonnet-5", "cs5") \
               .replace("gpt-4o", "4o").replace("|", "/")
        print(f"{tag:>10} " + " ".join(f"{v:>5.2f}" for v in col(p)))

    print("\nTRENDS (start -> end)")
    print("  judge vs judge:")
    for p in jj:
        print(f"    {p:34} {_trend(col(p))}")
    print("  judge vs oracle (hidden from selection):")
    for p in jo:
        print(f"    {p:34} {_trend(col(p))}")

    # Which judges drive the drift: mean overlap each judge holds with the
    # others, early vs late. A judge whose mean overlap collapses is the one
    # walking away from consensus.
    print("\nPER-JUDGE MEAN OVERLAP WITH THE OTHER JUDGES (early vs late)")
    early_i, late_i = 0, len(cps) - 1
    for m in models:
        ps = [p for p in jj if m in p.split("|")]
        e = sum(cps[early_i]["overlap"][p]["elite_jaccard"] for p in ps) / len(ps)
        l = sum(cps[late_i]["overlap"][p]["elite_jaccard"] for p in ps) / len(ps)
        vo = f"{m}|ORACLE"
        eo = cps[early_i]["overlap"][vo]["elite_jaccard"]
        lo = cps[late_i]["overlap"][vo]["elite_jaccard"]
        print(f"  {m:18} peers {e:.2f} -> {l:.2f}   oracle {eo:.2f} -> {lo:.2f}")

    out = {
        "candidates": s["candidates"], "trials": s["trials"],
        "pareto_degenerate": s.get("pareto_degenerate", False),
        "budgets": budgets,
        "elite_jaccard": {p: col(p) for p in jj + jo},
        "member_jaccard": {p: col(p, "member_jaccard") for p in jj + jo},
        "trends": {p: _trend(col(p)) for p in jj + jo},
    }
    outfile = RESULTS / f"qd_{args.tag}_analysis.json"
    outfile.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {outfile}")


if __name__ == "__main__":
    main()
