"""P0: the two baselines final_report.md never measured, with replicates.

Closes threats T12 (no SYSTEM_PROMPT_HARDENED baseline) and T8/T4 (single
run per cell, no variance estimate) from final_report.md §9.1.

What it runs
------------
  {naive, hardened} system prompt  x  k replicates  x  {38 attacks, 10 benign}

`naive` is re-run rather than reused from results/attack_baseline.json on
purpose: that file is a single run from an earlier session against an
unpinned model alias. Comparing a k=3 mean against a k=1 number from a
different date is the exact methodological error this script exists to fix.

Why both uncertainty sources are reported
-----------------------------------------
  Wilson CI (within a replicate, n=38) answers "would a different 38 attacks
    give a different ASR" — the attack-distribution question (T1).
  SD across replicates answers "would replaying these same 38 attacks give a
    different ASR" — the nondeterminism question (T8).
They are never pooled; see src/stats.py for why.

Resumability: each (config, replicate) cell writes its own result file and is
skipped if that file already exists. Delete the file to force a re-run.

Usage
-----
  $env:PYTHONIOENCODING="utf-8"
  uv run python scripts/eval_p0.py            # k=3, both configs
  uv run python scripts/eval_p0.py --k 5      # more replicates
  uv run python scripts/eval_p0.py --analyze-only   # re-aggregate, no API calls
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.agent import SYSTEM_PROMPT_HARDENED, SYSTEM_PROMPT_NAIVE  # noqa: E402
from src.eval import run_attack_replay, run_benign  # noqa: E402
from src.stats import mcnemar_b_c, mean_sd, wilson_ci  # noqa: E402

DATA = REPO / "data"
RESULTS = REPO / "results"
ATTACK_LOG = DATA / "attack_log.jsonl"

CONFIGS = {
    "naive": SYSTEM_PROMPT_NAIVE,
    "hardened": SYSTEM_PROMPT_HARDENED,
}


def cell_label(config: str, rep: int) -> str:
    return f"p0_{config}_r{rep}"


def run_cell(config: str, rep: int, model: str) -> None:
    """Run one (config, replicate) cell unless its outputs already exist."""
    label = cell_label(config, rep)
    atk_path = RESULTS / f"attack_{label}.json"
    ben_path = RESULTS / f"benign_{label}.json"
    prompt = CONFIGS[config]

    if atk_path.exists() and ben_path.exists():
        print(f"[skip] {label} (both result files present)")
        return

    print(f"\n{'=' * 70}\n[cell] {label}  model={model}\n{'=' * 70}")
    t0 = time.time()
    if not atk_path.exists():
        run_attack_replay(ATTACK_LOG, model_name=model, label=label, system_prompt=prompt)
    if not ben_path.exists():
        run_benign(model_name=model, label=label, system_prompt=prompt)
    print(f"[cell] {label} done in {time.time() - t0:.0f}s")


def _load_rows(config: str, k: int) -> tuple[list[dict], list[dict]]:
    """Return (attack summaries, benign summaries) for all replicates of a config."""
    atk, ben = [], []
    for rep in range(1, k + 1):
        label = cell_label(config, rep)
        ap, bp = RESULTS / f"attack_{label}.json", RESULTS / f"benign_{label}.json"
        if ap.exists():
            atk.append(json.loads(ap.read_text(encoding="utf-8")))
        if bp.exists():
            ben.append(json.loads(bp.read_text(encoding="utf-8")))
    return atk, ben


def _row_key(d: dict) -> str:
    return f"{d['seed_id']}|{d.get('round', 0)}"


def _per_row_wins(attack_summaries: list[dict]) -> dict[str, list[bool]]:
    """row_key -> [attacker_won in each replicate]."""
    out: dict[str, list[bool]] = {}
    for s in attack_summaries:
        for d in s["details"]:
            if "attacker_won" in d:
                out.setdefault(_row_key(d), []).append(bool(d["attacker_won"]))
    return out


def _categories(attack_summaries: list[dict]) -> dict[str, str]:
    return {_row_key(d): d["category"]
            for s in attack_summaries for d in s["details"] if "category" in d}


def analyze(k: int) -> dict:
    report: dict = {"k": k, "configs": {}}

    for config in CONFIGS:
        atk, ben = _load_rows(config, k)
        if not atk:
            print(f"[warn] no attack results for {config}; skipping")
            continue

        asrs = [s["asr_overall"] for s in atk]
        m, sd = mean_sd(asrs)
        rep1 = atk[0]
        wins1 = round(rep1["asr_overall"] * rep1["n"])

        per_row = _per_row_wins(atk)
        cats = _categories(atk)
        # A row is "unstable" if the attacker won in some replicates but not all.
        unstable = {rk: v for rk, v in per_row.items() if 0 < sum(v) < len(v)}
        # Majority vote per row: the outcome we use for cross-config comparison,
        # since it averages out the nondeterminism the replicates measure.
        majority = {rk: sum(v) * 2 > len(v) for rk, v in per_row.items()}

        by_cat: dict[str, dict] = {}
        for cat in sorted(set(cats.values())):
            rks = [rk for rk, c in cats.items() if c == cat]
            wins = sum(majority[rk] for rk in rks)
            ci = wilson_ci(wins, len(rks))
            by_cat[cat] = {"n": len(rks), "wins_majority": wins,
                           "asr": ci.point, "ci_low": ci.low, "ci_high": ci.high,
                           "per_replicate_asr": [
                               sum(1 for d in s["details"]
                                   if d.get("category") == cat and d.get("attacker_won")) /
                               max(1, sum(1 for d in s["details"] if d.get("category") == cat))
                               for s in atk]}

        ben_rates = [b["pass_rate"] for b in ben]
        bm, bsd = mean_sd(ben_rates)
        # Which benign tasks failed, and in how many replicates.
        ben_fail: dict[str, int] = {}
        for b in ben:
            for r in b["results"]:
                if not r.get("passed"):
                    ben_fail[r["id"]] = ben_fail.get(r["id"], 0) + 1

        # Wilson CI is computed on the per-row majority-vote outcome, not on
        # one arbitrarily-chosen replicate. Anchoring it to replicate 1 made
        # the interval swing with whichever run happened to be ordered first
        # ([30.1, 60.3] vs [9.2, 33.4] for the same config on two stacks) —
        # that is an artefact of the ordering, not of the attack distribution.
        # Majority vote is also what the paired test below uses, so the CI and
        # the significance test now describe the same quantity.
        ci_maj = wilson_ci(sum(majority.values()), len(majority))
        ci_per_rep = [wilson_ci(round(a * rep1["n"]), rep1["n"]) for a in asrs]
        report["configs"][config] = {
            "system_prompt_name": rep1.get("system_prompt_name"),
            "system_prompt_sha8": rep1.get("system_prompt_sha8"),
            "n_attacks": rep1["n"],
            "n_replicates": len(atk),
            "asr_per_replicate": asrs,
            "asr_mean": m, "asr_sd": sd,
            "asr_majority_vote": sum(majority.values()) / max(1, len(majority)),
            "asr_wilson_ci_majority": {"point": ci_maj.point, "low": ci_maj.low,
                                       "high": ci_maj.high},
            "asr_wilson_ci_per_replicate": [{"point": c.point, "low": c.low, "high": c.high}
                                            for c in ci_per_rep],
            "unstable_rows": len(unstable),
            "unstable_row_ids": sorted(unstable),
            "by_category": by_cat,
            "benign_pass_per_replicate": ben_rates,
            "benign_pass_mean": bm, "benign_pass_sd": bsd,
            "benign_failures": dict(sorted(ben_fail.items())),
        }

    # Paired comparison, on majority-vote outcomes over shared rows.
    if {"naive", "hardened"} <= set(report["configs"]):
        atk_n, _ = _load_rows("naive", k)
        atk_h, _ = _load_rows("hardened", k)
        maj_n = {rk: sum(v) * 2 > len(v) for rk, v in _per_row_wins(atk_n).items()}
        maj_h = {rk: sum(v) * 2 > len(v) for rk, v in _per_row_wins(atk_h).items()}
        cats = _categories(atk_n)
        shared = sorted(set(maj_n) & set(maj_h))

        comp = {}
        for cat in ["ALL"] + sorted(set(cats.values())):
            rks = [rk for rk in shared if cat == "ALL" or cats.get(rk) == cat]
            pairs = [(maj_n[rk], maj_h[rk]) for rk in rks]
            mc = mcnemar_b_c(pairs)
            wn, wh = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
            comp[cat] = {
                "n": len(rks),
                "naive_wins": wn, "hardened_wins": wh,
                "naive_asr": wn / max(1, len(rks)), "hardened_asr": wh / max(1, len(rks)),
                "abs_reduction_pp": (wn - wh) / max(1, len(rks)) * 100,
                **mc,
            }
        report["paired_naive_vs_hardened"] = comp

    (RESULTS / "p0_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_report(report: dict) -> None:
    k = report["k"]
    print(f"\n{'=' * 74}\nP0 SUMMARY  (k={k} replicates)\n{'=' * 74}")

    for config, c in report["configs"].items():
        print(f"\n--- {config}  (prompt={c['system_prompt_name']}/{c['system_prompt_sha8']}, "
              f"{c['n_replicates']} replicates) ---")
        reps = ", ".join(f"{a:.1%}" for a in c["asr_per_replicate"])
        print(f"  ASR per replicate : {reps}")
        print(f"  ASR mean +/- SD   : {c['asr_mean']:.1%} +/- {c['asr_sd']:.1%}   <- run-to-run (T8)")
        ci = c["asr_wilson_ci_majority"]
        print(f"  majority-vote ASR : {c['asr_majority_vote']:.1%}  Wilson 95% CI "
              f"[{ci['low']:.1%}, {ci['high']:.1%}] on n={c['n_attacks']}"
              f"   <- attack-distribution (T1)")
        print(f"  unstable rows     : {c['unstable_rows']}/{c['n_attacks']} "
              f"flipped across replicates {c['unstable_row_ids'] if c['unstable_rows'] else ''}")
        print(f"  benign pass       : {c['benign_pass_mean']:.1%} +/- {c['benign_pass_sd']:.1%}"
              f"   failures={c['benign_failures']}")
        for cat, cc in c["by_category"].items():
            print(f"    {cat:<18} {cc['asr']:>6.1%}  [{cc['ci_low']:.1%}, {cc['ci_high']:.1%}]  n={cc['n']}")

    comp = report.get("paired_naive_vs_hardened")
    if comp:
        print(f"\n--- paired: does the hardened prompt alone reduce ASR? ---")
        print(f"  {'category':<18} {'naive':>8} {'hardened':>9} {'delta':>8} "
              f"{'b':>3} {'c':>3} {'p':>8}")
        for cat, v in comp.items():
            sig = " *" if v["p_value"] < 0.05 else ""
            print(f"  {cat:<18} {v['naive_asr']:>7.1%} {v['hardened_asr']:>9.1%} "
                  f"{v['abs_reduction_pp']:>7.1f}pp {v['b']:>3} {v['c']:>3} "
                  f"{v['p_value']:>8.4f}{sig}")
        print("  (b = naive won & hardened didn't; c = hardened won & naive didn't;"
              " * = p<0.05)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="replicates per config")
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18",
                    help="pin a dated snapshot so backend rolls don't shift means (T8)")
    ap.add_argument("--analyze-only", action="store_true",
                    help="re-aggregate existing result files without any API calls")
    args = ap.parse_args()

    if not args.analyze_only:
        if not ATTACK_LOG.exists():
            sys.exit(f"missing {ATTACK_LOG}; run src/redteam.py first")
        total = len(CONFIGS) * args.k
        print(f"P0 sweep: {len(CONFIGS)} configs x {args.k} replicates = {total} cells")
        print(f"          each cell = 38 attack replays + 10 benign tasks")
        print(f"          model = {args.model}")
        for rep in range(1, args.k + 1):
            for config in CONFIGS:
                run_cell(config, rep, args.model)

    print_report(analyze(args.k))
    print(f"\nwrote {RESULTS / 'p0_summary.json'}")


if __name__ == "__main__":
    main()
