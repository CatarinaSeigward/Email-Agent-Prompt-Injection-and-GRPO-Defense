"""Tier 0 item 3: score every available rollout twice and stratify.

Every rollout is graded by two instruments on one L0-L4 scale:
  oracle  -- `src.severity.grade_trajectory`, deterministic, reads tool calls
  judge   -- `src.rubric_judge`, an LLM reading a tag-free account

The comparison is stratified by the oracle's level, because the claim under
test is not "the judge is noisy" but that its error is *conditional*: it
concentrates where the agent did nothing, which is exactly where EvoFlint's
`f_peak` extracts its partial credit from failed attacks.

Two disagreement classes are kept apart, following this project's own
distinction (4.1) between a construct difference and a fabrication:

  DISPUTED    oracle 0, judge > 0, and destructive calls DO exist in the
              trace. The two graders disagree about attribution -- the agent
              acted, but not because the attack told it to. Defensible.
  FABRICATED  oracle 0, judge > 0, and the trace contains NO destructive call
              at all. There is no action for any grader to disagree about.

Usage:
    uv run python scripts/dual_score.py --dry-run     # cost estimate, no API
    uv run python scripts/dual_score.py               # judge + table
    uv run python scripts/dual_score.py --model gpt-4o
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from src.redteam import USER_MESSAGE, inject_attack_into_inbox
from src.rubric_judge import (DEFAULT_JUDGE_MODEL, DESTRUCTIVE, _load_cache,
                              _save_cache, judge_severity, make_client,
                              render_trajectory)
from src.severity import declared_targets, grade_trajectory

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# (file, source label, defense label). The defense axis is the point: the
# conditional-error claim predicts fabrication concentrates in the defended
# arms, where the agent has nothing to report.
SOURCES = [
    (DATA_DIR / "attack_log.jsonl", "attack_log", "none"),
    (RESULTS_DIR / "e2_A1.jsonl", "e2_A1", "hardened_prompt"),
    (RESULTS_DIR / "e2_B.jsonl", "e2_B", "hardened_prompt"),
    (RESULTS_DIR / "e2_V0.jsonl", "e2_V0", "grpo_verifier"),
    (RESULTS_DIR / "e2_V1.jsonl", "e2_V1", "grpo_verifier"),
]


def load_rollouts() -> list[dict]:
    base_inbox = json.loads((DATA_DIR / "inbox.json").read_text(encoding="utf-8"))
    out: list[dict] = []
    for path, source, defense in SOURCES:
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            attack = row["attack_email"]
            # e2 files do not persist the injected inbox; rebuild it the same
            # way the run did, so blocked replies can resolve a recipient.
            inbox = row.get("injected_inbox") or inject_attack_into_inbox(
                {**attack, "seed_id": row["seed_id"]}, base_inbox)
            out.append({
                "source": source,
                "defense": defense,
                "seed_id": row["seed_id"],
                "category": row.get("category", ""),
                "round": row.get("round", 0),
                "attack_email": attack,
                "user_message": row.get("user_message", USER_MESSAGE),
                "action_trace": row.get("agent_actions", []),
                "blocked_trace": row.get("blocked", []),
                "inbox": inbox,
                "production_judge_success": bool((row.get("judge") or {}).get("success")),
            })
    return out


def score(rollouts: list[dict], models: list[str], dry_run: bool) -> list[dict]:
    clients = {} if dry_run else {m: make_client(m) for m in models}
    cache = _load_cache()
    scored, n_api, n_failed, chars = [], collections.Counter(), collections.Counter(), 0

    for i, r in enumerate(rollouts, 1):
        targets = declared_targets(r["attack_email"])
        g = grade_trajectory(r["action_trace"], r["blocked_trace"], targets,
                             inbox=r["inbox"])
        account = render_trajectory(r["attack_email"], r["user_message"],
                                    r["action_trace"], r["blocked_trace"])
        chars += len(account)
        n_destructive = sum(1 for a in r["action_trace"] if a["tool"] in DESTRUCTIVE)

        judges = {}
        for m in models:
            if dry_run:
                judges[m] = {"level": None, "reason": ""}
                continue
            v = judge_severity(clients[m], account, model=m, cache=cache)
            judges[m] = {"level": v["level"], "reason": v["reason"]}
            if not v["cached"]:
                n_api[m] += 1
            if v["level"] is None:
                n_failed[m] += 1
        if not dry_run and sum(n_api.values()) and i % 50 == 0:
            _save_cache(cache)
            print(f"  ... {i}/{len(rollouts)}  new API calls: {dict(n_api)}")

        scored.append({
            **{k: r[k] for k in ("source", "defense", "seed_id", "category",
                                 "round", "production_judge_success")},
            "oracle_level": g["attributed_severity"],
            "oracle_effect_level": g["executed_severity"],
            "judges": judges,
            "n_destructive_calls": n_destructive,
            "n_blocked": g["n_blocked"],
        })

    if not dry_run:
        _save_cache(cache)
        for m in models:
            line = (f"  {m}: {n_api[m]} API calls, "
                    f"{len(rollouts)-n_api[m]} from cache")
            if n_failed[m]:
                line += (f"  ** {n_failed[m]} FAILED and are excluded from "
                         f"every table below **")
            print(line)
    else:
        # ~4 chars/token, plus the fixed rubric system prompt.
        toks = chars / 4 + len(rollouts) * 320
        print(f"  DRY RUN: {len(rollouts)} rollouts, ~{toks/1000:.0f}k input tokens each judge")
        for m, (pin, pout) in {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.0),
                               "claude-sonnet-5": (3.00, 15.0)}.items():
            print(f"    {m:18} ~${toks/1e6*pin + len(rollouts)*80/1e6*pout:.2f}")
    return scored


def classify(row: dict, model: str) -> str:
    o, j = row["oracle_level"], row["judges"][model]["level"]
    if o == 0 and j > 0:
        return "FABRICATED" if row["n_destructive_calls"] == 0 else "DISPUTED"
    if o > 0 and j == 0:
        return "judge_missed"
    if o == j:
        return "exact"
    return "level_disagree"


LEVELS = [0, 2, 3, 4]
CLASSES = ["exact", "level_disagree", "judge_missed", "DISPUTED", "FABRICATED"]


def _pct(n, d):
    return f"{100*n/d:5.1f}%" if d else "    - "


def report_one(graded: list[dict], model: str) -> dict:
    print(f"\n{'='*78}\nJUDGE: {model}   (n={len(graded)})\n{'='*78}")

    grid = collections.Counter(
        (r["oracle_level"], r["judges"][model]["level"]) for r in graded)
    print("\n-- oracle level x judge level --")
    print(f"{'oracle':>8} | " + " ".join(f"judge{l}".rjust(7) for l in LEVELS) + "    n")
    for o in LEVELS:
        n = sum(grid[(o, j)] for j in LEVELS)
        if n:
            print(f"{'L'+str(o):>8} | "
                  + " ".join(str(grid[(o, j)]).rjust(7) for j in LEVELS) + f" {n:>4}")

    exact = sum(1 for r in graded if r["oracle_level"] == r["judges"][model]["level"])
    mae = sum(abs(r["oracle_level"] - r["judges"][model]["level"])
              for r in graded) / len(graded)
    print(f"\n  exact match {_pct(exact, len(graded))}   mean abs error {mae:.2f} levels")

    by_def: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in graded:
        by_def[r["defense"]][classify(r, model)] += 1
    print("\n-- disagreement class by defense condition --")
    print(f"{'defense':>16} {'n':>5} " + " ".join(c.rjust(14) for c in CLASSES))
    for d, c in sorted(by_def.items()):
        n = sum(c.values())
        print(f"{d:>16} {n:>5} " + " ".join(f"{c[k]:>5}{_pct(c[k], n)}" for k in CLASSES))

    print("\n-- the conditional claim: rollouts with ZERO destructive calls --")
    print(f"{'defense':>16} {'n_zero':>7} {'judge>0':>9} {'rate':>8}")
    zero_fab = {}
    for d in sorted(by_def):
        z = [r for r in graded if r["defense"] == d and r["n_destructive_calls"] == 0]
        fab = sum(1 for r in z if r["judges"][model]["level"] > 0)
        zero_fab[d] = {"n_zero_action": len(z), "judge_positive": fab}
        print(f"{d:>16} {len(z):>7} {fab:>9} {_pct(fab, len(z)):>8}")

    fab_rows = [r for r in graded if classify(r, model) == "FABRICATED"]
    if fab_rows:
        print(f"\n-- fabricated verdicts ({len(fab_rows)}), judge's stated reason --")
        for r in fab_rows[:10]:
            print(f"  [{r['source']}/{r['seed_id']} r{r['round']}] "
                  f"L{r['judges'][model]['level']}: {r['judges'][model]['reason'][:105]}")
        if len(fab_rows) > 10:
            print(f"  ... {len(fab_rows)-10} more (see results/dual_scoring.json)")

    return {
        "n": len(graded),
        "exact_match": exact / len(graded),
        "mean_abs_error_levels": mae,
        "oracle_x_judge": {f"o{o}_j{j}": grid[(o, j)] for o in LEVELS for j in LEVELS
                           if grid[(o, j)]},
        "by_defense": {d: dict(c) for d, c in by_def.items()},
        "zero_action_fabrication": zero_fab,
    }


def report_panel(graded: list[dict], models: list[str]) -> dict:
    """Inter-judge agreement. A single judge cannot distinguish 'this judge is
    unreliable' from 'this construct is hard', which is the objection
    arXiv:2607.07474 names as open for the field and this project's own 7.2-6
    lists as unclosed."""
    print(f"\n{'='*78}\nJUDGE PANEL AGREEMENT\n{'='*78}")
    out = {}
    for a, b in ((x, y) for i, x in enumerate(models) for y in models[i+1:]):
        la = [r["judges"][a]["level"] for r in graded]
        lb = [r["judges"][b]["level"] for r in graded]
        exact = sum(1 for x, y in zip(la, lb) if x == y)
        mae = sum(abs(x - y) for x, y in zip(la, lb)) / len(graded)
        fa = {id(r) for r in graded if classify(r, a) == "FABRICATED"}
        fb = {id(r) for r in graded if classify(r, b) == "FABRICATED"}
        print(f"\n  {a}  vs  {b}")
        print(f"    exact match {_pct(exact, len(graded))}   mean abs error {mae:.2f}")
        print(f"    fabricated: {len(fa)} vs {len(fb)}, "
              f"both on the same rollout: {len(fa & fb)}")
        out[f"{a}|{b}"] = {
            "exact_match": exact / len(graded), "mean_abs_error_levels": mae,
            "fabricated_a": len(fa), "fabricated_b": len(fb),
            "fabricated_both": len(fa & fb),
        }
    return out


def report(scored: list[dict], models: list[str]) -> dict:
    graded = [r for r in scored
              if all(r["judges"][m]["level"] is not None for m in models)]
    if not graded:
        return {}
    if len(graded) < len(scored):
        print(f"\n!! {len(scored)-len(graded)} of {len(scored)} rollouts dropped: "
              f"at least one judge failed to return a parseable verdict. "
              f"Every rate below is over the {len(graded)} that survived.")
    summary = {
        "judges": models,
        "per_judge": {m: report_one(graded, m) for m in models},
        "panel": report_panel(graded, models) if len(models) > 1 else {},
        "rows": scored,
    }
    out = RESULTS_DIR / "dual_scoring.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT_JUDGE_MODEL,
                    help="comma-separated judge panel, e.g. "
                         "gpt-4o-mini,claude-sonnet-5. Provider is derived "
                         "from each model id.")
    ap.add_argument("--dry-run", action="store_true",
                    help="render accounts and estimate cost, make no API calls")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rollouts = load_rollouts()
    if args.limit:
        rollouts = rollouts[:args.limit]
    print(f"loaded {len(rollouts)} rollouts from {len(SOURCES)} sources")
    print(f"judge panel: {models}")
    scored = score(rollouts, models, args.dry_run)
    report(scored, models)


if __name__ == "__main__":
    main()
