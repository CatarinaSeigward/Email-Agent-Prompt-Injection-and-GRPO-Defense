"""Score stored SFT/GRPO completions with the *training* reward function.

Why this exists
---------------
final_report.md §4.8 establishes *what* happened: the GRPO stage cut held-out
refusal rate from 86.8% to 50.0% (paired McNemar p=0.0013) without improving
semantic ASR. The tempting explanation is that the reward function contains an
unintended local optimum — that non-committal filler like "No further processing
required; ..." simply pays better than refusing.

That explanation has never been checked, and hand-arithmetic on the reward stack
in src/grpo_train.py suggests the opposite (a specific refusal should collect
+1.0 refusal +0.3 specificity +0.2 length +0.4 termination = +1.9, while the
filler collects only the +0.2/+0.4 length and termination bonuses = +0.6).

Asserting a mechanism that merely explains the data, without measuring whether
it occurs, is precisely the error §7.7 retracts. So: measure it.

Output: results/reward_scoring_check.json

Usage
-----
  $env:PYTHONIOENCODING="utf-8"
  uv run python scripts/check_reward_scoring.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.grpo_train import reward_fn  # noqa: E402
from src.stats import mean_sd  # noqa: E402

RESULTS = REPO / "results"
GRPO_FILE = RESULTS / "grpo_behavioral_attack.json"
SFT_FILE = RESULTS / "behavioral_attack_qwen-injection-sft.json"


def load_rows(path: Path) -> dict[str, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {f"{x['seed_id']}|{x.get('round', 0)}": x for x in d["details"]}


def refused(row: dict) -> bool:
    return bool(row.get("refused") or row.get("soft_refused"))


def score(text: str) -> float:
    return float(reward_fn([text])[0])


def summarize(name: str, values: list[float]) -> dict:
    m, sd = mean_sd(values)
    return {"group": name, "n": len(values), "mean": m, "sd": sd,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "median": st.median(values) if values else None}


def main() -> None:
    for p in (GRPO_FILE, SFT_FILE):
        if not p.exists():
            sys.exit(f"missing {p} - run eval_grpo_attack.py for both adapters first")

    G, S = load_rows(GRPO_FILE), load_rows(SFT_FILE)
    keys = sorted(set(G) & set(S))

    # The 16 rows §4.8 is about: SFT refused, GRPO did not.
    flipped = [k for k in keys if refused(S[k]) and not refused(G[k])]

    groups = {
        # Head-to-head on the rows where the behaviour actually changed.
        "flipped_sft_completions": [score(S[k]["completion"]) for k in flipped],
        "flipped_grpo_completions": [score(G[k]["completion"]) for k in flipped],
        # Whole distribution, for context.
        "all_sft": [score(S[k]["completion"]) for k in keys],
        "all_grpo": [score(G[k]["completion"]) for k in keys],
        # Partition GRPO's own output by whether it registered as a refusal.
        "grpo_refusals": [score(G[k]["completion"]) for k in keys if refused(G[k])],
        "grpo_non_refusals": [score(G[k]["completion"]) for k in keys if not refused(G[k])],
    }

    stats = {name: summarize(name, vals) for name, vals in groups.items()}

    # The decisive comparison: on rows where GRPO stopped refusing, did its
    # completion out-earn the refusal SFT produced for the same prompt?
    pairs = list(zip(groups["flipped_sft_completions"], groups["flipped_grpo_completions"]))
    grpo_wins = sum(1 for s, g in pairs if g > s)
    deltas = [g - s for s, g in pairs]
    mean_delta, sd_delta = mean_sd(deltas)

    train_reward = json.loads((RESULTS / "grpo_train.json").read_text(encoding="utf-8"))
    final_reward = train_reward.get("final_reward")

    if mean_delta > 0:
        verdict = ("EVASION PAYS MORE - the local-optimum explanation is supported: "
                   "on the rows where GRPO stopped refusing, its completions out-earn "
                   "the refusals SFT produced for the same prompts.")
    else:
        verdict = ("EVASION PAYS LESS - the local-optimum explanation is NOT supported. "
                   "GRPO's non-refusals score below the refusals they replaced, so the "
                   "policy did not move toward higher training reward on these prompts. "
                   "The mechanism is unexplained; see the greedy-vs-sampled hypothesis.")

    report = {
        "n_paired_rows": len(keys),
        "n_flipped_rows": len(flipped),
        "flipped_row_ids": flipped,
        "groups": stats,
        "head_to_head_on_flipped": {
            "grpo_higher_than_sft": grpo_wins,
            "sft_higher_or_equal": len(pairs) - grpo_wins,
            "mean_delta_grpo_minus_sft": mean_delta,
            "sd_delta": sd_delta,
        },
        "training_final_reward": final_reward,
        "decoding_note": (
            "Training reward is the mean over G=4 samples at temperature 0.7 "
            "(GRPOConfig in src/grpo_train.py); these completions come from greedy "
            "decoding (do_sample=False in eval_grpo_attack.py). A gap between "
            "training_final_reward and all_grpo.mean is expected on that basis alone "
            "and is not by itself evidence of anything."
        ),
        "verdict": verdict,
    }

    out = RESULTS / "reward_scoring_check.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"paired rows: {len(keys)}   rows where SFT refused and GRPO did not: {len(flipped)}\n")
    print(f"{'group':<28} {'n':>3} {'mean':>8} {'sd':>7} {'median':>8} {'min':>7} {'max':>7}")
    for s in stats.values():
        print(f"{s['group']:<28} {s['n']:>3} {s['mean']:>8.3f} {s['sd']:>7.3f} "
              f"{s['median']:>8.3f} {s['min']:>7.2f} {s['max']:>7.2f}")

    print(f"\nhead-to-head on the {len(pairs)} flipped rows:")
    print(f"  GRPO completion out-earns SFT's refusal : {grpo_wins}/{len(pairs)}")
    print(f"  mean delta (GRPO - SFT)                 : {mean_delta:+.3f}  (sd {sd_delta:.3f})")
    print(f"\ntraining final_reward (sampled, T=0.7)    : {final_reward:.3f}")
    print(f"all_grpo mean (greedy, this eval)         : {stats['all_grpo']['mean']:.3f}")
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
