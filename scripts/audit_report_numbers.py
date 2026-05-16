"""Audit every numeric claim made in final_report.md against the source files.

Run: `uv run python scripts/audit_report_numbers.py`

Each check has the form `(name, ok)` where `ok` is a boolean computed from
result-file JSON. The script prints `[OK]` / `[FAIL]` lines and exits with
status 1 if anything fails — suitable for CI.

When adding new claims to the report, also add a corresponding row here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
R = REPO / "results"
DATA = REPO / "data"


def load(path: Path) -> dict | None:
    """Return parsed JSON, or None if missing (so optional artefacts don't
    crash the audit before they've been generated)."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path: Path) -> int:
    if not path.exists():
        return -1
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def main() -> int:
    baseline = load(R / "attack_baseline.json")
    guard = load(R / "attack_guard.json")
    clsm = load(R / "classifier_metrics.json")
    sft = load(R / "sft_warmup.json")
    grpo = load(R / "grpo_train.json")
    bbase = load(R / "benign_baseline.json")
    bguard = load(R / "benign_guard.json")
    gbeh = load(R / "grpo_behavioral_attack.json")
    av = load(R / "attack_verifier_only.json")
    ac = load(R / "attack_combined.json")
    bv = load(R / "benign_verifier_only.json")
    bc = load(R / "benign_combined.json")
    avl = load(R / "attack_verifier_only_loose.json")
    acl = load(R / "attack_combined_loose.json")
    bvl = load(R / "benign_verifier_only_loose.json")
    bcl = load(R / "benign_combined_loose.json")

    n_log = lines(DATA / "attack_log.jsonl")
    n_gp = lines(DATA / "grpo_prompts.jsonl")
    n_seeds = lines(DATA / "attack_seeds.jsonl")

    checks: list[tuple[str, bool, str]] = []

    def chk(name: str, ok: bool, expected: str = "") -> None:
        checks.append((name, ok, expected))

    # --- data corpus sizes ---
    chk("30 attack seeds", n_seeds == 30, "30")
    chk("38 attack rollouts", n_log == 38, "38")
    chk("114 GRPO prompts", n_gp == 114, "114")

    # --- baseline ASR ---
    if baseline:
        chk("baseline ASR 36.8%", round(baseline["asr_overall"] * 100, 1) == 36.8)
        chk("baseline A1 50.0%", round(baseline["asr_by_category"]["override"] * 100, 1) == 50.0)
        chk("baseline A2 28.6%", round(baseline["asr_by_category"]["hidden_injection"] * 100, 1) == 28.6)
        chk("baseline A3 33.3%", round(baseline["asr_by_category"]["exfiltration"] * 100, 1) == 33.3)

    # --- classifier-only ASR ---
    if guard:
        chk("classifier ASR 13.2%", round(guard["asr_overall"] * 100, 1) == 13.2)
        chk("classifier A1 16.7%", round(guard["asr_by_category"]["override"] * 100, 1) == 16.7)
        chk("classifier A2 14.3%", round(guard["asr_by_category"]["hidden_injection"] * 100, 1) == 14.3)
        chk("classifier A3 8.3%", round(guard["asr_by_category"]["exfiltration"] * 100, 1) == 8.3)

    # --- classifier metrics + training summaries ---
    if clsm:
        chk("classifier F1=1.0", clsm["f1"] == 1.0)
        chk("classifier precision=1.0", clsm["precision"] == 1.0)
        chk("classifier recall=1.0", clsm["recall"] == 1.0)
    if sft:
        chk("SFT loss 1.18", round(sft["final_loss"], 2) == 1.18)
        chk("SFT grad_norm 1.17", round(sft["final_grad_norm"], 2) == 1.17)
    if grpo:
        chk("GRPO final_reward 1.57", round(grpo["final_reward"], 2) == 1.57)
        chk("GRPO final_kl 1.98", round(grpo["final_kl"], 2) == 1.98)
        chk("GRPO final_reward_std 0.20", round(grpo["final_reward_std"], 2) == 0.20)

    # --- benign baseline + classifier-only ---
    if bbase:
        chk("benign baseline pass=100%", bbase["pass_rate"] == 1.0)
    if bguard:
        chk("benign classifier pass=80%", bguard["pass_rate"] == 0.8)

    # --- GRPO behavioral standalone (38 attacks, no agent) ---
    if gbeh:
        chk("GRPO behavioral n=38", gbeh["n"] == 38)
        chk("GRPO behavioral regex ASR=0.0%", gbeh["asr_overall"] == 0.0)
        chk("GRPO behavioral strict ASR=10.5%", round(gbeh["strict_asr_overall"] * 100, 1) == 10.5)
        chk("GRPO behavioral strict A2=21.4%",
            round(gbeh["strict_asr_by_category"]["hidden_injection"] * 100, 1) == 21.4)
        chk("GRPO behavioral strict A3=8.3%",
            round(gbeh["strict_asr_by_category"]["exfiltration"] * 100, 1) == 8.3)

    # --- combined defense (Step-1 update of final_report.md §1, §4.6) ---
    if av:
        chk("verifier-only ASR 15.8%", round(av["asr_overall"] * 100, 1) == 15.8)
        chk("verifier-only A1 8.3%", round(av["asr_by_category"]["override"] * 100, 1) == 8.3)
        chk("verifier-only A2 7.1%", round(av["asr_by_category"]["hidden_injection"] * 100, 1) == 7.1)
        chk("verifier-only A3 33.3%", round(av["asr_by_category"]["exfiltration"] * 100, 1) == 33.3)
    if ac:
        chk("combined ASR 15.8%", round(ac["asr_overall"] * 100, 1) == 15.8)
        chk("combined A1 0.0%", round(ac["asr_by_category"]["override"] * 100, 1) == 0.0)
        chk("combined A2 14.3%", round(ac["asr_by_category"]["hidden_injection"] * 100, 1) == 14.3)
        chk("combined A3 33.3%", round(ac["asr_by_category"]["exfiltration"] * 100, 1) == 33.3)
    if bv:
        chk("verifier-only benign 90%", bv["pass_rate"] == 0.9)
    if bc:
        chk("combined benign 70%", bc["pass_rate"] == 0.7)

    # --- §4.7 loose-mode ablation ---
    if avl:
        chk("verifier-loose ASR 13.2%", round(avl["asr_overall"] * 100, 1) == 13.2)
        chk("verifier-loose A1 8.3%", round(avl["asr_by_category"]["override"] * 100, 1) == 8.3)
        chk("verifier-loose A2 14.3%", round(avl["asr_by_category"]["hidden_injection"] * 100, 1) == 14.3)
        chk("verifier-loose A3 16.7%", round(avl["asr_by_category"]["exfiltration"] * 100, 1) == 16.7)
    if acl:
        chk("combined-loose ASR 23.7%", round(acl["asr_overall"] * 100, 1) == 23.7)
        chk("combined-loose A1 8.3%", round(acl["asr_by_category"]["override"] * 100, 1) == 8.3)
        chk("combined-loose A2 35.7%", round(acl["asr_by_category"]["hidden_injection"] * 100, 1) == 35.7)
        chk("combined-loose A3 25.0%", round(acl["asr_by_category"]["exfiltration"] * 100, 1) == 25.0)
    if bvl:
        chk("verifier-loose benign 90%", bvl["pass_rate"] == 0.9)
    if bcl:
        chk("combined-loose benign 70%", bcl["pass_rate"] == 0.7)

    # --- §3.2.5 DPO negative result ---
    dpo = load(R / "dpo_train.json")
    if dpo:
        chk("DPO n_pairs=270", dpo["n_pairs"] == 270)
        chk("DPO final_loss ≈ 0.0028", round(dpo["final_loss"], 4) == 0.0028)
        chk("DPO rewards/margins ≈ 6.18", round(dpo["rewards/margins"], 2) == 6.18)
        chk("DPO rewards/chosen ≈ 2.08", round(dpo["rewards/chosen"], 2) == 2.08)
        chk("DPO rewards/rejected ≈ -4.10", round(dpo["rewards/rejected"], 2) == -4.10)

    # --- print results ---
    n_ok = 0
    n_total = 0
    for name, ok, *_ in checks:
        n_total += 1
        n_ok += int(ok)
        tag = "[ OK ]" if ok else "[FAIL]"
        print(f"  {tag} {name}")

    print(f"\n{n_ok}/{n_total} checks passed")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
