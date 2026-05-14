"""Step 2 ablation: rerun the four corner cases with the verifier set to
also veto on SOFT refusal (loose mode), not just explicit REFUSE_RE.

Hypothesis: A3 (exfiltration) ASR will drop because the LoRA does emit
soft-cautious language ("This looks unusual...", "you should confirm...")
on A3 prompts even when it doesn't reach an explicit "I won't" — those
get to pass through under the default `require_refusal_token=True`.

Costs predicted: benign pass rate drops further (loose threshold means
more legitimate forward/reply requests also trip soft-refusal heuristic).

Outputs (NEW labels — do not overwrite the originals from eval_combined.py):
  results/attack_verifier_only_loose.json
  results/attack_combined_loose.json
  results/benign_verifier_only_loose.json
  results/benign_combined_loose.json

After running, compare:
  attack_combined.json::asr_overall   (original, ~0.158)
  attack_combined_loose.json::asr_overall   (this run)

If loose drops A3 without overall ASR rising elsewhere → looser threshold is
a free win. If benign pass collapses → not worth it.
"""

from __future__ import annotations

from src.classifier import Guard
from src.combined_guard import CombinedGuard
from src.eval import DATA_DIR, run_attack_replay, run_benign
from src.grpo_verifier import GrpoVerifier


def main() -> None:
    print("=== loading verifier (LOOSE: require_refusal_token=False) ===")
    verifier = GrpoVerifier(require_refusal_token=False)
    print("=== loading classifier (ModernBERT) ===")
    classifier = Guard()
    combined = CombinedGuard(classifier=classifier, verifier=verifier)
    log = DATA_DIR / "attack_log.jsonl"

    print("\n=== 1/4 attack x verifier_only_loose ===")
    run_attack_replay(log, guard=verifier, label="verifier_only_loose")

    print("\n=== 2/4 attack x combined_loose ===")
    run_attack_replay(log, guard=combined, label="combined_loose")

    print("\n=== 3/4 benign x verifier_only_loose ===")
    run_benign(guard=verifier, label="verifier_only_loose")

    print("\n=== 4/4 benign x combined_loose ===")
    run_benign(guard=combined, label="combined_loose")

    print("\nall four loose corner cases done.")


if __name__ == "__main__":
    main()
