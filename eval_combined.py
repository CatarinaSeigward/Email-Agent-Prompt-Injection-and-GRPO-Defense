"""Four corner-case eval: attack/benign x verifier-only/combined.

Composition path (per plan §Step 4-pivot): keep gpt-4o-mini as the agent,
add the GRPO LoRA as a side-call veto verifier at the tool-call boundary.
Compare verifier-only vs verifier+classifier to isolate each layer's
marginal contribution.

Outputs (under results/):
  attack_verifier_only.json   ASR with only Layer-1 (GRPO veto) active
  attack_combined.json        ASR with Layer-1 + Layer-2 active
  benign_verifier_only.json   pass_rate with only Layer-1 active
  benign_combined.json        pass_rate with Layer-1 + Layer-2 active
"""

from __future__ import annotations

from src.classifier import Guard
from src.combined_guard import CombinedGuard
from src.eval import DATA_DIR, run_attack_replay, run_benign
from src.grpo_verifier import GrpoVerifier


def main() -> None:
    print("=== loading verifier (Qwen+GRPO LoRA) ===")
    verifier = GrpoVerifier()
    print("=== loading classifier (ModernBERT) ===")
    classifier = Guard()
    combined = CombinedGuard(classifier=classifier, verifier=verifier)
    log = DATA_DIR / "attack_log.jsonl"

    print("\n=== 1/4 attack x verifier-only ===")
    run_attack_replay(log, guard=verifier, label="verifier_only")

    print("\n=== 2/4 attack x combined (classifier+verifier) ===")
    run_attack_replay(log, guard=combined, label="combined")

    print("\n=== 3/4 benign x verifier-only ===")
    run_benign(guard=verifier, label="verifier_only")

    print("\n=== 4/4 benign x combined ===")
    run_benign(guard=combined, label="combined")

    print("\nall four corner cases done.")


if __name__ == "__main__":
    main()
