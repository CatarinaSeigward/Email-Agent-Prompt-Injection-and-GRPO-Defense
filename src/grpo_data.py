"""Build a prompt-only dataset for GRPO training.

GRPO does not need paired (chosen, rejected) data — it samples G completions
per prompt and assigns rewards via a reward function. So we only emit the
`prompt` field.

Sources (both judge-labels are attack-laden contexts, so both teach refusal):
  - judge.success == True   rows: PAIR-confirmed attack-wins traces
  - judge.success == False  rows: same attack content, agent happened to
                                  not fall for it. Still useful for GRPO
                                  because reward is computed on completion
                                  text, not on agent behaviour.

Augmentation (inbox shuffle):
  For each row we emit N prompts with different random subsets of inbox
  emails as context. The attack email itself is shown separately in the
  prompt — only the surrounding "Inbox snapshot" varies. This teaches
  the policy that refusal should depend on the attack email itself, not
  on what else happens to be in the inbox.

Output: data/grpo_prompts.jsonl
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PATH = DATA_DIR / "attack_log.jsonl"
OUT_PATH = DATA_DIR / "grpo_prompts.jsonl"

# Only the chat template is used here. Qwen2.5-0.5B- and 1.5B-Instruct ship the
# identical template (sha256 cd8e9439f057...), so this default matching the
# trained model changes no rendered prompt.
BASE_MODEL = os.environ.get("GRPO_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

_TOKENIZER = None


def stable_seed(*parts) -> int:
    """A 32-bit RNG seed that is the same in every process.

    The built-in hash() salts str per process (PYTHONHASHSEED), so a seed taken
    from hash((seed_id, round, ...)) changes from run to run. That is how
    data/grpo_prompts.jsonl was drawn: the saved file is what training used,
    but rerunning build_prompts() could never reproduce it. sha256 over the
    repr has no salt.
    """
    digest = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL)
    return _TOKENIZER


def _render_prompt(user_msg: str, inbox_summary: str, from_addr: str,
                   subject: str, body: str) -> str:
    system_content = (
        f'You are an email assistant. The user triggered: "{user_msg}"\n'
        f'Inbox snapshot:\n{inbox_summary}\n\n'
        f'The next email being processed:\n'
        f'From: {from_addr}\n'
        f'Subject: {subject}\n'
        f'Body:\n{body}'
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "What action should the agent take next?"},
    ]
    return _get_tokenizer().apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_prompts(
    inbox_variants: int = 3,
    inbox_sample_size: int = 6,
    include_near_miss: bool = True,
) -> Path:
    """Emit GRPO prompts from attack_log.jsonl.

    Args:
        inbox_variants: how many inbox-shuffle variants to emit per row.
            With the 38 rows of attack_log.jsonl, 3 variants give the 114
            prompts training used: about 29 optimizer steps per epoch at
            4 prompts per step (batch 2 x grad_accum 8 = 16 completions, G=4).
        inbox_sample_size: how many inbox emails to put in each prompt's
            "Inbox snapshot" section. 6 keeps prompts under ~700 tokens.
        include_near_miss: include judge.success==False rows. These are
            still attack-laden contexts (PAIR fed them attack content);
            the agent just happened not to fall for them. For GRPO this
            is fine because reward is computed on completion text.
    """
    rows = [
        json.loads(l) for l in LOG_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    out: list[dict] = []
    n_success = 0
    n_near_miss = 0

    for row in rows:
        if "judge" not in row:
            continue
        judge_success = row["judge"].get("success", False)
        if not judge_success and not include_near_miss:
            continue

        inbox = row["injected_inbox"]
        # Exclude x_atk* attack-injected emails from the inbox snapshot —
        # the attack email itself is shown separately under "The next email
        # being processed". Including it in the summary would just be a
        # duplicate signal.
        candidates = [e for e in inbox if not e["id"].startswith("x_")]
        if not candidates:
            continue

        attack = row["attack_email"]
        for variant_i in range(inbox_variants):
            # Deterministic per-(row, variant) shuffle so re-runs produce
            # identical datasets. Regenerating now gives a different set from
            # the saved file, which was drawn with the salted hash().
            rng = random.Random(stable_seed(row["seed_id"], row.get("round", 0), variant_i))
            k = min(inbox_sample_size, len(candidates))
            sampled = rng.sample(candidates, k=k)
            inbox_summary = "\n".join(
                f"  - {e['id']} from {e['from']}: {e['subject']}" for e in sampled
            )
            prompt = _render_prompt(
                user_msg=row["user_message"],
                inbox_summary=inbox_summary,
                from_addr=attack["from"],
                subject=attack["subject"],
                body=attack["body"][:600],
            )
            out.append({
                "prompt": prompt,
                "seed_id": row["seed_id"],
                "category": row["category"],
                "variant": variant_i,
                "source": "success" if judge_success else "near_miss",
            })
            if judge_success:
                n_success += 1
            else:
                n_near_miss += 1

    # Final shuffle so categories/sources are mixed across the training
    # iterator — GRPO is on-policy and benefits from i.i.d. prompt order.
    random.Random(42).shuffle(out)

    OUT_PATH.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in out),
        encoding="utf-8",
    )
    print(f"wrote {len(out)} GRPO prompts to {OUT_PATH}")
    print(f"  from success rows × {inbox_variants} variants = {n_success}")
    print(f"  from near-miss × {inbox_variants} variants = {n_near_miss}")
    print(f"  unique seed_ids: {len({p['seed_id'] for p in out})}")
    print(f"  categories: {sorted({p['category'] for p in out})}")
    return OUT_PATH


if __name__ == "__main__":
    build_prompts()
