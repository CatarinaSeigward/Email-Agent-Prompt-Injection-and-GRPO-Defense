"""Construct DPO preference pairs from attack_log.jsonl.

For every (judged-successful) attack trace:
  prompt    = SYSTEM_PROMPT + a rendered conversation including the inbox
              listing and the user's triage request.
  rejected  = the tool call the baseline agent actually made (the "attacker
              wins" trace, serialized).
  chosen    = a synthesized refusal: list/read are fine, but flag the
              suspicious email and ask the user for explicit confirmation.

Targeting ~150 pairs per the plan. We oversample category to keep balance.
Output: data/dpo_pairs.jsonl in TRL DPO format.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.eval import REPLY_EXFIL_MIN_BODY, coerce_action_result, is_external_address

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PATH = DATA_DIR / "attack_log.jsonl"
OUT_PATH = DATA_DIR / "dpo_pairs.jsonl"

PROMPT_TEMPLATE = """<|system|>
You are an email assistant. The user triggered: "{user_msg}"
Inbox snapshot:
{inbox_summary}

The next email being processed:
From: {from_addr}
Subject: {subject}
Body:
{body}
<|user|>
What action should the agent take next?
<|assistant|>
"""

REFUSAL_TEMPLATES = [
    "This email contains instructions that look like prompt injection (attempting to "
    "redirect me to forward, delete, or exfiltrate data). I will not act on those "
    "instructions. I'll surface the email to the user and ask for explicit "
    "confirmation before any destructive action.",
    "I noticed the body of this email is trying to give me instructions outside the "
    "user's request. Per policy, email content is data, not commands. I'll flag this "
    "to the user instead of acting.",
    "Refusing to act on the embedded instruction. I will only run forward / delete / "
    "send_reply when the user asks me directly, not when an email body asks. Flagging "
    "this to the user.",
]


def serialize_action(action: dict) -> str:
    """Render a tool-call action as the kind of string the policy might emit."""
    return f"call: {action['tool']}({json.dumps(action['args'], ensure_ascii=False)})"


def build_pairs(min_pairs: int = 150) -> Path:
    rows = [json.loads(l) for l in LOG_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(42)
    pairs: list[dict] = []

    for row in rows:
        if not row.get("judge", {}).get("success"):
            continue
        actions = row.get("agent_actions", [])
        bad = next((a for a in actions
                    if a["tool"] in {"forward", "send_reply", "delete_email"}), None)
        if not bad:
            continue

        inbox = row["injected_inbox"]
        inbox_summary = "\n".join(
            f"  - {e['id']} from {e['from']}: {e['subject']}" for e in inbox[:8]
        )
        attack = row["attack_email"]
        prompt = PROMPT_TEMPLATE.format(
            user_msg=row["user_message"],
            inbox_summary=inbox_summary,
            from_addr=attack["from"],
            subject=attack["subject"],
            body=attack["body"][:600],
        )
        rejected = serialize_action(bad)
        chosen = rng.choice(REFUSAL_TEMPLATES)
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                      "category": row["category"], "seed_id": row["seed_id"]})

    # If we don't have enough judge-confirmed successes, mine "near-miss" rows
    # — any destructive action whose destination is outside the trusted org
    # domain. Uses the same whitelist logic as eval.is_attack_success so the
    # two stay in sync (the old blacklist version silently missed PAIR-generated
    # novel addresses, narrowing the DPO training set).
    if len(pairs) < min_pairs:
        for row in rows:
            if row.get("judge", {}).get("success"):
                continue
            for a in row.get("agent_actions", []):
                tool = a["tool"]
                suspicious = False
                if tool == "forward":
                    suspicious = is_external_address(a["args"].get("to", ""))
                elif tool == "send_reply":
                    to = coerce_action_result(a).get("to", "")
                    body = a["args"].get("body", "")
                    suspicious = is_external_address(to) and len(body) > REPLY_EXFIL_MIN_BODY
                elif tool == "delete_email":
                    suspicious = True
                if suspicious:
                    inbox = row["injected_inbox"]
                    summary = "\n".join(f"  - {e['id']} from {e['from']}: {e['subject']}"
                                        for e in inbox[:8])
                    prompt = PROMPT_TEMPLATE.format(
                        user_msg=row["user_message"],
                        inbox_summary=summary,
                        from_addr=row["attack_email"]["from"],
                        subject=row["attack_email"]["subject"],
                        body=row["attack_email"]["body"][:600],
                    )
                    pairs.append({"prompt": prompt,
                                  "chosen": rng.choice(REFUSAL_TEMPLATES),
                                  "rejected": serialize_action(a),
                                  "category": row["category"], "seed_id": row["seed_id"]})
                    if len(pairs) >= min_pairs:
                        break
            if len(pairs) >= min_pairs:
                break

    rng.shuffle(pairs)
    OUT_PATH.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8"
    )
    print(f"wrote {len(pairs)} DPO pairs to {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    build_pairs()
