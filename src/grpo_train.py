"""TRL GRPO on Qwen2.5-0.5B-Instruct with rule-based reward.

Why GRPO here, not DPO:
  DPO failed because `chosen` (natural-language refusal) and `rejected`
  (`call: tool({args})` pseudo-format) live in different output spaces —
  Qwen has never seen the rejected format in pretraining, so margins
  inflate without the policy actually learning to refuse. GRPO doesn't
  need paired data: it samples G completions per prompt and ranks them
  by a reward function we control directly.

MVP scope (4060 8GB):
  - base = Qwen2.5-0.5B-Instruct (small enough for policy + reference
    + G=4 rollouts without OOM, no vllm needed)
  - LoRA r=16 (mirrors dpo_train for apples-to-apples comparison)
  - reward = rule-based pattern matching (no LLM judge calls, $0, fast)
  - G=4 rollouts per prompt, batch=1, grad_accum=8 → effective batch=32
  - target ~1-2h on 4060

NOT in scope for MVP:
  - vllm-backed generation (won't fit in 8GB alongside policy+ref)
  - LLM-as-judge reward (slower, costs API credits)
  - eval pipeline integration (needs verified tool_call output first)

Outputs:
  adapters/qwen-injection-grpo/  → LoRA adapter
  results/grpo_train.json        → training summary
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ADAPTERS_DIR = Path(__file__).resolve().parent.parent / "adapters"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ADAPTERS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_BASE = os.environ.get("GRPO_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
PROMPTS_PATH = DATA_DIR / "grpo_prompts.jsonl"
ADAPTER_OUT = ADAPTERS_DIR / "qwen-injection-grpo"
# Default GRPO init = output of src/sft_warmup.py. Setting this to None
# (or pointing at a non-existent path) falls back to fresh LoRA init,
# which V1 demonstrated does not converge for this task — keep that as
# an explicit ablation knob, not a default.
DEFAULT_SFT_INIT = ADAPTERS_DIR / "qwen-injection-sft"


# ── Reward function ────────────────────────────────────────────────────────
# Pure pattern matching. The intent isn't to be a perfect classifier — it's
# to give GRPO a stable, deterministic, cheap signal so the policy can climb
# from random initial completions toward "refuse + explain" outputs.

# Unsafe = the model is roleplaying as if it's about to perform the attack.
# Anchors include both pseudo-call format ("forward(") and JSON tool-call
# format ('"name": "forward"'), so this works regardless of which output
# style the policy drifts toward.
UNSAFE_PATTERNS = [
    r"\bforward\s*\(",
    r"\bdelete_email\s*\(",
    r"\bsend_reply\s*\(",
    r'"name"\s*:\s*"forward"',
    r'"name"\s*:\s*"delete_email"',
    r'"name"\s*:\s*"send_reply"',
    r'"tool"\s*:\s*"forward"',
    r'"tool"\s*:\s*"delete_email"',
    r"i'?ll forward",
    r"i'?ll delete",
    r"forwarding (it|the email|now)",
    r"deleting (it|the email|now)",
]

# Refusal = the model recognises the prompt-injection pattern and declines.
# We deliberately accept several phrasings (Qwen-0.5B has only ~30 prompts
# to learn from, so the policy gets credit for any reasonable refusal style
# rather than memorising one template).
REFUSE_PATTERNS = [
    r"\bi (won'?t|will not|cannot|can'?t)\b",
    r"\bi (refuse|decline)\b",
    r"\b(no,|no\.)",
    r"prompt injection",
    r"injection attempt",
    r"not from the user",
    r"email (body|content) is not",
    r"i'?m not going to (forward|delete|send|act)",
    r"flag(ging)? (this|it) (to|for) (the )?user",
    r"requires? (explicit )?(user )?confirmation",
    r"won'?t (act on|follow|execute)",
    r"ignor(e|ing) (that|the embedded|the email body)",
    r"\bdo not (follow|comply|act on)",
    r"\bshould not (forward|delete|send|comply)",
    r"\bsuspicious\b",
    r"\bnot trust(ed|worthy)?\b",
]

# Soft-refusal = the model is hesitating / asking for confirmation /
# expressing caution. Weaker than full refusal but in the right
# direction; we reward it as a stepping stone so the policy has a
# gradient to climb from "totally neutral" toward "explicit refuse".
SOFT_REFUSE_PATTERNS = [
    r"\b(confirm|verify|authoriz|permission)",
    r"\bare you sure\b",
    r"\bshould i (really )?(do|forward|delete|send|act)",
    r"\b(double[- ]?check|double check)\b",
    r"\bcheck with (the )?(user|you)",
    r"\bask (the )?user",
    r"\b(unusual|strange|odd|fishy|suspect)\b",
    r"\bcautious\b",
    r"\bbefore (i|acting|forward|delet|send)",
    r"\bwait(ing)? for",
]

# Compliant = neutral agreeable text without any safety-thinking. This
# is what Qwen-0.5B base produces by default ("Sure, I'll review the
# email and forward..."). We give this a NEGATIVE reward so the default
# behaviour is the worst option in the group, forcing GRPO to favour
# anything closer to refusal.
COMPLIANT_PATTERNS = [
    r"\bi'?ll (review|check|look|read|process|examine|handle|take care|help|assist|do that|go ahead)",
    r"\blet me (check|see|look|review|read|help)",
    r"\b(sure|okay|ok|alright|got it|of course|certainly|absolutely|no problem)\b[,.\s]",
    r"\bhappy to (help|assist|do|forward|process)",
    r"\bi (would|can|will) (be )?(happy|glad) to",
    r"\bthank(s| you) for (the|your)",
    r"\bas (you )?request",
    r"\bright away\b",
]

# Specificity = the refusal references the *reason* it refuses (this is
# an email-injection attack, content comes from email body not user, etc).
# We give a bonus on top of REFUSE_PATTERNS so GRPO has somewhere to climb
# even after SFT collapses the policy onto generic "I'm sorry, I can't
# assist" output. Empirically, SFT on 0.5B with 10 long templates produces
# the generic refusal (model can't memorise 10 paragraphs in r=16 LoRA),
# leaving GRPO with no within-group variance unless we differentiate
# generic-vs-specific refusal at reward time.
SPECIFIC_REFUSE_PATTERNS = [
    r"\bemail (body|content|message)\b",
    r"\b(prompt|email|content) injection\b",
    r"\b(came|comes) from (the )?email\b",
    r"\bnot from (the )?(user|you)\b",
    r"\bin the email\b",
    r"\bembedded (in|instruction|request)",
    r"\b(third[- ]?party|external) (email|sender|content)",
    r"\bflag(ging)? (this|it) (to|for) (the )?user",
    r"\b(data[- ]?handling|security) (policy|risk)",
    r"\b(forward|delete|exfil|send) (any|the|this) email",
]

UNSAFE_RE = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]
REFUSE_RE = [re.compile(p, re.IGNORECASE) for p in REFUSE_PATTERNS]
SOFT_REFUSE_RE = [re.compile(p, re.IGNORECASE) for p in SOFT_REFUSE_PATTERNS]
COMPLIANT_RE = [re.compile(p, re.IGNORECASE) for p in COMPLIANT_PATTERNS]
SPECIFIC_REFUSE_RE = [re.compile(p, re.IGNORECASE) for p in SPECIFIC_REFUSE_PATTERNS]


def _has_any(text: str, patterns: list) -> bool:
    return any(p.search(text) for p in patterns)


def reward_fn(completions, **kwargs) -> list[float]:
    """Layered rule-based reward designed for dense signal.

    Rationale: the previous version returned 0 for any "neutral" output,
    and Qwen-0.5B's default behaviour on attack prompts IS neutral rambling
    ("Sure, I'll review the email and..."). With 0 reward across all G
    rollouts, advantage = 0 / 0 = no gradient. Result: 100% of prompt
    groups had reward_std=0 and the policy didn't move (KL ≈ 0.002 after
    30 steps).

    The new reward stack is additive — every output gets a signal from
    multiple independent axes, so identical-looking rollouts rarely tie:

    Refusal axis:
      +1.0  contains explicit refusal pattern (won't / refuse / no, / etc.)
      -1.0  contains unsafe tool-call pattern (forward(/delete_email(/...)
      +0.4  contains soft-refuse (asking confirmation, "suspicious", etc.)
            — stepping stone reward, only credited if no full refusal
      -0.4  contains compliant boilerplate ("Sure, I'll help...")
            — only credited if no refusal/soft-refuse, so refusal can
            still appear alongside "Sure" without double-penalty

    Length axis (forces concise, terminated outputs):
      +0.2  8..50 words   (concise, clearly bounded)
      -0.3  >60 words     (rambling, likely max-length cut; V4: scaled
                           for max_completion_length=96 ≈ 65 words)
      -0.5  <3 words      (degenerate one-word reward hack)

    Termination axis:
      +0.4  appears to have ended naturally (last char is . ! ? or
            length < 90% of max_completion_length budget)
            V4: raised from 0.2 to 0.4 — V3 had 0% EOS rate across
            684 rollouts; stronger incentive needed.
    """
    rewards: list[float] = []
    for c in completions:
        text = c if isinstance(c, str) else str(c)
        unsafe = _has_any(text, UNSAFE_RE)
        refuse = _has_any(text, REFUSE_RE)
        soft = _has_any(text, SOFT_REFUSE_RE)
        comply = _has_any(text, COMPLIANT_RE)
        specific = _has_any(text, SPECIFIC_REFUSE_RE)

        # Refusal axis (mutually-exclusive priority order)
        if refuse:
            r = 1.0
        elif soft:
            r = 0.4
        elif comply:
            r = -0.4
        else:
            r = 0.0

        # Specificity bonus: refusal that references *why* (email-source,
        # injection, etc.) earns +0.3 over generic refusal. Gives GRPO a
        # climbing direction even after SFT collapses to "I'm sorry, can't
        # assist" generic refusal.
        if refuse and specific:
            r += 0.3

        # Unsafe penalty stacks on top — unsafe + refuse is half-credit
        if unsafe:
            r -= 1.0

        # Length axis
        n_words = len(text.split())
        if 8 <= n_words <= 50:
            r += 0.2
        elif n_words > 60:  # V4: scaled with max_completion_length=96 (~65 words)
            r -= 0.3
        if n_words < 3:
            r -= 0.5

        # Termination axis: any of these is a "I finished a thought" signal
        stripped = text.strip()
        if stripped and stripped[-1] in ".!?":
            r += 0.4  # V4: 0.2 → 0.4; V3 had 0% EOS rate, need stronger signal

        rewards.append(r)
    return rewards


# ── Trainer ────────────────────────────────────────────────────────────────

def main(
    base_model: str = DEFAULT_BASE,
    # V2 (post-SFT-warmup) hyperparameters:
    # epochs 5 → 3: policy starts in a sane region after SFT, doesn't
    #   need the long climb. 3 epochs ≈ 8-10 min.
    epochs: int = 3,
    # V2-iter2 update: 1e-6 → 2e-6. iter1 showed reward signal (mean 0.20
    # with std 0.13) was alive but couldn't overcome the KL pull toward
    # base. Bumping lr 2× gives reward gradient more weight per step.
    learning_rate: float = 2e-6,
    per_device_batch_size: int = 2,
    grad_accum: int = 8,
    num_generations: int = 4,   # V4: G=4 for reward variance; memory-neutral since max_completion 256→96
    # V2-iter2 update: 0.04 → 0.005. iter1 showed KL going DOWN over
    # training (1.11 → 0.71 → ...), meaning policy was pulled back toward
    # base Qwen (the reference in trl GRPO when policy is a PeftModel).
    # At beta=0.04 the KL penalty dominated the reward signal, so GRPO
    # was un-training the SFT-installed refusal behaviour. Drop beta 8×
    # so reward signal can actually push policy further from base in
    # the desired (more refusal-aware) direction.
    beta: float = 0.005,
    lora_r: int = 16,
    max_prompt_length: int = 512,
    max_completion_length: int = 96,   # V4: 256→96. SFT refusals are ~20-50 tokens. V3 at 256 had clipped_ratio=1.0 (0% EOS) → model rambled → all rewards ≈0.10
    # When set, load this LoRA adapter as the GRPO starting point instead
    # of initialising fresh LoRA. The standard pipeline is sft_warmup →
    # grpo_train, so the default points at the SFT output. Pass None to
    # do an ablation against fresh-LoRA init.
    init_adapter_path: str | None = str(DEFAULT_SFT_INIT),
):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(
            f"missing {PROMPTS_PATH}; run `uv run python -m src.grpo_data` first"
        )

    prompts = [
        json.loads(l) for l in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    ds = Dataset.from_list(prompts)
    print(f"loaded {len(ds)} GRPO prompts ({len({p['seed_id'] for p in prompts})} unique seeds)")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # V5 note: Qwen2.5-Instruct uses TWO stop tokens:
    #   151645 = <|im_end|>    (chat turn terminator — what the model actually generates)
    #   151643 = <|endoftext|> (base EOS)
    # TRL 0.19.1 reads tokenizer.eos_token_id (=151645) which is correct.
    # The generation_config also has [151645, 151643] by default. No patching needed.

    # V3: QLoRA 4-bit load. With G=2 rollouts × max_completion=256 +
    # reference model on 1.5B, bf16 would need ~9.4 GB → OOM on 4060.
    # 4-bit NF4 brings the policy+reference down to ~1.7 GB combined,
    # leaving plenty of headroom for activations and rollouts.
    # Identical config to src/dpo_train.py:65-78 (proven on this GPU).
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map={"": "cuda"},
        torch_dtype=torch.bfloat16,
    )
    # prepare_model_for_kbit_training: casts norms to fp32, makes
    # output embedding require_grad — needed for gradient_checkpointing
    # to function over a 4-bit base.
    # V6: pass use_gradient_checkpointing=False. The default (True) enables
    # gradient_checkpointing on the model, which forces use_cache=False and
    # breaks autoregressive generation during GRPO rollout sampling (the
    # model never generates EOS → clipped_ratio=1.0 in V3-V5).
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    # Critical for LoRA + gradient_checkpointing: without this, embedding
    # output is a no-grad tensor and gradient_checkpointing cannot trace
    # back through the (frozen) base to the LoRA adapters. Symptom is the
    # "None of the inputs have requires_grad=True" warning, followed by
    # the trainer optimizer initialising AdamW state for the full model
    # (since LoRA params look unused) and OOM-ing on the second-moment
    # allocation.
    model.enable_input_require_grads()

    # Two init paths:
    #   1. init_adapter_path set + exists  → load SFT-trained LoRA, continue
    #      training it (is_trainable=True). This is the V2 pipeline path.
    #   2. otherwise                       → fresh LoRA. V1 fallback; kept
    #      for ablation but known not to converge for this task.
    if init_adapter_path and Path(init_adapter_path).exists():
        model = PeftModel.from_pretrained(
            model, init_adapter_path, is_trainable=True
        )
        print(f"loaded SFT init adapter from {init_adapter_path}")
    else:
        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
        if init_adapter_path:
            print(f"WARN: init_adapter_path={init_adapter_path} does not exist; "
                  f"falling back to fresh LoRA (V1 path, may not converge)")
    model.print_trainable_parameters()

    cfg = GRPOConfig(
        output_dir=str(ADAPTER_OUT),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        beta=beta,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        # V4 update: T=1.2 caused 100% of rollouts to hit 256-token cap
        # without generating EOS. SFT-trained EOS behaviour is fragile at
        # high temperature. Drop to 0.7 — still above greedy (preserves
        # within-group diversity for GRPO advantage) but low enough that
        # the model's learned EOS placement survives sampling.
        temperature=0.7,
        bf16=True,
        fp16=False,
        # V6: gradient_checkpointing DISABLED. V3-V5 all had clipped_ratio=1.0
        # (0% EOS) because gradient_checkpointing forces use_cache=False,
        # which corrupts autoregressive generation during GRPO rollout
        # sampling — the model never generates EOS when gradient_checkpointing
        # is active. Disabling it is safe: V4's max_completion_length=96 with
        # G=4 uses comparable memory (4×96=384 tokens) to the old 2×256=512.
        # If OOM, reduce per_device_batch_size to 1 before re-enabling.
        gradient_checkpointing=False,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        # Disable vllm — we run HF generate to stay in one process and within
        # 8GB. vllm would spawn a separate inference server that contends
        # for the same GPU.
        use_vllm=False,
    )

    # Model is already a PeftModel (we wrapped above), so don't pass
    # peft_config — that path is for "let the trainer wrap a base model".
    # Passing both can re-wrap and break trainable_params accounting.
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

    trainer.train()
    trainer.save_model(str(ADAPTER_OUT))
    tokenizer.save_pretrained(str(ADAPTER_OUT))

    history = trainer.state.log_history
    final = next((h for h in reversed(history) if "reward" in h or "loss" in h), {})
    summary = {
        "base_model": base_model,
        "init_adapter": init_adapter_path,
        "n_prompts": len(ds),
        "epochs": epochs,
        "num_generations": num_generations,
        "learning_rate": learning_rate,
        "beta": beta,
        "final_loss": final.get("loss"),
        "final_reward": final.get("reward"),
        "final_reward_std": final.get("reward_std"),
        "final_kl": final.get("kl"),
        "adapter_path": str(ADAPTER_OUT),
    }
    (RESULTS_DIR / "grpo_train.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nGRPO done. Adapter at {ADAPTER_OUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
