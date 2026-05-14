"""SFT warmup for Qwen2.5-1.5B-Instruct on injection refusal (V3, QLoRA 4-bit).

Why this exists:
  Fresh Qwen-0.5B-Instruct has near-zero probability of generating any
  safety-thinking text on attack prompts — pretraining taught it to be
  a helpful email assistant, not a security-aware one. Running GRPO
  directly on this base produced reward stuck at 0.21 and KL ≈ 0.001
  (policy didn't move), because no rollout ever landed in the high-
  reward "refuse" region for advantage to point to.

  This is the standard "RL without prior" failure. The standard fix:
  a small SFT pass first to teach the model the target behaviour, then
  RL refines. Pretrain → SFT → RL is the canonical RLHF recipe.

Data:
  data/grpo_prompts.jsonl (114 attack-laden prompts, already chat-formatted)
  × src.dpo_data.REFUSAL_TEMPLATES (10 hand-written refusal styles)
  = 1140 (prompt, completion) pairs

Output:
  adapters/qwen-injection-sft/  → LoRA adapter (mergeable with GRPO step)
  results/sft_warmup.json       → loss summary

V3 changes vs V2 (0.5B):
  - base model: Qwen2.5-0.5B → Qwen2.5-1.5B-Instruct (3× capacity)
  - quantization: bf16 → 4-bit NF4 (QLoRA) for memory budget on 4060 8GB
  - per_device_batch: 4 → 2, grad_accum: 4 → 8 (1.5B activations are ~3×
    larger, so smaller per-step batch with more accumulation)
  - same LoRA r=16, same 7 target_modules, same lr=2e-5, same 2 epochs
  - QLoRA loader follows the proven pattern in src/dpo_train.py:65-78

Time:
  ~12-18 min on RTX 4060 (1140 samples × 2 epochs / batch 16 ≈ 142 steps × 5-7s)
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ADAPTERS_DIR = Path(__file__).resolve().parent.parent / "adapters"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ADAPTERS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_BASE = os.environ.get("GRPO_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
PROMPTS_PATH = DATA_DIR / "grpo_prompts.jsonl"
ADAPTER_OUT = ADAPTERS_DIR / "qwen-injection-sft"


def main(
    base_model: str = DEFAULT_BASE,
    epochs: int = 2,
    learning_rate: float = 2e-5,         # standard SFT lr (GRPO uses 1e-6)
    # V3: per_device 4 → 2, grad_accum 4 → 8. 1.5B activations ~3× larger
    # than 0.5B, so smaller per-step batch with more accumulation keeps
    # effective batch = 16 while staying under 4060 8GB ceiling.
    per_device_batch_size: int = 2,
    grad_accum: int = 8,                 # effective batch = 16
    lora_r: int = 16,
    max_seq_length: int = 768,
):
    # torch must be imported before trl modules to avoid a startup
    # segfault path in trl 0.19's lazy loader on Windows.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    # REFUSAL_TEMPLATES is the same 10-template list used by dpo_data —
    # well-tested for diversity (length, register, structure).
    from src.dpo_data import REFUSAL_TEMPLATES

    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(
            f"missing {PROMPTS_PATH}; run `uv run python -m src.grpo_data` first"
        )

    prompts = [
        json.loads(l) for l in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Cartesian product: each prompt × each of 10 templates.
    # Rationale: with only 114 unique prompts, pairing each with all
    # templates (1140 samples) lets the model see refusal-style diversity
    # within the same context, preventing memorisation of one fixed
    # phrasing. Random-pairing (114 samples) was the alternative but
    # gives ~10× less data and risks SFT over-fitting to one style.
    samples: list[dict] = []
    for prompt_obj in prompts:
        for tpl in REFUSAL_TEMPLATES:
            # prompt already ends with <|im_start|>assistant\n thanks to
            # apply_chat_template(add_generation_prompt=True) in grpo_data.
            # Appending the refusal + EOS gives the model a complete
            # assistant turn to learn from.
            text = prompt_obj["prompt"] + tpl + tokenizer.eos_token
            samples.append({"text": text})

    random.Random(42).shuffle(samples)
    ds = Dataset.from_list(samples)
    print(f"built {len(ds)} SFT samples ({len(prompts)} prompts × {len(REFUSAL_TEMPLATES)} templates)")

    # V3 QLoRA 4-bit load: identical pattern to src/dpo_train.py:65-78
    # which was proven to fit 1.5B + LoRA training on 4060 8GB.
    # Order matters: load → prepare_for_kbit (enables grad-ckpt-compatible
    # state) → enable_input_require_grads → get_peft_model (in next step).
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
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

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
    model.print_trainable_parameters()

    cfg = SFTConfig(
        output_dir=str(ADAPTER_OUT),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        max_length=max_seq_length,
        dataset_text_field="text",
        packing=False,                 # one sample per sequence; cleaner gradients
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        # SFTTrainer auto-handles loss masking; we train on the full
        # sequence which includes the prompt — this is fine for warmup
        # since we just want refusal continuation to be learned.
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(ADAPTER_OUT))
    tokenizer.save_pretrained(str(ADAPTER_OUT))

    history = trainer.state.log_history
    final = next((h for h in reversed(history) if "loss" in h), {})
    summary = {
        "base_model": base_model,
        "n_samples": len(ds),
        "n_prompts": len(prompts),
        "n_templates": len(REFUSAL_TEMPLATES),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "final_loss": final.get("loss"),
        "final_grad_norm": final.get("grad_norm"),
        "adapter_path": str(ADAPTER_OUT),
    }
    (RESULTS_DIR / "sft_warmup.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nSFT done. Adapter at {ADAPTER_OUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
