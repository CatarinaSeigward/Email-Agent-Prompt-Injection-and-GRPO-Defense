"""TRL DPO + 4-bit QLoRA on Qwen2.5-1.5B-Instruct.

Per the plan:
  - 4060 (8GB) → 4-bit base + LoRA adapters
  - batch_size=1 + grad accumulation if OOM
  - target ~2h train time on the produced preference set

Outputs:
  adapters/qwen-injection-dpo/  → LoRA adapter (mergeable)
  results/dpo_train.json        → final loss + reward margins

Run after src/dpo_data.py has produced data/dpo_pairs.jsonl.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ADAPTERS_DIR = Path(__file__).resolve().parent.parent / "adapters"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ADAPTERS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_BASE = os.environ.get("DPO_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
PAIRS_PATH = DATA_DIR / "dpo_pairs.jsonl"
ADAPTER_OUT = ADAPTERS_DIR / "qwen-injection-dpo"


def main(
    base_model: str = DEFAULT_BASE,
    # Softened hyperparameters after observing 8 epochs / lr=1e-5 produced
    # margins=10.5 but generation behaviour unchanged + train_loss=9.4e-5
    # (severe overfit on training sequences). The fix isn't more aggressive
    # DPO — it's matching Qwen's native chat template in the prompt format
    # (see dpo_data._render_prompt). With correct prompt format, milder
    # training is sufficient and avoids damaging out-of-distribution
    # generation.
    epochs: int = 4,
    learning_rate: float = 5e-6,
    per_device_batch_size: int = 1,
    grad_accum: int = 8,
    beta: float = 0.2,
    lora_r: int = 16,
):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    if not PAIRS_PATH.exists():
        raise FileNotFoundError(f"missing {PAIRS_PATH}; run dpo_data.py first")

    pairs = [json.loads(l) for l in PAIRS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    ds = Dataset.from_list(pairs)
    print(f"loaded {len(ds)} preference pairs")

    # bf16 throughout — RTX 4060 (Ada Lovelace) supports bf16 natively, and
    # bf16 has the same dynamic range as fp32, so no GradScaler needed.
    # Mixing fp16 GradScaler with bf16 LoRA/quant outputs hits:
    #   "_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # NOTE: TRL >= 0.12 removed `max_prompt_length` — it now derives prompt
    # length from `max_length` automatically. Keep only the total budget.
    cfg = DPOConfig(
        output_dir=str(ADAPTER_OUT),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        beta=beta,
        max_length=1024,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True, fp16=False,
        gradient_checkpointing=True,
        report_to=[],
    )

    # NOTE: TRL >= 0.13 renamed `tokenizer` to `processing_class` in DPOTrainer
    # (following the same change in transformers.Trainer >= 4.46).
    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(ADAPTER_OUT))
    tokenizer.save_pretrained(str(ADAPTER_OUT))

    history = trainer.state.log_history
    final = next((h for h in reversed(history) if "loss" in h), {})
    summary = {
        "base_model": base_model,
        "n_pairs": len(ds),
        "epochs": epochs,
        "final_loss": final.get("loss"),
        "rewards/chosen": final.get("rewards/chosen"),
        "rewards/rejected": final.get("rewards/rejected"),
        "rewards/margins": final.get("rewards/margins"),
        "adapter_path": str(ADAPTER_OUT),
    }
    (RESULTS_DIR / "dpo_train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDPO done. Adapter at {ADAPTER_OUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
