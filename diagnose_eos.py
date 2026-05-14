"""Final test: does training mode (vs eval mode) kill EOS generation?"""
import json, torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "adapters/qwen-injection-sft"
tok = AutoTokenizer.from_pretrained(BASE)

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map={"": "cuda"}, torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()
model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=True)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

lines = open("data/grpo_prompts.jsonl", encoding="utf-8").readlines()
prompts = [json.loads(l)["prompt"] for l in lines[:2]]
prompts_repeated = [p for p in prompts for _ in range(4)]
pi = tok(text=prompts_repeated, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
prompt_ids = pi["input_ids"].to(model.device)
prompt_mask = pi["attention_mask"].to(model.device)

gc = GenerationConfig(
    max_new_tokens=96, do_sample=True, temperature=0.7,
    pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
    eos_token_id=tok.eos_token_id,
    top_p=None, top_k=None, min_p=None, repetition_penalty=None,
    cache_implementation=None,
)

# Test in TRAINING mode (as GRPO does)
model.train()
print(f"model.training: {model.training}")
print("=== TRAIN mode generation ===")
with torch.no_grad():
    pci = model.generate(prompt_ids, attention_mask=prompt_mask, generation_config=gc)
comp = pci[:, prompt_ids.size(1):]
is_eos = comp == tok.eos_token_id
print(f"shape: {comp.shape}, EOS per row: {is_eos.any(dim=1).tolist()}")
for i in range(8):
    non_pad = comp[i][comp[i] != tok.pad_token_id]
    has_eos = 151645 in non_pad.tolist()
    text = tok.decode(non_pad, skip_special_tokens=True).strip()
    print(f"  [{i}] len={len(non_pad)}, eos={has_eos}, text={text[:80]}...")

# Now compare with eval mode
model.eval()
print(f"\nmodel.training: {model.training}")
print("=== EVAL mode generation ===")
with torch.no_grad():
    pci = model.generate(prompt_ids, attention_mask=prompt_mask, generation_config=gc)
comp = pci[:, prompt_ids.size(1):]
is_eos = comp == tok.eos_token_id
print(f"shape: {comp.shape}, EOS per row: {is_eos.any(dim=1).tolist()}")
for i in range(8):
    non_pad = comp[i][comp[i] != tok.pad_token_id]
    has_eos = 151645 in non_pad.tolist()
    text = tok.decode(non_pad, skip_special_tokens=True).strip()
    print(f"  [{i}] len={len(non_pad)}, eos={has_eos}, text={text[:80]}...")
