# Convenience targets. PowerShell users can also call the python commands
# directly — see README.md for the full sequence.

.PHONY: help install benign-baseline redteam attack-baseline classifier guard-eval dpo-data dpo-train dpo-eval plots all

help:
	@echo "Targets:"
	@echo "  install            uv sync"
	@echo "  benign-baseline    run the 10 benign tasks against gpt-4o-mini"
	@echo "  redteam            PAIR campaign → data/attack_log.jsonl"
	@echo "  attack-baseline    replay attacks against baseline (no defense)"
	@echo "  classifier         train ModernBERT injection classifier"
	@echo "  guard-eval         re-run benign + attack with classifier guard"
	@echo "  dpo-data           build preference pairs from attack_log.jsonl"
	@echo "  dpo-train          QLoRA + DPO on Qwen2.5-1.5B"
	@echo "  dpo-eval           re-run benign + attack with DPO model"
	@echo "  plots              comparison.png + headline table"
	@echo "  all                everything end-to-end"

install:
	uv sync

benign-baseline:
	python -m src.eval benign

redteam:
	python -m src.redteam

attack-baseline:
	python -c "from pathlib import Path; from src.eval import run_attack_replay; run_attack_replay(Path('data/attack_log.jsonl'), label='baseline')"

classifier:
	python -m src.classifier

guard-eval:
	python -c "from src.classifier import load_guard; from src.eval import run_benign, run_attack_replay; from pathlib import Path; g=load_guard(); run_benign(guard=g, label='guard'); run_attack_replay(Path('data/attack_log.jsonl'), guard=g, label='guard')"

dpo-data:
	python -m src.dpo_data

dpo-train:
	python -m src.dpo_train

plots:
	python -m src.plots

all: benign-baseline redteam attack-baseline classifier guard-eval dpo-data dpo-train plots
