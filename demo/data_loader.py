"""Cached data loaders for the Streamlit demo.

Single source of truth: the JSON files under `results/` plus
`data/attack_log.jsonl` for attack email bodies. No state, no caching
beyond `@st.cache_data` (which keys on function arguments).
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
DATA = REPO / "data"

# Defense configurations in display order.
ATTACK_LABELS = [
    "baseline",
    "guard",
    "verifier_only",
    "verifier_only_loose",
    "combined",
    "combined_loose",
]
BENIGN_LABELS = ATTACK_LABELS  # symmetric

DISPLAY_NAME = {
    "baseline": "Baseline (no defense)",
    "guard": "Classifier only",
    "verifier_only": "Verifier (strict)",
    "verifier_only_loose": "Verifier (loose)",
    "combined": "Combined (strict)",
    "combined_loose": "Combined (loose)",
}


@st.cache_data
def load_attack(label: str) -> dict:
    """Load `results/attack_{label}.json`."""
    return json.loads((RESULTS / f"attack_{label}.json").read_text(encoding="utf-8"))


@st.cache_data
def load_benign(label: str) -> dict:
    """Load `results/benign_{label}.json`."""
    return json.loads((RESULTS / f"benign_{label}.json").read_text(encoding="utf-8"))


@st.cache_data
def load_grpo_behavioral() -> dict:
    """Load the GRPO standalone behavioral eval (`eval_grpo_attack.py` output)."""
    return json.loads((RESULTS / "grpo_behavioral_attack.json").read_text(encoding="utf-8"))


@st.cache_data
def load_grpo_train() -> dict:
    """Load the GRPO training summary."""
    return json.loads((RESULTS / "grpo_train.json").read_text(encoding="utf-8"))


@st.cache_data
def load_attack_emails() -> dict:
    """Map (seed_id, round) -> attack_email dict pulled from attack_log.jsonl."""
    rows = [
        json.loads(line)
        for line in (DATA / "attack_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {(r["seed_id"], r["round"]): r["attack_email"] for r in rows}


@st.cache_data
def load_attack_meta() -> list[tuple[str, str, int]]:
    """List of (seed_id, category, round) tuples in baseline order — used for the seed picker."""
    baseline = load_attack("baseline")
    return [(r["seed_id"], r["category"], r.get("round", 0)) for r in baseline["details"]]
