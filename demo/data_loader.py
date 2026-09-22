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

# Defense configurations the demo can load, in display order.
# The p0_* entries are the k=3 replicate runs behind report §1.2; r1 is shown
# because every replicate has the same 38 rows and the picker only needs one.
ATTACK_LABELS = [
    "p0_naive_r1",
    "p0_hardened_r1",
    "baseline",
    "guard",
    "verifier_only",
    "verifier_only_loose",
    "combined",
    "combined_loose",
]
BENIGN_LABELS = ATTACK_LABELS  # symmetric

# The replay tab offers only these. `baseline` duplicates the naive prompt, and
# the `_loose` variants belong to a comparison the report retracted (§2.5).
REPLAY_LABELS = ["p0_naive_r1", "p0_hardened_r1", "guard", "verifier_only", "combined"]

DISPLAY_NAME = {
    "p0_naive_r1": "Naive prompt (no defense)",
    "p0_hardened_r1": "Hardened prompt",
    "baseline": "Baseline (no defense)",
    "guard": "Classifier",
    "verifier_only": "RL verifier",
    "verifier_only_loose": "RL verifier (loose)",
    "combined": "Classifier + RL verifier",
    "combined_loose": "Classifier + RL verifier (loose)",
}


def _has(label: str) -> bool:
    return (RESULTS / f"attack_{label}.json").exists()


# Labels whose result files are actually present. Deployment to Streamlit Cloud
# syncs this repo by hand, so a partial push is a real failure mode: without this
# filter a single missing JSON takes the whole page down with FileNotFoundError.
# Every consumer should iterate this, not ATTACK_LABELS.
AVAILABLE_ATTACK_LABELS = [lbl for lbl in ATTACK_LABELS if _has(lbl)]


@st.cache_data
def load_attack(label: str) -> dict:
    """Load `results/attack_{label}.json`."""
    return json.loads((RESULTS / f"attack_{label}.json").read_text(encoding="utf-8"))


@st.cache_data
def load_benign(label: str) -> dict:
    """Load `results/benign_{label}.json`."""
    return json.loads((RESULTS / f"benign_{label}.json").read_text(encoding="utf-8"))


@st.cache_data
def load_result(name: str) -> dict | None:
    """Load an arbitrary `results/{name}.json`, or None if it has not been generated.

    Used for the audit and judge artefacts (report §2–§8), which are produced by
    the `scripts/*.py` drivers rather than by training. Returning None lets the
    page degrade to a "not generated" note instead of crashing.
    """
    p = RESULTS / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def load_grpo_behavioral() -> dict | None:
    """GRPO standalone behavioral eval (`eval_grpo_attack.py`), or None if absent."""
    return load_result("grpo_behavioral_attack")


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
    """(seed_id, category, round) tuples for the seed picker.

    Keyed off whichever attack file is actually deployed — every config replays
    the same 38 rows, so any of them gives the same seed list.
    """
    if not AVAILABLE_ATTACK_LABELS:
        return []
    src = "baseline" if "baseline" in AVAILABLE_ATTACK_LABELS else AVAILABLE_ATTACK_LABELS[0]
    return [(r["seed_id"], r["category"], r.get("round", 0))
            for r in load_attack(src)["details"]]
