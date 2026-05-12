"""Generate the README hero figure: 3-stage ASR comparison + benign pass rate.

Reads results/{benign,attack}_{baseline,guard,dpo}.json (whichever exist)
and produces results/comparison.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STAGES = ["baseline", "guard", "dpo"]
STAGE_LABEL = {"baseline": "Baseline", "guard": "+ Classifier Guard", "dpo": "+ DPO Repair"}
CATEGORIES = ["override", "hidden_injection", "exfiltration"]
CAT_LABEL = {"override": "A1 Override", "hidden_injection": "A2 Hidden",
             "exfiltration": "A3 Exfiltration"}


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def make_plots() -> Path:
    asr = {s: _load(f"attack_{s}.json") for s in STAGES}
    benign = {s: _load(f"benign_{s}.json") for s in STAGES}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.25
    x = np.arange(len(CATEGORIES))

    for i, stage in enumerate(STAGES):
        data = asr.get(stage)
        if not data:
            heights = [0] * len(CATEGORIES)
        else:
            heights = [data["asr_by_category"].get(c, 0) * 100 for c in CATEGORIES]
        bars = ax1.bar(x + (i - 1) * width, heights, width, label=STAGE_LABEL[stage])
        for b, h in zip(bars, heights):
            ax1.text(b.get_x() + b.get_width() / 2, h + 1, f"{h:.0f}%",
                     ha="center", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels([CAT_LABEL[c] for c in CATEGORIES])
    ax1.set_ylabel("Attack Success Rate (%)  ↓ better")
    ax1.set_title("Attack Success Rate by Category")
    ax1.set_ylim(0, 100)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    bx = np.arange(len(STAGES))
    benign_pct = [(benign[s]["pass_rate"] * 100) if benign.get(s) else 0 for s in STAGES]
    bars = ax2.bar(bx, benign_pct, color=["#4C72B0", "#55A868", "#C44E52"])
    for b, h in zip(bars, benign_pct):
        ax2.text(b.get_x() + b.get_width() / 2, h + 1, f"{h:.0f}%",
                 ha="center", fontsize=10)
    ax2.set_xticks(bx)
    ax2.set_xticklabels([STAGE_LABEL[s] for s in STAGES])
    ax2.set_ylabel("Benign Task Pass Rate (%)  ↑ better")
    ax2.set_title("Benign Pass Rate (over-refusal check)")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    return out


def make_table_md() -> str:
    """Render the README's headline metric table from result files."""
    asr = {s: _load(f"attack_{s}.json") for s in STAGES}
    benign = {s: _load(f"benign_{s}.json") for s in STAGES}

    def _fmt(v):
        return f"{v * 100:.0f}%" if isinstance(v, (int, float)) else "—"

    lines = [
        "| Attack | " + " | ".join(STAGE_LABEL[s] for s in STAGES) + " |",
        "|---|" + "|".join(["---"] * len(STAGES)) + "|",
    ]
    for c in CATEGORIES:
        row = [CAT_LABEL[c]]
        for s in STAGES:
            d = asr.get(s)
            row.append(_fmt(d["asr_by_category"].get(c)) if d else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| **Benign Pass Rate** | " + " | ".join(
        _fmt(benign[s]["pass_rate"]) if benign.get(s) else "—" for s in STAGES) + " |")
    out = "\n".join(lines)
    (RESULTS_DIR / "headline_table.md").write_text(out, encoding="utf-8")
    print(f"saved headline_table.md\n\n{out}")
    return out


if __name__ == "__main__":
    make_plots()
    make_table_md()
