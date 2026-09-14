"""Generate the README hero figure: ASR comparison + benign pass rate.

Reads results/{benign,attack}_<stage>.json (whichever exist) and produces
results/comparison.png.

The stage list previously ended in `dpo`, referring to a DPO-repaired agent
that was never deployed — §3.2.5 documents why that path was abandoned, and
`results/attack_dpo.json` has never existed. The figure therefore rendered a
column of zeros for a configuration that does not exist. It now plots the
configurations that were actually measured, with the replicated ones first.

Caveat carried into the figure itself: the trained-defense columns are single
runs on a harness with a 12.1 pp run-to-run SD (final_report.md §9.1 T8), so
the bar heights are not separable. The subtitle says so, because a bar chart
invites exactly the eyeball comparison the data cannot support.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
STAGES = ["p0_naive", "p0_hardened", "guard", "verifier_only_loose", "combined_loose"]
STAGE_LABEL = {
    "p0_naive": "Naive baseline\n(k=3)",
    "p0_hardened": "Hardened prompt\n(k=3)",
    "guard": "+ Classifier\n(k=1)",
    "verifier_only_loose": "+ Verifier loose\n(k=1)",
    "combined_loose": "Combined loose\n(k=1)",
}
CATEGORIES = ["override", "hidden_injection", "exfiltration"]
CAT_LABEL = {"override": "A1 Override", "hidden_injection": "A2 Hidden",
             "exfiltration": "A3 Exfiltration"}


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _replicated_stages() -> tuple[dict, dict]:
    """Pull the k=3 configs from p0_summary.json.

    These must NOT be read from a single `attack_p0_<cfg>_r1.json` file: that is
    one replicate, and labelling one draw as "k=3" is precisely the error this
    project spent a full audit correcting. (Concretely: naive replicate 1 shows
    A2 = 0%, while the majority-vote value across three replicates is 28.6%.)
    """
    summ = _load("p0_summary.json")
    if not summ:
        return {}, {}
    asr, benign = {}, {}
    for cfg, key in (("naive", "p0_naive"), ("hardened", "p0_hardened")):
        c = summ["configs"].get(cfg)
        if not c:
            continue
        asr[key] = {"asr_by_category": {k: v["asr"] for k, v in c["by_category"].items()},
                    "asr_overall": c["asr_majority_vote"]}
        benign[key] = {"pass_rate": c["benign_pass_mean"]}
    return asr, benign


def make_plots() -> Path:
    asr = {s: _load(f"attack_{s}.json") for s in STAGES}
    benign = {s: _load(f"benign_{s}.json") for s in STAGES}
    rep_asr, rep_benign = _replicated_stages()
    asr.update(rep_asr)
    benign.update(rep_benign)

    # Drop stages whose result files are absent rather than plotting them as
    # zeros — a missing measurement is not a measurement of zero.
    stages = [s for s in STAGES if asr.get(s)]
    if not stages:
        raise SystemExit("no attack_*.json found for any stage; run the evals first")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    width = 0.8 / len(stages)
    x = np.arange(len(CATEGORIES))
    offset0 = -(len(stages) - 1) / 2

    for i, stage in enumerate(stages):
        heights = [asr[stage]["asr_by_category"].get(c, 0) * 100 for c in CATEGORIES]
        bars = ax1.bar(x + (offset0 + i) * width, heights, width, label=STAGE_LABEL[stage])
        for b, h in zip(bars, heights):
            ax1.text(b.get_x() + b.get_width() / 2, h + 1, f"{h:.0f}",
                     ha="center", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([CAT_LABEL[c] for c in CATEGORIES])
    ax1.set_ylabel("Attack Success Rate (%)  ↓ better")
    ax1.set_title("Attack Success Rate by Category")
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    bx = np.arange(len(stages))
    benign_pct = [(benign[s]["pass_rate"] * 100) if benign.get(s) else 0 for s in stages]
    bars = ax2.bar(bx, benign_pct, color="#4C72B0")
    for b, h in zip(bars, benign_pct):
        ax2.text(b.get_x() + b.get_width() / 2, h + 1, f"{h:.0f}%",
                 ha="center", fontsize=9)
    ax2.set_xticks(bx)
    ax2.set_xticklabels([STAGE_LABEL[s] for s in stages], fontsize=8)
    ax2.set_ylabel("Benign Task Pass Rate (%)  ↑ better")
    ax2.set_title("Benign Pass Rate (over-refusal check)")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Single-run configurations (k=1) carry a ±12.1 pp run-to-run SD — "
        "differences under ~24 pp are not separable. See final_report.md §9.1 T8.",
        fontsize=9, y=0.02, color="#666666",
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    out = RESULTS_DIR / "comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    return out


def make_table_md() -> str:
    """Render the README's headline metric table from result files."""
    asr = {s: _load(f"attack_{s}.json") for s in STAGES}
    benign = {s: _load(f"benign_{s}.json") for s in STAGES}
    rep_asr, rep_benign = _replicated_stages()
    asr.update(rep_asr)
    benign.update(rep_benign)

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
