"""Generate the project pipeline diagram.

Hand-drawn (xkcd-style) matplotlib figure showing the 6-stage pipeline.

Outputs:
  docs/diagrams/pipeline.png
  docs/diagrams/pipeline.svg

Run: uv run python docs/diagrams/make_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent

LAYER1_BLUE = "#D8E4F4"
LAYER2_GREEN = "#D8E8D2"
EVAL_GREY = "#E0E0E8"
OUTPUT_TAN = "#F4E6CC"
TEXT_GREY = "#444444"


def box(ax, cx, cy, w, h, text, *, edge="black", fill="white", fs=10, weight="normal"):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.4,rounding_size=1",
        edgecolor=edge,
        facecolor=fill,
        linewidth=2.0,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight=weight)


def arrow(ax, x1, y1, x2, y2, *, color="black", lw=2.0, connectionstyle="arc3,rad=0"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->,head_width=0.5,head_length=0.8",
            color=color,
            lw=lw,
            connectionstyle=connectionstyle,
        ),
    )


def label(ax, x, y, text, *, color="black", fs=8, ha="left"):
    ax.text(x, y, text, fontsize=fs, color=color, style="italic", ha=ha, va="center")


def main() -> None:
    with plt.xkcd(scale=0.5, length=80, randomness=2):
        fig, ax = plt.subplots(figsize=(15, 11), dpi=120)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

        # Title
        ax.text(
            50, 97, "Project Pipeline — From Red-Team to Two-Layer Defense",
            ha="center", fontsize=18, fontweight="bold",
        )

        # ── Stage A: Red-team (top, full width) ────────────────────
        box(ax, 50, 90, 60, 5,
            "STAGE A  ·  Red-team campaign  (src/redteam.py)\n"
            "30 hand-crafted seeds  ->  PAIR loop (gpt-4o-mini)  ->  "
            "38 rollouts  ->  data/attack_log.jsonl",
            fs=10)

        # Branch down to two lanes
        arrow(ax, 35, 87.5, 23, 80)
        label(ax, 26, 84, "prompts")

        arrow(ax, 65, 87.5, 77, 80)
        label(ax, 70, 84, "labels + bodies")

        # ── Left lane: Layer 1 (SFT -> GRPO) ────────────────────────
        box(ax, 23, 75, 32, 5,
            "STAGE B  ·  SFT warmup  (src/sft_warmup.py)\n"
            "1,140 samples  ·  QLoRA r=16  ·  loss -> 1.18",
            fill=LAYER1_BLUE, fs=9)

        arrow(ax, 23, 72.5, 23, 67)
        label(ax, 24, 70, "warmed adapter")

        box(ax, 23, 62, 32, 5,
            "STAGE C  ·  GRPO RL  (src/grpo_train.py)\n"
            "3 epochs · G=4 · rule-based reward\n"
            "reward 0.86 -> 1.57  ·  KL ≈ 2",
            fill=LAYER1_BLUE, fs=9)

        arrow(ax, 23, 59.5, 23, 54)
        label(ax, 24, 57, "LoRA adapter")

        box(ax, 23, 50, 32, 4,
            "adapters/qwen-injection-grpo  (73 MB)",
            fill=LAYER1_BLUE, fs=9)

        # ── Right lane: Layer 2 (ModernBERT) ───────────────────────
        box(ax, 77, 75, 32, 5,
            "STAGE D  ·  Classifier  (src/classifier.py)\n"
            "ModernBERT-base  ·  80/20 seed-grouped\n"
            "F1 = 1.0 held-out  (benign FP 20% in deploy)",
            fill=LAYER2_GREEN, fs=9)

        arrow(ax, 77, 72.5, 77, 54)
        label(ax, 78, 64, "trained weights")

        box(ax, 77, 50, 32, 4,
            "adapters/injection-classifier  (575 MB)",
            fill=LAYER2_GREEN, fs=9)

        # ── Stage E: Composition ───────────────────────────────────
        arrow(ax, 23, 47.5, 42, 38)
        label(ax, 26, 42, "Layer 1")

        arrow(ax, 77, 47.5, 58, 38)
        label(ax, 67, 42, "Layer 2")

        box(ax, 50, 34, 42, 6,
            "STAGE E  ·  Composition  (src/combined_guard.py)\n"
            "verifier = side-call to Qwen+LoRA on tool-call context\n"
            "classifier OR verifier blocks -> tool returns blocked",
            fs=9)

        arrow(ax, 50, 31, 50, 25)
        label(ax, 51, 28, "runtime guard")

        # ── Stage F: 4 corner-case eval ────────────────────────────
        box(ax, 50, 21, 52, 7,
            "STAGE F  ·  4 corner-case evaluation\n"
            "(eval_combined.py  +  eval_combined_loose.py)\n"
            "attack × {verifier-only, combined} × {strict, loose}  +  benign\n"
            "-> results/attack_*.json (×6)   results/benign_*.json (×6)",
            fill=EVAL_GREY, fs=9)

        arrow(ax, 50, 17.5, 50, 12)
        label(ax, 51, 15, "measurements")

        # ── Stage G: Outputs ───────────────────────────────────────
        box(ax, 50, 8, 60, 5,
            "STAGE G  ·  Outputs\n"
            "final_report.md (770 lines)   ·   Streamlit demo   ·   "
            "audit script (51/51 numeric claims pass)",
            fill=OUTPUT_TAN, fs=10)

        # ── Side annotation: why side-call ─────────────────────────
        ax.text(
            5, 32,
            "Why side-call,\nnot LoRA-as-agent:\n\n"
            "• Qwen 1.5B tool-calling\n  is fragile in ReAct\n\n"
            "• LoRA trained on a\n  different prompt shape\n\n"
            "• Side-call evaluates the\n  LoRA on its training\n  distribution",
            fontsize=8, color=TEXT_GREY, style="italic",
            verticalalignment="center",
        )

        fig.savefig(OUT_DIR / "pipeline.png", dpi=120, bbox_inches="tight",
                    facecolor="white")
        fig.savefig(OUT_DIR / "pipeline.svg", bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

        print(f"wrote {OUT_DIR / 'pipeline.png'}")


if __name__ == "__main__":
    main()
