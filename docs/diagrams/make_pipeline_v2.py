"""Generate the project pipeline diagram (v2, post-audit).

Revision of make_pipeline.py to match the audited report (2026-08-07).

Stale numbers corrected (all re-measured 2026-08-21):
  - adapters/qwen-injection-grpo   73 MB  ->  86 MB
  - adapters/injection-classifier 575 MB  -> 574 MB
  - final_report.md               770 ln  -> 926 ln
  - audit script                  51/51   -> 61/61 (audit_report_numbers.py)
                                             + 79/79 (audit_p0_numbers.py)
    "51/51" survives only as a comment inside audit_report_numbers.py:114,
    describing the pre-audit run that passed while carrying a wrong row.

Structural changes:
  - Title no longer ends at "Two-Layer Defense". §1 calls the audit "the most
    important part"; v1 stopped one stage short of it.
  - STAGE G (self-audit) added: 61/61 + 79/79, T8/T9/T12 closed, 2 of 4
    headline claims retracted.
  - DEFENSE 0 (hardened system prompt) added. It bypasses the whole training
    pipeline and outperforms it (§4.9) — drawn as a lane that skips A-E.
  - DPO added as a dashed dead-end branch. v1 omitted it, but §9.2 rates it a
    Strong claim and §7.8 makes it 1 of 3 Goodhart instances.
  - STAGE F now shows both evaluation waves. v1 stopped at the k=1 corner
    cases and omitted scripts/eval_p0.py (k=3), eval_grpo_attack.py (§4.8)
    and src/strict_scorer.py (§4.2, reconstructed).
  - Composition labelled in runtime order (classifier ~5 ms first,
    verifier ~1 s on pass) instead of the Layer 1 / Layer 2 numbering, which
    runs opposite to the data flow.
  - "Why side-call" side note dropped; it is prose in §4.6/§5 and cost the
    figure a column.

Outputs:
  docs/diagrams/pipeline_v2.png
  docs/diagrams/pipeline_v2.svg

Run: uv run python docs/diagrams/make_pipeline_v2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent

LAYER1_BLUE = "#D8E4F4"
LAYER2_GREEN = "#D8E8D2"
PROMPT_BLUE = "#D6E6F5"
EVAL_GREY = "#E0E0E8"
AUDIT_RED = "#F7DCDC"
OUTPUT_TAN = "#F4E6CC"
TEXT_GREY = "#444444"
DEAD_GREY = "#EDEDED"
ATTACK_RED = "#C03030"


def box(ax, cx, cy, w, h, text, *, edge="black", fill="white", fs=10,
        weight="normal", dashed=False, tcolor="black"):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.4,rounding_size=1",
        edgecolor=edge,
        facecolor=fill,
        linewidth=2.0,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=tcolor)


def arrow(ax, x1, y1, x2, y2, *, color="black", lw=2.0,
          connectionstyle="arc3,rad=0", dashed=False):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->,head_width=0.5,head_length=0.8",
            color=color,
            lw=lw,
            connectionstyle=connectionstyle,
            linestyle="--" if dashed else "-",
        ),
    )


def label(ax, x, y, text, *, color="black", fs=8, ha="left"):
    ax.text(x, y, text, fontsize=fs, color=color, style="italic", ha=ha, va="center")


def main() -> None:
    with plt.xkcd(scale=0.5, length=80, randomness=2):
        fig, ax = plt.subplots(figsize=(16, 11.5), dpi=120)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

        ax.text(
            50, 97, "Project Pipeline — From Red-Team to a Self-Audit",
            ha="center", fontsize=18, fontweight="bold",
        )

        # ── STAGE A ───────────────────────────────────────────────
        box(ax, 50, 91, 64, 5,
            "STAGE A  ·  Red-team campaign  (src/redteam.py)\n"
            "30 hand-crafted seeds  ->  PAIR loop (gpt-4o-mini)  ->  "
            "38 rollouts  ->  data/attack_log.jsonl",
            fs=10)

        arrow(ax, 38, 88.5, 30, 83.5)
        label(ax, 30.5, 86.5, "prompts")
        arrow(ax, 66, 88.5, 76, 83.5)
        label(ax, 68, 86.5, "labels + bodies")
        arrow(ax, 24, 88.5, 12, 83.5, dashed=True, lw=1.4)

        # ── dead-end branch: DPO ──────────────────────────────────
        box(ax, 11, 78, 20, 7,
            "DPO  ·  abandoned\n"
            "margins 6.18, loss 0.003\n"
            "generation unchanged\n(§3.2.5)",
            fill=DEAD_GREY, fs=8, dashed=True, tcolor=TEXT_GREY)

        # ── left lane: Layer 1 training ───────────────────────────
        box(ax, 38, 79, 32, 6,
            "STAGE B  ·  SFT warmup  (src/sft_warmup.py)\n"
            "114 prompts x 10 templates = 1,140 samples\n"
            "QLoRA r=16  ·  loss -> 1.18",
            fill=LAYER1_BLUE, fs=8.5)
        arrow(ax, 38, 76, 38, 71.5)
        label(ax, 39.5, 73.7, "warmed adapter")

        box(ax, 38, 67.5, 32, 6.5,
            "STAGE C  ·  GRPO RL  (src/grpo_train.py)\n"
            "3 epochs · G=4 · rule-based regex reward\n"
            "reward 0.86 -> 1.57 · reward_std -71% · KL ~ 2",
            fill=LAYER1_BLUE, fs=8.5)
        label(ax, 20.5, 62.5, "every training metric improved —\n"
                              "STAGE G shows what that hid",
              color=ATTACK_RED, fs=7.5, ha="right")

        arrow(ax, 38, 64.2, 38, 60)
        box(ax, 38, 56.5, 32, 6,
            "adapters/qwen-injection-{sft, grpo}\n"
            "86 MB each  —  §4.8 compares them:\n"
            "refusal  SFT 86.8%  vs  GRPO 50.0%",
            fill=LAYER1_BLUE, fs=8.5)

        # ── right lane: Layer 2 training ──────────────────────────
        box(ax, 78, 79, 30, 6,
            "STAGE D  ·  Classifier  (src/classifier.py)\n"
            "ModernBERT-base  ·  80/20 seed-grouped\n"
            "F1 = 1.0 held-out  (benign FP 20% in deploy)",
            fill=LAYER2_GREEN, fs=8.5)
        arrow(ax, 78, 76, 78, 60)
        label(ax, 79.5, 68, "trained weights")
        box(ax, 78, 56.5, 30, 4.5,
            "adapters/injection-classifier  (574 MB)",
            fill=LAYER2_GREEN, fs=8.5)

        # ── STAGE E: composition, in runtime order ────────────────
        arrow(ax, 40, 53.5, 48, 48)
        arrow(ax, 76, 54.2, 66, 48)

        box(ax, 57, 44, 44, 6.5,
            "STAGE E  ·  Composition  (src/combined_guard.py)\n"
            "classifier ~5 ms runs first and short-circuits;\n"
            "verifier side-call to Qwen+LoRA ~1 s only on pass",
            fs=8.5)

        # ── DEFENSE 0: skips the entire pipeline ──────────────────
        box(ax, 14, 42, 25, 8,
            "DEFENSE 0\nhardened system prompt\n(src/agent.py, 14 lines)\n"
            "no training · no GPU",
            fill=PROMPT_BLUE, fs=8.5)
        arrow(ax, 20, 37.5, 30, 33.5, connectionstyle="arc3,rad=-0.15")
        label(ax, 18.5, 33.5, "never trained,\nnever measured until T12",
              color=TEXT_GREY, fs=7.5, ha="center")

        arrow(ax, 57, 40.5, 57, 35.5)
        label(ax, 58.5, 38, "runtime guard")

        # ── STAGE F: both evaluation waves ────────────────────────
        box(ax, 52, 30, 62, 9,
            "STAGE F  ·  Evaluation\n"
            "wave 1  (k=1)  eval_combined.py + eval_combined_loose.py\n"
            "               attack x {verifier-only, combined} x {strict, loose} + benign\n"
            "wave 2  (k=3)  scripts/eval_p0.py {naive, hardened}   ·   "
            "eval_grpo_attack.py {SFT, GRPO}\n"
            "               + src/strict_scorer.py  (rebuilt — the original was never committed)",
            fill=EVAL_GREY, fs=8)

        arrow(ax, 52, 25.5, 52, 21)
        label(ax, 53.5, 23.3, "measurements")

        # ── STAGE G: the self-audit ───────────────────────────────
        box(ax, 50, 15.5, 74, 9.5,
            "STAGE G  ·  Self-audit  (p0_analysis.md)\n"
            "audit_report_numbers.py 86/86   ·   audit_p0_numbers.py 79/79   ·   "
            "T8 / T9 / T12 closed\n\n"
            "RETRACTED  agent-retry paradox (p=0.39)   ·   loose > strict verifier (p=0.50)\n"
            "HELD  GRPO reward hacking   ·   DPO structural failure        "
            "NEW  prompt-only 0.0% ASR",
            fill=AUDIT_RED, fs=8.5)

        arrow(ax, 50, 10.5, 50, 6.5)

        # ── STAGE H: outputs ──────────────────────────────────────
        box(ax, 50, 3.5, 64, 4.5,
            "STAGE H  ·  Outputs\n"
            "final_report.md (1165 lines)   ·   Streamlit demo   ·   "
            "165 numeric claims re-checked on every run",
            fill=OUTPUT_TAN, fs=9)

        fig.savefig(OUT_DIR / "pipeline_v2.png", dpi=120, bbox_inches="tight",
                    facecolor="white")
        fig.savefig(OUT_DIR / "pipeline_v2.svg", bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

        print(f"wrote {OUT_DIR / 'pipeline_v2.png'}")


if __name__ == "__main__":
    main()
