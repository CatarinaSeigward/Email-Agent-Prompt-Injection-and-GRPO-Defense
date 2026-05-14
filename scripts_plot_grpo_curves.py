"""Render GRPO training-curve figures from the captured stdout log.

Reads the hand-pasted training log embedded below (kept in source rather than
parsing live logs so this script is hermetic and re-runnable). Emits three
PNGs under results/.

Usage: uv run python scripts_plot_grpo_curves.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

# Each tuple: (epoch, loss, reward, reward_std, clipped_ratio, kl, mean_length).
# Sourced verbatim from the GRPO V6 training run captured 2026-05-13.
HISTORY = [
    (0.18, 0.0722,  0.8562, 0.6881, 0.0500, 1.5516, 43.4125),
    (0.35, 0.0522,  0.9950, 0.6356, 0.0500, 1.6442, 42.4750),
    (0.53, -0.0069, 0.9300, 0.6697, 0.0750, 1.5881, 38.9875),
    (0.70, 0.1114,  1.0012, 0.6497, 0.1000, 9.8669, 40.1250),
    (0.88, 0.0845,  1.1287, 0.6274, 0.0250, 1.7112, 36.5000),
    (1.07, 0.0636,  1.1150, 0.6841, 0.0375, 1.7197, 38.6250),
    (1.25, -0.0059, 1.3300, 0.4841, 0.0250, 1.8897, 34.3250),
    (1.42, -0.0288, 1.3950, 0.4995, 0.0250, 1.9711, 37.0625),
    (1.60, 0.0453,  1.4362, 0.4428, 0.0125, 2.2981, 33.4500),
    (1.77, 0.0171,  1.3962, 0.4812, 0.0250, 1.9070, 33.7375),
    (1.95, -0.0146, 1.4612, 0.4363, 0.0250, 1.9969, 34.7125),
    (2.14, 0.0396,  1.4387, 0.4949, 0.0125, 2.5197, 34.6750),
    (2.32, 0.0245,  1.4462, 0.4155, 0.0375, 2.6697, 36.4875),
    (2.49, 0.0213,  1.5800, 0.2745, 0.0125, 2.2143, 31.1500),
    (2.67, 0.0698,  1.4800, 0.3333, 0.0375, 1.7674, 41.6000),
    (2.84, 0.0385,  1.4875, 0.4080, 0.0250, 2.0402, 32.5250),
    (2.98, None,    1.5703, 0.2747, 0.0000, 1.9778, 37.9219),
]


def _plot(xs, ys_list, labels, ylabel, title, out_name, ylim=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ys, label, color in zip(ys_list, labels, colors):
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    if len(labels) > 1:
        ax.legend()
    plt.tight_layout()
    out = RESULTS / out_name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")
    plt.close(fig)


def main() -> None:
    epochs = [h[0] for h in HISTORY]
    rewards = [h[2] for h in HISTORY]
    reward_stds = [h[3] for h in HISTORY]
    clipped = [h[4] * 100 for h in HISTORY]  # to percent
    kls = [h[5] for h in HISTORY]
    lengths = [h[6] for h in HISTORY]

    _plot(
        epochs,
        [rewards, reward_stds],
        ["mean reward", "reward std"],
        "Reward / std",
        "GRPO V6 — mean reward rises, std falls (convergence)",
        "grpo_reward_curve.png",
    )
    _plot(
        epochs,
        [clipped],
        ["clipped_ratio (%)"],
        "% rollouts truncated at max_completion_length=96",
        "GRPO V6 — clipped ratio collapses from 5% to 0% (EOS learned)",
        "grpo_clipped_ratio.png",
        ylim=(-1, 12),
    )
    _plot(
        epochs,
        [kls],
        ["KL(policy ‖ SFT init)"],
        "KL divergence",
        "GRPO V6 — KL stays bounded around 2 (β=0.005)",
        "grpo_kl.png",
    )
    _plot(
        epochs,
        [lengths],
        ["mean completion length (tokens)"],
        "Tokens",
        "GRPO V6 — completion length stays short (no rambling drift)",
        "grpo_length.png",
    )


if __name__ == "__main__":
    main()
