"""Figures for the BoN section, from results/bon_curves.json.

  results/bon_efficiency.png   (a) selection efficiency at n=32 per judge, sorted,
                                   95% CIs, between random (0) and oracle (1)
                               (b) static over-credit vs efficiency -- the
                                   calibration-vs-ranking picture (H-B, H-C)
  results/bon_curves.png       small multiples: held-out gold vs n, one panel
                               per judge, inside the random-to-oracle band

Form and colour follow the dataviz method: the story is one magnitude per
judge, so position encodes it and a single hue carries every mark -- seven
categorical hues would say nothing position does not. The curves are faceted
rather than overlaid (past ~4 converging series, small multiples). Palette:
one accent plus chrome greys, validated with the skill's validate_palette
(blue vs grey: CVD dE 15.9, normal-vision 17.8, both >= 3:1 on the surface; the
grey's chroma-floor FAIL is the point -- it is de-emphasis, not an identity).

Run: uv run python scripts/bon_plot.py [--curves PATH] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, BAND, ACCENT = "#e1e0d9", "#c3c2b7", "#f0efec", "#2a78d6"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": INK_2, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8, linestyle="-")   # solid hairline
    ax.set_axisbelow(True)


def efficiency_figure(c: dict, out: Path) -> None:
    judges = sorted(c["meta"]["judges"], key=lambda m: c["efficiency"][m]["32"])
    eff = [c["efficiency"][m]["32"] for m in judges]
    lo = [c["efficiency_ci95"][m]["32"][0] for m in judges]
    hi = [c["efficiency_ci95"][m]["32"][1] for m in judges]
    oc = [c["over_credit_n1"][m] for m in judges]

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) sorted dot plot with intervals, bracketed by random and oracle
    y = range(len(judges))
    for yi, l_, h_ in zip(y, lo, hi):
        a.plot([l_, h_], [yi, yi], color=ACCENT, linewidth=1.6, solid_capstyle="round")
    a.plot(eff, list(y), "o", color=ACCENT, markersize=7,
           markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for x_, label in ((0, "random"), (1, "oracle")):
        a.axvline(x_, color=INK_2, linewidth=0.9)
        # Above the plot area, so the reference line never runs through its label.
        a.text(x_, 1.01, label, transform=a.get_xaxis_transform(), ha="center",
               va="bottom", color=INK_2, fontsize=8, clip_on=False)
    # Direct labels on the extremes only; the table view carries the rest.
    for i in (0, len(judges) - 1):
        a.text(hi[i] + 0.03, i, f"{eff[i]:.2f}", va="center", color=INK, fontsize=8.5)
    a.set_yticks(list(y), judges)
    a.set_xlim(min(-0.1, min(lo) - 0.05), max(1.15, max(hi) + 0.12))
    a.set_ylim(-0.6, len(judges) - 0.1)
    a.set_xlabel("selection efficiency at n = 32  (95% CI)")
    a.set_title("How far each judge's pick moves you from random to oracle",
                loc="left", fontsize=9.5, pad=20)
    _grid(a, "x")

    # (b) static calibration vs selection efficiency
    b.plot(oc, eff, "o", color=ACCENT, markersize=7,
           markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for x_, y_, m in zip(oc, eff, judges):
        b.annotate(m, (x_, y_), xytext=(5, 4), textcoords="offset points",
                   color=INK_2, fontsize=7.5)
    b.margins(x=0.22, y=0.15)          # room for the longest judge name
    hb = c["H-B"]
    b.set_xlabel("over-credit at n = 1  (judge success rate − oracle)")
    b.set_ylabel("efficiency at n = 32")
    b.set_title(f"Calibration vs ranking   ρ = {hb['spearman_rho']:+.2f}, "
                f"p = {hb['p_one_sided_exact']:.3f}", loc="left", fontsize=9.5, pad=20)
    _grid(b, "both")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def curves_figure(c: dict, out: Path) -> None:
    judges = sorted(c["meta"]["judges"], key=lambda m: c["efficiency"][m]["32"])
    ns = c["meta"]["ns"]
    xs = [math.log2(n) for n in ns]
    floor = c["floor"]
    ceiling = [c["ceiling"][str(n)] for n in ns]
    top = max(max(ceiling), *(max(c["gold_by_judge"][m][str(n)] for n in ns) for m in judges))

    cols = 4
    rows = math.ceil(len(judges) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(9.6, 2.35 * rows), sharex=True, sharey=True)
    axes = [ax for row in (axes if rows > 1 else [axes]) for ax in row]
    for ax, m in zip(axes, judges):
        gold = [c["gold_by_judge"][m][str(n)] for n in ns]
        ax.fill_between(xs, floor, ceiling, color=BAND, linewidth=0)        # achievable
        ax.axhline(floor, color=MUTED, linewidth=0.9)                      # random
        ax.plot(xs, ceiling, color=INK_2, linewidth=1.1, linestyle=(0, (3, 2)))  # oracle
        ax.plot(xs, gold, color=ACCENT, linewidth=1.8, marker="o", markersize=4.5,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
        ax.set_title(f"{m}   eff {c['efficiency'][m]['32']:.2f}", loc="left", fontsize=8.5)
        ax.set_xticks(xs, [str(n) for n in ns])
        ax.set_ylim(0, top * 1.12)
        _grid(ax, "y")
    for ax in axes[len(judges):]:
        ax.axis("off")
    # A key in the spare panel instead of a legend repeated seven times.
    if len(axes) > len(judges):
        k = axes[len(judges)]
        k.plot([], [], color=ACCENT, linewidth=1.8, marker="o", markersize=4.5, label="judge's pick")
        k.plot([], [], color=INK_2, linewidth=1.1, linestyle=(0, (3, 2)), label="oracle's pick")
        k.plot([], [], color=MUTED, linewidth=0.9, label="random pick")
        k.legend(loc="center left", frameon=False, fontsize=8, labelcolor=INK_2)
    fig.supxlabel("best-of-n  (n attacks drawn, the one the selector scores highest is kept)",
                  fontsize=8.5, color=INK_2)
    fig.supylabel("held-out gold", fontsize=8.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", type=Path, default=ROOT / "results" / "bon_curves.json")
    ap.add_argument("--outdir", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    c = json.loads(args.curves.read_text(encoding="utf-8"))
    efficiency_figure(c, args.outdir / "bon_efficiency.png")
    curves_figure(c, args.outdir / "bon_curves.png")
    print(f"wrote {args.outdir / 'bon_efficiency.png'} and {args.outdir / 'bon_curves.png'}")


if __name__ == "__main__":
    main()
