#!/usr/bin/env python3
"""Plot paired EXL3/GGUF measurements, including every repetition."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    definitions = [
        ("exl3", "EXL3 3-bit HQ", "#adff67"),
        ("iq2-base", "GGUF UD-IQ2_XXS", "#52d1e6"),
        ("iq2-mtp", "GGUF UD-IQ2_XXS + MTP", "#9ae6f1"),
        ("iq3-base", "GGUF UD-IQ3_XXS", "#a894f9"),
        ("iq3-mtp", "GGUF UD-IQ3_XXS + MTP", "#d0bfff"),
    ]
    runs = []
    for name, label, color in definitions:
        file = args.results / f"{name}.json"
        if file.exists():
            data = json.loads(file.read_text())
            if data.get("status") == "DONE":
                runs.append((label, color, data))
    if not runs:
        raise ValueError("No completed runs")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#eaf4ef", "axes.labelcolor": "#a2bbae",
                         "xtick.color": "#a2bbae", "ytick.color": "#eaf4ef"})
    fig, ax = plt.subplots(figsize=(12, 8.4), facecolor="#07110d")
    ax.set_facecolor("#07110d")
    fig.subplots_adjust(left=.18, right=.92, top=.76, bottom=.13)
    fig.text(.06, .95, "FIELD NOTES / RTX 5090 / NEMOTRON 3.5 SUPER 120B",
             color="#adff67", fontsize=10, fontfamily="DejaVu Sans Mono")
    fig.text(.06, .90, "Generation speed", fontsize=27, weight="bold")
    fig.text(.06, .855, "Same inputs · 128K context · 512 output tokens · thinking off",
             color="#a2bbae", fontsize=11)
    count = len(runs)
    spacing = count + 1.5
    peak = max(max(d["summary"][str(size)]["decode_tok_s"]["all"])
               for _, _, d in runs for size in (128, 8192, 32768))
    centers = []
    for group, size in enumerate((128, 8192, 32768)):
        start = group * spacing
        center = start + (count - 1) / 2
        centers.append(center)
        ax.axhspan(start-.65, start+count-.35, color="#10231a", alpha=.45, zorder=0)
        for index, (label, color, data) in enumerate(runs):
            y = start + index
            values = data["summary"][str(size)]["decode_tok_s"]
            median, samples = values["median"], values["all"]
            ax.barh(y, median, height=.60, color=color, alpha=.38,
                    label=label if group == 0 else None, zorder=2)
            ax.plot([min(samples), max(samples)], [y, y], color=color, lw=1.5, zorder=3)
            ax.scatter(samples, [y]*len(samples), color=color, edgecolor="#07110d",
                       linewidth=.8, s=25, zorder=4)
            ax.plot([median, median], [y-.34, y+.34], color=color, lw=2, zorder=5)
            ax.text(peak*1.17, y, f"{median:.2f}", va="center", ha="right", color=color,
                    fontsize=10, weight="bold")
    ax.set_yticks(centers, ["128 tokens", "8K tokens", "32K tokens"])
    ax.tick_params(axis="both", length=0, pad=10)
    ax.set_xlim(0, peak*1.20)
    ax.set_ylim(2*spacing+count-.25, -.9)
    ax.set_xlabel("output tokens / second · higher is better", labelpad=13)
    ax.grid(axis="x", color="#2b483a", linewidth=.6, alpha=.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-.17, 1.02), ncol=3,
              frameon=False, fontsize=9, handlelength=1.2, columnspacing=2)
    fig.text(.06, .047, "Bars: median. Dots: all 3 runs, centered on each bar.",
             color="#a2bbae", fontsize=10)
    fig.text(.06, .018, "EXL3 50.0 GB · UD-IQ2_XXS 51.1 GB · UD-IQ3_XXS 59.8 GB · Different quantizations",
             color="#a2bbae", fontsize=9)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(args.out.with_suffix(suffix), dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
