#!/usr/bin/env python3
"""
Plot a pipeline funnel summary as a bar graph.

Reads a summary text file with lines like:
  input_pdb            232805 #grey
  s1_fasta             227952 #orange

and writes a single bar chart with the same counts.

Example:
  python3 20_plot_pipeline_funnel.py \
      --input data/input/pipeline_membership.txt \
      --output data/output/analysis/plots/pipeline_funnel_bar.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_INPUT = ROOT / "data" / "input" / "pipeline_membership.txt"
DEFAULT_OUTPUT = ROOT / "data" / "output" / "analysis" / "plots" / "pipeline_funnel_bar.png"

COLOR_MAP = {
    "grey": "dimgray",
    "gray": "dimgray",
    "orange": "darkorange",
    "blue": "steelblue",
}


def parse_summary(path: Path) -> tuple[list[str], list[int], list[str]]:
    if not path.is_file():
        raise SystemExit(f"Summary file not found: {path}")

    labels: list[str] = []
    counts: list[int] = []
    colors: list[str] = []

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("=") or line.endswith(":"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        if parts[-1].startswith("#"):
            color_name = parts[-1].lstrip("#").lower()
            count_token = parts[-2]
            label_tokens = parts[:-2]
        else:
            color_name = "gray"
            count_token = parts[-1]
            label_tokens = parts[:-1]

        try:
            count = int(count_token.replace(",", ""))
        except ValueError:
            continue

        label = " ".join(label_tokens).strip()
        labels.append(label)
        counts.append(count)
        colors.append(COLOR_MAP.get(color_name, "gray"))

    if not labels:
        raise SystemExit(f"No counts found in summary file: {path}")

    return labels, counts, colors


def plot_bar(labels: list[str], counts: list[int], colors: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))

    grouped_labels: list[str] = []
    grouped_counts: list[list[int]] = []
    grouped_colors: list[list[str]] = []
    grouped_sublabels: list[list[str]] = []

    def group_key(label: str) -> str:
        if label.startswith("Step 3"):
            return "Step 3"
        if label.startswith("Step 4 unique"):
            return "Step 4 unique"
        return label

    def display_label(label: str) -> str:
        if label.startswith("Step 3"):
            return "Step 3 -\nproteins with predictions"
        if label.startswith("Step 4 unique"):
            return "Step 4 unique"
        return label

    def sublabel(label: str) -> str:
        if label.startswith("Step 4 unique"):
            return label.removeprefix("Step 4 unique").strip()
        return ""

    for label, count, color in zip(labels, counts, colors):
        key = group_key(label)
        if grouped_labels and key == grouped_labels[-1]:
            grouped_counts[-1].append(count)
            grouped_colors[-1].append(color)
            grouped_sublabels[-1].append(sublabel(label))
        else:
            grouped_labels.append(key)
            grouped_counts.append([count])
            grouped_colors.append([color])
            grouped_sublabels.append([sublabel(label)])

    bar_width = 0.75 if max(len(group) for group in grouped_counts) == 1 else 0.32
    bar_positions: list[list[float]] = []

    for group_index, group_counts in enumerate(grouped_counts):
        n_bars = len(group_counts)
        if n_bars == 1:
            positions = [group_index]
        else:
            offsets = [
                (idx - (n_bars - 1) / 2) * bar_width
                for idx in range(n_bars)
            ]
            positions = [group_index + offset for offset in offsets]
        bar_positions.append(positions)
        for x_pos, count, color in zip(positions, group_counts, grouped_colors[group_index]):
            ax.bar(x_pos, count, width=bar_width, color=color, edgecolor="black", linewidth=0.8)

    ax.set_title("Counts of files in consequent steps", fontsize=14)
    ax.set_ylabel("Count", fontsize=12)
    #ax.set_xlabel("", fontsize=12)
    ax.set_xticks(range(len(grouped_labels)))
    xtick_labels = [display_label(label) for label in grouped_labels]
    ax.set_xticklabels(xtick_labels, rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(facecolor=COLOR_MAP["orange"], edgecolor="black", label="Seq2Pocket"),
        Patch(facecolor=COLOR_MAP["blue"], edgecolor="black", label="P2Rank"),
    ])

    ymax = max(counts)
    ax.set_ylim(0, ymax * 1.15 if ymax else 1)

    for positions, group_counts in zip(bar_positions, grouped_counts):
        for x_pos, count in zip(positions, group_counts):
            ax.text(
                x_pos,
                count,
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ymax = max(counts)
    for group_label, positions, sublabels in zip(grouped_labels, bar_positions, grouped_sublabels):
        if group_label == "Step 4 unique" and any(sublabels):
            label_y = ymax * 0.03
            for x_pos, label in zip(positions, sublabels):
                if not label:
                    continue
                ax.text(
                    x_pos,
                    label_y,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="black",
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to the pipeline funnel summary text file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to write the bar graph PNG")
    args = parser.parse_args()

    labels, counts, colors = parse_summary(args.input)
    plot_bar(labels, counts, colors, args.output)

    print(f"Read:  {args.input}")
    print(f"Wrote: {args.output}")
    print("Counts:")
    for label, count in zip(labels, counts):
        print(f"  {label:20s} {count:,}")


if __name__ == "__main__":
    main()