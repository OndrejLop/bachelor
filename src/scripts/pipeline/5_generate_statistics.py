"""
Generate statistics and plots from pipeline results.

This script analyzes outputs from previous pipeline steps:
1. Pipeline funnel: data attrition across steps (requires manual input)
2. Per-method pocket statistics: pocket count, size, and score distributions
4. Novel pocket counts per method
6. Summary table

Input:
  - P2Rank predictions:    data/input/P2Rank/{pdb_id}_predictions.csv
  - Seq2Pocket predictions: data/output/Seq2Pockets/{pdb_id}_predictions.csv
  - Comparison results:    data/output/results/novel_s2p_pockets.csv
                           data/output/results/p2r_unique_pockets.csv
  - Clustering skip log:   data/output/CS_predictions/skipped_clustering.txt (optional)

Output:
  - data/output/analysis/summary.txt
  - data/output/analysis/plots/*.png
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'axes.axisbelow': True,
})

S2P_COLOR = 'darkorange'
P2R_COLOR = 'steelblue'

def _method_color(name):
    n = name.lower()
    if 's2p' in n or 'seq2pocket' in n:
        return S2P_COLOR
    if 'p2r' in n or 'p2rank' in n:
        return P2R_COLOR
    return 'gray'

ROOT = Path(__file__).parent.parent.parent.parent


parser = argparse.ArgumentParser(description="Generate statistics and plots from pipeline results")
parser.add_argument("--results-dir", type=Path, default=None,
                    help="Path to step-4 output dir containing novel_s2p_pockets.csv / p2r_unique_pockets.csv "
                         "(default: data/output/results/)")
parser.add_argument("--s2p-dir", type=Path, default=None,
                    help="Path to Seq2Pocket predictions (default: data/output/Seq2Pockets/)")
parser.add_argument("--timestamp", action="store_true",
                    help="Write analysis output into a timestamped subdir of data/output/analysis/")
parser.add_argument("--exclude-file", type=Path, default=None,
                    help="Path to text file listing pdb_ids to exclude (one per line, # comments). "
                         "Default: none (no exclusions). "
                         "Recommended location when used: data/output/analysis/excluded_pdbs.txt")
args = parser.parse_args()

P2RANK_DIR  = ROOT / 'data' / 'input' / 'P2Rank'
S2P_DIR     = args.s2p_dir if args.s2p_dir else ROOT / 'data' / 'output' / 'Seq2Pockets'
CS_DIR      = ROOT / 'data' / 'output' / 'CS_predictions'  # legacy, used only for skipped_clustering.txt lookup
RESULTS_DIR = args.results_dir if args.results_dir else ROOT / 'data' / 'output' / 'results'
PDB_DIR     = ROOT / 'data' / 'input' / 'pdb'
FASTA_DIR   = ROOT / 'data' / 'intermediate' / 'fastas'
PRED_DIR    = ROOT / 'data' / 'intermediate' / 'predictions'

analysis_base = ROOT / 'data' / 'output' / 'analysis'
if args.timestamp:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATS_DIR = analysis_base / timestamp
else:
    STATS_DIR = analysis_base
PLOTS_DIR = STATS_DIR / 'plots'

STATS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_exclusions(path):
    if path is None or not path.exists():
        return set()
    excluded = set()
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line:
                excluded.add(line)
    return excluded


exclude_path = args.exclude_file
if exclude_path is not None and not exclude_path.is_absolute():
    exclude_path = ROOT / exclude_path
EXCLUDED_PDBS = _load_exclusions(exclude_path)

print(f"\n{'='*60}")
print(f"Statistics Generation Parameters:")
print(f"{'='*60}")
print(f"Seq2Pocket dir:  {S2P_DIR}")
print(f"P2Rank dir:      {P2RANK_DIR}")
print(f"Results dir:     {RESULTS_DIR}")
print(f"Output dir:      {STATS_DIR}")
print(f"Exclude file:    {exclude_path if exclude_path else '(none)'}")
print(f"Excluded PDBs:   {len(EXCLUDED_PDBS)}")
print(f"{'='*60}\n")

# ============================================================
# Helper functions
# ============================================================

def _stem_to_pdb_id(stem, strip_suffix=""):
    if strip_suffix and stem.endswith(strip_suffix):
        return stem[: -len(strip_suffix)]
    return stem


def count_files(directory, pattern, strip_suffix=""):
    """Count files matching a glob pattern in a directory, excluding EXCLUDED_PDBS."""
    if not directory.exists():
        return 0
    return sum(
        1 for p in directory.glob(pattern)
        if _stem_to_pdb_id(p.stem, strip_suffix) not in EXCLUDED_PDBS
    )

def load_pocket_csv(csv_path):
    """Load a pocket predictions CSV and parse residue lists."""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["residue_list"] = df["residue_ids"].apply(
        lambda x: x.strip().split() if pd.notna(x) else []
    )
    df["pocket_size"] = df["residue_list"].apply(len)
    return df

def collect_pocket_stats(predictions_dir, file_pattern):
    """
    Collect pocket statistics from all prediction CSV files in a directory.

    Returns:
        pockets_per_protein: list of pocket counts per protein
        pocket_sizes: list of all pocket sizes (residue count)
        pocket_scores: list of all pocket scores
        pocket_records: list of dicts with per-pocket metadata
                        (pdb_id, name, rank, size, score) for outlier analysis
    """
    pockets_per_protein = []
    pocket_sizes = []
    pocket_scores = []
    pocket_records = []

    for csv_path in sorted(predictions_dir.glob(file_pattern)):
        pdb_id = csv_path.stem.replace("_predictions", "")
        if pdb_id in EXCLUDED_PDBS:
            continue
        try:
            df = load_pocket_csv(csv_path)
        except Exception:
            continue
        pockets_per_protein.append(len(df))
        pocket_sizes.extend(df["pocket_size"].tolist())
        if "score" in df.columns:
            pocket_scores.extend(df["score"].astype(float).tolist())
        for _, r in df.iterrows():
            pocket_records.append({
                "pdb_id": pdb_id,
                "name": r["name"] if "name" in df.columns else "",
                "rank": int(r["rank"]) if "rank" in df.columns and pd.notna(r["rank"]) else -1,
                "size": int(r["pocket_size"]),
                "score": float(r["score"]) if "score" in df.columns and pd.notna(r["score"]) else float("nan"),
            })

    return pockets_per_protein, pocket_sizes, pocket_scores, pocket_records


def _needs_log2(data, ratio_threshold=20):
    """Return True if data range warrants log2 scale."""
    arr = np.array(data, dtype=float)
    arr = arr[arr > 0]
    if len(arr) < 10:
        return False
    return arr.max() / np.median(arr) > ratio_threshold


def _violin_box(ax, datasets, labels, colors, ylabel, use_log2=False):
    """
    Draw violin plot with overlaid compact boxplot.

    When use_log2=True, data is log2-transformed and y-tick labels show
    original-scale values (powers of 2).
    """
    if use_log2:
        plot_data = [np.log2(np.clip(d, 1, None).astype(float)) for d in datasets]
    else:
        plot_data = [np.array(d, dtype=float) for d in datasets]

    positions = list(range(1, len(datasets) + 1))

    parts = ax.violinplot(plot_data, positions=positions,
                          showmedians=False, showextrema=False)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)

    bp = ax.boxplot(plot_data, positions=positions, widths=0.18,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker='.', markersize=3, alpha=0.4))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=11)

    if use_log2:
        ax.set_ylabel(f"{ylabel} (log\u2082 scale)", fontsize=12)
        ymin, ymax = ax.get_ylim()
        ticks = np.arange(int(np.floor(ymin)), int(np.ceil(ymax)) + 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(2 ** int(t)) for t in ticks])
    else:
        ax.set_ylabel(ylabel, fontsize=12)


# ============================================================
# 1. Pipeline funnel
# ============================================================

def _read_novel_csv(filename):
    """Load a novel-pocket CSV and drop excluded pdb_ids."""
    csv_path = RESULTS_DIR / filename
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if EXCLUDED_PDBS and "pdb_id" in df.columns:
        df = df[~df["pdb_id"].astype(str).isin(EXCLUDED_PDBS)]
    return df


def _count_novel_proteins(filename):
    df = _read_novel_csv(filename)
    return 0 if df is None else len(df)


def pipeline_funnel(out):
    out.write("=" * 60 + "\n")
    out.write("1. PIPELINE FUNNEL\n")
    out.write("=" * 60 + "\n\n")

    n_pdb = count_files(PDB_DIR, "*.pdb")
    n_p2r_pockets = count_files(P2RANK_DIR, "*_predictions.csv", strip_suffix="_predictions")
    n_s2p_pockets = count_files(S2P_DIR, "*_predictions.csv", strip_suffix="_predictions")
    n_s2p_unique = _count_novel_proteins("novel_s2p_pockets.csv")
    n_p2r_unique = _count_novel_proteins("p2r_unique_pockets.csv")

    out.write(f"  Inputs:\n")
    out.write(f"    PDB files:                     {n_pdb}\n")
    out.write(f"    P2Rank predictions:            {n_p2r_pockets}\n")
    out.write(f"  Seq2Pocket processed:\n")
    out.write(f"    Structures with S2P pockets:   {n_s2p_pockets}\n")
    out.write(f"  Proteins with unique pockets:\n")
    out.write(f"    S2P-unique:                    {n_s2p_unique}\n")
    out.write(f"    P2R-unique:                    {n_p2r_unique}\n")

    # Try to read skipped_clustering.txt for attrition details
    skip_file = CS_DIR / "skipped_clustering.txt"
    if skip_file.exists():
        out.write(f"\n  Clustering skip breakdown ({skip_file.name}):\n")
        with open(skip_file) as f:
            for line in f:
                out.write(f"    {line.rstrip()}\n")

    out.write("\n")

    return {
        "n_pdb": n_pdb,
        "n_s2p_pockets": n_s2p_pockets,
        "n_p2r_pockets": n_p2r_pockets,
        "n_s2p_unique": n_s2p_unique,
        "n_p2r_unique": n_p2r_unique,
    }

def plot_funnel(funnel, out_path):
    stage_labels = ["Inputs", "Seq2Pocket\nprocessed", "Proteins with\nunique pockets"]
    # Per stage: list of (label, count, color)
    stage_bars = [
        [("PDB", funnel["n_pdb"], 'dimgray'),
         ("P2Rank", funnel["n_p2r_pockets"], P2R_COLOR)],
        [("S2P", funnel["n_s2p_pockets"], S2P_COLOR)],
        [("S2P-unique", funnel["n_s2p_unique"], S2P_COLOR),
         ("P2R-unique", funnel["n_p2r_unique"], P2R_COLOR)],
    ]

    if sum(c for group in stage_bars for _, c, _ in group) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.35
    xticks = []
    for stage_idx, group in enumerate(stage_bars):
        n = len(group)
        offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * bar_width
        for (label, count, color), off in zip(group, offsets):
            x = stage_idx + off
            ax.bar(x, count, width=bar_width, color=color,
                   edgecolor='black', label=label)
            ax.text(x, count, f'{count:,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        xticks.append(stage_idx)

    ax.set_xticks(xticks)
    ax.set_xticklabels(stage_labels, fontsize=11)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Pipeline Funnel: Data Attrition Across Steps", fontsize=14)
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ============================================================
# 2. Per-method pocket statistics
# ============================================================

def per_method_stats(out):
    out.write("=" * 60 + "\n")
    out.write("2. PER-METHOD POCKET STATISTICS\n")
    out.write("=" * 60 + "\n\n")

    results = {}
    for method, directory, pattern in [
        ("Seq2Pocket", S2P_DIR, "*_predictions.csv"),
        ("P2Rank", P2RANK_DIR, "*_predictions.csv"),
    ]:
        pockets_per_protein, pocket_sizes, pocket_scores, pocket_records = collect_pocket_stats(directory, pattern)

        if not pockets_per_protein:
            out.write(f"  {method}: no data found\n\n")
            results[method] = None
            continue

        out.write(f"  --- {method} ---\n")
        out.write(f"  Proteins analyzed:       {len(pockets_per_protein)}\n")
        out.write(f"  Total pockets:           {sum(pockets_per_protein)}\n")
        out.write(f"  Pockets per protein:\n")
        out.write(f"    Mean:   {np.mean(pockets_per_protein):.2f}\n")
        out.write(f"    Median: {np.median(pockets_per_protein):.0f}\n")
        out.write(f"    Min:    {np.min(pockets_per_protein)}\n")
        out.write(f"    Max:    {np.max(pockets_per_protein)}\n")
        out.write(f"  Pocket size (residues):\n")
        out.write(f"    Mean:   {np.mean(pocket_sizes):.2f}\n")
        out.write(f"    Median: {np.median(pocket_sizes):.0f}\n")
        out.write(f"    Min:    {np.min(pocket_sizes)}\n")
        out.write(f"    Max:    {np.max(pocket_sizes)}\n")
        if pocket_scores:
            out.write(f"  Pocket score:\n")
            out.write(f"    Mean:   {np.mean(pocket_scores):.4f}\n")
            out.write(f"    Median: {np.median(pocket_scores):.4f}\n")
            out.write(f"    Min:    {np.min(pocket_scores):.4f}\n")
            out.write(f"    Max:    {np.max(pocket_scores):.4f}\n")
        out.write("\n")

        results[method] = {
            "pockets_per_protein": pockets_per_protein,
            "pocket_sizes": pocket_sizes,
            "pocket_scores": pocket_scores,
            "pocket_records": pocket_records,
        }

    return results

def plot_pocket_distributions(results, out_dir, novel_stats=None):
    novel_stats = novel_stats or {}
    for method, data in results.items():
        if data is None:
            continue

        label = method.lower().replace(" ", "_")
        color = _method_color(method)

        # Pockets per protein histogram
        fig, ax = plt.subplots(figsize=(8, 5))
        max_val = max(data["pockets_per_protein"])
        bins = range(0, max_val + 2)
        ax.hist(data["pockets_per_protein"], bins=bins, color=color,
                edgecolor='black', alpha=0.8, align='left')
        ax.set_xlabel("Number of Pockets", fontsize=12)
        ax.set_ylabel("Number of Proteins", fontsize=12)
        ax.set_title(f"{method}: Pockets per Protein", fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir / f"{label}_pockets_per_protein.png", dpi=150)
        plt.close(fig)

        # Pocket size violin plot: all pockets + unique (if available)
        novel_key = f"{method}-unique"
        novel_sizes = (novel_stats.get(novel_key) or {}).get("sizes", [])
        datasets = [data["pocket_sizes"]]
        vlabels = ["All"]
        colors = [color]
        combined = list(data["pocket_sizes"])
        if novel_sizes:
            datasets.append(novel_sizes)
            vlabels.append("Unique")
            colors.append(color)
            combined.extend(novel_sizes)

        fig, ax = plt.subplots(figsize=(8, 5))
        log2 = _needs_log2(combined)
        _violin_box(ax, datasets, vlabels, colors,
                    "Pocket Size (residues)", use_log2=log2)
        ax.set_title(f"{method}: Pocket Size Distribution", fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir / f"{label}_pocket_sizes.png", dpi=150)
        plt.close(fig)

    # Combined comparison: pockets per protein
    valid = {m: d for m, d in results.items() if d is not None}
    if len(valid) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        for ax, (method, data) in zip(axes, valid.items()):
            max_val = max(data["pockets_per_protein"])
            bins = range(0, max_val + 2)
            ax.hist(data["pockets_per_protein"], bins=bins, color=_method_color(method),
                    edgecolor='black', alpha=0.8, align='left')
            ax.set_xlabel("Number of Pockets", fontsize=12)
            ax.set_title(method, fontsize=13)
            ax.set_yscale('log', base=2)
        axes[0].set_ylabel("Number of Proteins (log\u2082 scale)", fontsize=12)
        fig.suptitle("Pockets per Protein: Method Comparison", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_dir / "comparison_pockets_per_protein.png", dpi=150)
        plt.close(fig)

        # Combined: pocket size violin plot (log2 scale when high variance)
        fig, ax = plt.subplots(figsize=(8, 5))
        all_sizes = [s for d in valid.values() for s in d["pocket_sizes"]]
        log2 = _needs_log2(all_sizes)
        _violin_box(ax,
                    [d["pocket_sizes"] for d in valid.values()],
                    list(valid.keys()),
                    [_method_color(m) for m in valid.keys()],
                    "Pocket Size (residues)", use_log2=log2)
        ax.set_title("Pocket Size Comparison", fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir / "comparison_pocket_sizes.png", dpi=150)
        plt.close(fig)

# ============================================================
# 3. Pocket size outliers
# ============================================================

TOP_N_OUTLIERS = 5

def pocket_outliers(method_stats, out, out_dir):
    out.write("=" * 60 + "\n")
    out.write(f"3. POCKET SIZE OUTLIERS (top {TOP_N_OUTLIERS} by residue count)\n")
    out.write("=" * 60 + "\n\n")

    rows = []
    for method, data in method_stats.items():
        if not data or not data.get("pocket_records"):
            out.write(f"  {method}: no records\n\n")
            continue

        top = sorted(data["pocket_records"], key=lambda r: r["size"], reverse=True)[:TOP_N_OUTLIERS]
        out.write(f"  --- {method} ---\n")
        out.write(f"  {'pdb_id':<12} {'name':<10} {'rank':>5} {'size':>6} {'score':>10}\n")
        for r in top:
            score = f"{r['score']:.4f}" if not np.isnan(r['score']) else "N/A"
            out.write(f"  {r['pdb_id']:<12} {str(r['name']):<10} {r['rank']:>5} {r['size']:>6} {score:>10}\n")
            rows.append({"method": method, **r})
        out.write("\n")

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "pocket_outliers.csv", index=False)


def zero_size_pockets(method_stats, out, out_dir):
    out.write("=" * 60 + "\n")
    out.write("3b. ZERO-SIZE POCKETS (size == 0 residues)\n")
    out.write("=" * 60 + "\n\n")

    rows = []
    for method, data in method_stats.items():
        if not data or not data.get("pocket_records"):
            continue

        zeros = [r for r in data["pocket_records"] if r["size"] == 0]
        out.write(f"  --- {method} ---\n")
        out.write(f"  Zero-size pockets: {len(zeros)}\n")
        if zeros:
            proteins = sorted({r['pdb_id'] for r in zeros})
            out.write(f"  Affected proteins: {len(proteins)}\n")
            out.write(f"  {'pdb_id':<12} {'name':<10} {'rank':>5} {'score':>10}\n")
            for r in zeros:
                score = f"{r['score']:.4f}" if not np.isnan(r['score']) else "N/A"
                out.write(f"  {r['pdb_id']:<12} {str(r['name']):<10} {r['rank']:>5} {score:>10}\n")
                rows.append({"method": method, **r})
        out.write("\n")

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "zero_size_pockets.csv", index=False)


# ============================================================
# 4. Novel pocket counts
# ============================================================

def novel_pocket_stats(out):
    out.write("=" * 60 + "\n")
    out.write("4. NOVEL POCKET COUNTS\n")
    out.write("=" * 60 + "\n\n")

    results = {}
    for label, filename in [
        ("Seq2Pocket-unique", "novel_s2p_pockets.csv"),
        ("P2Rank-unique", "novel_p2r_pockets.csv"),
    ]:
        df = _read_novel_csv(filename)
        if df is None:
            alt = filename.replace("novel_", "").replace("pockets", "unique_pockets")
            df = _read_novel_csv(alt)
        if df is None:
            out.write(f"  {label}: file not found ({filename})\n")
            results[label] = None
            continue

        n_proteins = len(df)
        # Each row has space-separated pocket numbers in 'pockets' column
        total_pockets = sum(len(str(row).split()) for row in df["pockets"])
        sizes_flat = []
        for row in df["sizes"]:
            sizes_flat.extend([int(x) for x in str(row).split()])

        out.write(f"  --- {label} ---\n")
        out.write(f"  Proteins with novel pockets: {n_proteins}\n")
        out.write(f"  Total novel pockets:         {total_pockets}\n")
        if sizes_flat:
            out.write(f"  Novel pocket size (residues):\n")
            out.write(f"    Mean:   {np.mean(sizes_flat):.2f}\n")
            out.write(f"    Median: {np.median(sizes_flat):.0f}\n")
            out.write(f"    Min:    {np.min(sizes_flat)}\n")
            out.write(f"    Max:    {np.max(sizes_flat)}\n")
        out.write("\n")

        results[label] = {
            "n_proteins": n_proteins,
            "total_pockets": total_pockets,
            "sizes": sizes_flat,
        }

    return results

def plot_novel_pockets(results, out_dir):
    # Bar chart: total novel pockets per method
    valid = {m: d for m, d in results.items() if d is not None}
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = list(valid.keys())
    counts = [d["total_pockets"] for d in valid.values()]
    colors = [_method_color(m) for m in methods]
    bars = ax.bar(methods, counts, color=colors, edgecolor='black')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of Novel Pockets", fontsize=12)
    ax.set_title("Novel Pockets by Method", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "novel_pockets_count.png", dpi=150)
    plt.close(fig)

    # Novel pocket size distributions
    has_sizes = {m: d for m, d in valid.items() if d["sizes"]}
    if has_sizes:
        fig, ax = plt.subplots(figsize=(8, 5))
        all_novel = [s for d in has_sizes.values() for s in d["sizes"]]
        log2 = _needs_log2(all_novel)
        _violin_box(ax,
                    [d["sizes"] for d in has_sizes.values()],
                    list(has_sizes.keys()),
                    [_method_color(m) for m in has_sizes.keys()],
                    "Pocket Size (residues)", use_log2=log2)
        ax.set_title("Novel Pocket Size Distribution", fontsize=14)
        plt.tight_layout()
        fig.savefig(out_dir / "novel_pocket_sizes.png", dpi=150)
        plt.close(fig)

# ============================================================
# 6. Summary table
# ============================================================

def summary_table(funnel, method_stats, novel_stats, out):
    out.write("=" * 60 + "\n")
    out.write("6. SUMMARY TABLE\n")
    out.write("=" * 60 + "\n\n")

    header = f"{'Metric':<40} {'Seq2Pocket':>12} {'P2Rank':>12}"
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")

    def row(label, s2p_val, p2r_val):
        s2p_str = str(s2p_val) if s2p_val is not None else "N/A"
        p2r_str = str(p2r_val) if p2r_val is not None else "N/A"
        out.write(f"{label:<40} {s2p_str:>12} {p2r_str:>12}\n")

    s2p = method_stats.get("Seq2Pocket")
    p2r = method_stats.get("P2Rank")

    row("Proteins analyzed",
        len(s2p["pockets_per_protein"]) if s2p else None,
        len(p2r["pockets_per_protein"]) if p2r else None)
    row("Total pockets",
        sum(s2p["pockets_per_protein"]) if s2p else None,
        sum(p2r["pockets_per_protein"]) if p2r else None)
    row("Mean pockets/protein",
        f"{np.mean(s2p['pockets_per_protein']):.2f}" if s2p else None,
        f"{np.mean(p2r['pockets_per_protein']):.2f}" if p2r else None)
    row("Median pocket size (residues)",
        f"{np.median(s2p['pocket_sizes']):.0f}" if s2p else None,
        f"{np.median(p2r['pocket_sizes']):.0f}" if p2r else None)
    row("Mean pocket size (residues)",
        f"{np.mean(s2p['pocket_sizes']):.2f}" if s2p else None,
        f"{np.mean(p2r['pocket_sizes']):.2f}" if p2r else None)

    s2p_novel = novel_stats.get("Seq2Pocket-unique")
    p2r_novel = novel_stats.get("P2Rank-unique")
    row("Novel pockets (total)",
        s2p_novel["total_pockets"] if s2p_novel else None,
        p2r_novel["total_pockets"] if p2r_novel else None)
    row("Proteins with novel pockets",
        s2p_novel["n_proteins"] if s2p_novel else None,
        p2r_novel["n_proteins"] if p2r_novel else None)

    out.write("\n")

# ============================================================
# 5. Length-based distributions & novel ratio
# ============================================================

def load_protein_lengths(residues_dir):
    """Return {pdb_id: seq_length} from S2P _residues.csv row counts."""
    lengths = {}
    files = list(residues_dir.glob("*_residues.csv"))
    print(f"  Scanning {len(files):,} residue files for sequence lengths...")
    for p in files:
        pdb_id = p.stem.replace("_residues", "")
        if pdb_id in EXCLUDED_PDBS:
            continue
        try:
            with open(p, 'rb') as f:
                n = sum(1 for _ in f) - 1
        except Exception:
            continue
        if n > 0:
            lengths[pdb_id] = n
    return lengths


STANDARD_AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
               "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
_STANDARD_AA_SET = set(STANDARD_AA)


def collect_aa_composition(directory, res_col, pocket_col):
    """Count AA types for residues assigned to any pocket (>0) across all files."""
    counts = Counter()
    if not directory.exists():
        return counts
    files = list(directory.glob("*_residues.csv"))
    print(f"  Scanning {len(files):,} residue files in {directory.name} for AA composition...")
    for p in files:
        pdb_id = p.stem.replace("_residues", "")
        if pdb_id in EXCLUDED_PDBS:
            continue
        try:
            df = pd.read_csv(p, skipinitialspace=True)
        except Exception:
            continue
        if res_col not in df.columns or pocket_col not in df.columns:
            continue
        sel = df[pd.to_numeric(df[pocket_col], errors="coerce").fillna(0) > 0]
        for aa in sel[res_col].astype(str).str.strip().str.upper():
            if aa in _STANDARD_AA_SET:
                counts[aa] += 1
    return counts


def plot_aa_composition(aa_counts_by_method, out_path):
    """Grouped bar chart: AA fraction of binding residues, per method."""
    methods = [m for m, c in aa_counts_by_method.items() if c and sum(c.values()) > 0]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(STANDARD_AA))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        counts = aa_counts_by_method[m]
        total = sum(counts.values())
        fracs = [100 * counts.get(aa, 0) / total for aa in STANDARD_AA]
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, fracs, width, color=_method_color(m),
               edgecolor='black', label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(STANDARD_AA, rotation=0, fontsize=10)
    ax.set_xlabel("Amino Acid", fontsize=12)
    ax.set_ylabel("% of Binding Residues", fontsize=12)
    ax.set_title("Amino Acid Composition of Binding Residues", fontsize=14)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_skip_log(skip_file):
    """Parse skipped_clustering.txt into {label: count}."""
    if not skip_file.exists():
        return {}
    counts = {}
    with open(skip_file) as f:
        for line in f:
            if ':' not in line:
                continue
            label, _, val = line.partition(':')
            try:
                counts[label.strip()] = int(val.strip().split()[0])
            except (ValueError, IndexError):
                continue
    return counts


def plot_skip_breakdown(skip_counts, out_path):
    """Horizontal stacked bar: single bar with per-reason segments."""
    if not skip_counts or sum(skip_counts.values()) == 0:
        return
    order = [k for k in [
        "Processed successfully",
        "Skipped (no binding res)",
        "Skipped (no PDB file)",
        "Skipped (no CA atoms)",
        "Skipped (residue mismatch)",
        "Skipped (no surface points)",
        "Skipped (error)",
    ] if k in skip_counts and skip_counts[k] > 0]
    if not order:
        return

    palette = ['#2ca02c', '#d62728', '#7f7f7f', '#9467bd', '#8c564b', '#e377c2', '#ff7f0e']
    fig, ax = plt.subplots(figsize=(10, 2.5))
    left = 0
    total = sum(skip_counts[k] for k in order)
    for k, color in zip(order, palette):
        v = skip_counts[k]
        ax.barh(0, v, left=left, color=color, edgecolor='black',
                label=f"{k} ({v:,}; {100*v/total:.1f}%)")
        if v / total > 0.03:
            ax.text(left + v / 2, 0, f"{v:,}", ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        left += v
    ax.set_yticks([])
    ax.set_xlabel("Number of Proteins", fontsize=12)
    ax.set_title("Clustering Pipeline Outcomes (Step 3)", fontsize=14)
    ax.set_xlim(0, total)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
              ncol=2, fontsize=9, frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def threshold_sweep(directories, thresholds):
    """Scan _residues.csv files per method, count binding residues at each threshold.

    directories: dict {method_name: (Path, prob_col)}
    Returns {method: {"thresholds": [...], "total_binding": [...], "mean_per_protein": [...]}}
    """
    result = {}
    for method, (directory, prob_col) in directories.items():
        if not directory.exists():
            result[method] = None
            continue
        files = list(directory.glob("*_residues.csv"))
        print(f"  Threshold sweep: scanning {len(files):,} files in {directory.name}...")
        totals = np.zeros(len(thresholds), dtype=np.int64)
        per_protein_counts = [[] for _ in thresholds]
        for p in files:
            pdb_id = p.stem.replace("_residues", "")
            if pdb_id in EXCLUDED_PDBS:
                continue
            try:
                df = pd.read_csv(p, skipinitialspace=True, usecols=lambda c: c.strip() == prob_col)
            except Exception:
                continue
            if prob_col not in df.columns or df.empty:
                continue
            probs = pd.to_numeric(df[prob_col], errors="coerce").dropna().to_numpy()
            for i, t in enumerate(thresholds):
                c = int((probs >= t).sum())
                totals[i] += c
                per_protein_counts[i].append(c)
        result[method] = {
            "thresholds": list(thresholds),
            "total_binding": totals.tolist(),
            "mean_per_protein": [float(np.mean(x)) if x else 0.0 for x in per_protein_counts],
        }
    return result


def plot_threshold_sweep(sweep, out_path):
    valid = {m: d for m, d in sweep.items() if d is not None}
    if not valid:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for method, d in valid.items():
        color = _method_color(method)
        ax1.plot(d["thresholds"], d["total_binding"], color=color,
                 label=method, lw=2, marker='o', markersize=4)
        ax2.plot(d["thresholds"], d["mean_per_protein"], color=color,
                 label=method, lw=2, marker='o', markersize=4)

    all_totals = [v for d in valid.values() for v in d["total_binding"] if v > 0]
    all_means = [v for d in valid.values() for v in d["mean_per_protein"] if v > 0]

    ax1.set_xlabel("Probability Threshold", fontsize=12)
    ax1.set_ylabel("Total Binding Residues", fontsize=12)
    ax1.set_title("Binding Residues (total)", fontsize=13)
    if _needs_log2(all_totals):
        ax1.set_yscale('log', base=2)
    ax1.legend()

    ax2.set_xlabel("Probability Threshold", fontsize=12)
    ax2.set_ylabel("Mean Binding Residues per Protein", fontsize=12)
    ax2.set_title("Binding Residues (mean per protein)", fontsize=13)
    if _needs_log2(all_means):
        ax2.set_yscale('log', base=2)
    ax2.legend()

    fig.suptitle("Probability Threshold Sweep (residue-level)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_protein_summary(method_stats, lengths):
    """{method: DataFrame[pdb_id, length, n_pockets, mean_pocket_size, binding_frac]}."""
    result = {}
    for method, data in method_stats.items():
        if not data or not data.get("pocket_records"):
            result[method] = None
            continue
        per_pdb = defaultdict(lambda: {"n_pockets": 0, "sizes": []})
        for r in data["pocket_records"]:
            per_pdb[r["pdb_id"]]["n_pockets"] += 1
            per_pdb[r["pdb_id"]]["sizes"].append(r["size"])
        rows = []
        for pdb_id, d in per_pdb.items():
            if pdb_id not in lengths:
                continue
            L = lengths[pdb_id]
            rows.append({
                "pdb_id": pdb_id,
                "length": L,
                "n_pockets": d["n_pockets"],
                "mean_pocket_size": float(np.mean(d["sizes"])) if d["sizes"] else 0.0,
                "binding_frac": sum(d["sizes"]) / L if L > 0 else 0.0,
            })
        result[method] = pd.DataFrame(rows)
    return result


def _binned_mean(df, x_col, y_col, bin_edges):
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.clip(np.digitize(df[x_col], bin_edges) - 1, 0, len(centers) - 1)
    means = np.array([
        df.loc[bin_idx == i, y_col].mean() if (bin_idx == i).any() else np.nan
        for i in range(len(centers))
    ])
    return centers, means


def _length_bins(max_len, use_log2, n_bins=25):
    """Return bin edges for sequence length, log-spaced when use_log2."""
    if use_log2:
        return np.geomspace(max(1, 10), max_len + 50, n_bins)
    return np.arange(0, max_len + 50, 50)


def plot_length_distribution(per_protein, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    series = []
    max_len = 0
    all_lengths = []
    for method, df in per_protein.items():
        if df is None or df.empty:
            continue
        series.append((method, df))
        max_len = max(max_len, int(df["length"].max()))
        all_lengths.extend(df["length"].tolist())
    if not series:
        plt.close(fig); return
    log_x = _needs_log2(all_lengths)
    edges = _length_bins(max_len, log_x)
    centers = 0.5 * (edges[:-1] + edges[1:])
    all_counts = []
    for method, df in series:
        counts, _ = np.histogram(df["length"], bins=edges)
        all_counts.extend(counts.tolist())
        ax.plot(centers, counts, color=_method_color(method),
                label=method, lw=1.8)
    ax.set_xlabel("Sequence Length (residues)", fontsize=12)
    ax.set_ylabel("Number of Proteins", fontsize=12)
    ax.set_title("Protein Size Distribution", fontsize=14)
    if log_x:
        ax.set_xscale('log', base=2)
    if _needs_log2(all_counts):
        ax.set_yscale('log', base=2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pockets_vs_length(per_protein, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    series = []
    max_len = 0
    all_lengths = []
    for method, df in per_protein.items():
        if df is None or df.empty:
            continue
        series.append((method, df))
        max_len = max(max_len, int(df["length"].max()))
        all_lengths.extend(df["length"].tolist())
    if not series:
        plt.close(fig); return
    log_x = _needs_log2(all_lengths)
    edges = _length_bins(max_len, log_x)
    all_means = []
    for method, df in series:
        centers, means = _binned_mean(df, "length", "n_pockets", edges)
        all_means.extend(means.tolist())
        ax.plot(centers, means, color=_method_color(method),
                label=method, lw=1.8)
    ax.set_xlabel("Sequence Length (residues)", fontsize=12)
    ax.set_ylabel("Mean Number of Pockets", fontsize=12)
    ax.set_title("Pockets per Protein vs Sequence Length", fontsize=14)
    if log_x:
        ax.set_xscale('log', base=2)
    if _needs_log2(all_means):
        ax.set_yscale('log', base=2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pocket_size_vs_length(per_protein, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    series = []
    max_len = 0
    all_lengths = []
    for method, df in per_protein.items():
        if df is None or df.empty:
            continue
        series.append((method, df))
        max_len = max(max_len, int(df["length"].max()))
        all_lengths.extend(df["length"].tolist())
    if not series:
        plt.close(fig); return
    log_x = _needs_log2(all_lengths)
    edges = _length_bins(max_len, log_x)
    all_means = []
    for method, df in series:
        centers, means = _binned_mean(df, "length", "mean_pocket_size", edges)
        all_means.extend(means.tolist())
        ax.plot(centers, means, color=_method_color(method),
                label=method, lw=1.8)
    ax.set_xlabel("Sequence Length (residues)", fontsize=12)
    ax.set_ylabel("Mean Pocket Size (residues)", fontsize=12)
    ax.set_title("Pocket Size vs Sequence Length", fontsize=14)
    if log_x:
        ax.set_xscale('log', base=2)
    if _needs_log2(all_means):
        ax.set_yscale('log', base=2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_binding_fraction(per_protein, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    edges = np.arange(0, 101, 5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    series = []
    for method, df in per_protein.items():
        if df is None or df.empty:
            continue
        series.append((method, df))
    if not series:
        plt.close(fig); return
    all_counts = []
    for method, df in series:
        pct = (df["binding_frac"] * 100).clip(upper=100)
        counts, _ = np.histogram(pct, bins=edges)
        all_counts.extend(counts.tolist())
        ax.plot(centers, counts, color=_method_color(method),
                label=method, lw=1.8)
    ax.set_xlabel("% Binding Residues", fontsize=12)
    ax.set_ylabel("Number of Proteins", fontsize=12)
    ax.set_title("Distribution of Binding Residue Fraction", fontsize=14)
    if _needs_log2(all_counts):
        ax.set_yscale('log', base=2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_novel_ratio(method_stats, novel_stats, out_path):
    novel_map = {
        "Seq2Pocket": (novel_stats.get("Seq2Pocket-unique") or {}).get("total_pockets", 0),
        "P2Rank":     (novel_stats.get("P2Rank-unique") or {}).get("total_pockets", 0),
    }
    methods = [m for m, d in method_stats.items() if d and sum(d["pockets_per_protein"]) > 0]
    if not methods:
        return

    totals = [sum(method_stats[m]["pockets_per_protein"]) for m in methods]
    shared = [t - novel_map.get(m, 0) for t, m in zip(totals, methods)]
    colors = [_method_color(m) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.35
    b1 = ax.bar(x - width/2, totals, width, color=colors,
                edgecolor='black', label="Total pockets")
    b2 = ax.bar(x + width/2, shared, width, color=colors,
                edgecolor='black', alpha=0.5, label="Shared with other method")
    for bar, v in list(zip(b1, totals)) + list(zip(b2, shared)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Number of Pockets", fontsize=12)
    ax.set_title("Total vs Shared Pockets", fontsize=14)
    if _needs_log2(totals + shared):
        ax.set_yscale('log', base=2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import sys

    summary_path = STATS_DIR / "summary.txt"
    with open(summary_path, 'w') as out:
        out.write("Pipeline Statistics Report\n")
        out.write(f"Generated from: {ROOT}\n\n")

        # 1. Funnel
        funnel = pipeline_funnel(out)
        plot_funnel(funnel, PLOTS_DIR / "pipeline_funnel.png")

        # 2. Per-method stats
        method_stats = per_method_stats(out)

        # 3. Pocket size outliers
        pocket_outliers(method_stats, out, STATS_DIR)
        zero_size_pockets(method_stats, out, STATS_DIR)

        # 4. Novel pockets
        novel_stats = novel_pocket_stats(out)
        plot_pocket_distributions(method_stats, PLOTS_DIR, novel_stats)
        plot_novel_pockets(novel_stats, PLOTS_DIR)

        # 5. Length-based distributions & novel ratio
        lengths = load_protein_lengths(S2P_DIR)
        per_protein = per_protein_summary(method_stats, lengths)
        plot_length_distribution(per_protein, PLOTS_DIR / "protein_size_distribution.png")
        plot_pockets_vs_length(per_protein, PLOTS_DIR / "pockets_vs_length.png")
        plot_pocket_size_vs_length(per_protein, PLOTS_DIR / "pocket_size_vs_length.png")
        plot_binding_fraction(per_protein, PLOTS_DIR / "binding_fraction_distribution.png")
        plot_novel_ratio(method_stats, novel_stats, PLOTS_DIR / "novel_pocket_ratio.png")

        aa_counts = {
            "Seq2Pocket": collect_aa_composition(S2P_DIR, "residue_type", "pocket_number"),
            "P2Rank":     collect_aa_composition(P2RANK_DIR, "residue_name", "pocket"),
        }
        plot_aa_composition(aa_counts, PLOTS_DIR / "aa_composition.png")

        skip_breakdown = parse_skip_log(CS_DIR / "skipped_clustering.txt")
        plot_skip_breakdown(skip_breakdown, PLOTS_DIR / "clustering_skip_breakdown.png")

        sweep = threshold_sweep(
            {"Seq2Pocket": (S2P_DIR, "probability"),
             "P2Rank":     (P2RANK_DIR, "probability")},
            thresholds=np.round(np.arange(0.1, 0.95, 0.05), 2),
        )
        plot_threshold_sweep(sweep, PLOTS_DIR / "threshold_sweep.png")

        # 6. Summary table
        summary_table(funnel, method_stats, novel_stats, out)

    print(f"Statistics written to: {summary_path}")
    print(f"Plots saved to:       {PLOTS_DIR}")

    # Print summary to stdout as well
    with open(summary_path) as f:
        print(f.read())

# ============================================================
# SUGGESTIONS TODO
# ============================================================
#
# 3. METHOD AGREEMENT (requires matched PDB IDs between methods)
#    - Per protein: count overlapping vs unique pockets
#    - Jaccard similarity of residue sets for matched pockets
#    - Fraction of proteins where both methods agree on >= 1 pocket
#    - Heatmap of agreement scores across proteins
#
# 5. RESIDUE-LEVEL ANALYSIS (requires _residues.csv from step 3)
#    - Distribution of binding probabilities across all residues
#    - Fraction of residues predicted as binding per protein
#    - Correlation between binding fraction and protein length
#    - ROC / PR curves if ground truth labels available
#
# Additional ideas:
#    - Pocket score vs pocket size scatter plot
#    - Per-chain analysis (multi-chain proteins)
#    - Export statistics as LaTeX tables for thesis
