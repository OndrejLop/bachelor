#!/usr/bin/env python3
"""
Tool 18 — SASA / residue and pocket protrusion distributions per category.

Categorizes every predicted pocket from a step-4 comparison run into one
of four buckets:

  unique_p2r — P2R pocket listed in {pdb_id}/p2r/unmatched_pockets.csv
  shared_p2r — P2R pocket NOT listed there (matched against an S2P pocket)
  unique_s2p — S2P pocket listed in {pdb_id}/s2p/unmatched_pockets.csv  (or legacy cs/)
  shared_s2p — S2P pocket NOT listed there (matched against a P2R pocket)

Per PDB the tool computes (once, on the first model with hetatms and
hydrogens removed):
  - per-residue SASA via biotite's vectorized Shrake-Rupley
    (`biotite.structure.sasa`, default probe 1.4 Å, single-radius VdW
    table). ~20× faster than Bio.PDB.SASA.ShrakeRupley at equal point counts.
  - per-residue neighbor count: number of heavy atoms of *other* residues
    within --neighbor-radius Å of the residue's CA, via
    `biotite.structure.CellList`. Higher count = more buried (i.e. lower
    protrusion).

For each pocket it then emits two normalized metrics:
  sasa_per_residue   = sum(residue SASA over pocket residues) / n_residues
  mean_neighbors     = mean(neighbor count over pocket residues)

Outputs (under --output-dir, default data/output/analysis/sasa/):
  pocket_metrics.csv          long-form, one row per pocket
  sasa_distributions.png      KDE per category — SASA / residue
  sasa_violin.png             violin plot — SASA / residue
  sasa_box.png                box plot — SASA / residue (alternative view)
  protrusion_distributions.png  KDE per category — mean neighbor count
  protrusion_violin.png       violin plot — mean neighbor count
  protrusion_box.png          box plot — mean neighbor count (alternative view)
  sasa_vs_protrusion.png      scatter SASA vs mean neighbors, colored by category
  metrics_summary.txt         n / mean / median / IQR / std per category, both metrics

Reads pocket size bounds from {results_dir}/run_metadata.json when present
(falls back to 3..70). Override with --min-pocket-size / --max-pocket-size.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_RESULTS = ROOT / 'data' / 'output' / 'results'
DEFAULT_OUTPUT = ROOT / 'data' / 'output' / 'analysis' / 'sasa'
DEFAULT_PDB_DIR = ROOT / 'data' / 'input' / 'pdb'
DEFAULT_P2R_DIR = ROOT / 'data' / 'input' / 'P2Rank'
DEFAULT_S2P_DIR = ROOT / 'data' / 'output' / 'Seq2Pockets'

def _savefig(fig, path: Path, dpi: int = 150):
    """Save to png/, pdf/, svg/ subfolders relative to path.parent."""
    stem = path.stem
    for subdir, ext in [('png', '.png'), ('pdf', '.pdf'), ('svg', '.svg')]:
        d = path.parent / subdir
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}{ext}", dpi=dpi, bbox_inches='tight')


CATEGORIES = ('unique_p2r', 'shared_p2r', 'unique_s2p', 'shared_s2p')
CATEGORY_COLORS = {
    'unique_p2r': 'steelblue',
    'shared_p2r': '#aec7e8',
    'unique_s2p': 'darkorange',
    'shared_s2p': '#ff9896',
}


def parse_residue_ids(s):
    """'A_113 A_30' -> [('A', 113), ...]. Skips malformed tokens."""
    if not isinstance(s, str):
        return []
    out = []
    for tok in s.strip().split():
        chain, _, resnum = tok.partition('_')
        if not resnum:
            continue
        try:
            out.append((chain, int(resnum)))
        except ValueError:
            continue
    return out


def load_predictions(csv_path: Path) -> dict[str, list[tuple[str, int]]]:
    """{pocket_name: [(chain, resseq), ...]}.  Strips column whitespace."""
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    out: dict[str, list[tuple[str, int]]] = {}
    for _, row in df.iterrows():
        name = str(row.get('name', '')).strip()
        if not name:
            continue
        out[name] = parse_residue_ids(row.get('residue_ids', ''))
    return out


def load_unmatched_names(csv_path: Path) -> set[str]:
    if not csv_path.is_file():
        return set()
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    return {str(p).strip() for p in df.get('pocket', []) if str(p).strip()}


def find_pdb_file(pdb_dir: Path, pdb_id: str) -> Path | None:
    direct = pdb_dir / f"{pdb_id}.pdb"
    if direct.is_file():
        return direct
    matches = list(pdb_dir.glob(f"{pdb_id}*"))
    return matches[0] if matches else None


def load_atom_array(pdb_path: Path):
    """Read first model, drop hetatms and hydrogens. Returns a biotite
    AtomArray. Much faster than Bio.PDB for SASA + neighbor search."""
    pf = pdb_io.PDBFile.read(str(pdb_path))
    atoms = pf.get_structure(model=1)
    atoms = atoms[~atoms.hetero]
    if hasattr(atoms, 'element'):
        atoms = atoms[atoms.element != 'H']
    return atoms


def compute_residue_sasa(atoms, probe_radius: float, point_number: int) -> dict:
    """{(chain_id, resseq): residue SASA total}. Uses biotite's vectorized
    Shrake-Rupley (~20× faster than Bio.PDB.SASA.ShrakeRupley)."""
    if len(atoms) == 0:
        return {}
    sasa_per_atom = struc.sasa(atoms, probe_radius=probe_radius,
                               point_number=point_number)
    sasa_per_residue = struc.apply_residue_wise(atoms, sasa_per_atom, np.nansum)
    if sasa_per_residue is None:
        return {}
    starts = struc.get_residue_starts(atoms)
    chain_ids = atoms.chain_id[starts]
    res_ids = atoms.res_id[starts]
    return {(str(c), int(r)): float(s)
            for c, r, s in zip(chain_ids, res_ids, sasa_per_residue)}


def compute_residue_neighbor_counts(atoms, radius: float) -> dict:
    """Per-residue protrusion proxy: count heavy atoms of OTHER residues
    within `radius` Å of this residue's CA atom.

    Higher count = more buried = lower protrusion. Residues without a CA
    are skipped. Uses biotite's CellList — O(N) average neighbor lookup."""
    if len(atoms) == 0:
        return {}
    ca_mask = atoms.atom_name == 'CA'
    ca_idx = np.where(ca_mask)[0]
    if len(ca_idx) == 0:
        return {}
    cell_list = struc.CellList(atoms, cell_size=radius)
    chain_ids = atoms.chain_id
    res_ids = atoms.res_id
    out = {}
    for i in ca_idx:
        nearby = cell_list.get_atoms(atoms.coord[i], radius=radius)
        c, r = str(chain_ids[i]), int(res_ids[i])
        n_self = int(np.sum((chain_ids[nearby] == c) & (res_ids[nearby] == r)))
        out[(c, r)] = int(len(nearby) - n_self)
    return out


def pocket_sasa(residues: list[tuple[str, int]], sasa_map: dict) -> tuple[float, int, int]:
    """(total_sasa, n_resolved, n_missing) over the residues."""
    total = 0.0
    resolved = missing = 0
    for key in residues:
        if key in sasa_map:
            total += sasa_map[key]
            resolved += 1
        else:
            missing += 1
    return total, resolved, missing


def pocket_neighbor_mean(residues: list[tuple[str, int]], neighbor_map: dict) -> tuple[float, int]:
    """(mean_neighbors_over_resolved_residues, n_resolved). Returns
    (NaN, 0) if no residue is found in neighbor_map."""
    vals = [neighbor_map[k] for k in residues if k in neighbor_map]
    if not vals:
        return float('nan'), 0
    return sum(vals) / len(vals), len(vals)


def _row_for_pocket(pdb_id, method, category, name, residues,
                    sasa_map, neighbor_map):
    sasa_total, sasa_resolved, sasa_missing = pocket_sasa(residues, sasa_map)
    nbr_mean, nbr_resolved = pocket_neighbor_mean(residues, neighbor_map)
    return {
        'pdb_id': pdb_id,
        'method': method,
        'category': category,
        'pocket': name,
        'n_residues': len(residues),
        'n_residues_resolved': sasa_resolved,
        'n_residues_missing': sasa_missing,
        'sasa_total': sasa_total,
        'sasa_per_residue': sasa_total / len(residues),
        'mean_neighbors': nbr_mean,
        'n_residues_with_ca': nbr_resolved,
    }


def categorize_pdb(pdb_id, p2r_csv, s2p_csv, pdb_path, results_dir,
                   lo, hi, probe_radius, n_points, neighbor_radius):
    p2r_all = load_predictions(p2r_csv)
    s2p_all = load_predictions(s2p_csv)
    p2r_all = {n: r for n, r in p2r_all.items() if lo <= len(r) <= hi}
    s2p_all = {n: r for n, r in s2p_all.items() if lo <= len(r) <= hi}
    if not p2r_all and not s2p_all:
        return []

    uniq_p2r = load_unmatched_names(results_dir / pdb_id / 'p2r' / 'unmatched_pockets.csv')
    uniq_s2p = (load_unmatched_names(results_dir / pdb_id / 's2p' / 'unmatched_pockets.csv')
                | load_unmatched_names(results_dir / pdb_id / 'cs' / 'unmatched_pockets.csv'))

    atoms = load_atom_array(pdb_path)
    sasa_map = compute_residue_sasa(atoms, probe_radius, n_points)
    neighbor_map = compute_residue_neighbor_counts(atoms, neighbor_radius)

    rows = []
    for name, residues in p2r_all.items():
        if not residues:
            continue
        cat = 'unique_p2r' if name in uniq_p2r else 'shared_p2r'
        rows.append(_row_for_pocket(pdb_id, 'p2r', cat, name, residues,
                                    sasa_map, neighbor_map))
    for name, residues in s2p_all.items():
        if not residues:
            continue
        cat = 'unique_s2p' if name in uniq_s2p else 'shared_s2p'
        rows.append(_row_for_pocket(pdb_id, 's2p', cat, name, residues,
                                    sasa_map, neighbor_map))
    return rows


def write_summary(df: pd.DataFrame, metrics: list[tuple[str, str]], path: Path) -> str:
    """metrics: list of (column, axis_label)."""
    lines = ["Pocket metrics summary", "=" * 60]
    for column, label in metrics:
        lines.append("")
        lines.append(f"{label}")
        lines.append("-" * len(label))
        lines.append(f"{'category':<14} {'n':>6} {'mean':>10} {'median':>10} "
                     f"{'q25':>10} {'q75':>10} {'std':>10}")
        for cat in CATEGORIES:
            sub = df.loc[df['category'] == cat, column].dropna()
            if sub.empty:
                lines.append(f"{cat:<14} {0:>6} {'-':>10} {'-':>10} {'-':>10} {'-':>10} {'-':>10}")
                continue
            lines.append(f"{cat:<14} {len(sub):>6d} "
                         f"{sub.mean():>10.3f} {sub.median():>10.3f} "
                         f"{sub.quantile(.25):>10.3f} {sub.quantile(.75):>10.3f} "
                         f"{sub.std():>10.3f}")
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    return text


def plot_distributions(df: pd.DataFrame, column: str, axis_label: str,
                       title: str, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    all_vals = df[column].dropna().to_numpy()
    if len(all_vals) == 0:
        plt.close(fig)
        return
    upper = float(pd.Series(all_vals).quantile(0.99))
    bins = [b for b in [i * upper / 40 for i in range(41)] if b > 0] or 40

    for cat in CATEGORIES:
        sub = df.loc[df['category'] == cat, column].dropna().to_numpy()
        if len(sub) < 2:
            continue
        color = CATEGORY_COLORS[cat]
        label = f"{cat} (n={len(sub)})"
        ax.hist(sub, bins=bins, density=True, histtype='stepfilled',
                color=color, alpha=0.30, label=label)
        ax.hist(sub, bins=bins, density=True, histtype='step',
                color=color, linewidth=1.5)

    ax.set_xlabel(axis_label)
    ax.set_ylabel('density')
    ax.set_xlim(0, upper if upper > 0 else None)
    ax.legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


def plot_violin(df: pd.DataFrame, column: str, axis_label: str,
                title: str, path: Path):
    order = [c for c in CATEGORIES
             if df.loc[df['category'] == c, column].notna().sum() >= 2]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [df.loc[df['category'] == c, column].dropna().to_numpy() for c in order]
    parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.85)
    for body, cat in zip(parts['bodies'], order):
        body.set_facecolor(CATEGORY_COLORS[cat])
        body.set_edgecolor('black')
        body.set_alpha(0.65)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{c}\n(n={len(d)})" for c, d in zip(order, data)])
    ax.set_ylabel(axis_label)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                 x_label: str, y_label: str, title: str, path: Path):
    """2D scatter of two metrics, points colored by category. Useful for
    spotting whether unique vs shared pockets cluster in metric space."""
    sub = df[[x_col, y_col, 'category']].dropna()
    if len(sub) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for cat in CATEGORIES:
        s = sub[sub['category'] == cat]
        if s.empty:
            continue
        ax.scatter(s[x_col], s[y_col], s=18, alpha=0.55,
                   color=CATEGORY_COLORS[cat], edgecolor='none',
                   label=f"{cat} (n={len(s)})", rasterized=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


def plot_box(df: pd.DataFrame, column: str, axis_label: str,
             title: str, path: Path):
    """Categorical box plot — alternative view to plot_violin. No scatter
    overlay (that was what made the original boxplot crowded)."""
    order = [c for c in CATEGORIES
             if df.loc[df['category'] == c, column].notna().sum() >= 1]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [df.loc[df['category'] == c, column].dropna().to_numpy() for c in order]
    bp = ax.boxplot(data, tick_labels=[f"{c}\n(n={len(d)})"
                                       for c, d in zip(order, data)],
                    showfliers=True, widths=0.55, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='.', markersize=2, alpha=0.4))
    for patch, cat in zip(bp['boxes'], order):
        patch.set_facecolor(CATEGORY_COLORS[cat])
        patch.set_alpha(0.65)
    for flier in bp['fliers']:
        flier.set_rasterized(True)
    ax.set_ylabel(axis_label)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    _savefig(fig, path)
    plt.close(fig)


def _looks_like_step4_dir(d: Path) -> bool:
    """True when d looks like a step-4 output: has run_metadata.json,
    an aggregate CSV, or any per-PDB subdir."""
    if not d.is_dir():
        return False
    if (d / "run_metadata.json").is_file():
        return True
    for marker in ("novel_s2p_pockets.csv", "novel_cs_pockets.csv",
                   "p2r_unique_pockets.csv"):
        if (d / marker).is_file():
            return True
    for child in d.iterdir():
        if child.is_dir() and child.name.lower().startswith("pdb"):
            return True
    return False


def resolve_results_dir(path: Path) -> Path:
    """Return the actual step-4 output dir.

    A relative path that doesn't exist as-is is also tried as a subdir of
    DEFAULT_RESULTS (mirrors step 5's convention, e.g.
    `--results-dir 20260506_161049_max_res0_pct0.0_size3-70`).

    If the resolved path looks like a step-4 output, keep it; otherwise pick
    the most recent timestamped subdir that does."""
    if not path.exists() and not path.is_absolute():
        candidate = DEFAULT_RESULTS / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        sys.exit(f"--results-dir does not exist: {path}")
    if _looks_like_step4_dir(path):
        return path
    timestamped = sorted([d for d in path.iterdir()
                          if d.is_dir() and not d.name.lower().startswith("pdb")],
                         reverse=True)
    for d in timestamped:
        if _looks_like_step4_dir(d):
            return d
    sys.exit(f"No step-4 output found under {path} (looked for run_metadata.json, "
             f"aggregate CSVs, or pdb*/ subdirs in the dir and any timestamped subdir)")


def resolve_bounds(args, results_dir: Path):
    if args.min_pocket_size is not None and args.max_pocket_size is not None:
        return args.min_pocket_size, args.max_pocket_size
    meta_path = results_dir / 'run_metadata.json'
    meta = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            print(f"  [WARN] {meta_path} is not valid JSON; using defaults")
    lo = args.min_pocket_size if args.min_pocket_size is not None else meta.get('min_pocket_size', 3)
    hi = args.max_pocket_size if args.max_pocket_size is not None else meta.get('max_pocket_size', 70)
    return lo, hi


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS)
    ap.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument('--pdb-dir', type=Path, default=DEFAULT_PDB_DIR)
    ap.add_argument('--p2r-dir', type=Path, default=DEFAULT_P2R_DIR)
    ap.add_argument('--s2p-dir', type=Path, default=DEFAULT_S2P_DIR)
    ap.add_argument('--probe-radius', type=float, default=1.4,
                    help='ShrakeRupley probe radius in Å (default: 1.4)')
    ap.add_argument('--n-points', type=int, default=100,
                    help='ShrakeRupley sphere sample points per atom (default: 100). '
                         'biotite is fast enough that 960 is also viable, but 100 is '
                         'plenty for residue-aggregated values.')
    ap.add_argument('--neighbor-radius', type=float, default=10.0,
                    help='Radius (Å) for the protrusion proxy: count heavy atoms '
                         'of other residues within this distance of CA (default: 10)')
    ap.add_argument('--min-pocket-size', type=int, default=None,
                    help='Override min pocket size (default: from run_metadata.json or 3)')
    ap.add_argument('--max-pocket-size', type=int, default=None,
                    help='Override max pocket size (default: from run_metadata.json or 70)')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process only the first N PDBs (smoke testing)')
    ap.add_argument('--timestamp', action='store_true',
                    help='Append timestamp + param suffix to --output-dir (prevents overwrites)')
    args = ap.parse_args()

    if not args.p2r_dir.is_dir():
        sys.exit(f"P2R prediction dir not found: {args.p2r_dir}")
    if not args.pdb_dir.is_dir():
        sys.exit(f"PDB input dir not found: {args.pdb_dir}")

    resolved_results_dir = resolve_results_dir(args.results_dir)
    if resolved_results_dir != args.results_dir:
        print(f"  [auto] --results-dir resolved to {resolved_results_dir}")
    args.results_dir = resolved_results_dir

    lo, hi = resolve_bounds(args, args.results_dir)
    if lo > hi:
        sys.exit(f"min_pocket_size ({lo}) > max_pocket_size ({hi})")

    if args.timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = (f"probe{args.probe_radius:g}"
                  f"_nbr{args.neighbor_radius:g}"
                  f"_size{lo}-{hi}")
        args.output_dir = args.output_dir / f"{ts}_{suffix}"

    print(f"Pocket size bounds: [{lo}, {hi}] residues")
    print(f"Probe radius: {args.probe_radius} Å, n_points: {args.n_points}")
    print(f"Neighbor radius (protrusion): {args.neighbor_radius} Å")
    print(f"Results dir:   {args.results_dir}")
    print(f"PDB dir:       {args.pdb_dir}")
    print(f"P2R dir:       {args.p2r_dir}")
    print(f"S2P dir:       {args.s2p_dir}")
    print(f"Output dir:    {args.output_dir}")

    s2p_lookup = {}
    for f in args.s2p_dir.rglob("*_predictions.csv"):
        pid = f.stem.replace('_predictions', '')
        s2p_lookup.setdefault(pid, f)
    print(f"\nFound {len(s2p_lookup)} S2P prediction files")

    pdb_ids = []
    for p2r_csv in sorted(args.p2r_dir.glob("*_predictions.csv")):
        pid = p2r_csv.stem.replace('_predictions', '')
        if pid in s2p_lookup:
            pdb_ids.append(pid)
    print(f"PDBs with both P2R and S2P predictions: {len(pdb_ids)}")
    if args.limit:
        pdb_ids = pdb_ids[:args.limit]
        print(f"--limit applied: processing {len(pdb_ids)} PDBs")

    rows = []
    n_no_pdb = n_failed = 0
    t0 = time.time()
    for i, pid in enumerate(pdb_ids, 1):
        pdb_path = find_pdb_file(args.pdb_dir, pid)
        if pdb_path is None:
            n_no_pdb += 1
            continue
        try:
            new_rows = categorize_pdb(
                pid, args.p2r_dir / f"{pid}_predictions.csv", s2p_lookup[pid],
                pdb_path, args.results_dir,
                lo, hi, args.probe_radius, args.n_points, args.neighbor_radius,
            )
            rows.extend(new_rows)
        except Exception as e:
            n_failed += 1
            print(f"  [WARN] {pid}: {e}")
        if i % 25 == 0 or i == len(pdb_ids):
            dt = time.time() - t0
            print(f"  {i}/{len(pdb_ids)} processed  ({dt:.1f}s, {len(rows)} pockets)")

    print(f"\nSkipped (no PDB structure): {n_no_pdb}")
    print(f"Failed (exceptions):        {n_failed}")
    print(f"Total pockets categorized:  {len(rows)}")
    if not rows:
        sys.exit("No pockets were categorized — nothing to plot.")

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'pocket_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    sasa_label = 'SASA / residue (Å²)'
    nbr_label = f'mean neighbors within {args.neighbor_radius:g} Å of CA'
    metrics = [('sasa_per_residue', sasa_label),
               ('mean_neighbors', nbr_label)]

    summary_text = write_summary(df, metrics, args.output_dir / 'metrics_summary.txt')
    print(f"Wrote {args.output_dir / 'metrics_summary.txt'}")
    print()
    print(summary_text)

    plot_distributions(df, 'sasa_per_residue', sasa_label,
                       'SASA / residue distribution by pocket category',
                       args.output_dir / 'sasa_distributions.png')
    plot_violin(df, 'sasa_per_residue', sasa_label,
                'SASA / residue per pocket category',
                args.output_dir / 'sasa_violin.png')
    plot_box(df, 'sasa_per_residue', sasa_label,
             'SASA / residue per pocket category',
             args.output_dir / 'sasa_box.png')
    plot_distributions(df, 'mean_neighbors', nbr_label,
                       'Pocket protrusion (mean CA neighbor count) by category',
                       args.output_dir / 'protrusion_distributions.png')
    plot_violin(df, 'mean_neighbors', nbr_label,
                'Pocket protrusion per category',
                args.output_dir / 'protrusion_violin.png')
    plot_box(df, 'mean_neighbors', nbr_label,
             'Pocket protrusion per category',
             args.output_dir / 'protrusion_box.png')
    plot_scatter(df, 'sasa_per_residue', 'mean_neighbors',
                 sasa_label, nbr_label,
                 'SASA / residue vs protrusion (mean neighbor count)',
                 args.output_dir / 'sasa_vs_protrusion.png')
    for name in ('sasa_distributions.png', 'sasa_violin.png', 'sasa_box.png',
                 'protrusion_distributions.png', 'protrusion_violin.png', 'protrusion_box.png',
                 'sasa_vs_protrusion.png'):
        print(f"Plot: {args.output_dir / name}")


if __name__ == '__main__':
    main()
