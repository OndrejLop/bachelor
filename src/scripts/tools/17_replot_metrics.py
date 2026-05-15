#!/usr/bin/env python3
"""
Tool 17 — replot tool 18's metrics from an existing pocket_metrics.csv as violin plots.

Useful when you have the metrics table from a previous tool-18 run that produced
the older box-plot style and want violin plots without re-running SASA.

Reads a long-form CSV produced by tool 18 (`18_pocket_sasa.py`) and writes:

  sasa_violin.png         violin per category — SASA / residue
  protrusion_violin.png   violin per category — mean neighbor count
  sasa_vs_protrusion.png  scatter SASA vs mean neighbors, colored by category

The CSV must have columns: category, sasa_per_residue, mean_neighbors.
"""
#example use
#python3 17_replot_metrics.py --metrics-csv "/Users/ondrejlopatka/Local files/bachelor_local/git/data/output/sasa/all_box_probe1.4_nbr10_size3-70/pocket_metrics.csv"


import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

TOOL18_PATH = Path(__file__).resolve().parent / '18_pocket_sasa.py'
REQUIRED_COLUMNS = {'category', 'sasa_per_residue', 'mean_neighbors'}


def load_tool18():
    """Import tool 18 by path (its module name starts with a digit, so a
    plain `import` doesn't work)."""
    if not TOOL18_PATH.is_file():
        sys.exit(f"Cannot find {TOOL18_PATH} — is tool 18 still in src/scripts/tools/?")
    spec = importlib.util.spec_from_file_location("tool18", TOOL18_PATH)
    if spec is None or spec.loader is None:
        sys.exit(f"Failed to load module spec from {TOOL18_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--metrics-csv', type=Path, required=True,
                    help='Path to pocket_metrics.csv from a tool-18 run')
    ap.add_argument('--output-dir', type=Path, default=None,
                    help='Where to write plots (default: same dir as --metrics-csv)')
    ap.add_argument('--neighbor-radius', type=float, default=10.0,
                    help='Used only for the protrusion plot axis label '
                         '(should match the radius the original run used; default: 10)')
    args = ap.parse_args()

    if not args.metrics_csv.is_file():
        sys.exit(f"Metrics CSV not found: {args.metrics_csv}")

    df = pd.read_csv(args.metrics_csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"Metrics CSV is missing required columns: {sorted(missing)}")

    out_dir = args.output_dir or args.metrics_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source CSV:  {args.metrics_csv}")
    print(f"Rows:        {len(df)}")
    print(f"Output dir:  {out_dir}")
    print(f"Neighbor-radius label: {args.neighbor_radius:g} Å")

    tool18 = load_tool18()

    sasa_label = 'SASA / residue (Å²)'
    nbr_label = f'mean neighbors within {args.neighbor_radius:g} Å of CA'

    tool18.plot_violin(df, 'sasa_per_residue', sasa_label,
                       'SASA / residue per pocket category',
                       out_dir / 'sasa_violin.png')
    tool18.plot_violin(df, 'mean_neighbors', nbr_label,
                       'Pocket protrusion per category',
                       out_dir / 'protrusion_violin.png')
    tool18.plot_scatter(df, 'sasa_per_residue', 'mean_neighbors',
                        sasa_label, nbr_label,
                        'SASA / residue vs protrusion (mean neighbor count)',
                        out_dir / 'sasa_vs_protrusion.png')

    for name in ('sasa_violin.png', 'protrusion_violin.png',
                 'sasa_vs_protrusion.png'):
        print(f"Plot: {out_dir / name}")


if __name__ == '__main__':
    main()
