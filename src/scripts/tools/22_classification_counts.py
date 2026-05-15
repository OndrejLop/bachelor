#!/usr/bin/env python3
"""
Tool 22 — Count proteins per classification category for a given results run.

Reads the per-PDB subdirectories from a step-4 results directory, looks each
PDB up in pdb_classification.csv (tool 14), and writes a CSV with one row per
category showing how many proteins landed there.

Outputs (--output-dir, default data/output/analysis/):
  classification_counts.csv   — classification, n_proteins, n_with_ec
"""
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

RESULTS_ROOT   = ROOT / "data" / "output" / "results"
DEFAULT_OUTPUT = ROOT / "data" / "output" / "analysis"
DEFAULT_CLASS  = ROOT / "data" / "intermediate" / "pdb_classification.csv"

parser = argparse.ArgumentParser(description="Tool 22 — classification category counts.")
parser.add_argument("--results-dir", type=Path, default=None,
                    help="Step-4 results directory containing per-PDB subdirs. "
                         "Absolute or relative to data/output/results/. "
                         "Default: auto-select (newest timestamped or flat).")
parser.add_argument("--classification-csv", type=Path, default=DEFAULT_CLASS)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
args = parser.parse_args()


def _find_results_dir(root):
    if (root / "novel_s2p_pockets.csv").exists():
        return root
    candidates = sorted(
        [d for d in root.iterdir() if d.is_dir() and not d.name.lower().startswith("pdb")],
        reverse=True)
    for d in candidates:
        if (d / "novel_s2p_pockets.csv").exists():
            return d
    return None


def _resolve_results_dir(path):
    if path is None:
        d = _find_results_dir(RESULTS_ROOT)
        if d is None:
            parser.error(f"No results directory found under {RESULTS_ROOT}")
        return d
    candidate = path if path.is_absolute() else RESULTS_ROOT / path
    candidate = candidate.resolve()
    if not candidate.exists():
        parser.error(f"--results-dir does not exist: {candidate}")
    return candidate


results_dir = _resolve_results_dir(args.results_dir)

print(f"Results dir:        {results_dir}")
print(f"Classification CSV: {args.classification_csv}")
print(f"Output dir:         {args.output_dir}")

# ── Collect PDB IDs from per-PDB subdirectories ────────────────────────────────
pdb_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
raw_ids  = [d.name for d in pdb_dirs]

def _normalize(name):
    n = name.lower()
    return n[3:] if n.startswith("pdb") else n

pdb_ids = [_normalize(n) for n in raw_ids]
print(f"PDB subdirs found:  {len(pdb_ids)}")

# ── Load classification ────────────────────────────────────────────────────────
if not args.classification_csv.exists():
    parser.error(f"Classification CSV not found: {args.classification_csv}")

cls_df = pd.read_csv(args.classification_csv, dtype=str).fillna("")
cls_df["pdb_id"] = cls_df["pdb_id"].str.lower().str.strip()

results_df = pd.DataFrame({"pdb_id": pdb_ids})
merged = results_df.merge(cls_df, on="pdb_id", how="left")

n_unclassified = merged["classification"].isna().sum() + (merged["classification"] == "").sum()
if n_unclassified:
    print(f"  [WARN] {n_unclassified} PDB(s) not found in classification CSV → counted as 'UNCLASSIFIED'")
merged["classification"] = merged["classification"].fillna("UNCLASSIFIED").replace("", "UNCLASSIFIED")
merged["has_ec"] = merged["ec_numbers"].notna() & (merged["ec_numbers"] != "")

# ── Count per category ─────────────────────────────────────────────────────────
counts = (
    merged.groupby("classification", sort=False)
    .agg(n_proteins=("pdb_id", "count"),
         n_with_ec=("has_ec", "sum"))
    .reset_index()
    .sort_values("n_proteins", ascending=False)
    .reset_index(drop=True)
)
counts["n_with_ec"] = counts["n_with_ec"].astype(int)

merged["classification_upper"] = merged["classification"].str.upper()
counts_ci = (
    merged.groupby("classification_upper", sort=False)
    .agg(n_proteins=("pdb_id", "count"),
         n_with_ec=("has_ec", "sum"))
    .reset_index()
    .rename(columns={"classification_upper": "classification"})
    .sort_values("n_proteins", ascending=False)
    .reset_index(drop=True)
)
counts_ci["n_with_ec"] = counts_ci["n_with_ec"].astype(int)

# ── Save ───────────────────────────────────────────────────────────────────────
args.output_dir.mkdir(parents=True, exist_ok=True)
out_path    = args.output_dir / "classification_counts.csv"
out_path_ci = args.output_dir / "classification_counts_ci.csv"
counts.to_csv(out_path, index=False)
counts_ci.to_csv(out_path_ci, index=False)

print(f"\nCategories found (exact):            {len(counts)}")
print(f"Categories found (case-insensitive): {len(counts_ci)}")
print(f"Total proteins: {counts['n_proteins'].sum()}")
print(f"  → {out_path}")
print(f"  → {out_path_ci}")
