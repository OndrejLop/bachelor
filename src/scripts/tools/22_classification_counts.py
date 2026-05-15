#!/usr/bin/env python3
"""
Tool 22 — Count proteins and pockets per classification category.

Reads the per-PDB subdirectories from a step-4 results directory, looks each
PDB up in pdb_classification.csv (tool 14), and writes a case-insensitive CSV
with one row per category.

Output (--output-dir, default data/output/analysis/):
  classification_counts_ci.csv — classification (uppercased), n_proteins,
                                  n_with_ec, n_s2p_unique_pockets,
                                  n_s2p_shared_pockets, n_p2r_unique_pockets,
                                  n_p2r_shared_pockets
"""
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

RESULTS_ROOT   = ROOT / "data" / "output" / "results"
DEFAULT_OUTPUT = ROOT / "data" / "output" / "analysis"
DEFAULT_CLASS  = ROOT / "data" / "intermediate" / "pdb_classification.csv"
DEFAULT_S2P    = ROOT / "data" / "output" / "Seq2Pockets"
DEFAULT_P2R    = ROOT / "data" / "input"  / "P2Rank"

parser = argparse.ArgumentParser(description="Tool 22 — classification category counts.")
parser.add_argument("--results-dir", type=Path, default=None,
                    help="Step-4 results directory containing per-PDB subdirs. "
                         "Absolute or relative to data/output/results/. "
                         "Default: auto-select (newest timestamped or flat).")
parser.add_argument("--classification-csv", type=Path, default=DEFAULT_CLASS)
parser.add_argument("--s2p-dir", type=Path, default=DEFAULT_S2P,
                    help="Seq2Pockets output dir (for total pocket counts).")
parser.add_argument("--p2r-dir", type=Path, default=DEFAULT_P2R,
                    help="P2Rank predictions dir (for total pocket counts).")
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
print(f"S2P dir:            {args.s2p_dir}")
print(f"P2R dir:            {args.p2r_dir}")
print(f"Output dir:         {args.output_dir}")


def _normalize(name):
    n = name.lower()
    return n[3:] if n.startswith("pdb") else n


# ── Collect PDB IDs from per-PDB subdirectories ────────────────────────────────
pdb_ids = [_normalize(d.name) for d in results_dir.iterdir() if d.is_dir()]
print(f"PDB subdirs found:  {len(pdb_ids)}")

# ── Load classification ────────────────────────────────────────────────────────
if not args.classification_csv.exists():
    parser.error(f"Classification CSV not found: {args.classification_csv}")

cls_df = pd.read_csv(args.classification_csv, dtype=str).fillna("")
cls_df["pdb_id"] = cls_df["pdb_id"].str.lower().str.strip()

merged = pd.DataFrame({"pdb_id": pdb_ids}).merge(cls_df, on="pdb_id", how="left")

n_unclassified = merged["classification"].isna().sum() + (merged["classification"] == "").sum()
if n_unclassified:
    print(f"  [WARN] {n_unclassified} PDB(s) not in classification CSV → 'UNCLASSIFIED'")
merged["classification"] = merged["classification"].fillna("UNCLASSIFIED").replace("", "UNCLASSIFIED")
merged["has_ec"] = merged["ec_numbers"].notna() & (merged["ec_numbers"] != "")
merged["classification_upper"] = merged["classification"].str.upper()

# ── Unique pocket counts from novel CSVs ──────────────────────────────────────
def _load_unique_counts(csv_path):
    """Return {pdb_id: n_unique_pockets} from a novel/unique CSV."""
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path, dtype=str)
    out = {}
    for _, row in df.iterrows():
        pid = _normalize(str(row.get("pdb_id", "")).strip())
        pockets = str(row.get("pockets", "")).split()
        out[pid] = len(pockets)
    return out


s2p_unique = _load_unique_counts(results_dir / "novel_s2p_pockets.csv")
p2r_unique = _load_unique_counts(results_dir / "p2r_unique_pockets.csv")
print(f"S2P-unique proteins: {len(s2p_unique)}  |  P2R-unique proteins: {len(p2r_unique)}")

# ── Total pocket counts (fast line-counting, no column parsing) ───────────────
def _count_total_pockets(directory):
    """Return {pdb_id: n_pockets} by counting CSV rows in *_predictions.csv."""
    counts = {}
    if not directory.exists():
        return counts
    for f in directory.rglob("*_predictions.csv"):
        pid = _normalize(f.stem[: -len("_predictions")])
        try:
            with open(f, "rb") as fh:
                n = sum(1 for _ in fh) - 1  # subtract header
            if n > 0:
                counts[pid] = n
        except Exception:
            continue
    return counts


print("Counting total pockets per PDB (line-counting)…")
s2p_total = _count_total_pockets(args.s2p_dir)
p2r_total = _count_total_pockets(args.p2r_dir)
print(f"S2P total indexed: {len(s2p_total)}  |  P2R total indexed: {len(p2r_total)}")

# ── Attach pocket counts to per-PDB frame ─────────────────────────────────────
merged["s2p_unique"]  = merged["pdb_id"].map(s2p_unique).fillna(0).astype(int)
merged["s2p_total"]   = merged["pdb_id"].map(s2p_total).fillna(0).astype(int)
merged["p2r_unique"]  = merged["pdb_id"].map(p2r_unique).fillna(0).astype(int)
merged["p2r_total"]   = merged["pdb_id"].map(p2r_total).fillna(0).astype(int)
merged["s2p_shared"]  = (merged["s2p_total"] - merged["s2p_unique"]).clip(lower=0)
merged["p2r_shared"]  = (merged["p2r_total"] - merged["p2r_unique"]).clip(lower=0)

# ── Aggregate case-insensitively ───────────────────────────────────────────────
counts_ci = (
    merged.groupby("classification_upper", sort=False)
    .agg(
        n_proteins          =("pdb_id",     "count"),
        n_with_ec           =("has_ec",      "sum"),
        n_s2p_unique_pockets=("s2p_unique",  "sum"),
        n_s2p_shared_pockets=("s2p_shared",  "sum"),
        n_p2r_unique_pockets=("p2r_unique",  "sum"),
        n_p2r_shared_pockets=("p2r_shared",  "sum"),
    )
    .reset_index()
    .rename(columns={"classification_upper": "classification"})
    .sort_values("n_proteins", ascending=False)
    .reset_index(drop=True)
)
for col in ("n_with_ec", "n_s2p_unique_pockets", "n_s2p_shared_pockets",
            "n_p2r_unique_pockets", "n_p2r_shared_pockets"):
    counts_ci[col] = counts_ci[col].astype(int)

# ── Save ───────────────────────────────────────────────────────────────────────
args.output_dir.mkdir(parents=True, exist_ok=True)
out_path_ci = args.output_dir / "classification_counts_ci.csv"
counts_ci.to_csv(out_path_ci, index=False)

print(f"\nCategories (case-insensitive): {len(counts_ci)}")
print(f"Total proteins: {counts_ci['n_proteins'].sum()}")
print(f"  → {out_path_ci}")
