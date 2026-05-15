#!/usr/bin/env python3
"""
Tool 24 — Find cryptic S2P-unique pockets passing burial thresholds.

Iterates over proteins listed in novel_s2p_pockets.csv from a step-4 results
directory, computes SASA/residue and mean neighbor count (10 Å default) for
each S2P-unique pocket, and collects the first N that pass both thresholds:

  sasa_per_residue  < --max-sasa      (default 80 Å²/residue)
  mean_neighbors    < --max-neighbors (default 33)

Stops as soon as --limit pockets are collected (default 10).

Outputs (--output-dir, default data/intermediate/smallcluster):
  cryptic_pockets.csv    — one row per qualifying pocket with all metrics
  cryptic_residues.csv   — residue list for each qualifying pocket
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_RESULTS  = ROOT / "data" / "output" / "results"
DEFAULT_S2P_DIR  = ROOT / "data" / "output" / "Seq2Pockets"
DEFAULT_PDB_DIR  = ROOT / "data" / "input"  / "pdb"
DEFAULT_OUTPUT   = ROOT / "data" / "intermediate" / "smallcluster"

# ── import helpers from tool 18 (no duplication) ──────────────────────────────
_t18_path = ROOT / "src" / "scripts" / "tools" / "18_pocket_sasa.py"
_spec = importlib.util.spec_from_file_location("tool18", _t18_path)
_t18  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t18)

load_predictions           = _t18.load_predictions
load_unmatched_names       = _t18.load_unmatched_names
find_pdb_file              = _t18.find_pdb_file
load_atom_array            = _t18.load_atom_array
compute_residue_sasa       = _t18.compute_residue_sasa
compute_residue_neighbor_counts = _t18.compute_residue_neighbor_counts
pocket_sasa                = _t18.pocket_sasa
pocket_neighbor_mean       = _t18.pocket_neighbor_mean

# ── args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--results-dir", type=Path,
                    default=DEFAULT_RESULTS / "max_res2_pct20.0_size3-70",
                    help="Step-4 results directory. Absolute or relative to "
                         "data/output/results/. Default: max_res2_pct20.0_size3-70")
parser.add_argument("--s2p-dir",  type=Path, default=DEFAULT_S2P_DIR)
parser.add_argument("--pdb-dir",  type=Path, default=DEFAULT_PDB_DIR)
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
parser.add_argument("--max-sasa", type=float, default=80.0,
                    help="Upper bound on sasa_per_residue (default: 80 Å²/residue)")
parser.add_argument("--max-neighbors", type=float, default=33.0,
                    help="Upper bound on mean neighbor count within --neighbor-radius "
                         "(default: 33)")
parser.add_argument("--neighbor-radius", type=float, default=10.0,
                    help="Radius in Å for protrusion proxy (default: 10)")
parser.add_argument("--probe-radius", type=float, default=1.4,
                    help="Shrake-Rupley probe radius in Å (default: 1.4)")
parser.add_argument("--n-points", type=int, default=100,
                    help="Shrake-Rupley sphere points per atom (default: 100)")
parser.add_argument("--limit", type=int, default=10,
                    help="Stop after this many qualifying pockets (default: 10)")
args = parser.parse_args()

# Resolve results dir (try as subdir of DEFAULT_RESULTS if not absolute/existing)
results_dir = args.results_dir
if not results_dir.exists() and not results_dir.is_absolute():
    results_dir = DEFAULT_RESULTS / results_dir
if not results_dir.exists():
    parser.error(f"--results-dir does not exist: {results_dir}")

novel_csv = results_dir / "novel_s2p_pockets.csv"
if not novel_csv.exists():
    parser.error(f"No novel_s2p_pockets.csv in {results_dir}")

print(f"\n{'='*60}")
print(f"Tool 24 — Cryptic pocket finder")
print(f"{'='*60}")
print(f"Results dir:     {results_dir}")
print(f"S2P dir:         {args.s2p_dir}")
print(f"PDB dir:         {args.pdb_dir}")
print(f"Output dir:      {args.output_dir}")
print(f"Thresholds:      SASA/residue < {args.max_sasa},  "
      f"mean neighbors ({args.neighbor_radius} Å) < {args.max_neighbors}")
print(f"Collecting up to {args.limit} pockets\n")

# ── Build S2P predictions lookup {pdb_id: path} ───────────────────────────────
s2p_lookup: dict[str, Path] = {}
for f in args.s2p_dir.rglob("*_predictions.csv"):
    pid = f.stem[: -len("_predictions")]
    s2p_lookup.setdefault(pid, f)
print(f"S2P prediction files indexed: {len(s2p_lookup)}")

# ── Read novel CSV ─────────────────────────────────────────────────────────────
novel_df = pd.read_csv(novel_csv, dtype=str)
print(f"Proteins with S2P-unique pockets: {len(novel_df)}\n")

# ── Iterate, compute, filter ───────────────────────────────────────────────────
pocket_rows: list[dict] = []
residue_rows: list[dict] = []
n_no_s2p = n_no_pdb = n_failed = n_checked = 0

for _, row in novel_df.iterrows():
    if len(pocket_rows) >= args.limit:
        break

    pdb_id = str(row["pdb_id"]).strip()

    if pdb_id not in s2p_lookup:
        n_no_s2p += 1
        continue

    pdb_path = find_pdb_file(args.pdb_dir, pdb_id)
    if pdb_path is None:
        n_no_pdb += 1
        continue

    # Unique pocket names for this PDB
    uniq_names = (
        load_unmatched_names(results_dir / f"pdb{pdb_id}" / "s2p" / "unmatched_pockets.csv")
        | load_unmatched_names(results_dir / f"pdb{pdb_id}" / "cs"  / "unmatched_pockets.csv")
        | load_unmatched_names(results_dir / pdb_id / "s2p" / "unmatched_pockets.csv")
        | load_unmatched_names(results_dir / pdb_id / "cs"  / "unmatched_pockets.csv")
    )
    if not uniq_names:
        continue

    all_preds = load_predictions(s2p_lookup[pdb_id])
    uniq_preds = {name: res for name, res in all_preds.items()
                  if name in uniq_names and res}
    if not uniq_preds:
        continue

    try:
        atoms        = load_atom_array(pdb_path)
        sasa_map     = compute_residue_sasa(atoms, args.probe_radius, args.n_points)
        neighbor_map = compute_residue_neighbor_counts(atoms, args.neighbor_radius)
    except Exception as e:
        print(f"  [WARN] {pdb_id}: structure load/compute failed: {e}")
        n_failed += 1
        continue

    for pocket_name, residues in uniq_preds.items():
        if len(pocket_rows) >= args.limit:
            break
        n_checked += 1

        sasa_total, sasa_res, _ = pocket_sasa(residues, sasa_map)
        if sasa_res == 0:
            continue
        sasa_per_res = sasa_total / sasa_res

        nbr_mean, nbr_res = pocket_neighbor_mean(residues, neighbor_map)
        if np.isnan(nbr_mean):
            continue

        if sasa_per_res >= args.max_sasa or nbr_mean >= args.max_neighbors:
            continue

        print(f"  FOUND [{len(pocket_rows)+1}/{args.limit}] "
              f"{pdb_id}  {pocket_name}  "
              f"SASA/res={sasa_per_res:.1f}  neighbors={nbr_mean:.1f}  "
              f"n_res={len(residues)}")

        pocket_rows.append({
            "pdb_id":            pdb_id,
            "pocket_name":       pocket_name,
            "pdb_file":          str(pdb_path),
            "n_residues":        len(residues),
            "sasa_total":        round(sasa_total, 3),
            "sasa_per_residue":  round(sasa_per_res, 3),
            "mean_neighbors":    round(nbr_mean, 3),
            "n_residues_with_ca": nbr_res,
        })
        for chain, resseq in residues:
            residue_rows.append({
                "pdb_id":      pdb_id,
                "pocket_name": pocket_name,
                "chain":       chain,
                "resseq":      resseq,
            })

print(f"\nPockets checked: {n_checked}")
print(f"Qualifying:      {len(pocket_rows)}")
print(f"No S2P CSV:      {n_no_s2p}  |  No PDB file: {n_no_pdb}  |  Failed: {n_failed}")

# ── Save ───────────────────────────────────────────────────────────────────────
args.output_dir.mkdir(parents=True, exist_ok=True)

out_pockets  = args.output_dir / "cryptic_pockets.csv"
out_residues = args.output_dir / "cryptic_residues.csv"

pd.DataFrame(pocket_rows).to_csv(out_pockets, index=False)
pd.DataFrame(residue_rows).to_csv(out_residues, index=False)

print(f"\n  → {out_pockets}  ({len(pocket_rows)} pockets)")
print(f"  → {out_residues}  ({len(residue_rows)} residues)")
print("\nDone (24).")
