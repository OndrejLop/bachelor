#!/usr/bin/env python3
"""
Build a per-PDB membership matrix across pipeline steps.

Scans each step's input/output directory and writes a CSV with one row
per PDB ID (union of every ID seen anywhere) and boolean columns for
each step. Useful for tracing attrition — where in the pipeline does a
given PDB drop out?

Columns:
  pdb_id            4-char lowercase PDB ID
  input_pdb         data/input/pdb/pdb{id}.pdb exists
  s1_fasta          any data/intermediate/fastas/pdb{id}_*.fasta
  s2_predictions    any data/intermediate/predictions/pdb{id}_*_predictions.csv
  s3_s2p            any data/output/Seq2Pockets*/**/pdb{id}_predictions.csv
  s3_p2r            data/input/P2Rank/pdb{id}_predictions.csv exists
  s4_compared       data/output/results/pdb{id}/ exists and non-empty
  s5_included       s4_compared AND pdb_id not in exclusion list

Output: data/intermediate/pipeline_membership.csv
"""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
PDB_DIR         = ROOT / 'data' / 'input' / 'pdb'
FASTA_DIR       = ROOT / 'data' / 'intermediate' / 'fastas'
PRED_DIR        = ROOT / 'data' / 'intermediate' / 'predictions'
S2P_BASE        = ROOT / 'data' / 'output'  # scan Seq2Pockets* dirs under here
P2R_DIR         = ROOT / 'data' / 'input' / 'P2Rank'
RESULTS_DIR     = ROOT / 'data' / 'output' / 'results'
DEFAULT_EXCLUDE = ROOT / 'data' / 'input' / 'excluded_pdbs.txt'
DEFAULT_OUT     = ROOT / 'data' / 'intermediate' / 'pipeline_membership.csv'


def strip_prefix(stem: str) -> str:
    return stem[3:].lower() if stem.lower().startswith("pdb") else stem.lower()


def ids_from_glob(pattern_dir: Path, pattern: str, suffix_strip: str = "") -> set[str]:
    """Collect {pdb_id} from filenames matching pattern. suffix_strip removes
    trailing non-id parts (e.g. '_A' from 'pdb1a00_A.fasta' stem)."""
    if not pattern_dir.exists():
        return set()
    out = set()
    for f in pattern_dir.glob(pattern):
        stem = f.stem
        if suffix_strip and "_" in stem:
            stem = stem.rsplit("_", 1)[0] if suffix_strip == "chain" else stem
        if suffix_strip == "twoparts":  # e.g. pdb1a00_A_predictions -> pdb1a00
            stem = stem.rsplit("_", 2)[0]
        if suffix_strip == "preds":  # e.g. pdb1a00_predictions -> pdb1a00
            if stem.endswith("_predictions"):
                stem = stem[: -len("_predictions")]
        out.add(strip_prefix(stem))
    return out


def collect_ids() -> dict[str, dict[str, bool]]:
    # Each collector returns the set of lowercase 4-char IDs present at that step
    input_ids  = {strip_prefix(f.stem) for f in PDB_DIR.glob("*.pdb")} if PDB_DIR.exists() else set()
    fasta_ids  = ids_from_glob(FASTA_DIR, "*.fasta", suffix_strip="chain")
    pred_ids   = ids_from_glob(PRED_DIR, "*_predictions.csv", suffix_strip="twoparts")

    # S2P: scan all Seq2Pockets* dirs under data/output/ (handles timestamped subdirs)
    s2p_ids = set()
    for base in S2P_BASE.glob("Seq2Pockets*"):
        if not base.is_dir():
            continue
        for f in base.rglob("*_predictions.csv"):
            s2p_ids.add(strip_prefix(f.stem.replace("_predictions", "")))

    p2r_ids = ids_from_glob(P2R_DIR, "*_predictions.csv", suffix_strip="preds")

    s4_ids = set()
    if RESULTS_DIR.exists():
        for d in RESULTS_DIR.iterdir():
            if d.is_dir() and any(d.iterdir()):
                s4_ids.add(strip_prefix(d.name))

    all_ids = input_ids | fasta_ids | pred_ids | s2p_ids | p2r_ids | s4_ids

    return {
        "all_ids":     all_ids,
        "input_pdb":   input_ids,
        "s1_fasta":    fasta_ids,
        "s2_predictions": pred_ids,
        "s3_s2p":      s2p_ids,
        "s3_p2r":      p2r_ids,
        "s4_compared": s4_ids,
    }


def load_exclusions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {strip_prefix(line.strip()) for line in f if line.strip() and not line.startswith("#")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUDE)
    args = ap.parse_args()

    sets = collect_ids()
    excluded = load_exclusions(args.exclude)
    all_ids = sorted(sets["all_ids"])

    columns = ["input_pdb", "s1_fasta", "s2_predictions",
               "s3_s2p", "s3_p2r", "s4_compared", "s5_included"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_id", *columns])
        totals = {c: 0 for c in columns}
        for pid in all_ids:
            row = {
                "input_pdb":      pid in sets["input_pdb"],
                "s1_fasta":       pid in sets["s1_fasta"],
                "s2_predictions": pid in sets["s2_predictions"],
                "s3_s2p":         pid in sets["s3_s2p"],
                "s3_p2r":         pid in sets["s3_p2r"],
                "s4_compared":    pid in sets["s4_compared"],
            }
            row["s5_included"] = row["s4_compared"] and pid not in excluded
            writer.writerow([pid] + [int(row[c]) for c in columns])
            for c in columns:
                totals[c] += int(row[c])

    print(f"Wrote {len(all_ids)} rows → {args.output}")
    print(f"Excluded PDBs loaded: {len(excluded)}")
    print(f"\nPer-step counts (attrition funnel):")
    for c in columns:
        print(f"  {c:20s} {totals[c]}")


if __name__ == "__main__":
    main()
