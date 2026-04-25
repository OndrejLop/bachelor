#!/usr/bin/env python3
"""
Per-PDB membership matrix across pipeline steps — two modes.

--make (default): scan each step's input/output dir and write the CSV.
--take: read the existing CSV and print PDB IDs matching column filters.

Columns:
  pdb_id            4-char lowercase PDB ID
  input_pdb         data/input/pdb/pdb{id}.pdb exists
  s1_fasta          any data/intermediate/fastas/pdb{id}_*.fasta
  s2_predictions    any data/intermediate/predictions/pdb{id}_*_predictions.csv
  s3_s2p            any data/output/Seq2Pockets*/**/pdb{id}_predictions.csv
  s3_p2r            data/input/P2Rank/pdb{id}_predictions.csv exists
  s4_compared       data/output/results/[timestamp/]pdb{id}/ exists and non-empty
  s5_included       s4_compared AND pdb_id not in exclusion list

Examples:
  # Scan and (re)build the matrix
  python3 15_pipeline_membership.py --make

  # PDBs that have both input and P2Rank outputs
  python3 15_pipeline_membership.py --take --input_pdb --s3_p2r

  # PDBs that reached step 3 (S2P) but dropped out of step 4
  python3 15_pipeline_membership.py --take --s3_s2p --not-s4_compared

CSV path: data/intermediate/pipeline_membership.csv
"""
import argparse
import csv
import sys
from pathlib import Path

COLUMNS = ["input_pdb", "s1_fasta", "s2_predictions",
           "s3_s2p", "s3_p2r", "s4_compared", "s5_included"]

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PDB_DIR         = ROOT / 'data' / 'input' / 'pdb'
FASTA_DIR       = ROOT / 'data' / 'intermediate' / 'fastas'
PRED_DIR        = ROOT / 'data' / 'intermediate' / 'predictions'
S2P_BASE        = ROOT / 'data' / 'output'  # scan Seq2Pockets* dirs under here
P2R_DIR         = ROOT / 'data' / 'input' / 'P2Rank'
RESULTS_DIR     = ROOT / 'data' / 'output' / 'results'
DEFAULT_EXCLUDE = ROOT / 'data' / 'output' / 'analysis' / 'excluded_pdbs.txt'
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
            if not d.is_dir():
                continue
            if d.name.lower().startswith("pdb"):
                # Flat: results/pdb{id}/
                if any(d.iterdir()):
                    s4_ids.add(strip_prefix(d.name))
            else:
                # Timestamped: results/{timestamp}_{params}/pdb{id}/
                for sub in d.iterdir():
                    if sub.is_dir() and sub.name.lower().startswith("pdb") and any(sub.iterdir()):
                        s4_ids.add(strip_prefix(sub.name))

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


def run_make(args):
    sets = collect_ids()
    excluded = load_exclusions(args.exclude)
    all_ids = sorted(sets["all_ids"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["pdb_id", *COLUMNS])
        totals = {c: 0 for c in COLUMNS}
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
            writer.writerow([pid] + [int(row[c]) for c in COLUMNS])
            for c in COLUMNS:
                totals[c] += int(row[c])

    summary_lines = [
        f"Rows: {len(all_ids)}",
        f"Excluded PDBs loaded: {len(excluded)}",
        "",
        "Per-step counts (attrition funnel):",
    ]
    for c in COLUMNS:
        summary_lines.append(f"  {c:20s} {totals[c]}")

    summary_path = args.output.with_suffix(".txt")
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"Wrote {len(all_ids)} rows → {args.output}")
    print(f"Wrote summary → {summary_path}")
    print(f"Excluded PDBs loaded: {len(excluded)}")
    print(f"\nPer-step counts (attrition funnel):")
    for c in COLUMNS:
        print(f"  {c:20s} {totals[c]}")


def run_take(args):
    if not args.output.exists():
        sys.exit(f"Membership CSV not found: {args.output}  (run with --make first)")

    filters = {}
    for col in COLUMNS:
        pos = getattr(args, col)
        neg = getattr(args, f"not_{col}")
        if pos and neg:
            sys.exit(f"Cannot require both {col}=1 and {col}=0")
        if pos:
            filters[col] = "1"
        elif neg:
            filters[col] = "0"

    if not filters:
        sys.exit("--take requires at least one column filter (e.g. --input_pdb, --not-s3_p2r)")

    matching = []
    with open(args.output, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if all(row[col] == val for col, val in filters.items()):
                matching.append(row["pdb_id"])

    filter_desc = " AND ".join(f"{c}={v}" for c, v in filters.items())
    print(f"# Filter: {filter_desc}", file=sys.stderr)
    print(f"# Matches: {len(matching)}", file=sys.stderr)

    if args.take_output:
        # Bare filename (no path separator) → default under data/intermediate/
        out_path = args.take_output
        if out_path.parent == Path("."):
            out_path = DEFAULT_OUT.parent / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(matching) + ("\n" if matching else ""))
        print(f"# Wrote to {out_path}", file=sys.stderr)
    else:
        for pid in matching:
            print(pid)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--make", action="store_true",
                      help="Scan dirs and build the membership CSV (default)")
    mode.add_argument("--take", action="store_true",
                      help="Read existing CSV and print PDBs matching column filters")

    ap.add_argument("--output", type=Path, default=DEFAULT_OUT,
                    help="Path to the membership CSV (read in --take, written in --make)")
    ap.add_argument("--exclude", type=Path, default=DEFAULT_EXCLUDE)
    ap.add_argument("--take-output", type=Path, default=None,
                    help="--take: write PDB list to file instead of stdout")

    for col in COLUMNS:
        ap.add_argument(f"--{col}", action="store_true",
                        help=f"--take: require {col}=1")
        ap.add_argument(f"--not-{col}", action="store_true", dest=f"not_{col}",
                        help=f"--take: require {col}=0")

    args = ap.parse_args()

    if args.take:
        run_take(args)
    else:
        run_make(args)


if __name__ == "__main__":
    main()
