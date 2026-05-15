#!/usr/bin/env python3
"""
Tool 21.3: Heavy — EC group stats + EC AA composition from residue files.

Part 6 (EC group stats) reads 02_s2p_pockets.csv and 03_p2r_pockets.csv
written by 21_2; no prediction-file re-scan.

Part 7 (EC AA composition) scans all *_residues.csv files — this is the
expensive step (~hours for 10k proteins).

Outputs (--output-dir, default data/intermediate/):
  06_ec_group_stats.csv    — per EC class protein/pocket counts + rates
  07_ec_aa_composition.csv — AA composition per EC class × method × category

Run order: submit after 21_2 has finished (needs 02/03 CSVs for Part 6).
Part 7 is independent of 21_2 but shares the same output-dir for convenience.

Requires: data/intermediate/pdb_classification.csv (tool 14).
Optional:  data/intermediate/pipeline_membership.csv (tool 15, to restrict to
           comparable proteins; falls back to all classified if absent).
"""
import argparse
import json
from collections import Counter
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

parser = argparse.ArgumentParser(description="Tool 21.3 — EC group stats + EC AA composition.")
parser.add_argument("--results-dir", type=Path, default=None,
                    help="Comparison-results dir under data/output/results/ or absolute path.")
parser.add_argument("--s2p-dir", type=Path,
                    default=ROOT / "data" / "output" / "Seq2Pockets",
                    help="Directory containing S2P *_residues.csv files.")
parser.add_argument("--p2r-dir", type=Path,
                    default=ROOT / "data" / "input" / "P2Rank",
                    help="Directory containing P2Rank *_residues.csv files.")
parser.add_argument("--exclude-file", type=Path,
                    default=ROOT / "data" / "output" / "analysis" / "excluded_pdbs.txt")
parser.add_argument("--min-pocket-size", type=int, default=None)
parser.add_argument("--max-pocket-size", type=int, default=None)
parser.add_argument("--output-dir", type=Path,
                    default=ROOT / "data" / "intermediate",
                    help="Must point to the same dir used by 21_2 (reads 02/03 CSVs from here).")
parser.add_argument("--skip-aa", action="store_true",
                    help="Skip Part 7 (EC AA composition) — useful to just regenerate Part 6.")
args = parser.parse_args()

RESULTS_ROOT = ROOT / "data" / "output" / "results"

STANDARD_AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
               "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
_STANDARD_AA_SET = set(STANDARD_AA)

EC_NAMES = {
    "1": "Oxidoreductases", "2": "Transferases", "3": "Hydrolases",
    "4": "Lyases",          "5": "Isomerases",   "6": "Ligases",
    "7": "Translocases",
}


def _load_exclusions(path):
    if path is None or not path.exists():
        return set()
    excluded = set()
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                excluded.add(line)
    return excluded


EXCLUDED_PDBS = _load_exclusions(args.exclude_file)


def _find_novel_dir(results_root):
    if (results_root / "novel_s2p_pockets.csv").exists():
        return results_root
    if not results_root.exists():
        return None
    for d in sorted([x for x in results_root.iterdir()
                     if x.is_dir() and not x.name.lower().startswith("pdb")],
                    reverse=True):
        if (d / "novel_s2p_pockets.csv").exists():
            return d
    return None


def _resolve_results_dir(path):
    if path is None:
        return _find_novel_dir(RESULTS_ROOT)
    candidate = path if path.is_absolute() else RESULTS_ROOT / path
    candidate = candidate.resolve()
    if not candidate.exists():
        parser.error(f"--results-dir does not exist: {candidate}")
    return candidate


SELECTED_RESULTS_DIR = _resolve_results_dir(args.results_dir)


def _resolve_pocket_size_bounds(results_dir):
    meta = {}
    if results_dir is not None:
        meta_path = results_dir / "run_metadata.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                pass
    lo = args.min_pocket_size if args.min_pocket_size is not None else meta.get("min_pocket_size", 3)
    hi = args.max_pocket_size if args.max_pocket_size is not None else meta.get("max_pocket_size", 70)
    return lo, hi


MIN_POCKET_SIZE, MAX_POCKET_SIZE = _resolve_pocket_size_bounds(SELECTED_RESULTS_DIR)

print(f"\n{'='*60}")
print(f"Tool 21.3 — EC Group Stats + EC AA Composition")
print(f"{'='*60}")
print(f"S2P dir:     {args.s2p_dir}")
print(f"P2R dir:     {args.p2r_dir}")
print(f"Results dir: {SELECTED_RESULTS_DIR or '(not found)'}")
print(f"Excluded:    {len(EXCLUDED_PDBS)} PDBs")
print(f"Output dir:  {args.output_dir}  (reads 02/03 CSVs from here)")
print(f"Skip AA:     {args.skip_aa}")
print(f"{'='*60}\n")

args.output_dir.mkdir(parents=True, exist_ok=True)


# ── shared helpers ────────────────────────────────────────────────────────────

def _stem_to_pdb_id(stem, strip_suffix=""):
    if strip_suffix and stem.endswith(strip_suffix):
        return stem[: -len(strip_suffix)]
    return stem


def _normalize_pdb_id(pid):
    p = str(pid).strip().lower()
    return p[3:] if p.startswith("pdb") else p


def files_by_pdb(directory, pattern, strip_suffix=""):
    out = {}
    if not directory.exists():
        return out
    for p in directory.rglob(pattern):
        pid = _stem_to_pdb_id(p.stem, strip_suffix)
        if pid in EXCLUDED_PDBS or pid in out:
            continue
        out[pid] = p
    return out


def _load_novel_csv(filename):
    if SELECTED_RESULTS_DIR is None:
        return None
    path = SELECTED_RESULTS_DIR / filename
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if EXCLUDED_PDBS and "pdb_id" in df.columns:
        df = df[~df["pdb_id"].astype(str).isin(EXCLUDED_PDBS)]
    return df.reset_index(drop=True)


def _load_unique_pocket_sets(filename):
    df = _load_novel_csv(filename)
    if df is None or df.empty:
        return {}
    result = {}
    for _, r in df.iterrows():
        pid = _normalize_pdb_id(str(r["pdb_id"]))
        pockets = {int(p) for p in str(r.get("pockets", "")).split() if p.isdigit()}
        if pockets:
            result[pid] = pockets
    return result


def _load_classification_df():
    path = ROOT / "data" / "intermediate" / "pdb_classification.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str).fillna("")
    df["pdb_id"] = df["pdb_id"].map(_normalize_pdb_id)
    if "ec_numbers" in df.columns:
        df["ec_top"] = df["ec_numbers"].apply(
            lambda s: s.split(";")[0].strip() if s and s.strip() else "")
    else:
        df["ec_top"] = ""
    return df[df["ec_top"] != ""][["pdb_id", "ec_top"]].reset_index(drop=True)


def _load_comparable_set():
    path = ROOT / "data" / "intermediate" / "pipeline_membership.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"pdb_id": str})
    if "s3_s2p" not in df.columns or "s3_p2r" not in df.columns:
        return None
    mask = (df["s3_s2p"].astype(int) == 1) & (df["s3_p2r"].astype(int) == 1)
    return set(df.loc[mask, "pdb_id"].map(_normalize_pdb_id))


# ── Part 6: EC group stats ────────────────────────────────────────────────────

def build_ec_group_stats(classif, comparable, s2p_uniq, p2r_uniq):
    """Reads 02/03 CSVs from output-dir; no prediction-file re-scan."""
    if classif is None:
        print("  [SKIP] pdb_classification.csv not found")
        return None

    if comparable is not None:
        classif = classif[classif["pdb_id"].isin(comparable)].copy()
        print(f"  Restricted to {len(classif)} comparable, classified proteins")
    else:
        print(f"  No pipeline_membership.csv — using all {len(classif)} classified proteins")

    # Read pocket tables from 21_2 output
    def _read_pocket_counts(fname):
        p = args.output_dir / fname
        if not p.exists():
            print(f"  [WARN] {fname} not found in output-dir — pocket totals will be 0")
            return {}
        df = pd.read_csv(p, usecols=["pdb_id"])
        df["_pid"] = df["pdb_id"].map(_normalize_pdb_id)
        return df.groupby("_pid").size().to_dict()

    s2p_counts = _read_pocket_counts("02_s2p_pockets.csv")
    p2r_counts = _read_pocket_counts("03_p2r_pockets.csv")

    rows = []
    for ec_top, grp in classif.groupby("ec_top"):
        pids = set(grp["pdb_id"])
        n = len(pids)
        n_s2p_tot          = sum(s2p_counts.get(p, 0) for p in pids)
        n_p2r_tot          = sum(p2r_counts.get(p, 0) for p in pids)
        n_s2p_uniq_prots   = sum(1 for p in pids if p in s2p_uniq)
        n_p2r_uniq_prots   = sum(1 for p in pids if p in p2r_uniq)
        n_s2p_uniq_pockets = sum(len(s2p_uniq[p]) for p in pids if p in s2p_uniq)
        n_p2r_uniq_pockets = sum(len(p2r_uniq[p]) for p in pids if p in p2r_uniq)
        rows.append({
            "ec_class":              ec_top,
            "ec_name":               EC_NAMES.get(str(ec_top), "Unknown"),
            "n_comparable_proteins": n,
            "n_s2p_pockets_total":   n_s2p_tot,
            "n_p2r_pockets_total":   n_p2r_tot,
            "n_with_s2p_unique":     n_s2p_uniq_prots,
            "n_with_p2r_unique":     n_p2r_uniq_prots,
            "n_s2p_unique_pockets":  n_s2p_uniq_pockets,
            "n_p2r_unique_pockets":  n_p2r_uniq_pockets,
            "s2p_unique_rate":       n_s2p_uniq_prots / n if n else float("nan"),
            "p2r_unique_rate":       n_p2r_uniq_prots / n if n else float("nan"),
            "s2p_novelty_rate":      n_s2p_uniq_pockets / n_s2p_tot if n_s2p_tot else float("nan"),
            "p2r_novelty_rate":      n_p2r_uniq_pockets / n_p2r_tot if n_p2r_tot else float("nan"),
        })

    if not rows:
        print("  [WARN] no EC-classified proteins found")
        return None
    return (pd.DataFrame(rows)
              .sort_values("n_comparable_proteins", ascending=False)
              .reset_index(drop=True))


# ── Part 7: EC AA composition ─────────────────────────────────────────────────

def build_ec_aa_composition(classif, comparable, s2p_uniq, p2r_uniq):
    if classif is None:
        print("  [SKIP] pdb_classification.csv not found")
        return None

    if comparable is not None:
        classif = classif[classif["pdb_id"].isin(comparable)].copy()

    ec_lookup  = dict(zip(classif["pdb_id"], classif["ec_top"]))

    counts: dict[tuple, Counter] = {}

    def _add(ec, method, category, aa):
        key = (ec, method, category)
        if key not in counts:
            counts[key] = Counter()
        counts[key][aa] += 1

    # S2P residues
    s2p_res = files_by_pdb(args.s2p_dir, "*_residues.csv", "_residues")
    total_s2p = len(s2p_res)
    print(f"  Scanning {total_s2p:,} S2P residue files…")
    for i, (raw_pid, path) in enumerate(s2p_res.items(), 1):
        pid = _normalize_pdb_id(raw_pid)
        ec  = ec_lookup.get(pid)
        if not ec:
            continue
        try:
            df = pd.read_csv(path, skipinitialspace=True)
        except Exception:
            continue
        if "residue_type" not in df.columns or "pocket_number" not in df.columns:
            continue
        uniq_set = s2p_uniq.get(pid, set())
        pkt = pd.to_numeric(df["pocket_number"], errors="coerce").fillna(0)
        for aa, p in zip(df["residue_type"].astype(str).str.strip().str.upper(), pkt):
            if aa not in _STANDARD_AA_SET:
                continue
            if p > 0:
                _add(ec, "S2P", "all", aa)
            if int(p) in uniq_set:
                _add(ec, "S2P", "unique", aa)
        if i % 500 == 0:
            print(f"    {i:,}/{total_s2p:,} S2P files processed…")

    # P2Rank residues
    p2r_res = files_by_pdb(args.p2r_dir, "*_residues.csv", "_residues")
    total_p2r = len(p2r_res)
    print(f"  Scanning {total_p2r:,} P2Rank residue files…")
    for i, (raw_pid, path) in enumerate(p2r_res.items(), 1):
        pid = _normalize_pdb_id(raw_pid)
        ec  = ec_lookup.get(pid)
        if not ec:
            continue
        try:
            df = pd.read_csv(path, skipinitialspace=True)
        except Exception:
            continue
        if "residue_name" not in df.columns or "pocket" not in df.columns:
            continue
        uniq_set = p2r_uniq.get(pid, set())
        pkt = pd.to_numeric(df["pocket"], errors="coerce").fillna(0)
        for aa, p in zip(df["residue_name"].astype(str).str.strip().str.upper(), pkt):
            if aa not in _STANDARD_AA_SET:
                continue
            if p > 0:
                _add(ec, "P2Rank", "all", aa)
            if int(p) in uniq_set:
                _add(ec, "P2Rank", "unique", aa)
        if i % 500 == 0:
            print(f"    {i:,}/{total_p2r:,} P2Rank files processed…")

    if not counts:
        print("  [WARN] no EC-classified residues found")
        return None

    rows = []
    for (ec, method, category), counter in counts.items():
        total = sum(counter.values())
        for aa in STANDARD_AA:
            n = counter.get(aa, 0)
            rows.append({
                "ec_class":               ec,
                "ec_name":                EC_NAMES.get(str(ec), "Unknown"),
                "method":                 method,
                "category":               category,
                "amino_acid":             aa,
                "count":                  n,
                "total_binding_residues": total,
                "fraction":               n / total if total else float("nan"),
            })

    return (pd.DataFrame(rows)
              .sort_values(["ec_class", "method", "category", "amino_acid"])
              .reset_index(drop=True))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load shared data once (avoid redundant I/O if both parts run)
    classif    = _load_classification_df()
    comparable = _load_comparable_set()
    s2p_uniq   = _load_unique_pocket_sets("novel_s2p_pockets.csv")
    p2r_uniq   = _load_unique_pocket_sets("p2r_unique_pockets.csv")

    print("Part 6: EC group stats…")
    ec_stats = build_ec_group_stats(classif, comparable, s2p_uniq, p2r_uniq)
    if ec_stats is not None:
        out6 = args.output_dir / "06_ec_group_stats.csv"
        ec_stats.to_csv(out6, index=False)
        print(f"  → {out6}  ({len(ec_stats)} EC classes)\n")
    else:
        print()

    if args.skip_aa:
        print("Part 7: skipped (--skip-aa).")
    else:
        print("Part 7: EC AA composition (heavy)…")
        ec_aa = build_ec_aa_composition(classif, comparable, s2p_uniq, p2r_uniq)
        if ec_aa is not None:
            out7 = args.output_dir / "07_ec_aa_composition.csv"
            ec_aa.to_csv(out7, index=False)
            print(f"  → {out7}  ({len(ec_aa):,} rows)\n")

    print("Done (21.3).")
