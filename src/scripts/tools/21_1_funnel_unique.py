#!/usr/bin/env python3
"""
Tool 21.1: Fast — funnel counts + unique pocket tables.

No prediction/residue file reading; uses only file-listing (rglob for counts)
and the small novel CSVs from step 4.  Runs in minutes even for 10k proteins.

Outputs (--output-dir, default data/intermediate/):
  01_funnel.csv        — dataset funnel counts + skip breakdown + novel pocket aggregates
  04_s2p_unique.csv    — one row per S2P-unique pocket
  05_p2r_unique.csv    — one row per P2R-unique pocket

Run order: independent, can run alongside 21_2.
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

parser = argparse.ArgumentParser(description="Tool 21.1 — funnel counts + unique pocket tables.")
parser.add_argument("--results-dir", type=Path, default=None,
                    help="Comparison-results dir under data/output/results/ or absolute path.")
parser.add_argument("--s2p-dir", type=Path,
                    default=ROOT / "data" / "output" / "Seq2Pockets")
parser.add_argument("--p2r-dir", type=Path,
                    default=ROOT / "data" / "input" / "P2Rank")
parser.add_argument("--exclude-file", type=Path,
                    default=ROOT / "data" / "output" / "analysis" / "excluded_pdbs.txt")
parser.add_argument("--min-pocket-size", type=int, default=None)
parser.add_argument("--max-pocket-size", type=int, default=None)
parser.add_argument("--output-dir", type=Path,
                    default=ROOT / "data" / "intermediate")
args = parser.parse_args()

RESULTS_ROOT = ROOT / "data" / "output" / "results"


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
    if lo > hi:
        parser.error(f"--min-pocket-size ({lo}) must be <= --max-pocket-size ({hi})")
    return lo, hi


MIN_POCKET_SIZE, MAX_POCKET_SIZE = _resolve_pocket_size_bounds(SELECTED_RESULTS_DIR)

print(f"\n{'='*60}")
print(f"Tool 21.1 — Funnel + Unique Pockets")
print(f"{'='*60}")
print(f"S2P dir:     {args.s2p_dir}")
print(f"P2R dir:     {args.p2r_dir}")
print(f"Results dir: {SELECTED_RESULTS_DIR or '(not found)'}")
print(f"Excluded:    {len(EXCLUDED_PDBS)} PDBs")
print(f"Output dir:  {args.output_dir}")
print(f"{'='*60}\n")

args.output_dir.mkdir(parents=True, exist_ok=True)


def _stem_to_pdb_id(stem, strip_suffix=""):
    if strip_suffix and stem.endswith(strip_suffix):
        return stem[: -len(strip_suffix)]
    return stem


def ids_from_dir(directory, pattern, strip_suffix=""):
    if not directory.exists():
        return set()
    return {
        pid for p in directory.rglob(pattern)
        if (pid := _stem_to_pdb_id(p.stem, strip_suffix)) not in EXCLUDED_PDBS
    }


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


def explode_novel_csv(filename):
    df = _load_novel_csv(filename)
    if df is None or df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        pockets = str(r.get("pockets", "")).split()
        sizes   = str(r.get("sizes",   "")).split()
        for pkt, sz in zip(pockets, sizes):
            try:
                rows.append({"pdb_id": r["pdb_id"],
                             "pocket_number": int(pkt),
                             "pocket_size":   int(sz)})
            except ValueError:
                continue
    return rows


def parse_skip_log(skip_path):
    counts = {}
    if skip_path.is_dir():
        files = list(skip_path.rglob("skipped_clustering.txt"))
    elif skip_path.is_file():
        files = [skip_path]
    else:
        return {}
    for f in files:
        with open(f) as fh:
            for line in fh:
                if ":" not in line:
                    continue
                label, _, val = line.partition(":")
                try:
                    counts[label.strip()] = counts.get(label.strip(), 0) + int(val.strip().split()[0])
                except (ValueError, IndexError):
                    continue
    return counts


def _agg(values, prefix, section, rows):
    if not values:
        return
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return
    for metric, val in [
        ("n",      len(arr)),
        ("mean",   float(np.mean(arr))),
        ("median", float(np.median(arr))),
        ("std",    float(np.std(arr))),
        ("min",    float(np.min(arr))),
        ("p25",    float(np.percentile(arr, 25))),
        ("p75",    float(np.percentile(arr, 75))),
        ("max",    float(np.max(arr))),
    ]:
        rows.append({"section": section, "metric": f"{prefix}_{metric}", "value": val})


if __name__ == "__main__":
    rows = []

    # Funnel counts — file listing only, no CSV reading
    print("Counting files…")
    pdb_ids    = ids_from_dir(ROOT / "data" / "input" / "pdb", "*.pdb")
    p2r_ids    = ids_from_dir(args.p2r_dir, "*_predictions.csv", "_predictions")
    s2p_ids    = ids_from_dir(args.s2p_dir, "*_predictions.csv", "_predictions")
    comparable = s2p_ids & p2r_ids

    s2p_novel_df = _load_novel_csv("novel_s2p_pockets.csv")
    p2r_novel_df = _load_novel_csv("p2r_unique_pockets.csv")

    for metric, val in [
        ("n_pdb_files",           len(pdb_ids)),
        ("n_p2r_proteins",        len(p2r_ids)),
        ("n_s2p_proteins",        len(s2p_ids)),
        ("n_comparable",          len(comparable)),
        ("n_s2p_unique_proteins", 0 if s2p_novel_df is None else len(s2p_novel_df)),
        ("n_p2r_unique_proteins", 0 if p2r_novel_df is None else len(p2r_novel_df)),
        ("pocket_size_min",       MIN_POCKET_SIZE),
        ("pocket_size_max",       MAX_POCKET_SIZE),
    ]:
        rows.append({"section": "funnel", "metric": metric, "value": val})

    skip = parse_skip_log(args.s2p_dir)
    for label, count in skip.items():
        rows.append({"section": "funnel_skip",
                     "metric": label.lower().replace(" ", "_").replace("(", "").replace(")", ""),
                     "value": count})

    # Novel pocket aggregate stats
    for label, filename in [("s2p_unique", "novel_s2p_pockets.csv"),
                             ("p2r_unique", "p2r_unique_pockets.csv")]:
        df = _load_novel_csv(filename)
        if df is None or df.empty:
            rows.append({"section": label, "metric": "available", "value": 0})
            continue
        rows.append({"section": label, "metric": "available",      "value": 1})
        rows.append({"section": label, "metric": "n_proteins",     "value": len(df)})
        all_pockets = [p for row in df["pockets"].fillna("").astype(str)
                       for p in row.split() if p]
        all_sizes   = [int(s) for row in df["sizes"].fillna("").astype(str)
                       for s in row.split() if s.isdigit()]
        rows.append({"section": label, "metric": "n_pockets_total", "value": len(all_pockets)})
        _agg(all_sizes, "pocket_size", label, rows)

    out1 = args.output_dir / "01_funnel.csv"
    pd.DataFrame(rows)[["section", "metric", "value"]].to_csv(out1, index=False)
    print(f"  → {out1}  ({len(rows)} rows)")

    # Unique pocket tables (just reading two small CSVs)
    for filename, outname, label in [
        ("novel_s2p_pockets.csv", "04_s2p_unique.csv", "S2P-unique"),
        ("p2r_unique_pockets.csv", "05_p2r_unique.csv", "P2R-unique"),
    ]:
        recs = explode_novel_csv(filename)
        out = args.output_dir / outname
        if recs:
            pd.DataFrame(recs).to_csv(out, index=False)
        else:
            pd.DataFrame(columns=["pdb_id", "pocket_number", "pocket_size"]).to_csv(out, index=False)
            print(f"    [WARN] no {label} pockets found")
        print(f"  → {out}  ({len(recs)} rows)")

    print("\nDone (21.1).")
