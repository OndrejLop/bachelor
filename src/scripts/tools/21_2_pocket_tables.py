#!/usr/bin/env python3
"""
Tool 21.2: Medium — all-pockets tables + per-method aggregate stats.

Scans all *_predictions.csv files for S2P and P2Rank.  Each method is scanned
ONCE; the same pass produces both the per-pocket table and the aggregate stats
row (no redundant double-scan as in the monolithic 21.py).

Outputs (--output-dir, default data/intermediate/):
  02_s2p_pockets.csv   — one row per S2P pocket (pdb_id, rank, size, score)
  03_p2r_pockets.csv   — one row per P2Rank pocket
  01_method_stats.csv  — per-method aggregate stats (section/metric/value)

Run order: independent, can run alongside 21_1.
21_3 reads 02/03 CSVs written here, so submit 21_3 after this finishes.
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

parser = argparse.ArgumentParser(description="Tool 21.2 — pocket tables + method aggregate stats.")
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
print(f"Tool 21.2 — Pocket Tables + Method Stats")
print(f"{'='*60}")
print(f"S2P dir:     {args.s2p_dir}")
print(f"P2R dir:     {args.p2r_dir}")
print(f"Excluded:    {len(EXCLUDED_PDBS)} PDBs")
print(f"Pocket size: [{MIN_POCKET_SIZE}, {MAX_POCKET_SIZE}]")
print(f"Output dir:  {args.output_dir}")
print(f"{'='*60}\n")

args.output_dir.mkdir(parents=True, exist_ok=True)


def _stem_to_pdb_id(stem, strip_suffix=""):
    if strip_suffix and stem.endswith(strip_suffix):
        return stem[: -len(strip_suffix)]
    return stem


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


def ids_from_dir(directory, pattern, strip_suffix=""):
    if not directory.exists():
        return set()
    return {
        pid for p in directory.rglob(pattern)
        if (pid := _stem_to_pdb_id(p.stem, strip_suffix)) not in EXCLUDED_PDBS
    }


def scan_predictions(directory, method_label, comparable: set):
    """Single-pass scan: returns (pocket_records_df, agg_stats_rows).

    pocket_df includes an `is_comparable` column (True when the protein has
    predictions from both methods).  Aggregate stats are computed for all
    proteins AND for comparable-only, written as separate sections.
    """
    records = []
    seen = set()
    n_files = 0

    all_csv = sorted(directory.rglob("*_predictions.csv")) if directory.exists() else []
    total = len(all_csv)
    print(f"  {method_label}: {total:,} prediction files to scan…")

    for csv_path in all_csv:
        pdb_id = _stem_to_pdb_id(csv_path.stem, "_predictions")
        if pdb_id in EXCLUDED_PDBS or pdb_id in seen:
            continue
        seen.add(pdb_id)
        n_files += 1
        is_comp = pdb_id in comparable

        try:
            df = pd.read_csv(csv_path, skipinitialspace=True)
            df.columns = df.columns.str.strip()
            df["residue_list"] = df["residue_ids"].apply(
                lambda x: x.strip().split() if pd.notna(x) else [])
            df["pocket_size"] = df["residue_list"].apply(len)
            df = df[(df["pocket_size"] >= MIN_POCKET_SIZE) &
                    (df["pocket_size"] <= MAX_POCKET_SIZE)]
        except Exception as e:
            print(f"    [WARN] {csv_path.name}: {e}")
            continue

        for r in df.to_dict("records"):
            records.append({
                "pdb_id":         pdb_id,
                "is_comparable":  is_comp,
                "pocket_name":    r.get("name", ""),
                "rank":           int(r["rank"]) if "rank" in r and pd.notna(r.get("rank")) else -1,
                "pocket_size":    int(r["pocket_size"]),
                "pocket_score":   (float(r["score"])
                                   if "score" in r and pd.notna(r.get("score"))
                                   else float("nan")),
            })

        if n_files % 500 == 0:
            print(f"    {n_files:,}/{total:,} processed…")

    n_comp = len({r["pdb_id"] for r in records if r["is_comparable"]})
    print(f"  {method_label}: {n_files:,} proteins total, {n_comp:,} comparable, "
          f"{len(records):,} pockets total")

    pocket_df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["pdb_id", "is_comparable", "pocket_name", "rank", "pocket_size", "pocket_score"])

    section = "s2p" if "s2p" in method_label.lower() else "p2r"
    stat_rows = []

    def _stats_for(df_sub, sec):
        if df_sub.empty:
            return
        ppp    = df_sub.groupby("pdb_id").size().tolist()
        sizes  = df_sub["pocket_size"].tolist()
        scores = df_sub["pocket_score"].dropna().tolist()
        stat_rows.append({"section": sec, "metric": "n_proteins",      "value": len(ppp)})
        stat_rows.append({"section": sec, "metric": "n_pockets_total", "value": len(df_sub)})
        _agg(ppp,    "pockets_per_protein", sec, stat_rows)
        _agg(sizes,  "pocket_size",         sec, stat_rows)
        if scores:
            _agg(scores, "pocket_score",    sec, stat_rows)

    _stats_for(pocket_df,                                    section)
    _stats_for(pocket_df[pocket_df["is_comparable"]], f"{section}_comparable")

    return pocket_df, stat_rows


if __name__ == "__main__":
    # Compute comparable set up front (file listing only — no CSV reads)
    print("Computing comparable set…")
    s2p_ids    = ids_from_dir(args.s2p_dir, "*_predictions.csv", "_predictions")
    p2r_ids    = ids_from_dir(args.p2r_dir, "*_predictions.csv", "_predictions")
    comparable = s2p_ids & p2r_ids
    print(f"  S2P: {len(s2p_ids):,}  P2R: {len(p2r_ids):,}  comparable: {len(comparable):,}\n")

    all_stat_rows = []

    print("Part 2: S2P pocket table…")
    s2p_df, s2p_stats = scan_predictions(args.s2p_dir, "S2P", comparable)
    out2 = args.output_dir / "02_s2p_pockets.csv"
    s2p_df.to_csv(out2, index=False)
    print(f"  → {out2}  ({len(s2p_df):,} rows)\n")
    all_stat_rows.extend(s2p_stats)

    print("Part 3: P2Rank pocket table…")
    p2r_df, p2r_stats = scan_predictions(args.p2r_dir, "P2Rank", comparable)
    out3 = args.output_dir / "03_p2r_pockets.csv"
    p2r_df.to_csv(out3, index=False)
    print(f"  → {out3}  ({len(p2r_df):,} rows)\n")
    all_stat_rows.extend(p2r_stats)

    out_stats = args.output_dir / "01_method_stats.csv"
    pd.DataFrame(all_stat_rows)[["section", "metric", "value"]].to_csv(out_stats, index=False)
    print(f"  → {out_stats}  ({len(all_stat_rows)} rows)")

    print("\nDone (21.2).  Submit 21_3 after this finishes.")
