#!/usr/bin/env python3
"""
Tool 21: Export pipeline statistics as comprehensive CSV tables.

Reads the same inputs as pipeline step 5 (generate_statistics.py) but
produces seven flat CSV files instead of plots, so results can be re-plotted
or inspected without re-running the full pipeline.

Output files (written to --output-dir, default data/intermediate/):
  01_overall_stats.csv    — dataset funnel + per-method aggregate statistics
  02_s2p_pockets.csv      — one row per Seq2Pocket pocket (all proteins)
  03_p2r_pockets.csv      — one row per P2Rank pocket (all proteins)
  04_s2p_unique.csv       — one row per S2P-unique (novel) pocket
  05_p2r_unique.csv       — one row per P2R-unique pocket
  06_ec_group_stats.csv   — per EC class protein/pocket counts (requires tool 14 + 15)
  07_ec_aa_composition.csv — AA composition per EC class × method × category (all/unique)

Input:
  --s2p-dir      data/output/Seq2Pockets/          (S2P _predictions.csv files)
  --p2r-dir      data/input/P2Rank/                (P2Rank _predictions.csv files)
  --results-dir  data/output/results/<run>/        (novel_s2p_pockets.csv etc.)
  --exclude-file data/output/analysis/excluded_pdbs.txt
  --min-pocket-size / --max-pocket-size            (defaults from run_metadata.json)
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent

parser = argparse.ArgumentParser(
    description="Export pipeline statistics as five CSV tables (no plots).")
parser.add_argument("--results-dir", type=Path, default=None,
                    help="Comparison-results subdirectory under data/output/results/ "
                         "or an absolute path.  Default: auto-select newest timestamped run.")
parser.add_argument("--s2p-dir", type=Path,
                    default=ROOT / "data" / "output" / "Seq2Pockets",
                    help="Directory containing S2P *_predictions.csv files.")
parser.add_argument("--p2r-dir", type=Path,
                    default=ROOT / "data" / "input" / "P2Rank",
                    help="Directory containing P2Rank *_predictions.csv files.")
parser.add_argument("--exclude-file", type=Path,
                    default=ROOT / "data" / "output" / "analysis" / "excluded_pdbs.txt",
                    help="Text file listing pdb_ids to exclude (one per line, # comments). "
                         "Pass an empty path to disable.")
parser.add_argument("--min-pocket-size", type=int, default=None,
                    help="Drop pockets with fewer than this many residues.")
parser.add_argument("--max-pocket-size", type=int, default=None,
                    help="Drop pockets with more than this many residues.")
parser.add_argument("--output-dir", type=Path,
                    default=ROOT / "data" / "intermediate",
                    help="Directory for the output CSV files.")
args = parser.parse_args()

RESULTS_ROOT = ROOT / "data" / "output" / "results"


# ── exclusions ────────────────────────────────────────────────────────────────

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


# ── results-dir resolution ────────────────────────────────────────────────────

def _find_novel_dir(results_root: Path) -> Path | None:
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


def _resolve_results_dir(path: Path | None) -> Path | None:
    if path is None:
        return _find_novel_dir(RESULTS_ROOT)
    candidate = path if path.is_absolute() else RESULTS_ROOT / path
    candidate = candidate.resolve()
    if not candidate.exists():
        parser.error(f"--results-dir does not exist: {candidate}")
    return candidate


SELECTED_RESULTS_DIR = _resolve_results_dir(args.results_dir)


# ── pocket-size bounds ────────────────────────────────────────────────────────

def _resolve_pocket_size_bounds(results_dir: Path | None) -> tuple[int, int]:
    meta = {}
    if results_dir is not None:
        meta_path = results_dir / "run_metadata.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                print(f"  [WARN] {meta_path} is not valid JSON; ignoring")
    lo = args.min_pocket_size if args.min_pocket_size is not None else meta.get("min_pocket_size", 3)
    hi = args.max_pocket_size if args.max_pocket_size is not None else meta.get("max_pocket_size", 70)
    if lo > hi:
        parser.error(f"--min-pocket-size ({lo}) must be <= --max-pocket-size ({hi})")
    return lo, hi


MIN_POCKET_SIZE, MAX_POCKET_SIZE = _resolve_pocket_size_bounds(SELECTED_RESULTS_DIR)

print(f"\n{'='*60}")
print(f"Tool 21 — Export Statistics Tables")
print(f"{'='*60}")
print(f"S2P dir:       {args.s2p_dir}")
print(f"P2R dir:       {args.p2r_dir}")
print(f"Results dir:   {SELECTED_RESULTS_DIR or '(not found)'}")
print(f"Excluded PDBs: {len(EXCLUDED_PDBS)}")
print(f"Pocket size:   [{MIN_POCKET_SIZE}, {MAX_POCKET_SIZE}]")
print(f"Output dir:    {args.output_dir}")
print(f"{'='*60}\n")

args.output_dir.mkdir(parents=True, exist_ok=True)


# ── file-scanning helpers ─────────────────────────────────────────────────────

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


# ── pocket loading ────────────────────────────────────────────────────────────

def load_pocket_csv(csv_path):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["residue_list"] = df["residue_ids"].apply(
        lambda x: x.strip().split() if pd.notna(x) else []
    )
    df["pocket_size"] = df["residue_list"].apply(len)
    in_range = (df["pocket_size"] >= MIN_POCKET_SIZE) & (df["pocket_size"] <= MAX_POCKET_SIZE)
    return df[in_range].reset_index(drop=True)


def collect_pocket_records(predictions_dir, file_pattern):
    """Return list of per-pocket dicts: pdb_id, pocket_name, rank, pocket_size, pocket_score."""
    records = []
    seen = set()
    for csv_path in sorted(predictions_dir.rglob(file_pattern)):
        pdb_id = _stem_to_pdb_id(csv_path.stem, "_predictions")
        if pdb_id in EXCLUDED_PDBS or pdb_id in seen:
            continue
        seen.add(pdb_id)
        try:
            df = load_pocket_csv(csv_path)
        except Exception as e:
            print(f"  [WARN] could not load {csv_path}: {e}")
            continue
        for _, r in df.iterrows():
            records.append({
                "pdb_id": pdb_id,
                "pocket_name": r.get("name", ""),
                "rank": int(r["rank"]) if "rank" in df.columns and pd.notna(r.get("rank")) else -1,
                "pocket_size": int(r["pocket_size"]),
                "pocket_score": (float(r["score"])
                                 if "score" in df.columns and pd.notna(r.get("score"))
                                 else float("nan")),
            })
    return records


# ── novel/unique pocket loading ───────────────────────────────────────────────

def _load_novel_csv(filename: str) -> pd.DataFrame | None:
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


def explode_novel_csv(filename: str) -> list[dict]:
    """Expand the space-separated pocket/size columns to one row per pocket."""
    df = _load_novel_csv(filename)
    if df is None or df.empty:
        return []
    rows = []
    for _, r in df.iterrows():
        pockets = str(r.get("pockets", "")).split()
        sizes   = str(r.get("sizes", "")).split()
        # pair up; if lengths differ, zip to shorter
        for pkt, sz in zip(pockets, sizes):
            try:
                rows.append({
                    "pdb_id": r["pdb_id"],
                    "pocket_number": int(pkt),
                    "pocket_size": int(sz),
                })
            except ValueError:
                continue
    return rows


# ── skip-log parsing (for overall stats) ─────────────────────────────────────

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


# ── EC / classification constants ────────────────────────────────────────────

STANDARD_AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
               "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
_STANDARD_AA_SET = set(STANDARD_AA)

EC_NAMES = {
    "1": "Oxidoreductases", "2": "Transferases", "3": "Hydrolases",
    "4": "Lyases",          "5": "Isomerases",   "6": "Ligases",
    "7": "Translocases",
}


def _normalize_pdb_id(pid: str) -> str:
    """Strip 'pdb'/'PDB' prefix and lowercase — matches step 5 / tool 14 convention."""
    p = str(pid).strip().lower()
    return p[3:] if p.startswith("pdb") else p


def _load_classification_df() -> pd.DataFrame | None:
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


def _load_comparable_set() -> set[str] | None:
    """Return normalized pdb_ids of proteins comparable in both methods, or None if file absent."""
    path = ROOT / "data" / "intermediate" / "pipeline_membership.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"pdb_id": str})
    if "s3_s2p" not in df.columns or "s3_p2r" not in df.columns:
        return None
    mask = (df["s3_s2p"].astype(int) == 1) & (df["s3_p2r"].astype(int) == 1)
    return set(df.loc[mask, "pdb_id"].map(_normalize_pdb_id))


def _load_unique_pocket_sets(filename: str) -> dict[str, set[int]]:
    """Parse a novel/unique CSV into {normalized_pdb_id: set(pocket_numbers)}."""
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


# ── aggregate stats helper ────────────────────────────────────────────────────

def _agg(values, prefix, section, rows):
    """Append descriptive stats rows for a numeric list to `rows`."""
    if not values:
        return
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return
    for metric, val in [
        ("n", len(arr)),
        ("mean",   float(np.mean(arr))),
        ("median", float(np.median(arr))),
        ("std",    float(np.std(arr))),
        ("min",    float(np.min(arr))),
        ("p25",    float(np.percentile(arr, 25))),
        ("p75",    float(np.percentile(arr, 75))),
        ("max",    float(np.max(arr))),
    ]:
        rows.append({"section": section, "metric": f"{prefix}_{metric}", "value": val})


# ============================================================
# Part 1: overall stats
# ============================================================

def build_overall_stats() -> pd.DataFrame:
    rows = []

    # funnel counts
    pdb_ids  = ids_from_dir(ROOT / "data" / "input" / "pdb", "*.pdb")
    p2r_ids  = ids_from_dir(args.p2r_dir, "*_predictions.csv", "_predictions")
    s2p_ids  = ids_from_dir(args.s2p_dir, "*_predictions.csv", "_predictions")
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

    # clustering skip breakdown
    skip = parse_skip_log(args.s2p_dir)
    for label, count in skip.items():
        rows.append({"section": "funnel_skip",
                     "metric": label.lower().replace(" ", "_").replace("(", "").replace(")", ""),
                     "value": count})

    # per-method pocket stats
    for method, directory in [("s2p", args.s2p_dir), ("p2r", args.p2r_dir)]:
        print(f"  Collecting {method.upper()} pocket records for overall stats…")
        recs = collect_pocket_records(directory, "*_predictions.csv")
        proteins  = list({r["pdb_id"] for r in recs})
        ppp       = [sum(1 for r in recs if r["pdb_id"] == pid) for pid in proteins]
        sizes     = [r["pocket_size"] for r in recs]
        scores    = [r["pocket_score"] for r in recs if not np.isnan(r["pocket_score"])]

        rows.append({"section": method, "metric": "n_proteins",    "value": len(proteins)})
        rows.append({"section": method, "metric": "n_pockets_total","value": len(recs)})
        _agg(ppp,    "pockets_per_protein", method, rows)
        _agg(sizes,  "pocket_size",         method, rows)
        if scores:
            _agg(scores, "pocket_score",    method, rows)

    # novel pocket aggregate stats
    for label, filename in [("s2p_unique", "novel_s2p_pockets.csv"),
                             ("p2r_unique", "p2r_unique_pockets.csv")]:
        df = _load_novel_csv(filename)
        if df is None or df.empty:
            rows.append({"section": label, "metric": "available", "value": 0})
            continue
        rows.append({"section": label, "metric": "available",        "value": 1})
        rows.append({"section": label, "metric": "n_proteins",        "value": len(df)})
        all_pockets = [p for row in df["pockets"].fillna("").astype(str)
                       for p in row.split() if p]
        all_sizes   = [int(s) for row in df["sizes"].fillna("").astype(str)
                       for s in row.split() if s.isdigit()]
        rows.append({"section": label, "metric": "n_pockets_total", "value": len(all_pockets)})
        _agg(all_sizes, "pocket_size", label, rows)

    return pd.DataFrame(rows)[["section", "metric", "value"]]


# ============================================================
# Part 2 & 3: all S2P / P2R pockets (per-pocket rows)
# ============================================================

def build_method_table(directory, label) -> pd.DataFrame:
    print(f"  Collecting {label} per-pocket rows…")
    recs = collect_pocket_records(directory, "*_predictions.csv")
    if not recs:
        print(f"    [WARN] no {label} pockets found")
        return pd.DataFrame(columns=["pdb_id", "pocket_name", "rank", "pocket_size", "pocket_score"])
    return pd.DataFrame(recs)


# ============================================================
# Part 4 & 5: unique/novel pockets (per-pocket rows)
# ============================================================

def build_unique_table(filename, label) -> pd.DataFrame:
    print(f"  Expanding {label} unique pocket rows…")
    rows = explode_novel_csv(filename)
    if not rows:
        print(f"    [WARN] no {label} unique pockets found (file: {filename})")
        return pd.DataFrame(columns=["pdb_id", "pocket_number", "pocket_size"])
    return pd.DataFrame(rows)


# ============================================================
# Part 6: EC group counts
# ============================================================

def build_ec_group_stats(s2p_df: pd.DataFrame, p2r_df: pd.DataFrame, classif, comparable, s2p_uniq, p2r_uniq) -> pd.DataFrame | None:
    """Per EC top-level class: protein/pocket counts and unique-pocket rates.

    Requires data/intermediate/pdb_classification.csv (tool 14).
    Optionally uses pipeline_membership.csv (tool 15) to restrict to proteins
    present in both methods; falls back to all classified proteins if absent.
    """
    if classif is None:
        print("  [SKIP] pdb_classification.csv not found — skipping EC group stats")
        return None

    if comparable is not None:
        classif = classif[classif["pdb_id"].isin(comparable)].copy()
        print(f"  EC stats restricted to {len(classif)} comparable, classified proteins")
    else:
        print(f"  pipeline_membership.csv absent — using all {len(classif)} classified proteins")

    # Per-pdb_id totals from parts 2/3 (normalize ids to match classification)
    def _pdb_pocket_counts(df):
        if df.empty:
            return {}
        tmp = df.copy()
        tmp["_pid"] = tmp["pdb_id"].map(_normalize_pdb_id)
        return tmp.groupby("_pid").size().to_dict()

    s2p_counts = _pdb_pocket_counts(s2p_df)
    p2r_counts = _pdb_pocket_counts(p2r_df)

    rows = []
    for ec_top, grp in classif.groupby("ec_top"):
        pids = set(grp["pdb_id"])
        n = len(pids)
        n_s2p_tot   = sum(s2p_counts.get(p, 0) for p in pids)
        n_p2r_tot   = sum(p2r_counts.get(p, 0) for p in pids)
        n_s2p_uniq_prots = sum(1 for p in pids if p in s2p_uniq)
        n_p2r_uniq_prots = sum(1 for p in pids if p in p2r_uniq)
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
            "s2p_novelty_rate":      (n_s2p_uniq_pockets / n_s2p_tot
                                      if n_s2p_tot else float("nan")),
            "p2r_novelty_rate":      (n_p2r_uniq_pockets / n_p2r_tot
                                      if n_p2r_tot else float("nan")),
        })

    if not rows:
        print("  [WARN] no EC-classified proteins found after filtering")
        return None
    return (pd.DataFrame(rows)
              .sort_values("n_comparable_proteins", ascending=False)
              .reset_index(drop=True))


# ============================================================
# Part 7: AA composition per EC class × method × category
# ============================================================

def build_ec_aa_composition(classif, comparable, s2p_uniq, p2r_uniq) -> pd.DataFrame | None:
    """AA composition of binding residues broken down by EC class, method (S2P / P2Rank),
    and category (all pockets / unique pockets only).

    Requires pdb_classification.csv.  Scans *_residues.csv files in s2p_dir and p2r_dir.
    """
    from collections import Counter

    if classif is None:
        print("  [SKIP] pdb_classification.csv not found — skipping EC AA composition")
        return None

    if comparable is not None:
        classif = classif[classif["pdb_id"].isin(comparable)].copy()

    ec_lookup = dict(zip(classif["pdb_id"], classif["ec_top"]))  # normalized_id → ec_top

    # counts[(ec_top, method, category)] = Counter({aa: n})
    counts: dict[tuple, Counter] = {}

    def _add(ec, method, category, aa):
        key = (ec, method, category)
        if key not in counts:
            counts[key] = Counter()
        counts[key][aa] += 1

    # ── S2P residues ──────────────────────────────────────────────────────────
    s2p_res = files_by_pdb(args.s2p_dir, "*_residues.csv", "_residues")
    print(f"  Scanning {len(s2p_res):,} S2P residue files for EC AA composition…")
    for raw_pid, path in s2p_res.items():
        pid = _normalize_pdb_id(raw_pid)
        ec = ec_lookup.get(pid)
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

    # ── P2Rank residues ───────────────────────────────────────────────────────
    p2r_res = files_by_pdb(args.p2r_dir, "*_residues.csv", "_residues")
    print(f"  Scanning {len(p2r_res):,} P2Rank residue files for EC AA composition…")
    for raw_pid, path in p2r_res.items():
        pid = _normalize_pdb_id(raw_pid)
        ec = ec_lookup.get(pid)
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

    if not counts:
        print("  [WARN] no EC-classified residues found")
        return None

    rows = []
    for (ec, method, category), counter in counts.items():
        total = sum(counter.values())
        for aa in STANDARD_AA:
            n = counter.get(aa, 0)
            rows.append({
                "ec_class":              ec,
                "ec_name":               EC_NAMES.get(str(ec), "Unknown"),
                "method":                method,
                "category":              category,
                "amino_acid":            aa,
                "count":                 n,
                "total_binding_residues": total,
                "fraction":              n / total if total else float("nan"),
            })

    return (pd.DataFrame(rows)
              .sort_values(["ec_class", "method", "category", "amino_acid"])
              .reset_index(drop=True))


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Part 1: overall statistics…")
    overall = build_overall_stats()
    out1 = args.output_dir / "01_overall_stats.csv"
    overall.to_csv(out1, index=False)
    print(f"  → {out1}  ({len(overall)} rows)\n")

    print("Part 2: all S2P pockets…")
    s2p_all = build_method_table(args.s2p_dir, "S2P")
    out2 = args.output_dir / "02_s2p_pockets.csv"
    s2p_all.to_csv(out2, index=False)
    print(f"  → {out2}  ({len(s2p_all)} rows)\n")

    print("Part 3: all P2Rank pockets…")
    p2r_all = build_method_table(args.p2r_dir, "P2Rank")
    out3 = args.output_dir / "03_p2r_pockets.csv"
    p2r_all.to_csv(out3, index=False)
    print(f"  → {out3}  ({len(p2r_all)} rows)\n")

    print("Part 4: S2P-unique pockets…")
    s2p_unique = build_unique_table("novel_s2p_pockets.csv", "S2P-unique")
    out4 = args.output_dir / "04_s2p_unique.csv"
    s2p_unique.to_csv(out4, index=False)
    print(f"  → {out4}  ({len(s2p_unique)} rows)\n")

    print("Part 5: P2R-unique pockets…")
    p2r_unique = build_unique_table("p2r_unique_pockets.csv", "P2R-unique")
    out5 = args.output_dir / "05_p2r_unique.csv"
    p2r_unique.to_csv(out5, index=False)
    print(f"  → {out5}  ({len(p2r_unique)} rows)\n")

    # Parts 6 & 7 require pdb_classification.csv from tool 14
    # Load once to avoid redundant I/O if both parts run
    classif    = _load_classification_df()
    comparable = _load_comparable_set()
    s2p_uniq   = _load_unique_pocket_sets("novel_s2p_pockets.csv")
    p2r_uniq   = _load_unique_pocket_sets("p2r_unique_pockets.csv")

    print("Part 6: EC group counts…")
    ec_stats = build_ec_group_stats(s2p_all, p2r_all, classif, comparable, s2p_uniq, p2r_uniq)
    if ec_stats is not None:
        out6 = args.output_dir / "06_ec_group_stats.csv"
        ec_stats.to_csv(out6, index=False)
        print(f"  → {out6}  ({len(ec_stats)} EC classes)\n")
    else:
        out6 = None
        print()

    print("Part 7: EC AA composition…")
    ec_aa = build_ec_aa_composition(classif, comparable, s2p_uniq, p2r_uniq)
    if ec_aa is not None:
        out7 = args.output_dir / "07_ec_aa_composition.csv"
        ec_aa.to_csv(out7, index=False)
        print(f"  → {out7}  ({len(ec_aa)} rows)\n")
    else:
        out7 = None
        print()

    print("Done.")
    print(f"  01_overall_stats.csv    — {len(overall)} metric rows")
    print(f"  02_s2p_pockets.csv      — {len(s2p_all)} pockets")
    print(f"  03_p2r_pockets.csv      — {len(p2r_all)} pockets")
    print(f"  04_s2p_unique.csv       — {len(s2p_unique)} pockets")
    print(f"  05_p2r_unique.csv       — {len(p2r_unique)} pockets")
    print(f"  06_ec_group_stats.csv   — {len(ec_stats) if ec_stats is not None else 'skipped'} EC classes")
    print(f"  07_ec_aa_composition.csv — {len(ec_aa) if ec_aa is not None else 'skipped'} rows")
