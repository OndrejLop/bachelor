"""
Compare P2Rank and Seq2Pocket binding pocket predictions.

This script:
1. Loads pocket predictions from P2Rank and Seq2Pocket
2. Identifies novel pockets (limited overlap with other method)
3. Generates summary statistics (PDB ID, pocket number, pocket size)
4. Writes PyMOL visualization scripts for manual inspection
5. Saves unmatched pocket lists to CSV

Output format for comparison:
  - pdb_id: Protein identifier
  - pockets: Space-separated pocket numbers
  - sizes: Space-separated pocket residue counts

Two output files:
  - novel_s2p_pockets.csv: Seq2Pocket predictions with limited P2Rank overlap
  - p2r_unique_pockets.csv: P2Rank predictions with limited Seq2Pocket overlap

PyMOL scripts are generated in pdb-specific subdirectories for visualization.
"""
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
import json

parser = argparse.ArgumentParser(description="Compare P2Rank and Seq2Pocket pocket predictions")
parser.add_argument("--max-overlap-residues", type=int, default=0,
                    help="Maximum overlapping residues to still consider pocket unique (default: 0)")
parser.add_argument("--max-overlap-percent", type=float, default=0,
                    help="Maximum overlap percentage to consider pocket unique, 0-100 (default: 0=disabled)")
parser.add_argument("--min-pocket-size", type=int, default=3,
                    help="Drop pockets with fewer than this many residues before comparison (default: 3)")
parser.add_argument("--max-pocket-size", type=int, default=70,
                    help="Drop pockets with more than this many residues before comparison (default: 70)")
parser.add_argument("--timestamp", action="store_true",
                    help="Add timestamp to output directory (prevents overwrites)")
parser.add_argument("--resume-after", type=str, default=None,
                    help="Skip PDB IDs up to and including this one (resume from next)")
args = parser.parse_args()

if args.min_pocket_size > args.max_pocket_size:
    parser.error(f"--min-pocket-size ({args.min_pocket_size}) must be <= --max-pocket-size ({args.max_pocket_size})")

def load_pockets(csv_path):
    """
    Load pocket definitions from CSV file.

    Converts residue_ids column (space-separated chain_residue pairs like "A_101 B_45")
    into frozensets for efficient overlap detection. Pockets whose residue count
    falls outside [args.min_pocket_size, args.max_pocket_size] are dropped here,
    so they never participate in the overlap comparison.

    Args:
        csv_path (str): Path to predictions CSV

    Returns:
        pd.DataFrame: Columns [name, rank, score, residue_ids, atom_ids, residue_set]
    """
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df["residue_set"] = df["residue_ids"].apply(
        lambda x: frozenset(x.strip().split()) if pd.notna(x) else frozenset()
    )
    sizes = df["residue_set"].map(len)
    in_range = (sizes >= args.min_pocket_size) & (sizes <= args.max_pocket_size)
    return df[in_range].reset_index(drop=True)

def find_unmatched(source_df, target_df):
    """
    Returns rows from source_df that are sufficiently different (unique).

    A pocket is considered unique if its overlap with ALL target pockets
    stays below the threshold criteria:
    - Overlap <= max_overlap_residues AND
    - (max_overlap_percent == 0 OR overlap_percent <= max_overlap_percent)

    Args:
        source_df (pd.DataFrame): Source pockets to check
        target_df (pd.DataFrame): Target pockets to match against

    Returns:
        pd.DataFrame: Unique pockets from source_df
    """
    unmatched = []
    for _, row in source_df.iterrows():
        is_unique = True
        for _, t in target_df.iterrows():
            overlap = row["residue_set"] & t["residue_set"]
            overlap_count = len(overlap)

            # Check if overlap exceeds max residues
            if overlap_count > args.max_overlap_residues:
                # Check percentage threshold (if specified)
                if args.max_overlap_percent > 0:
                    source_size = len(row["residue_set"])
                    target_size = len(t["residue_set"])
                    overlap_percent = (overlap_count / min(source_size, target_size)) * 100 if min(source_size, target_size) > 0 else 0
                    if overlap_percent <= args.max_overlap_percent:
                        continue  # Still unique

                is_unique = False
                break

        if is_unique:
            unmatched.append(row)
    return pd.DataFrame(unmatched) if unmatched else pd.DataFrame()

def save_unmatched(df, out_path):
    """
    Save unmatched pockets to CSV.

    Args:
        df (pd.DataFrame): DataFrame with unmatched pockets
        out_path (Path): Directory where unmatched_pockets.csv will be saved
    """
    if df.empty:
        return
    out_path.mkdir(parents=True, exist_ok=True)
    out = df[["name", "residue_ids"]].copy()
    out.columns = ["pocket", "residue_ids"]
    out.to_csv(out_path / "unmatched_pockets.csv", index=False)

def residue_ids_to_selection(residue_ids_str):
    """
    Convert residue ID string to PyMOL selection.

    Input format: "A_101 B_45"
    Output format: "(chain A and resi 101) or (chain B and resi 45)"

    Args:
        residue_ids_str (str): Space-separated residue identifiers

    Returns:
        str: PyMOL selection expression
    """
    parts = []
    for res in residue_ids_str.strip().split():
        tokens = res.split("_", 1)
        if len(tokens) == 2:
            chain, resi = tokens
            parts.append(f"(chain {chain} and resi {resi})")
        else:
            parts.append(f"(resi {res})")
    return " or ".join(parts)

def write_pymol_script(pdb_id, unmatched_df, pdb_dir, out_path, source_label):
    """
    Generate PyMOL visualization script for pockets.

    Creates a script that loads the protein, colors pockets, and zooms to selection.

    Args:
        pdb_id (str): PDB identifier
        unmatched_df (pd.DataFrame): Pockets to visualize
        pdb_dir (Path): Directory containing PDB files
        out_path (Path): Output script path
        source_label (str): Source method label (e.g. 's2p', 'p2r')
    """
    pdb_files = list(pdb_dir.glob(f"{pdb_id}*"))
    pdb_file  = pdb_files[0] if pdb_files else pdb_dir / f"{pdb_id}.pdb"
    lines = [
        f"load {pdb_file}, {pdb_id}",
        "hide everything",
        "show cartoon, all",
        "color grey80, all",
        "",
    ]
    sel_names = []
    for i, (_, row) in enumerate(unmatched_df.iterrows()):
        num   = ''.join(filter(str.isdigit, str(row["name"])))
        sname = f"{source_label}_pocket{num}"
        color = PYMOL_COLORS[i % len(PYMOL_COLORS)]
        sel   = residue_ids_to_selection(str(row["residue_ids"]))
        lines.append(f"select {sname}, {pdb_id} and ({sel})")
        lines.append(f"color {color}, {sname}")
        lines.append(f"show sticks, {sname}")
        lines.append("")
        sel_names.append(sname)
    if sel_names:
        all_sel = " or ".join(sel_names)
        lines.append(f"zoom {all_sel}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))

# --- Paths ---
ROOT         = Path(__file__).parent.parent.parent.parent
p2rank_dir   = ROOT / 'data' / 'input' / 'P2Rank'
s2p_dir       = ROOT / 'data' / 'output' / 'Seq2Pockets'
pdb_dir      = ROOT / 'data' / 'input' / 'pdb'
base_out_dir = ROOT / 'data' / 'output' / 'results'

# --- Create output directory with optional timestamp ---
if args.timestamp:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_suffix = (f"max_res{args.max_overlap_residues}"
                    f"_pct{args.max_overlap_percent}"
                    f"_size{args.min_pocket_size}-{args.max_pocket_size}")
    out_base_dir = base_out_dir / f"{timestamp}_{param_suffix}"
else:
    out_base_dir = base_out_dir

out_base_dir.mkdir(parents=True, exist_ok=True)

# --- Save run metadata ---
run_metadata = {
    "timestamp": datetime.now().isoformat(),
    "max_overlap_residues": args.max_overlap_residues,
    "max_overlap_percent": args.max_overlap_percent,
    "min_pocket_size": args.min_pocket_size,
    "max_pocket_size": args.max_pocket_size,
    "output_dir": str(out_base_dir),
}
metadata_path = out_base_dir / "run_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(run_metadata, f, indent=2)

print(f"\n{'='*60}")
print(f"Pocket Comparison Run Parameters:")
print(f"{'='*60}")
print(f"Max overlap residues: {args.max_overlap_residues}")
print(f"Max overlap percent:  {args.max_overlap_percent}%")
print(f"Pocket size bounds:   [{args.min_pocket_size}, {args.max_pocket_size}] residues")
print(f"Output directory:     {out_base_dir}")
print(f"Metadata saved:       {metadata_path}")
print(f"{'='*60}\n")

# --- PyMOL color palette for visualization ---
PYMOL_COLORS = ["red", "blue", "green", "yellow", "magenta", "cyan", "orange", "violet", "salmon", "limon"]

# --- Build lookup of all Seq2Pocket prediction files (top-level + subdirectories) ---
s2p_lookup = {}
for s2p_csv in s2p_dir.rglob("*_predictions.csv"):
    pdb_id = s2p_csv.stem.replace('_predictions', '')
    s2p_lookup[pdb_id] = s2p_csv
print(f"Found {len(s2p_lookup)} Seq2Pocket prediction files across {s2p_dir}")

s2p_log_rows  = []
p2r_log_rows = []

resumed = args.resume_after is None

for p2r_csv in sorted(p2rank_dir.glob("*_predictions.csv")):
    pdb_id = p2r_csv.stem.replace('_predictions', '')

    if not resumed:
        if pdb_id == args.resume_after:
            resumed = True
            print(f"Resuming after {pdb_id}...")
        continue

    if pdb_id not in s2p_lookup:
        print(f"Missing Seq2Pocket file for {pdb_id}, skipping.")
        continue
    s2p_csv = s2p_lookup[pdb_id]

    p2r_df = load_pockets(p2r_csv)
    s2p_df  = load_pockets(s2p_csv)

    # Unmatched pockets
    p2r_unmatched = find_unmatched(p2r_df, s2p_df)
    s2p_unmatched  = find_unmatched(s2p_df,  p2r_df)

    if p2r_unmatched.empty and s2p_unmatched.empty:
        print(f"{pdb_id}: all pockets matched, no output.")
        continue

    save_unmatched(p2r_unmatched, out_base_dir / pdb_id / "p2r")
    save_unmatched(s2p_unmatched,  out_base_dir / pdb_id / "s2p")
    print(f"{pdb_id}: saved {len(p2r_unmatched)} unmatched P2R and {len(s2p_unmatched)} unmatched S2P pockets.")

    def pocket_number(name):
        return ''.join(filter(str.isdigit, str(name)))

    if not s2p_unmatched.empty:
        s2p_log_rows.append({
            "pdb_id":  pdb_id,
            "pockets": " ".join(pocket_number(r["name"]) for _, r in s2p_unmatched.iterrows()),
            "sizes":   " ".join(str(len(r["residue_set"])) for _, r in s2p_unmatched.iterrows()),
        })
        try:
            write_pymol_script(pdb_id, s2p_unmatched, pdb_dir,
                               out_base_dir / pdb_id / "s2p_novel.pml", "s2p")
        except Exception as e:
            print(f"  [WARN] {pdb_id}: failed to write S2P PyMOL script: {e}")
    if not p2r_unmatched.empty:
        p2r_log_rows.append({
            "pdb_id":  pdb_id,
            "pockets": " ".join(pocket_number(r["name"]) for _, r in p2r_unmatched.iterrows()),
            "sizes":   " ".join(str(len(r["residue_set"])) for _, r in p2r_unmatched.iterrows()),
        })
        try:
            write_pymol_script(pdb_id, p2r_unmatched, pdb_dir,
                               out_base_dir / pdb_id / "p2r_novel.pml", "p2r")
        except Exception as e:
            print(f"  [WARN] {pdb_id}: failed to write P2R PyMOL script: {e}")

out_base_dir.mkdir(parents=True, exist_ok=True)
if s2p_log_rows:
    pd.DataFrame(s2p_log_rows).to_csv(out_base_dir / "novel_s2p_pockets.csv", index=False)
    print(f"\nNovel S2P pockets saved -> {out_base_dir / 'novel_s2p_pockets.csv'}")
if p2r_log_rows:
    pd.DataFrame(p2r_log_rows).to_csv(out_base_dir / "p2r_unique_pockets.csv", index=False)
    print(f"Novel P2R pockets saved -> {out_base_dir / 'p2r_unique_pockets.csv'}")