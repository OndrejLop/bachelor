"""
Compare P2Rank and CryptoSite binding pocket predictions.

This script:
1. Loads pocket predictions from P2Rank and CryptoSite
2. Identifies novel pockets (limited overlap with other method)
3. Generates summary statistics (PDB ID, pocket number, pocket size)
4. Writes PyMOL visualization scripts for manual inspection
5. Saves unmatched pocket lists to CSV

Output format for comparison:
  - pdb_id: Protein identifier
  - pockets: Space-separated pocket numbers
  - sizes: Space-separated pocket residue counts

Two output files:
  - novel_cs_pockets.csv: CryptoSite predictions with limited P2Rank overlap
  - p2r_unique_pockets.csv: P2Rank predictions with limited CryptoSite overlap

PyMOL scripts are generated in pdb-specific subdirectories for visualization.
"""
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
import json

parser = argparse.ArgumentParser(description="Compare P2Rank and CryptoSite pocket predictions")
parser.add_argument("--max-overlap-residues", type=int, default=0,
                    help="Maximum overlapping residues to still consider pocket unique (default: 0)")
parser.add_argument("--max-overlap-percent", type=float, default=0,
                    help="Maximum overlap percentage to consider pocket unique, 0-100 (default: 0=disabled)")
parser.add_argument("--timestamp", action="store_true",
                    help="Add timestamp to output directory (prevents overwrites)")
args = parser.parse_args()

def load_pockets(csv_path):
    """
    Load pocket definitions from CSV file.

    Converts residue_ids column (space-separated chain_residue pairs like "A_101 B_45")
    into frozensets for efficient overlap detection.

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
    return df

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
        chain, resi = res.split("_")
        parts.append(f"(chain {chain} and resi {resi})")
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
        source_label (str): Source method label (e.g. 'cs', 'p2r')
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
ROOT         = Path(__file__).parent.parent.parent
p2rank_dir   = ROOT / 'data' / 'input' / 'P2Rank'
cs_dir       = ROOT / 'data' / 'output' / 'CS_predictions'
pdb_dir      = ROOT / 'data' / 'input' / 'pdb'
base_out_dir = ROOT / 'data' / 'output' / 'results'

# --- Create output directory with optional timestamp ---
if args.timestamp:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_suffix = f"max_res{args.max_overlap_residues}_pct{args.max_overlap_percent}"
    out_base_dir = base_out_dir / f"{timestamp}_{param_suffix}"
else:
    out_base_dir = base_out_dir

out_base_dir.mkdir(parents=True, exist_ok=True)

# --- Save run metadata ---
run_metadata = {
    "timestamp": datetime.now().isoformat(),
    "max_overlap_residues": args.max_overlap_residues,
    "max_overlap_percent": args.max_overlap_percent,
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
print(f"Output directory:     {out_base_dir}")
print(f"Metadata saved:       {metadata_path}")
print(f"{'='*60}\n")

# --- PyMOL color palette for visualization ---
PYMOL_COLORS = ["red", "blue", "green", "yellow", "magenta", "cyan", "orange", "violet", "salmon", "limon"]

cs_log_rows  = []
p2r_log_rows = []

for p2r_csv in p2rank_dir.glob("*_predictions.csv"):
    pdb_id = p2r_csv.stem.replace('_predictions', '')
    cs_csv = cs_dir / f'{pdb_id}_predictions.csv'

    if not cs_csv.exists():
        print(f"Missing CryptoSite file for {pdb_id}, skipping.")
        continue

    p2r_df = load_pockets(p2r_csv)
    cs_df  = load_pockets(cs_csv)

    # Unmatched pockets
    p2r_unmatched = find_unmatched(p2r_df, cs_df)
    cs_unmatched  = find_unmatched(cs_df,  p2r_df)

    if p2r_unmatched.empty and cs_unmatched.empty:
        print(f"{pdb_id}: all pockets matched, no output.")
        continue

    save_unmatched(p2r_unmatched, out_base_dir / pdb_id / "p2r")
    save_unmatched(cs_unmatched,  out_base_dir / pdb_id / "cs")
    print(f"{pdb_id}: saved {len(p2r_unmatched)} unmatched P2R and {len(cs_unmatched)} unmatched CS pockets.")

    def pocket_number(name):
        return ''.join(filter(str.isdigit, str(name)))

    if not cs_unmatched.empty:
        cs_log_rows.append({
            "pdb_id":  pdb_id,
            "pockets": " ".join(pocket_number(r["name"]) for _, r in cs_unmatched.iterrows()),
            "sizes":   " ".join(str(len(r["residue_set"])) for _, r in cs_unmatched.iterrows()),
        })
        write_pymol_script(pdb_id, cs_unmatched, pdb_dir,
                           out_base_dir / pdb_id / "cs_novel.pml", "cs")
    if not p2r_unmatched.empty:
        p2r_log_rows.append({
            "pdb_id":  pdb_id,
            "pockets": " ".join(pocket_number(r["name"]) for _, r in p2r_unmatched.iterrows()),
            "sizes":   " ".join(str(len(r["residue_set"])) for _, r in p2r_unmatched.iterrows()),
        })
        write_pymol_script(pdb_id, p2r_unmatched, pdb_dir,
                           out_base_dir / pdb_id / "p2r_novel.pml", "p2r")

out_base_dir.mkdir(parents=True, exist_ok=True)
if cs_log_rows:
    pd.DataFrame(cs_log_rows).to_csv(out_base_dir / "novel_cs_pockets.csv", index=False)
    print(f"\nNovel CS pockets saved -> {out_base_dir / 'novel_cs_pockets.csv'}")
if p2r_log_rows:
    pd.DataFrame(p2r_log_rows).to_csv(out_base_dir / "p2r_unique_pockets.csv", index=False)
    print(f"Novel P2R pockets saved -> {out_base_dir / 'p2r_unique_pockets.csv'}")