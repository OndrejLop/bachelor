#!/usr/bin/env python3
"""
Prepare per-PDB inputs for both predictors.

For each PDB file in data/input/pdb/:
  - extract one FASTA per chain -> data/intermediate/fastas/   (input for S2P / step 2)

Then build a P2Rank dataset (.ds) listing every PDB that doesn't already have
a P2Rank prediction in data/input/P2Rank/, written to
data/intermediate/p2rank_dataset.ds (input for step 2's P2Rank invocation).

Standard residues are converted to 1-letter codes. Non-standard residues are
marked with 'X'. Water molecules and ligands are excluded.
"""
import argparse
import os
import sys
import glob
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from pathlib import Path

ROOT       = Path(__file__).parent.parent.parent.parent
source_dir = ROOT / 'data' / 'input' / 'pdb'
out_dir    = ROOT / 'data' / 'intermediate' / 'fastas'
P2R_DIR    = ROOT / 'data' / 'input' / 'P2Rank'
DS_PATH    = ROOT / 'data' / 'intermediate' / 'p2rank_dataset.ds'
out_dir.mkdir(parents=True, exist_ok=True)
DS_PATH.parent.mkdir(parents=True, exist_ok=True)

def extract_sequence(pdb_file):
    parser = PDBParser(QUIET=True)
    try:
        # Get the base filename without extension to use as protein ID
        base_name = os.path.basename(pdb_file)
        protein_id = os.path.splitext(base_name)[0]
        
        structure = parser.get_structure(protein_id, pdb_file)
        
        for model in structure:
            for chain in model:
                sequence = []
                for residue in chain:
                    # Only process amino acids (exclude hetero atoms like water, ligands)
                    if residue.id[0] == ' ':
                        try:
                            # Convert 3-letter code to 1-letter code
                            aa = seq1(residue.get_resname())
                            sequence.append(aa)
                        except KeyError:
                            # Unknown residue, use 'X'
                            sequence.append('X')
                
                if sequence:
                    seq_string = ''.join(sequence)
                    output_filename = f"{protein_id}_{chain.id}.fasta"
                    print(f"Saving {len(sequence)} residues for chain {chain.id} to {output_filename}")
                    
                    with open(out_dir / output_filename, 'w') as f:
                        f.write(f">{protein_id}_{chain.id}\n")
                        f.write(seq_string + '\n')
                else:
                    print(f"No residues found in chain {chain.id}, skipping.")
                    
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}", file=sys.stderr)

def needs_p2rank_prediction(pdb_path):
    """True if data/input/P2Rank/{stem}_predictions.csv is missing."""
    stem = Path(pdb_path).stem  # e.g. "pdb1abc"
    return not (P2R_DIR / f"{stem}_predictions.csv").exists()


def write_p2rank_dataset(pdb_files, ds_path):
    """Write a P2Rank dataset (.ds) listing only PDBs missing P2Rank predictions.
    Returns the number of entries."""
    needing = [p for p in pdb_files if needs_p2rank_prediction(p)]
    with open(ds_path, 'w') as f:
        f.write("HEADER:\n")
        f.write("PARAM.PROTEIN_LOADER_FORCE_BIOJAVA_LIBRARY=true\n\n")
        for p in needing:
            f.write(f"{Path(p).resolve()}\n")
    return len(needing)


def main():
    """Main entry point. Searches for PDB files and extracts sequences."""
    pdb_files = []

    # Search for PDB files with .pdb extensions
    pdb_files.extend(glob.glob(os.path.join(source_dir, '*.pdb')))

    if not pdb_files:
        print(f"No PDB files found in {source_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdb_files)} PDB files to process\n")

    for pdb_file in pdb_files:
        print(f"{'='*60}")
        print(f"Processing: {pdb_file}")
        extract_sequence(pdb_file)

    # Build P2Rank dataset for step 2 (only PDBs without an existing P2R prediction)
    n = write_p2rank_dataset(pdb_files, DS_PATH)
    print(f"\n{'='*60}")
    print(f"P2Rank dataset: {n} of {len(pdb_files)} PDBs need a fresh prediction")
    print(f"  written to {DS_PATH}")

if __name__ == "__main__":
    main()