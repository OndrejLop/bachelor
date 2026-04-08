#!/usr/bin/env python3
"""
Extract sequences from PDB files.

This script reads PDB files from data/input/pdb/ and extracts protein sequences
for each chain. Outputs FASTA files to data/intermediate/fastas/.

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

ROOT       = Path(__file__).parent.parent.parent
source_dir = ROOT / 'data' / 'input' / 'pdb'
out_dir    = ROOT / 'data' / 'intermediate' / 'fastas'
out_dir.mkdir(parents=True, exist_ok=True)

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
                            # Unknown residue, skip or use 'X'
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

def main():
    """Main entry point. Searches for PDB files and extracts sequences."""
    pdb_files = []

    # Search for PDB files with .ent and .pdb extensions
    pdb_files.extend(glob.glob(os.path.join(source_dir, '*.ent')))
    pdb_files.extend(glob.glob(os.path.join(source_dir, '*.pdb')))
    
    if not pdb_files:
        print(f"No PDB files found in {source_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(pdb_files)} PDB files to process\n")
    
    for pdb_file in pdb_files:
        print(f"{'='*60}")
        print(f"Processing: {pdb_file}")
        print(f"{'='*60}")
        extract_sequence(pdb_file)

if __name__ == "__main__":
    main()