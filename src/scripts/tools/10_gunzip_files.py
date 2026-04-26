#!/usr/bin/env python3
import os
import gzip
import shutil

#gunzip and extract PDB files in /data/download

def gunzip_files():
    # Specify input and output directories here
    input_dir = "/home/lopatkao/bachelor/p2rank/datasets/pdb"
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'data', 'input', 'pdb')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        return
    
    # Find all .gz files recursively (handles subdirectory structure like pdb/1a/pdb1a00.pdb.gz)
    gz_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.gz'):
                gz_files.append(os.path.join(root, f))

    if not gz_files:
        print(f"Warning: No .gz files found in '{input_dir}'.")
        return

    print(f"Found {len(gz_files)} .gz files to process.")

    for input_path in gz_files:
        gz_file = os.path.basename(input_path)
        
        # Create output filename: remove .gz and .ent, add .pdb
        output_filename = gz_file.replace('.gz', '').replace('.ent', '')
        if not output_filename.endswith('.pdb'):
            output_filename = output_filename + '.pdb'
        
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing {gz_file} -> {output_filename}...", end=" ")
        
        try:
            with gzip.open(input_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print("✓ Done")
        except Exception as e:
            print(f"✗ Failed: {e}")

if __name__ == "__main__":
    gunzip_files()
