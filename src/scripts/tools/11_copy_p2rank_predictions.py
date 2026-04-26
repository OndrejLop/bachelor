"""
Copy P2Rank prediction files to the pipeline input directory.

Source format:  .../predictions/pdbid.ent.gz_predictions.csv
Target format:  .../P2Rank/pdbid_predictions.csv

Input:  /home/lopatkao/bachelor/p2rank/predictions/PDBe-p2rank-2.4-conservation-hmm/predictions/
Output: /home/lopatkao/bachelor/git/data/input/P2Rank/
"""
import os
import shutil
from pathlib import Path

SRC_DIR = Path("/home/lopatkao/bachelor/p2rank/predictions/PDBe-p2rank-2.4-conservation-hmm/predictions")
DST_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'input' / 'P2Rank'

DST_DIR.mkdir(parents=True, exist_ok=True)

csv_files = list(SRC_DIR.glob("*.ent.gz_predictions.csv"))
if not csv_files:
    raise SystemExit(f"No prediction files found in: {SRC_DIR}")

print(f"Found {len(csv_files)} prediction files. Copying...")

ok = 0
failed = 0
for src_path in csv_files:
    # pdbid.ent.gz_predictions.csv -> pdbid_predictions.csv
    new_name = src_path.name.replace(".ent.gz", "")
    dst_path = DST_DIR / new_name
    try:
        shutil.copy2(src_path, dst_path)
        ok += 1
    except Exception as e:
        print(f"  Failed: {src_path.name} -> {e}")
        failed += 1

print(f"\nDone. Copied: {ok}, Failed: {failed}")
print(f"Output: {DST_DIR}")