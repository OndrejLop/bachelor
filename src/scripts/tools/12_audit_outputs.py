#!/usr/bin/env python3
"""
Audit CS_predictions output: missing PDBs, duplicates across run dirs, corrupted pairs.

A PDB is OK in a run dir if both {pdb_id}_predictions.csv and {pdb_id}_residues.csv
exist and are non-empty. Anything else is corrupted.

Use --delete-corrupted to remove partial/zero-byte pairs.
"""
import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent
INPUT_DIR = ROOT / 'data' / 'input' / 'pdb'
OUTPUT_DIR = ROOT / 'data' / 'output' / 'CS_predictions'
SEQ2POCKETS_DIR = ROOT / 'data' / 'output' / 'Seq2Pockets'

PAIR_SUFFIXES = ('_predictions.csv', '_residues.csv')


def input_pdb_ids():
    return sorted({p.stem for p in INPUT_DIR.glob('*.pdb')})


def scan_run_dir(run_dir: Path):
    """Return (ok_ids, corrupted_ids) for this run dir."""
    by_id = defaultdict(dict)
    for f in run_dir.iterdir():
        if not f.is_file():
            continue
        for suf in PAIR_SUFFIXES:
            if f.name.endswith(suf):
                pdb_id = f.name[: -len(suf)]
                by_id[pdb_id][suf] = f
                break

    ok, corrupted = [], []
    for pdb_id, files in by_id.items():
        has_both = len(files) == 2
        non_empty = all(f.stat().st_size > 0 for f in files.values())
        if has_both and non_empty:
            ok.append(pdb_id)
        else:
            corrupted.append((pdb_id, files))
    return sorted(ok), corrupted


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--delete-corrupted', action='store_true',
                    help='Remove partial/zero-byte files (default: dry-run)')
    ap.add_argument('--consolidate', action='store_true',
                    help=f'Copy clustered prediction pairs into {SEQ2POCKETS_DIR.name}/ '
                         '(newest run dir wins for duplicates)')
    args = ap.parse_args()

    inputs = set(input_pdb_ids())
    print(f"Input PDBs: {len(inputs)}")

    run_dirs = sorted(d for d in OUTPUT_DIR.iterdir() if d.is_dir())
    print(f"Run dirs:   {len(run_dirs)}\n")

    id_to_dirs = defaultdict(list)   # pdb_id -> [run_dir, ...]
    all_corrupted = []               # (run_dir, pdb_id, files_dict)

    for rd in run_dirs:
        ok, corrupted = scan_run_dir(rd)
        print(f"  {rd.name}: {len(ok)} ok, {len(corrupted)} corrupted")
        for pid in ok:
            id_to_dirs[pid].append(rd)
        for pid, files in corrupted:
            all_corrupted.append((rd, pid, files))

    covered = set(id_to_dirs)
    missing = sorted(inputs - covered)
    extra = sorted(covered - inputs)
    duplicates = {pid: dirs for pid, dirs in id_to_dirs.items() if len(dirs) > 1}

    print(f"\n{'='*40}")
    print(f"COVERAGE")
    print(f"{'='*40}")
    print(f"  Input PDBs:           {len(inputs)}")
    print(f"  Clustered (covered):  {len(covered & inputs)}  ({100*len(covered & inputs)/len(inputs):.1f}%)")
    print(f"  Missing:              {len(missing)}")
    print(f"  Duplicated:           {len(duplicates)}")
    print(f"  Corrupted entries:    {len(all_corrupted)}")

    print(f"\n=== MISSING ({len(missing)}) ===")
    for pid in missing:
        print(f"  {pid}")

    print(f"\n=== DUPLICATES ({len(duplicates)}) ===")
    for pid, dirs in sorted(duplicates.items()):
        print(f"  {pid}: {[d.name for d in dirs]}")

    print(f"\n=== CORRUPTED ({len(all_corrupted)}) ===")
    for rd, pid, files in all_corrupted:
        reasons = []
        if len(files) < 2:
            reasons.append(f"missing {set(PAIR_SUFFIXES) - set(files)}")
        for suf, f in files.items():
            if f.stat().st_size == 0:
                reasons.append(f"{suf} empty")
        print(f"  {rd.name}/{pid}: {', '.join(reasons)}")

    if extra:
        print(f"\n=== EXTRA (in output, not in input) ({len(extra)}) ===")
        for pid in extra:
            print(f"  {pid}")

    if args.delete_corrupted and all_corrupted:
        print(f"\nDeleting {len(all_corrupted)} corrupted entries...")
        for rd, pid, files in all_corrupted:
            for f in files.values():
                print(f"  rm {f}")
                f.unlink()
    elif all_corrupted:
        print("\n(Pass --delete-corrupted to remove them.)")

    if args.consolidate:
        SEQ2POCKETS_DIR.mkdir(parents=True, exist_ok=True)
        copied, skipped = 0, 0
        # run_dirs is sorted ascending — newest (latest timestamp prefix) is last
        for pid, dirs in id_to_dirs.items():
            src_dir = sorted(dirs)[-1]
            for suf in PAIR_SUFFIXES:
                src = src_dir / f"{pid}{suf}"
                dst = SEQ2POCKETS_DIR / f"{pid}{suf}"
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    skipped += 1
                    continue
                shutil.copy2(src, dst)
                copied += 1
        print(f"\nConsolidated → {SEQ2POCKETS_DIR}")
        print(f"  Copied:  {copied}")
        print(f"  Skipped (already up to date): {skipped}")


if __name__ == '__main__':
    main()
