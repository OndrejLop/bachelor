"""
Scan data/input/pdb/ and report PDB files that contain no protein residues.
Writes the list to data/intermediate/no_protein_pdbs.txt and prints a summary.
Uses raw line scanning (no BioPython structure parsing) for speed.
"""
import argparse
import multiprocessing as mp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PDB_DIR = ROOT / "data" / "input" / "pdb"
OUT_FILE = ROOT / "data" / "intermediate" / "no_protein_pdbs.txt"

# Standard 20 AA + common modified residues BioPython's is_aa(standard=False) covers
_AA = frozenset({
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'MSE', 'SEC', 'PYL', 'CSO', 'HYP', 'MLY', 'FME', 'CME', 'OCS', 'KCX',
    'HID', 'HIE', 'HIP', 'CYX', 'LYN', 'ASH', 'GLH', 'UNK',
})


def _check(path: Path) -> tuple[str, bool]:
    try:
        with open(path, 'r', errors='ignore') as f:
            for line in f:
                rec = line[:6]
                if rec == 'ATOM  ':
                    # residue name is at columns 18-20 (0-indexed: 17:20)
                    if line[17:20] in _AA:
                        return path.stem, True
                elif rec == 'SEQRES':
                    # tokens after the header are residue names
                    if any(tok in _AA for tok in line[19:].split()):
                        return path.stem, True
    except Exception as e:
        print(f"  [error] {path.name}: {e}", flush=True)
    return path.stem, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", type=Path, default=PDB_DIR)
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    files = sorted(args.pdb_dir.glob("*.pdb"))
    total = len(files)
    print(f"Scanning {total} PDB files with {args.workers} workers ...", flush=True)

    no_protein = []
    done = 0
    with mp.Pool(args.workers) as pool:
        for stem, has_protein in pool.imap_unordered(_check, files, chunksize=128):
            done += 1
            if not has_protein:
                no_protein.append(stem)
            if done % 10000 == 0:
                print(f"  {done}/{total} done, {len(no_protein)} no-protein so far", flush=True)

    no_protein.sort()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(no_protein) + ("\n" if no_protein else ""))

    print(f"\nTotal:      {total}")
    print(f"No protein: {len(no_protein)}")
    print(f"Written to: {args.out}")


if __name__ == "__main__":
    main()
