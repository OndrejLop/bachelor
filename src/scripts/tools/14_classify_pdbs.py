#!/usr/bin/env python3
"""
Fetch RCSB header classification (and EC numbers) for every PDB ID.

Source of PDB IDs: data/input/pdb/*.pdb  (names like pdb1a00.pdb -> "1a00").
Queries the RCSB GraphQL API in batches and writes a CSV mapping each
PDB ID to its header classification string (e.g. "HYDROLASE") and EC
numbers. Idempotent — re-running skips IDs already in the output file.

Output columns: pdb_id, classification, ec_numbers
  - classification: struct_keywords.pdbx_keywords (single uppercase label)
  - ec_numbers: semicolon-separated list of top-level EC classes across
    polymer entities (empty for non-enzymes)
"""
import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PDB_DIR = ROOT / 'data' / 'input' / 'pdb'
OUTPUT_CSV = ROOT / 'data' / 'intermediate' / 'pdb_classification.csv'

RCSB_URL = "https://data.rcsb.org/graphql"
QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct_keywords { pdbx_keywords }
    polymer_entities {
      rcsb_polymer_entity { rcsb_ec_lineage { id } }
    }
  }
}
"""


def collect_pdb_ids(pdb_dir: Path) -> list[str]:
    ids = set()
    for f in pdb_dir.glob("*.pdb"):
        stem = f.stem
        ids.add(stem[3:].upper() if stem.lower().startswith("pdb") else stem.upper())
    return sorted(ids)


def load_existing(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            pid = row.get("pdb_id", "").strip().upper()
            if pid:
                done.add(pid)
    return done


def query_batch(ids: list[str], timeout: int = 60) -> list[dict]:
    payload = json.dumps({"query": QUERY, "variables": {"ids": ids}}).encode()
    req = urllib.request.Request(
        RCSB_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    if "errors" in body:
        raise RuntimeError(f"GraphQL error: {body['errors']}")
    return body["data"]["entries"] or []


def extract_row(entry: dict) -> tuple[str, str, str]:
    pdb_id = entry["rcsb_id"].upper()
    kw = (entry.get("struct_keywords") or {}).get("pdbx_keywords") or ""
    ec_top = set()
    for pe in (entry.get("polymer_entities") or []):
        lineage = (pe.get("rcsb_polymer_entity") or {}).get("rcsb_ec_lineage") or []
        for node in lineage:
            # EC IDs look like "3", "3.1", "3.1.1", "3.1.1.1" — keep top level only
            eid = str(node.get("id") or "")
            if eid and "." not in eid:
                ec_top.add(eid)
    return pdb_id, kw.strip(), ";".join(sorted(ec_top))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=100,
                        help="PDB IDs per GraphQL query (default 100)")
    parser.add_argument("--sleep", type=float, default=0.1,
                        help="Seconds between requests (default 0.1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after fetching N new entries (for testing)")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--retries", type=int, default=3,
                        help="Retries per batch on network error (default 3)")
    args = parser.parse_args()

    if not PDB_DIR.exists():
        print(f"PDB dir not found: {PDB_DIR}")
        sys.exit(1)

    all_ids = collect_pdb_ids(PDB_DIR)
    done = load_existing(args.output)
    todo = [i for i in all_ids if i not in done]
    print(f"Total PDBs: {len(all_ids)}  already classified: {len(done)}  to fetch: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.output.exists()
    fetched = 0
    with open(args.output, "a", newline='') as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["pdb_id", "classification", "ec_numbers"])

        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            entries = None
            for attempt in range(args.retries):
                try:
                    entries = query_batch(batch)
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, RuntimeError) as e:
                    wait = 2 ** attempt
                    print(f"  [retry {attempt+1}/{args.retries}] {e} — sleeping {wait}s")
                    time.sleep(wait)
            if entries is None:
                print(f"  [FAIL] batch starting {batch[0]} after {args.retries} retries — skipping")
                continue

            returned_ids = set()
            for entry in entries:
                pid, kw, ec = extract_row(entry)
                writer.writerow([pid, kw, ec])
                returned_ids.add(pid)
            # RCSB omits obsolete/unknown IDs — record them so we don't retry forever
            for pid in batch:
                if pid not in returned_ids:
                    writer.writerow([pid, "", ""])
            f.flush()
            fetched += len(batch)

            if (i // args.batch_size) % 10 == 0:
                print(f"  progress: {fetched}/{len(todo)}")

            if args.limit and fetched >= args.limit:
                print(f"Hit --limit {args.limit}; stopping.")
                break
            time.sleep(args.sleep)

    print(f"Done. Wrote {fetched} new rows to {args.output}")


if __name__ == "__main__":
    main()
