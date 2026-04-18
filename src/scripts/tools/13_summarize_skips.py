#!/usr/bin/env python3
"""
Summarize clustering skip counts from cluster_*.log files in src/sbatch/.

Two sources per log:
1. Final "Clustering Summary" block (written only if the job finished cleanly)
2. Scattered [SKIP]/[ERROR] lines (counted as fallback for killed jobs)

If the final summary is present, it's used verbatim. Otherwise counts are
derived from scattered markers and flagged as partial.
"""
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent.parent
SBATCH_DIR = ROOT / 'src' / 'sbatch'

HEADER = "Pocket Clustering Run Parameters:"

SUMMARY_KEYS = [
    ("Processed successfully",     "processed_ok"),
    ("Skipped (no PDB file)",      "no_pdb_file"),
    ("Skipped (no binding res)",   "no_binding_residues"),
    ("Skipped (no CA atoms)",      "no_ca_atoms"),
    ("Skipped (residue mismatch)", "residue_count_mismatch"),
    ("Skipped (no surface points)","no_surface_points"),
    ("Skipped (error)",            "error"),
]

SKIP_PATTERNS = {
    "no_ca_atoms":            re.compile(r"\[SKIP\].*no CA atoms found"),
    "residue_count_mismatch": re.compile(r"\[SKIP\].*residues but predictions have"),
    "no_binding_residues":    re.compile(r"\[SKIP\].*no binding residues above threshold"),
    "no_surface_points":      re.compile(r"\[SKIP\].*no surface points found"),
    "error":                  re.compile(r"\[ERROR\]"),
}


def parse_final_summary(text: str) -> dict | None:
    if "Clustering Summary" not in text:
        return None
    counts = {}
    for label, key in SUMMARY_KEYS:
        m = re.search(rf"{re.escape(label)}:\s+(\d+)", text)
        if m:
            counts[key] = int(m.group(1))
    m = re.search(r"Total PDB IDs:\s+(\d+)", text)
    if m:
        counts["total"] = int(m.group(1))
    return counts if counts else None


def count_markers(text: str) -> dict:
    counts = {k: len(p.findall(text)) for k, p in SKIP_PATTERNS.items()}
    counts["processed_attempts"] = len(re.findall(r"^Processing \S+\.\.\.", text, re.MULTILINE))
    return counts


def main():
    all_logs = sorted(SBATCH_DIR.glob("*.log"))
    logs = [p for p in all_logs if HEADER in p.read_text(errors='replace')]
    if not logs:
        print(f"No logs with header '{HEADER}' in {SBATCH_DIR}")
        return

    totals = Counter()
    print(f"{'='*70}")
    print(f"Per-log summary")
    print(f"{'='*70}")

    for log in logs:
        text = log.read_text(errors='replace')
        final = parse_final_summary(text)

        if final:
            print(f"\n[{log.name}] (completed)")
            for _, key in SUMMARY_KEYS:
                val = final.get(key, 0)
                print(f"  {key:25s} {val}")
                totals[key] += val
            if "total" in final:
                totals["total"] += final["total"]
        else:
            markers = count_markers(text)
            print(f"\n[{log.name}] (partial — no final summary, counted markers)")
            print(f"  processed_attempts:       {markers['processed_attempts']}")
            for key in ("no_ca_atoms", "residue_count_mismatch",
                        "no_binding_residues", "no_surface_points", "error"):
                print(f"  {key:25s} {markers[key]}")
                totals[key] += markers[key]
            totals["processed_attempts"] += markers["processed_attempts"]

    print(f"\n{'='*70}")
    print(f"TOTAL across {len(logs)} log(s)")
    print(f"{'='*70}")
    for key in ("total", "processed_ok", "processed_attempts",
                "no_pdb_file", "no_binding_residues", "no_ca_atoms",
                "residue_count_mismatch", "no_surface_points", "error"):
        if totals[key]:
            print(f"  {key:25s} {totals[key]}")


if __name__ == "__main__":
    main()
