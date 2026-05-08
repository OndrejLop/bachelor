# Protein-Ligand Binding Site Prediction Pipeline

**Bachelor's thesis project — integrating sequence- and structure-based binding-site predictors.**

This repository runs two binding-site predictors over the same set of proteins
and compares their results pocket-by-pocket:

- **[Seq2Pocket](https://github.com/skrhakv/seq2pocket) (S2P)** — sequence-based, ESM2-3B fine-tuned classifier +
  smoothing model + 3D MeanShift clustering on Fibonacci-lattice surface points.
- **[P2Rank](https://github.com/rdk/p2rank) (P2R)** — structure-based baseline (external Java tool).

Outputs per-pocket and per-residue predictions for each method, then computes
overlap statistics — which pockets each method finds that the other misses.
A separate analysis tool computes per-pocket SASA and protrusion (heavy-atom
neighbor count) distributions across the resulting pocket categories.

> **Naming convention** — the modern names are `Seq2Pocket` / `S2P` /
> `s2p` and `P2Rank` / `P2R` / `p2r`. The legacy form `CryptoShow` /
> `CS` / `cs` may still appear in older outputs (e.g. `data/output/CS_predictions/`).

## Features

- Structure-based pocket prediction via P2Rank
- Sequence-based cryptic-pocket prediction via fine-tuned ESM2-3B (Seq2Pocket)
- Pocket-level comparison with configurable overlap thresholds and pocket-size bounds
- Per-pocket SASA + protrusion distributions across unique/shared categories
- PyMOL `.pml` visualization scripts and aggregate plots/CSVs

## Quick start

### Requirements

- **Python 3.12** — required by the pinned `torch==2.6.0` in
  `src/utilities/requirements.txt`. Newer Python versions are not tested
  and are not guaranteed to install.
- Java 17–24 (for P2Rank)
- PyTorch with CUDA (GPU strongly recommended for step 2)
- Disk: the fine-tuned ESM2-3B checkpoint and the P2Rank install both
  need to fit alongside your inputs — budget at least 7 GB.

### Supported platforms

The pipeline is developed and run on **Linux** (cluster + workstation).
**Windows is not currently supported** — several scripts assume POSIX
paths and the conda env layout used on the cluster. macOS works for
local development of individual scripts but the end-to-end pipeline has
only been validated on Linux.

### Install

```bash
git clone https://github.com/OndrejLop/Cryptic-S2P-to-P2R-comparison
cd Cryptic-S2P-to-P2R-comparison

pip install -r src/utilities/requirements.txt

# Downloads models, P2Rank, and creates the data/ directory layout
python3 src/scripts/pipeline/0_setup.py
```

Place your PDB structures in `data/input/pdb/`.

> **Testing only — try the pipeline without your own data.** A small set
> of protein PDBs is bundled under `data/input/testing_data/` *for testing
> purposes*. Move (or copy) them into `data/input/pdb/` before running
> step 1:
>
> ```bash
> mv data/input/testing_data/*.pdb data/input/pdb/
> ```
>
> This is not the production input path — for real runs, drop your own
> structures into `data/input/pdb/` directly.

### Run the full pipeline

```bash
# Local — sequential
python3 src/scripts/pipeline/1_extract_sequence.py
python3 src/scripts/pipeline/2_predict_residues.py
python3 src/scripts/pipeline/3_cluster_pockets.py
python3 src/scripts/pipeline/4_compare_pockets.py
python3 src/scripts/pipeline/5_generate_statistics.py

# Cluster (SLURM) — one wrapper per step
sbatch src/sbatch/1.sh
sbatch src/sbatch/2.sh
sbatch src/sbatch/3.sh           # or 3_chunks.sh for SLURM array over chunks
sbatch src/sbatch/4.sh
sbatch src/sbatch/5.sh
```

All scripts compute paths from `__file__`, so the SLURM wrappers `cd` to the
repo root before invoking Python via the project's conda env.

## Pipeline

| Step | Script | Inputs | Outputs |
|---|---|---|---|
| 0 | `0_setup.py` | (none — fresh clone) | `data/models/`, `data/tools/P2Rank/`, optional input dirs |
| 1 | `1_extract_sequence.py` | `data/input/pdb/*.pdb` | `data/intermediate/fastas/{stem}_{chain}.fasta`, `data/intermediate/p2rank_dataset.ds` |
| 2 | `2_predict_residues.py` | fastas + `.ds` | `data/intermediate/predictions/`, `data/intermediate/embeddings/`, `data/input/P2Rank/*.csv` |
| 3 | `3_cluster_pockets.py` | predictions + embeddings + PDBs | `data/output/Seq2Pockets/[subdir/]{pdb_id}_{predictions,residues}.csv` |
| 4 | `4_compare_pockets.py` | S2P + P2R outputs | `data/output/results/[subdir/]{pdb_id}/{p2r,s2p}/unmatched_pockets.csv`, `novel_s2p_pockets.csv`, `p2r_unique_pockets.csv`, `run_metadata.json` |
| 5 | `5_generate_statistics.py` | results + tools 14/15 outputs | `data/output/analysis/summary.txt`, `plots/*.png`, `classification_stats*.csv` |

### Notable per-step parameters

```bash
# Step 3 — clustering
python3 src/scripts/pipeline/3_cluster_pockets.py \
    --decision-threshold 0.65 \
    --distance-threshold 12 \
    --timestamp

# Step 4 — comparison + size filter
python3 src/scripts/pipeline/4_compare_pockets.py \
    --max-overlap-residues 3 \
    --max-overlap-percent 25 \
    --min-pocket-size 3 \
    --max-pocket-size 70 \
    --timestamp

# Step 5 — statistics; size bounds auto-loaded from results/run_metadata.json
python3 src/scripts/pipeline/5_generate_statistics.py \
    --results-dir 20260506_161049_max_res0_pct0.0_size3-70 \
    --exclude-file data/output/analysis/excluded_pdbs.txt
```

The `--min-pocket-size` / `--max-pocket-size` flags (default 3..70 inclusive)
drop pockets outside that residue-count range *before* the comparison, so they
never participate in overlap matching. Step 5 reads the same bounds from
`{results_dir}/run_metadata.json` so its plots stay consistent with what
step 4 actually compared.

## Tools

Utilities for inspecting, post-processing, or analyzing pipeline outputs.
They are not pipeline steps and can be run independently.

| Tool | Purpose |
|---|---|
| `10_gunzip_files.py` | Decompress `.ent.gz` PDBe files into the input dir |
| `11_copy_p2rank_predictions.py` | Stage P2Rank prediction CSVs into `data/input/P2Rank/` |
| `12_audit_outputs.py` | Find missing / duplicated / corrupted prediction pairs |
| `13_summarize_skips.py` | Roll up reasons clustering skipped a PDB |
| `14_classify_pdbs.py` | Fetch RCSB header classification + EC numbers (GraphQL) |
| `15_pipeline_membership.py` | Build per-PDB membership matrix across pipeline steps |
| `16_prankweb_diff.py` | Diff prankweb tarball contents vs membership; extract their CSVs |
| `18_pocket_sasa.py` | Per-pocket SASA + protrusion distributions across unique/shared categories |

## Data layout

```
data/
├── input/
│   ├── pdb/                            input PDB structures
│   ├── testing_data/                   bundled protein PDBs for testing only — move into pdb/ to use
│   └── P2Rank/                         P2Rank prediction CSVs (input to step 4)
├── intermediate/
│   ├── fastas/                         extracted sequences
│   ├── predictions/                    ESM2 per-residue binding probabilities
│   ├── embeddings/                     ESM2 token embeddings
│   ├── p2rank_dataset.ds               list of files for batch P2Rank invocation
│   └── pipeline_membership.csv         per-PDB step-presence matrix (tool 15)
├── models/
│   ├── 3B-model.pt                     fine-tuned ESM2-3B checkpoint
│   └── smoother.pt                     smoothing MLP
├── tools/
│   └── P2Rank/                         local P2Rank install (downloaded by step 0)
└── output/
    ├── Seq2Pockets/                    S2P pocket predictions (flat or chunked subdirs)
    ├── results/                        step 4 comparison output (timestamped subdirs allowed)
    └── analysis/                       step 5 + tool 18 plots, summaries, stats
```

Each leaf `data/` directory has its own `.gitignore` (`*` + keep `.gitignore`)
so the structure is git-tracked but data files are not.

## Conventions

- **PDB ID format** — files in `data/input/pdb/` are conventionally
  `pdb{id}.pdb` (PDBe-style); AFDB-style `AF-XXX-F1-model_v4.pdb` also works.
  Per-PDB result directories are named `pdb{id}/`.
- **Step 3 chunk-mode output** — `Seq2Pockets/{prefix}_{timestamp}_{job_id}_decision{X}_dist{Y}/`,
  where the prefix encodes `--resume-after` / `--stop-before` / `--pdb-list` so
  array-job chunks sort cleanly.
- **Step 4 / 5 timestamped output** — when step 4 is run with `--timestamp`,
  step 5 (and tools 17 and 18) auto-discover the most recent
  `data/output/results/{ts}_*` subdir.
- **Excluded PDBs** — step 5 reads `data/output/analysis/excluded_pdbs.txt`
  (one PDB id per line, `#` comments) and applies the exclusion to most data paths.

## License

Bachelor's thesis project. See repository for any third-party-tool licenses
(P2Rank, ESM, biotite, biopython).
