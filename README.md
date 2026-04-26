# Protein-Ligand Binding Site Prediction Pipeline

**Integration of sequence and structure-based approaches for protein-ligand binding site prediction**

This repository contains a complete bioinformatics pipeline for predicting protein-ligand binding sites using fine-tuned ESM2 embeddings combined with 3D structural clustering.

## Overview

The pipeline predicts binding sites by:
1. **Extracting sequences** from PDB protein structures
2. **Running ESM2 3B model** pretrained on evolutionary sequence data for binding site classification
3. **Applying smoothing** using local structural context to refine predictions
4. **Clustering pockets** in 3D space using surface points and MeanShift algorithm
5. **Comparing results** against alternative methods (P2Rank) for validation

**Key Innovation:** Combines protein language model embeddings (ESM2) with structure-based 3D clustering to identify spatially coherent binding pockets.

## Features

✅ **Sequence-based prediction** using fine-tuned ESM2 3B model
✅ **Structure-based clustering** using Fibonacci lattice surface points
✅ **Smoothing refinement** using surrounding residue context
✅ **Multi-chain support** for complex protein structures
✅ **Comparative analysis** with P2Rank predictions
✅ **PyMOL visualization** scripts for manual inspection
✅ **Parameter tracking** with timestamped runs to prevent overwrites
✅ **HPC-ready** SLURM integration for cluster processing

## Quick Start

### Requirements

- Python 3.12+
- PyTorch with CUDA support (GPU recommended for predictions)
- ESM2 3B model (≈5GB) 
- PDB structure files

### Installation

```bash
# Clone repository
git clone <repo-url>
cd /git

# Install dependencies
pip install -r requirements.txt

# Download models https://owncloud.cesnet.cz/index.php/s/sAvrV3RiTWKFmc1
mkdir -p data/models
# Place 3B-model.pt and smoother.pt in data/models/
```

### Usage

**Complete pipeline:**
```bash
# Step 1: Extract sequences from PDB files
python src/scripts/01_extract_sequence.py

# Step 2: Run ESM2 predictions (GPU recommended)
python src/scripts/02_predict_residues.py

# Step 3: Cluster pockets with smoothing
python src/scripts/03_cluster_pockets.py --decision-threshold 0.7 --distance-threshold 10

# Step 4: Compare with P2Rank
python src/scripts/04_compare_pockets.py --max-overlap-residues 0
```

**Individual steps:**
```bash
# Filter sequences by length (optional)
python scripts/00_filter_fastas.py

# Clustering with custom parameters, timestamped output
python src/scripts/03_cluster_pockets.py \
  --decision-threshold 0.65 \
  --distance-threshold 12 \
  --timestamp

# Compare with flexible overlap threshold
python src/scripts/04_compare_pockets.py \
  --max-overlap-residues 3 \
  --max-overlap-percent 25 \
  --timestamp
```

## Data Directory Structure

```
data/
├── input/
│   ├── pdb/                     # Input PDB protein structures
│   └── P2Rank/                  # P2Rank predictions for comparison
├── intermediate/
│   ├── fastas/                  # Extracted FASTA sequences (≤1024 residues)
│   ├── fastas_long/             # Filtered long sequences (>1024 residues)
│   ├── predictions/             # ESM2 binding probabilities (per-residue)
│   └── embeddings/              # ESM2 token embeddings (1024-dim)
├── models/
│   ├── 3B-model.pt              # Fine-tuned ESM2 3B model
│   └── smoother.pt              # Smoothing refinement model
└── output/
    ├── Seq2Pockets/          # Final binding site predictions
    └── results/                 # Comparison results and visualizations
```

## Pipeline Description

For detailed documentation, see [PIPELINE.md](PIPELINE.md)

### Step 1: Extract Sequences
**Input:** PDB files
**Output:** FASTA files (one per chain)
**Purpose:** Convert 3D structures to sequence format for model inference

### Step 2: Run Predictions
**Input:** FASTA sequences
**Output:** Binding probabilities + ESM2 embeddings
**Purpose:** Generate initial binding site predictions using fine-tuned ESM2 3B
**Runtime:** ~100-500ms per sequence on GPU

### Step 3: Cluster Pockets
**Input:** Predictions, embeddings, PDB structures
**Output:** Cluster assignments with 3D pocket definitions
**Purpose:** Group predicted residues into spatially coherent binding pockets
**Algorithm:**
- Smoothing: refine predictions using local context
- Clustering: MeanShift on 3D surface points (Fibonacci lattice)
- Voting: propagate labels from points → atoms → residues

**Configurable parameters:**
- `--decision-threshold`: Binding probability cutoff (default: 0.7)
- `--distance-threshold`: Neighbor search distance in Å (default: 10)
- `--timestamp`: Prevent output overwrites (optional)

### Step 4: Compare Pockets
**Input:** Seq2Pockets predictions, P2Rank predictions, PDB structures
**Output:** Novel pockets summary + PyMOL visualization scripts
**Purpose:** Identify differences between methods

**Configurable parameters:**
- `--max-overlap-residues`: Max shared residues for "unique" pocket (default: 0)
- `--max-overlap-percent`: Max overlap % relative to smaller pocket (default: 0)
- `--timestamp`: Prevent output overwrites (optional)

## Key Concepts

### ESM2 Embeddings
The ESM2 3B model generates 1024-dimensional embeddings representing protein sequences at the token level. These capture evolutionary patterns from massive protein databases, providing rich representations for binding site prediction.

### 3D Surface Clustering
Rather than clustering residues directly, the pipeline generates surface points around each predicted binding residue using a Fibonacci lattice sphere. MeanShift clustering on these 3D points identifies spatially coherent regions, which are then mapped back to residues.

**Advantages:**
- Accounts for 3D spatial geometry
- Identifies connected pocket regions
- Robust to coordinate uncertainties
- Natural multi-pocket separation

### Smoothing Refinement
The smoothing model uses ESM2 embeddings and spatial proximity to refine predictions. For each borderline residue (probability < decision_threshold), it queries nearby predicted binding residues and uses a trained neural network to decide inclusion.

**Effect:**
- Fills small gaps in predictions
- Maintains spatial coherence
- Preserves low-confidence but structural valid sites

## Output Examples

### Predictions CSV
```csv
name,rank,score,residue_ids,atom_ids
pocket1,1,0.85,A_101 A_102 A_105,1234 1235 1248
pocket2,2,0.72,A_200 A_201,1456 1457
```

### Residues CSV
```csv
chain_id,residue_id,residue_type,probability,pocket_number
A,101,I,0.95,1
A,102,M,0.88,1
A,200,K,0.75,2
```

### Comparison CSV
```csv
pdb_id,pockets,sizes
1abc,1 3,42 28
2def,2,35
```

## Configuration & Parameters

### Tunable Parameters (defaults)

**Sequence filtering:**
- Maximum sequence length: 1024 residues (ESM2 limitation)

**Prediction thresholding:**
- Decision threshold: 0.7 (P > 0.7 → predicted binding)

**Smoothing phase:**
- Distance threshold: 10 Å (neighbor search)
- Smoothing decision threshold: loaded from model

**Clustering phase:**
- MeanShift bandwidth: 10 Å
- SASA points per atom: ~50
- Probe radius: 1.6 Å

**Comparison phase:**
- Max overlap residues: 0 (strict by default)
- Max overlap percent: 0 (disabled by default)

All configurable parameters can be set via command-line flags. See [PIPELINE.md](PIPELINE.md) for details.

## Utilities

### `utils.py`
Single shared module used by the pipeline:
- `cluster_atoms_by_surface()`: MeanShift clustering of SASA surface points with majority-vote label propagation to atoms
- `aal_prot`: set of standard protein residue names
- `POINTS_DENSITY_PER_ATOM`, `PROBE_RADIUS`: SASA calculation constants
- `CryptoBenchClassifier`: smoothing model architecture (3-layer MLP on concatenated ESM-2 embeddings)
- `SMOOTHING_DECISION_THRESHOLD`: threshold for the smoothing model

## HPC/Cluster Usage

All intensive steps have SBATCH templates:

```bash
# Submit prediction job to SLURM
sbatch src/sbatch/run_02_predict_residues.sh

# Monitor status
squeue -u $USER
sacct -u $USER --format=JobID,JobName,State

# View logs
tail -f slurm-<job_id>.out
```

**SBATCH configuration:**
- GPU partition with 1 GPU per job
- 32-64GB memory
- 2-4 hour time limits

## Performance

| Step | Input | Runtime | Notes |
|------|-------|---------|-------|
| 01 Extract | 1000 PDBs | ~30s | Single-threaded |
| 02 Predict | 1000 sequences | ~5-10 min | GPU dependent |
| 03 Cluster | 1000 proteins | ~5-15 min | CPU-bound |
| 04 Compare | 1000 results | ~2-5 min | Single-threaded |

**Scaling:** Pipeline handles 700k+ files with multi-threaded filtering and parallel job submission.

## Workflow Examples

### Example 1: Standard Run
```bash
# Use defaults, process all proteins
python src/scripts/03_cluster_pockets.py
python src/scripts/04_compare_pockets.py
```

### Example 2: Parameter Sweep
```bash
# Test different decision thresholds with timestamping
for threshold in 0.6 0.65 0.7 0.75 0.8; do
  python src/scripts/03_cluster_pockets.py \
    --decision-threshold $threshold \
    --timestamp
done

# Compare results
ls data/output/Seq2Pockets/
```

### Example 3: Lenient Overlap Matching
```bash
# Allow some flexibility in pocket definitions
python src/scripts/04_compare_pockets.py \
  --max-overlap-residues 5 \
  --max-overlap-percent 30 \
  --timestamp
```

## Model Files

### 3B-model.pt
Fine-tuned ESM2 3B model with three task heads:
- Binding site classifier
- pLDDT score regressor
- Distance regressor

Size: ~2GB
Download: [Provide link or instructions]

### smoother.pt
Refinement model for smoothing phase:
- Concatenates current + surrounding embeddings
- Binary classification (include/exclude residue)

Size: ~50MB
Download: [Provide link or instructions]

## Validation & Comparison

The pipeline includes built-in comparison with P2Rank predictions to:
- Identify method-specific predictions
- Validate pocket definitions
- Assess consistency across approaches
- Generate PyMOL scripts for manual review

**Metrics:**
- Residue overlap (count)
- Overlap percentage (relative to pocket size)
- Pocket size distributions
- Method agreement statistics