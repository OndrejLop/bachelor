"""
Run ESM2 3B model predictions on protein sequences (all lengths).

This script handles both short (≤1024) and long (>1024 residue) sequences:

Short sequences (≤1024 residues):
- Predicted once with full context

Long sequences (>1024 residues):
- Use sliding window approach (1024-residue windows, 100-residue stride)
- Window 1 [0-1023]: keep first 500 residues
- Window 2 [100-1123]: keep residues 500-599 (100-residue overlap)
- ...repeat with 100-residue stride...
- Final window: keep last 500 residues
- Overlapping regions averaged for smooth predictions

Input: FASTA files with any sequence length from data/intermediate/fastas/
Output: CSV files with binding probabilities (one per sequence)
        NPY files with ESM2 embeddings (one per sequence)
"""
from transformers import AutoTokenizer
import torch
import numpy as np
from transformers import EsmModel
import torch.nn as nn
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

ROOT       = Path(__file__).parent.parent.parent.parent
MODEL_PATH = ROOT / 'data' / 'models' / '3B-model.pt'
FASTA_DIR  = ROOT / 'data' / 'intermediate' / 'fastas'  # Read from main fastas directory
out_dir    = ROOT / 'data' / 'intermediate' / 'predictions'
emb_dir    = ROOT / 'data' / 'intermediate' / 'embeddings'
P2R_DIR    = ROOT / 'data' / 'input' / 'P2Rank'
DS_PATH    = ROOT / 'data' / 'intermediate' / 'p2rank_dataset.ds'
P2RANK_BIN = Path(os.environ.get("P2RANK_BIN", str(ROOT / 'data' / 'tools' / 'P2Rank' / 'prank')))
sys.path.append(str(ROOT / 'src' / 'utilities'))

DROPOUT = 0.3
OUTPUT_SIZE = 1

class MultitaskFinetunedEsmModel(nn.Module):
    """
    Multi-task ESM2 fine-tuned model for protein binding site prediction.

    The model uses ESM2 3B pre-trained embeddings and applies three task-specific
    linear heads for predicting binding sites, pLDDT scores, and distance values.

    Attributes:
        llm: ESM2 3B pre-trained model
        dropout: Dropout layer (p=0.3)
        classifier: Linear head for binding site prediction (1 output)
        plDDT_regressor: Linear head for pLDDT score prediction (1 output)
        distance_regressor: Linear head for distance prediction (1 output)
    """
    def __init__(self, esm_model: str) -> None:
        super().__init__()
        self.llm = EsmModel.from_pretrained(esm_model)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(self.llm.config.hidden_size, OUTPUT_SIZE)
        self.plDDT_regressor = nn.Linear(self.llm.config.hidden_size, OUTPUT_SIZE)
        self.distance_regressor = nn.Linear(self.llm.config.hidden_size, OUTPUT_SIZE)

    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            batch (dict): Dictionary with 'input_ids' and 'attention_mask' tensors

        Returns:
            Tuple[torch.Tensor]: (binding_predictions, pLDDT_predictions, distance_predictions, embeddings)
        """
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        token_embeddings = self.llm(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self.classifier(token_embeddings), self.plDDT_regressor(token_embeddings), self.distance_regressor(token_embeddings), token_embeddings


MODEL_NAME = 'facebook/esm2_t36_3B_UR50D'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 1024
WINDOW_STRIDE = 100
KEEP_START = 500  # Keep first N residues from first window
KEEP_END = 500    # Keep last N residues from last window

from Bio import SeqIO

finetuned_model = torch.load(MODEL_PATH, weights_only=False, map_location='cpu')
finetuned_model.__class__ = MultitaskFinetunedEsmModel
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
finetuned_model = finetuned_model.to(device)
finetuned_model.eval()

out_dir.mkdir(parents=True, exist_ok=True)
emb_dir.mkdir(parents=True, exist_ok=True)

# find files matching pattern
fasta_files = sorted(FASTA_DIR.glob('*.fasta'))
if not fasta_files:
    raise SystemExit(f"No FASTA files found in: {FASTA_DIR}")

print(f"Found {len(fasta_files)} FASTA file(s). Processing (short and long sequences)...\n")


def predict_long_sequence(sequence, tokenizer, model, device):
    """
    Predict binding sites for protein sequences (all lengths).

    DECISION SWITCH based on sequence length:
    - If ≤1024 residues: Predict once with full context
    - If >1024 residues: Use sliding window approach (1024 positions long windows, 100 stride)

    Args:
        sequence (str): Protein sequence (any length)
        tokenizer: ESM2 tokenizer
        model: ESM2 model
        device: torch device

    Returns:
        Tuple of (predictions, embeddings) for full sequence
    """
    seq_len = len(sequence)

    # Get embedding dimension from model config
    embedding_dim = finetuned_model.llm.config.hidden_size  # ESM2 3B = 2560

    # Initialize arrays to accumulate predictions and embeddings
    all_predictions = np.zeros(seq_len)
    all_embeddings = np.zeros((seq_len, embedding_dim), dtype=np.float32)
    prediction_counts = np.zeros(seq_len)  # Track how many times each residue is predicted

    # === DECISION SWITCH ===
    if seq_len <= MAX_LENGTH:
        # SHORT SEQUENCE: Predict once
        window_starts = [0]
        print(f"  Length: {seq_len} residues (≤{MAX_LENGTH}) → Single prediction")
    else:
        # LONG SEQUENCE: Sliding window approach
        print(f"  Length: {seq_len} residues (>{MAX_LENGTH}) → Sliding window")

        window_starts = []
        pos = 0
        # Add windows with stride until the next window wouldn't reach far enough
        while pos + MAX_LENGTH < seq_len:
            window_starts.append(pos)
            pos += WINDOW_STRIDE
        # Add final window positioned to reach the end exactly
        final_window_start = seq_len - MAX_LENGTH
        if not window_starts or final_window_start > window_starts[-1]:
            window_starts.append(final_window_start)

    print(f"  Windows: {len(window_starts)}")

    for w_idx, start_pos in enumerate(window_starts):
        end_pos = min(start_pos + MAX_LENGTH, seq_len)
        window_seq = sequence[start_pos:end_pos]

        # Tokenize window
        tokenized = tokenizer(
            window_seq, max_length=MAX_LENGTH, padding='max_length', truncation=True
        )
        tokenized = {k: torch.tensor([v]).to(device) for k, v in tokenized.items()}

        # Predict
        with torch.no_grad():
            output, _, _, embeddings = model(tokenized)
        output = output.flatten()

        mask = (tokenized['attention_mask'] == 1).flatten()
        preds = torch.round(torch.sigmoid(output[mask][1:-1]) * 1000) / 1000
        preds = preds.detach().cpu().numpy()
        emb = embeddings[0][mask][1:-1].detach().cpu().numpy()

        # Determine which positions to keep from this window
        actual_window_len = end_pos - start_pos
        is_first_window = (w_idx == 0)
        is_last_window = (w_idx == len(window_starts) - 1)

        if seq_len <= MAX_LENGTH:
            # Short sequence: single window covers all
            firm_start = 0
            firm_end   = actual_window_len
            pred_start = 0
            pred_end   = actual_window_len
        elif is_first_window:
            # First window: firm [0, KEEP_START), averaging transition [KEEP_START, KEEP_START+STRIDE)
            firm_start = 0
            firm_end   = min(KEEP_START, actual_window_len)
            pred_start = 0
            pred_end   = min(KEEP_START + WINDOW_STRIDE, actual_window_len)
        elif is_last_window:
            # Last window: averaging transition [..., firm_start), firm [firm_start, end)
            firm_start = max(0, actual_window_len - KEEP_END)
            firm_end   = actual_window_len
            pred_start = max(0, firm_start - WINDOW_STRIDE)
            pred_end   = actual_window_len
        else:
            # Middle windows: transition on both sides of the 100-residue firm region
            firm_start = KEEP_START
            firm_end   = min(KEEP_START + WINDOW_STRIDE, actual_window_len)
            pred_start = max(0, KEEP_START - WINDOW_STRIDE)
            pred_end   = min(KEEP_START + 2 * WINDOW_STRIDE, actual_window_len)

        # Accumulate predictions (including transition zones for averaging)
        for i in range(pred_start, pred_end):
            global_pos = start_pos + i
            if i < len(preds) and global_pos < seq_len:
                all_predictions[global_pos] += preds[i]
                prediction_counts[global_pos] += 1

        # Assign embeddings (same region as predictions — later windows overwrite transition zone)
        # No averaging: embeddings are context-dependent and cannot be meaningfully averaged
        for i in range(pred_start, pred_end):
            global_pos = start_pos + i
            if i < len(emb) and global_pos < seq_len:
                all_embeddings[global_pos] = emb[i]

        print(f"    Window {w_idx + 1}/{len(window_starts)}: seq [{start_pos}-{end_pos-1}], firm [{start_pos+firm_start}-{start_pos+firm_end-1}], pred [{start_pos+pred_start}-{start_pos+pred_end-1}]")

    # Average overlapping predictions (scalars only — embeddings are already assigned)
    for i in range(seq_len):
        if prediction_counts[i] > 0:
            all_predictions[i] /= prediction_counts[i]

    return all_predictions, all_embeddings


for fasta_file in fasta_files:
    print(f"\n{'='*60}")
    print(f"Processing: {fasta_file.name}")
    print(f"{'='*60}")

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)

        preds, embeddings = predict_long_sequence(sequence, tokenizer, finetuned_model, device)

        # Build output filename
        fasta_stem = Path(fasta_file).stem
        record_id = record.id.replace("/", "_")
        out_path = out_dir / f"{fasta_stem}_predictions.csv"

        # Save predictions as CSV
        np.savetxt(out_path, preds, fmt="%.3f", delimiter=',')

        # Save embeddings as NPY
        np.save(emb_dir / f"{fasta_stem}_embeddings.npy", embeddings)

        print(f"  Saved predictions to: {out_path}")
        print(f"  Saved embeddings to: {emb_dir / f'{fasta_stem}_embeddings.npy'}\n")

print(f"\n{'='*60}")
print(f"S2P prediction complete.")
print(f"{'='*60}")


# ---------------- P2Rank batch step ----------------

def _ds_count(ds_path):
    """Count non-header, non-empty lines in a P2Rank .ds file."""
    if not ds_path.exists():
        return 0
    n = 0
    with open(ds_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("HEADER:") or line.startswith("PARAM."):
                continue
            n += 1
    return n


def run_p2rank_batch(ds_path, p2r_dir, prank_bin):
    """Invoke P2Rank once on the dataset and copy outputs into p2r_dir."""
    p2r_dir.mkdir(parents=True, exist_ok=True)
    if not prank_bin.exists():
        print(f"  [P2R] prank binary not found at {prank_bin}")
        print(f"  [P2R] set P2RANK_BIN env var or run step 0 to install P2Rank")
        return
    with tempfile.TemporaryDirectory(prefix="p2r_run_") as tmp:
        tmp = Path(tmp)
        cmd = [str(prank_bin), "predict", str(ds_path), "-o", str(tmp)]
        print(f"  $ {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  [P2R] prank exited with status {e.returncode}; partial outputs (if any) will still be copied")

        # P2Rank writes per-input output files like <input_basename>_predictions.csv
        # somewhere under the output dir. Walk recursively and rename to canonical.
        copied_pred = copied_res = 0
        for pred_src in tmp.rglob("*_predictions.csv"):
            stem = pred_src.stem.replace("_predictions", "")  # e.g. "pdb1abc.pdb" or "pdb1abc"
            pdb_id = Path(stem).stem  # strip ".pdb" if present
            target_pred = p2r_dir / f"{pdb_id}_predictions.csv"
            shutil.copy2(pred_src, target_pred)
            copied_pred += 1
            res_src = pred_src.with_name(pred_src.name.replace("_predictions.csv",
                                                                 "_residues.csv"))
            if res_src.exists():
                shutil.copy2(res_src, p2r_dir / f"{pdb_id}_residues.csv")
                copied_res += 1
        print(f"  [P2R] copied {copied_pred} predictions / {copied_res} residue files -> {p2r_dir}")


print(f"\n{'='*60}")
print(f"P2Rank batch prediction")
print(f"{'='*60}")
n_needing = _ds_count(DS_PATH)
print(f"  Dataset: {DS_PATH}  ({n_needing} PDBs needing prediction)")
if n_needing == 0:
    print(f"  Nothing to do — all PDBs already have P2Rank predictions in {P2R_DIR}")
else:
    run_p2rank_batch(DS_PATH, P2R_DIR, P2RANK_BIN)

print(f"\n{'='*60}")
print(f"All predictions complete.")
print(f"{'='*60}")
