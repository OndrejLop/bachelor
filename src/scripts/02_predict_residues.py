from transformers import AutoTokenizer
import torch
import numpy as np
from transformers import EsmModel
import torch.nn as nn
from pathlib import Path
import sys

ROOT     = Path(__file__).parent.parent.parent
MODEL_PATH = ROOT / 'data' / 'models' / '3B-model.pt'
FASTA_DIR  = ROOT / 'data' / 'intermediate' / 'fastas'
out_dir    = ROOT / 'data' / 'intermediate' / 'predictions'
emb_dir    = ROOT / 'data' / 'intermediate' / 'embeddings'
sys.path.append(str(ROOT / 'src' / 'utilities'))

DROPOUT = 0.3
OUTPUT_SIZE = 1

class MultitaskFinetunedEsmModel(nn.Module):
    def __init__(self, esm_model: str) -> None:
        super().__init__()
        self.llm = EsmModel.from_pretrained(esm_model)
        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(self.llm.config.hidden_size, OUTPUT_SIZE)
        self.plDDT_regressor = nn.Linear(self.llm.config.hidden_size, OUTPUT_SIZE)
        self.distance_regressor = nn.Linear(self.llm.config.hidden_size, OUTPUT_SIZE)

    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        token_embeddings = self.llm(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self.classifier(token_embeddings), self.plDDT_regressor(token_embeddings), self.distance_regressor(token_embeddings), token_embeddings

MODEL_NAME = 'facebook/esm2_t36_3B_UR50D'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 1024

from Bio import SeqIO

finetuned_model = torch.load(MODEL_PATH, weights_only=False)
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

print(f"Found {len(fasta_files)} FASTA file(s). Processing...")

for fasta_file in fasta_files:
    #print(f"Processing file: {fasta_file}")
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)

        # tokenize the sequence
        tokenized_sequences = tokenizer(
            sequence, max_length=MAX_LENGTH, padding='max_length', truncation=True
        )
        tokenized_sequences = {k: torch.tensor([v]).to(device) for k, v in tokenized_sequences.items()}

        # predict (no grad)
        with torch.no_grad():
            output, _, _, embeddings = finetuned_model(tokenized_sequences)
        output = output.flatten()

        mask = (tokenized_sequences['attention_mask'] == 1).flatten()
        preds = torch.round(torch.sigmoid(output[mask][1:-1]) * 1000) / 1000
        preds = preds.detach().cpu().numpy()

        # build an output filename using fasta file stem and record id
        fasta_stem = Path(fasta_file).stem
        record_id = record.id.replace("/", "_")
        out_path = out_dir / f"{fasta_stem}_predictions.csv"

        # save predictions as CSV
        np.savetxt(out_path, preds, fmt="%.3f", delimiter=',')

        # save per-residue embeddings (excluding special tokens) as .npy
        emb = embeddings[0][mask][1:-1].detach().cpu().numpy()
        np.save(emb_dir / f"{fasta_stem}_embeddings.npy", emb)
        #print(f"Saved predictions for record '{record.id}' to: {out_path}")