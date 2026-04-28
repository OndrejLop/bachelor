from transformers import EsmModel
import torch
import torch.nn as nn
import numpy as np
import csv
from datasets import Dataset
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')
torch.manual_seed(42)

DROPOUT = 0.3
OUTPUT_SIZE = 1
MAX_LENGTH = 1024
LABEL_PAD_TOKEN_ID = -100

LIGYSIS_OVERLAPPING_SEQUENCES = {'P28907', 'P40925', 'Q8TDX7', 'P62068', 'P13929', 'P18669', 'P22303', 'P23141', 'P46597', 'P68106', 'Q86WA6', 'P13686', 'P00480', 'O75530', 'P50053', 'Q13303', 'O75317', 'P15328', 'P08236', 'Q9H477', 'P52895', 'P09936', 'P14207', 'P30613', 'P14618', 'P42330', 'P17900', 'Q04828', 'P12004', 'Q8IVL8', 'Q7L266', 'P51857', 'Q9BUT1', 'P30838', 'O43924', 'P09104', 'P07864', 'Q9BYC5', 'P06733', 'P15121', 'P17516', 'P40926', 'O60218', 'P00338', 'P07738', 'P07195'}

# MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
# MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MODEL_NAME = 'facebook/esm2_t36_3B_UR50D'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

OUT_CHANNELS = 128
KERNEL_SIZE = 15
NUM_LAYERS = 3

class MultitaskFinetunedEsmModelWithCnn(nn.Module):
    def __init__(self, esm_model: str) -> None:
        super().__init__()
        self.llm = EsmModel.from_pretrained(esm_model) #, torch_dtype=torch.bfloat16)

        conv_layers = [nn.Conv1d(1, OUT_CHANNELS, KERNEL_SIZE, padding=KERNEL_SIZE // 2), 
                      nn.ReLU()]
        for i in range(NUM_LAYERS - 1):
            conv_layers.append(nn.Conv1d(OUT_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, padding=KERNEL_SIZE // 2))
            conv_layers.append(nn.ReLU())
            # conv_layers.append(nn.BatchNorm1d(OUT_CHANNELS))
        self.pLDDT_conv = nn.Sequential(*conv_layers)

        self.dropout = nn.Dropout(DROPOUT)

        input_size = self.llm.config.hidden_size + OUT_CHANNELS
        self.classifier = nn.Linear(input_size, OUTPUT_SIZE) # , dtype=torch.bfloat16)
        self.plDDT_regressor = nn.Linear(input_size, OUTPUT_SIZE) # , dtype=torch.bfloat16)
        self.distance_regressor = nn.Linear(input_size, OUTPUT_SIZE) # , dtype=torch.bfloat16)

    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        input_ids, attention_mask, pLDDT = batch["input_ids"], batch["attention_mask"], batch["plDDTs"].half()

        token_embeddings = self.llm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        token_embeddings = self.dropout(token_embeddings)
        
        pLDDT = pLDDT.unsqueeze(1)
        processed_pLDDT = self.pLDDT_conv(pLDDT)
        processed_pLDDT = processed_pLDDT.reshape(token_embeddings.shape[0], token_embeddings.shape[1], OUT_CHANNELS)
        assert processed_pLDDT.shape[0] == token_embeddings.shape[0]
        assert processed_pLDDT.shape[1] == token_embeddings.shape[1]

        token_embeddings = torch.cat((token_embeddings, processed_pLDDT), dim=2)

        assert processed_pLDDT.shape[1] == token_embeddings.shape[1]
        return self.classifier(token_embeddings), self.plDDT_regressor(token_embeddings), self.distance_regressor(token_embeddings)

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
        
        return self.classifier(token_embeddings), self.plDDT_regressor(token_embeddings), self.distance_regressor(token_embeddings)

class FinetunedEsmModel(nn.Module):
    def __init__(self, esm_model: str) -> None:
        super().__init__()
        self.llm = EsmModel.from_pretrained(esm_model)

        self.dropout = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(self.llm.config.hidden_size, OUTPUT_SIZE)
        
    def forward(self, batch: dict[str, np.ndarray]) -> torch.Tensor:
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        token_embeddings = self.llm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        return self.classifier(token_embeddings)

def process_sequence_dataset(annotation_path, tokenizer, distances_scaler=None, plDDT_scaler=None, distances_path=None, 
                               plDDT_path=None, uniprot_ids=False, load_ids=False):
    sequences = []
    labels = []
    distances = []
    plDDTs = []
    ids = []

    with open(annotation_path) as f:
        reader = csv.reader(f, delimiter=";")

        for row in reader:
            if not uniprot_ids:
                protein_id = row[0].lower() + row[1]
            else:
                protein_id = row[0]
            
            if protein_id in LIGYSIS_OVERLAPPING_SEQUENCES:
                continue
            sequence = row[4]
            # max sequence length of ESM2
            if len(sequence) > MAX_LENGTH: continue 
            
            indices = [int(residue[1:]) for residue in row[3].split(' ')]
            label = np.zeros(len(sequence))
            label[indices] = 1

            # load distances only if the path is provided
            if distances_path:
                assert distances_scaler is not None
                distance = np.load(f'{distances_path}/{protein_id}.npy')
                distance[distance == -1] = 0.5
                distance = np.clip(distance, 0.0, 10.0, dtype=np.float32)
    
                if len(distance) != len(sequence): 
                    print(f'{protein_id} doesn\'t match. Skipping ...')
                    break
                
                # scale the distance
                distance = distances_scaler.transform(distance.reshape(-1, 1)).reshape(1, -1)[0]
                distances.append(distance)

            # load plDDT only if the path is provided
            if plDDT_path:
                assert plDDT_scaler is not None
                plDDT = np.load(f'{plDDT_path}/{protein_id}.npy')
                plDDT[plDDT == -100.0] = 0.8
    
                if len(plDDT) != len(sequence): 
                    print(f'{protein_id} doesn\'t match. Skipping ...')
                    break
                
                # scale the distance
                plDDT = plDDT_scaler.transform(plDDT.reshape(-1, 1)).reshape(1, -1)[0]
                plDDTs.append(plDDT)

            ids.append([protein_id for _ in range(len(sequence))])
            sequences.append(sequence)
            labels.append(label)

    train_tokenized = tokenizer(sequences)
    
    dataset = Dataset.from_dict(train_tokenized)
    dataset = dataset.add_column("labels", labels)
    if load_ids:
        dataset = dataset.add_column("ids", ids)
    if len(distances) > 0:
        dataset = dataset.add_column("distances", distances)
    if len(plDDTs) > 0:
        dataset = dataset.add_column("plDDTs", plDDTs)
    
    return dataset


def train_scaler(annotation_path, distances_path=None, plDDT_path=None, uniprot_ids=False):
    values = []

    with open(annotation_path) as f:
        reader = csv.reader(f, delimiter=";")

        for row in reader:
            if not uniprot_ids:
                protein_id = row[0].lower() + row[1]
            else:
                protein_id = row[0]
            
            if distances_path:
                distance = np.load(f'{distances_path}/{protein_id}.npy')
                distance[distance == -1] = 0.5
                values.append(distance)
            elif plDDT_path:
                plDDT = np.load(f'{plDDT_path}/{protein_id}.npy')
                plDDT[plDDT == -100.0] = 0.75
                values.append(plDDT)
            else:
                raise ValueError("Either distances_path or plDDT_path must be provided.")

    scaler = MinMaxScaler()
    scaler.fit(np.concatenate(values).reshape(-1, 1))
    return scaler

def collate_fn(batch, tokenizer):
    label_names = ['labels']
    if "distances" in batch[0].keys():
        label_names.append('distances')
    if "plDDTs" in batch[0].keys():
        label_names.append('plDDTs')
    if "ids" in batch[0].keys():
        label_names.append('ids')

    labels = {label_name: [feature[label_name] for feature in batch] for label_name in label_names}
    no_labels_features = [{k: v for k, v in feature.items() if k not in label_names} for feature in batch]

    batch = tokenizer.pad(
        no_labels_features,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt").to(device)
    
    sequence_length = batch["input_ids"].shape[1]

    for label_name in label_names:
        batch[label_name] = [[LABEL_PAD_TOKEN_ID] + list(label) + [LABEL_PAD_TOKEN_ID] * (sequence_length - len(label)-1) for label in labels[label_name]]
        if label_name != 'ids':
            batch[label_name] = torch.tensor(batch[label_name]).to(device)
    return batch
