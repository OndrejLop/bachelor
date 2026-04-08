"""
Cluster predicted binding sites and apply smoothing.

This script:
1. Reads ESM2 predictions and embeddings from intermediate files
2. Applies a smoothing model to expand binding site predictions based on
   surrounding residues within distance threshold
3. Clusters predicted residues into distinct binding pockets using 3D surface
   points and MeanShift clustering
4. Generates cluster assignments with atom and residue mappings
5. Saves pocket predictions and residue-level binding probabilities

The smoothing model uses residue embeddings and local context to identify
residues that should be reclassified as binding sites if they're close to
predicted binding residues.

Input:
  - Predictions: CSV files (one float per residue)
  - Embeddings: NPY files (1024-dim per residue)
  - PDB files: 3D structure coordinates

Output:
  - {pdb_id}_predictions.csv: Pocket definitions with residue and atom lists
  - {pdb_id}_residues.csv: Per-residue binding probabilities and pocket assignments
"""
from collections import Counter
import numpy as np
import sys
import torch
from pathlib import Path
import argparse
from datetime import datetime
import json

ROOT       = Path(__file__).parent.parent.parent

sys.path.append(str(ROOT / 'src' / 'utilities'))
import clustering_utils
import eval_utils
from eval_utils import CryptoBenchClassifier

parser = argparse.ArgumentParser(description="Cluster predicted binding sites with smoothing")
parser.add_argument("--decision-threshold", type=float, default=0.7,
                    help="Binding probability threshold (default: 0.7)")
parser.add_argument("--distance-threshold", type=float, default=10,
                    help="Spatial distance for neighbor search in Angstroms (default: 10)")
parser.add_argument("--timestamp", action="store_true",
                    help="Add timestamp to output directory (prevents overwrites)")
args = parser.parse_args()

POSITIVE_DISTANCE_THRESHOLD = args.distance_threshold
DECISION_THRESHOLD = args.decision_threshold

PREDICTIONS_DIR = ROOT / 'data' / 'intermediate' / 'predictions'
EMBEDDINGS_DIR  = ROOT / 'data' / 'intermediate' / 'embeddings'
PDB_DIR         = ROOT / 'data' / 'input' / 'pdb'
base_output_dir = ROOT / 'data' / 'output' / 'CS_predictions'
MODEL_PATH      = ROOT / 'data' / 'models' / '3B-model.pt'
SMOOTHING_MODEL_PATH = ROOT / 'data' / 'models' / 'smoother.pt'

# --- Create output directory with optional timestamp ---
if args.timestamp:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_suffix = f"decision{args.decision_threshold}_dist{args.distance_threshold}"
    OUTPUT_DIR = base_output_dir / f"{timestamp}_{param_suffix}"
else:
    OUTPUT_DIR = base_output_dir

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Save run metadata ---
run_metadata = {
    "timestamp": datetime.now().isoformat(),
    "decision_threshold": args.decision_threshold,
    "distance_threshold": args.distance_threshold,
    "output_dir": str(OUTPUT_DIR),
}
metadata_path = OUTPUT_DIR / "run_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(run_metadata, f, indent=2)

print(f"\n{'='*60}")
print(f"Pocket Clustering Run Parameters:")
print(f"{'='*60}")
print(f"Decision threshold:   {args.decision_threshold}")
print(f"Distance threshold:   {args.distance_threshold} Å")
print(f"Output directory:     {OUTPUT_DIR}")
print(f"Metadata saved:       {metadata_path}")
print(f"{'='*60}\n")


def map_residue_numbering_to_auth(pdb_path: str, binding_residues: dict[np.ndarray], binding_scores: dict[np.ndarray]) -> dict[list[int]]:
    """
    Map the binding residues from zero-based numbering (0=first residue, 1=second residue, etc.) to the auth labeling (residue labeling from the PDB file).
    Args:
        pdb_path (str): Path to the PDB file.
        binding_residues (dict[np.ndarray]): Dictionary of binding residues, keys are chain IDs and values are arrays of residue indices (zero-based).
        auth (bool): Whether to use author fields.
    Returns:
        dict[list[int]]: Dictionary of binding residues in the auth labeling, keys are chain IDs and values are lists of residue numbers.
    """
    import biotite.structure.io.pdb as pdb
    from biotite.structure.io.pdb import get_structure
    from biotite.structure import get_residues
    
    cif_file = pdb.PDBFile.read(pdb_path)
    
    protein = get_structure(cif_file, model=1) #, use_author_fields=False) - UNCOMMENT THIS TO USE LABEL_SEQ_ID FIELDS
    protein = protein[(protein.atom_name == "CA") 
                        & (protein.element == "C") ]
    
    mapped_residues = {}
    mapped_scores = {}
    for chain_id in binding_residues.keys():
    
        protein_chain = protein[protein.chain_id == chain_id]
        mapped_residues[chain_id] = []
        mapped_scores[chain_id] = []
        residue_ids, _ = get_residues(protein_chain)
        
        # loop over all residues in chain and check if the residue index matches the binding residue index, if so, add the auth residue number to the mapped residues list
        for i, residue_id in enumerate(residue_ids):
            residue_index = np.where(binding_residues[chain_id] == i)[0] # get positions where the residue index matches the binding residue index
            if len(residue_index) > 0:
                mapped_residues[chain_id].append(residue_id)
                mapped_scores[chain_id].append(binding_scores[chain_id][i])
    
        assert len(mapped_residues[chain_id]) == len(binding_residues[chain_id]), f"Chain {chain_id} has different number of residues in mapped residues and original binding residues"
        assert len(mapped_scores[chain_id]) == len(binding_residues[chain_id]), f"Chain {chain_id} has different number of scores in mapped scores and original binding residues"
    
    return mapped_residues, mapped_scores

def keep_only_standard_residues(structure):
    """Keep only standard protein residues in the structure."""
    for chain in list(structure):
        for residue in list(chain):
            if residue.get_resname() not in clustering_utils.aal_prot:
                chain.detach_child(residue.id)
    return structure

def _attach_sasa_points(struct, n_points, probe_radius):
    """
    Attach 3D surface points to each atom for later clustering.

    Uses a Fibonacci lattice sphere around each atom, filtered to only include
    points that are exposed (not inside neighboring atoms).

    Algorithm:
    1. Generate n_points points on a unit sphere using Fibonacci lattice distribution
    2. For each atom: scale points by van der Waals radius + probe radius
    3. Use KDTree to find neighboring atoms within interaction distance
    4. For each neighboring atom, remove points that fall inside it
    5. Store exposed points in atom.sasa_points

    Args:
        struct (PDB.Structure): Biopython structure
        n_points (int): Number of surface points per atom (~50)
        probe_radius (float): Solvent probe radius in Angstroms (~1.6)

    Side effect:
        Sets atom.sasa_points attribute containing numpy array of exposed 3D coordinates
    """
    from scipy.spatial import KDTree
    vdw_radii = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8, 'P': 1.8,
                 'H': 1.2, 'SE': 1.9, 'FE': 1.4, 'ZN': 1.39, 'MG': 1.73}
    default_radius = 1.5
    golden = (1 + 5 ** 0.5) / 2
    idx = np.arange(n_points)
    theta = np.arccos(1 - 2 * (idx + 0.5) / n_points)
    phi = 2 * np.pi * idx / golden
    unit_sphere = np.column_stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    all_atoms = list(struct.get_atoms())
    all_coords = np.array([a.get_vector().get_array() for a in all_atoms])
    all_radii = np.array([vdw_radii.get(a.element, default_radius) + probe_radius for a in all_atoms])
    max_radius = float(all_radii.max())
    tree = KDTree(all_coords)
    for i, atom in enumerate(all_atoms):
        center = all_coords[i]; r = all_radii[i]; pts = unit_sphere * r + center
        neighbor_idx = [j for j in tree.query_ball_point(center, r + max_radius) if j != i]
        if neighbor_idx:
            nc = all_coords[neighbor_idx]; nr = all_radii[neighbor_idx]
            exposed = np.ones(len(pts), dtype=bool)
            for nc_j, nr_j in zip(nc, nr):
                exposed &= np.linalg.norm(pts - nc_j, axis=1) >= nr_j
            atom.sasa_points = pts[exposed]
        else:
            atom.sasa_points = pts
        if len(atom.sasa_points) == 0:
            atom.sasa_points = center.reshape(1, 3)

def get_protein_surface_points(pdb_path, predicted_binding_sites):
    """
    Extract 3D surface points from predicted binding site residues.

    For each atom in predicted binding residues, collects the pre-computed
    SASA surface points (from _attach_sasa_points).

    Args:
        pdb_path (str): Path to PDB file
        predicted_binding_sites (dict[str, list]): Chain ID -> list of binding residue IDs

    Returns:
        Tuple containing:
        - surface_points (np.array): (N, 3) array of 3D coordinates
        - map_surface_points_to_atom_id (np.array): (N,) array mapping each point to atom serial number
        - map_atoms_to_residue_id (dict): atom_id -> (chain_id, residue_id)
        - atom_coords (dict): atom_id -> coordinate vector
        - residue_coords (dict): (chain_id, residue_id) -> coordinate vector

    Returns empty array if no surface points found.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley

    p = PDBParser(QUIET=1)
    struct = p.get_structure("protein", pdb_path)
    struct = struct[0]
    struct = keep_only_standard_residues(struct)

    # compute SASA and attach 3D surface points
    sr = ShrakeRupley(n_points=clustering_utils.POINTS_DENSITY_PER_ATOM, probe_radius=clustering_utils.PROBE_RADIUS)
    try:
        sr.compute(struct, level="A")
    except ValueError:
        return np.empty((0, 3)), np.array([]), {}, {}, {}
    _attach_sasa_points(struct, clustering_utils.POINTS_DENSITY_PER_ATOM, clustering_utils.PROBE_RADIUS)

    surface_points = []
    map_surface_points_to_atom_id = []
    atom_coords = {}
    residue_coords = {}
    map_atoms_to_residue_id = {}
    for residue in struct.get_residues():
        # consider only residues from predicted binding sites
        residue_chain = residue.get_full_id()[2]
        residue_id = residue.get_id()[1]
        
        if 'CA' in residue:
            residue_coords[(residue_chain, residue_id)] = residue['CA'].get_vector()
        else:
            # if no CA atom, use the first atom's coordinates
            first_atom = next(residue.get_atoms())
            residue_coords[(residue_chain, residue_id)] = first_atom.get_vector()
        
        if residue_chain not in predicted_binding_sites or residue.get_id()[1] not in predicted_binding_sites[residue_chain]:
            continue
        
        # get surface points for each atom in the residue
        for atom in residue.get_atoms():
            atom_id = atom.get_serial_number()
            surface_points.append(atom.sasa_points)
            map_surface_points_to_atom_id.extend([atom_id] * len(atom.sasa_points))
            atom_coords[atom_id] = atom.get_vector()
            map_atoms_to_residue_id[atom_id] = (residue_chain, residue_id)

    if not surface_points:
        return np.empty((0, 3)), np.array([]), map_atoms_to_residue_id, atom_coords, residue_coords
    surface_points = np.vstack(surface_points)
    map_surface_points_to_atom_id = np.array(map_surface_points_to_atom_id)
    return surface_points, map_surface_points_to_atom_id, map_atoms_to_residue_id, atom_coords, residue_coords



def execute_atom_clustering(pdb_path, predictions, probabilities, eps=10):
    """
    Execute atom-level clustering based on predicted binding residues.
    Args:
        pdb_path: Path to the PDB file.
        chain_id: Chain identifier of the protein.
        predictions: List of predicted binding residue IDs (mmCIF numbering).
        probabilities: List of probabilities/scores for the predicted binding residues.
    Returns:
        clusters: Dict {cluster_id: [atom_id, ...], ...}
        cluster_residues: List of Lists [[residue_id, ...], ...] for each cluster. The ordering corresponds to cluster IDs.
        cluster_scores: List of average scores for each cluster. List has size of N, where N is number of clusters, and the ordering corresponds to cluster IDs.
        atom_coords: Dict {atom_id: np.array([x,y,z])}
    """
    mapped_prediction, mapped_scores = map_residue_numbering_to_auth(pdb_path=pdb_path,\
                                                binding_residues=predictions, \
                                                binding_scores=probabilities)
    # 2. Get surface points and their mapping to atoms, atom coordinates, and atom to residue mapping
    all_points, map_point_to_atom, map_atoms_to_residue_id, atom_coords, residue_coords = get_protein_surface_points(pdb_path, mapped_prediction)

    if all_points.shape[0] == 0:
        return None, None, None, None, None

    # 3. Cluster surface points and propagate labels to atoms    
    atom_labels = clustering_utils.cluster_atoms_by_surface(
        all_points, map_point_to_atom, eps=eps)

    # get cluster dictionary {cluster_id: [atom_id, ...], ...}
    clusters = {}
    for atom_index, cluster_label in atom_labels.items():
        if cluster_label not in clusters:
            clusters[cluster_label] = []
        clusters[cluster_label].append(atom_index)

    # 4. Voting: the residue gets the label of the majority of its atoms
    cluster_scores = [[] for _ in range(max(clusters) + 1)]
    cluster_residues = [[] for _ in range(max(clusters) + 1)]
    auth_predictions = {}
    for chain_id, pred in mapped_prediction.items():
        auth_predictions[chain_id] = np.array(pred)


    # 4.1 For each atom in each cluster, get its residue and score
    for atom_id, cluster_label in atom_labels.items():
        chain_id, residue_id = map_atoms_to_residue_id[atom_id] # this is auth residue id
        score = mapped_scores[chain_id][np.where(auth_predictions[chain_id] == int(residue_id))[0][0]]
        cluster_scores[cluster_label].append(score)
        cluster_residues[cluster_label].append(f'{chain_id}_{residue_id}')

    # 4.2 Vote
    # Reformat auth_predictions to be a list of strings in the format "chain_residueid", e.g. "A_123"
    reformated_auth_predictions = []
    for chain_id, pred in auth_predictions.items():
        reformated_auth_predictions.extend([f'{chain_id}_{res_id}' for res_id in pred])
        
    residue_voting = {residue: [0 for _ in range(len(cluster_residues))] for residue in reformated_auth_predictions}
    for i, labels in enumerate(cluster_residues):
        counts = Counter(labels)
        for residue, number_of_occurences in counts.items():
            residue_voting[residue][i] = number_of_occurences
    
    residue_clusters = {i: [] for i in range(len(cluster_residues))}
    # 4.3 get residue cluster assignment based on voting
    for residue, votes in residue_voting.items():
        cluster = np.argmax(votes)
        residue_clusters[cluster].append(residue)
    
    # 5. Compute average cluster scores
    final_cluster_scores = []
    for scores in cluster_scores:
        if len(scores) == 0:
            final_cluster_scores.append(0.0)
        else:
            final_cluster_scores.append(np.mean(scores))
    cluster_scores = final_cluster_scores

    return clusters, residue_clusters, cluster_scores, atom_coords, residue_coords

def compute_distance_matrix(pdb_path, chain_id):
    import biotite.database.rcsb as rcsb
    import biotite.structure.io.pdb as pdb
    from biotite.structure.io.pdb import get_structure
    from biotite.structure import get_residues
    from scipy.spatial import distance_matrix
    
    pdb_file = pdb.PDBFile.read(pdb_path)
    protein = get_structure(pdb_file, model=1)
    protein = protein[(protein.atom_name == "CA")
                        & (protein.element == "C")
                        & (protein.chain_id == chain_id) ]
    if len(protein) == 0:
        return None
    _, residue_types = get_residues(protein)

    # To calculate embeddings later
    # sequence = ''.join([cryptoshow_utils.three_to_one(residue_type) for residue_type in residue_types])
    # with open(f'{PATH}/data/sequences/pdb1a00_{chain_id}.txt', 'w') as f:
    #     f.write(sequence)

    coords = protein.coord
    dist_matrix = distance_matrix(coords, coords)

    return dist_matrix

from collections import defaultdict

smoothing_model = torch.load(SMOOTHING_MODEL_PATH, weights_only=False)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# discover all PDB IDs and their chains from prediction files
pdb_chains = defaultdict(list)
for pred_file in sorted(Path(PREDICTIONS_DIR).glob('*_predictions.csv')):
    pdb_id, chain_id, _ = pred_file.stem.rsplit('_', 2)
    pdb_chains[pdb_id].append(chain_id)

def run_assertions(clusters):
    for cluster_id, atoms in clusters.items():
        for atom in atoms:
            for cluster_iid, atoms in clusters.items():
                if cluster_id != cluster_iid:
                    assert atom not in atoms, f'Atom {atom} is in both cluster {cluster_id} and cluster {cluster_iid}'

def get_residue_ids(pdb_path, chain_id):
    import biotite.structure.io.pdb as pdb
    from biotite.structure.io.pdb import get_structure
    from biotite.structure import get_residues

    pdb_file = pdb.PDBFile.read(pdb_path)
    protein = get_structure(pdb_file, model=1)
    protein = protein[(protein.atom_name == "CA")
                        & (protein.element == "C")
                        & (protein.chain_id == chain_id)]
    if len(protein) == 0:
        return [], []
    residue_ids, residue_types = get_residues(protein)
    return residue_ids, residue_types

def output_predictions(clusters, residue_clusters, cluster_scores, pdb_id):
    with open(f'{OUTPUT_DIR}/{pdb_id}_predictions.csv', 'w') as f:
        f.write("name,rank,score,residue_ids,atom_ids\n")
        for rank, cluster_id in enumerate(sorted(clusters.keys(), key=lambda x: cluster_scores[x], reverse=True)):
            cluster_atoms, cluster_residues, cluster_score = clusters[cluster_id], residue_clusters[cluster_id], cluster_scores[cluster_id]
            f.write(f"pocket{rank+1},{rank+1},{cluster_score},{' '.join(map(str, cluster_residues))},{' '.join(map(str, cluster_atoms))}\n")

def output_residues(pocket_residues_to_pocket_number, probabilities, pdb_id, pdb_path):
    sanity_check_residues2 = set()
    with open(f'{OUTPUT_DIR}/{pdb_id}_residues.csv', 'w') as f:
        f.write("chain_id,residue_id,residue_type,probability,pocket_number\n")
        for chain_id in probabilities.keys():
            residue_ids, residue_types = get_residue_ids(pdb_path=pdb_path, chain_id=chain_id)
            if len(residue_ids) != len(probabilities[chain_id]):
                print(f'  [SKIP] {pdb_id} chain {chain_id}: PDB has {len(residue_ids)} residues but predictions have {len(probabilities[chain_id])}')
                continue
            for i, (residue_id, residue_type) in enumerate(zip(residue_ids, residue_types)):
                if f'{chain_id}_{residue_id}' in pocket_residues_to_pocket_number:
                    sanity_check_residues2.add(f'{chain_id}_{residue_id}')
                    f.write(f'{chain_id},{residue_id},{residue_type},{probabilities[chain_id][i]},{pocket_residues_to_pocket_number[f"{chain_id}_{residue_id}"]}\n')
                else:
                    f.write(f'{chain_id},{residue_id},{residue_type},{probabilities[chain_id][i]},0\n')
    return sanity_check_residues2

for PDB_ID, chain_ids in pdb_chains.items():
    print(f'Processing {PDB_ID}...')
    predictions = {}
    smoothed_predictions = {}
    probabilities = {}
    PDB_PATH = Path(PDB_DIR) / f'{PDB_ID}.pdb'

    for chain_id in chain_ids:
        with open(f'{PREDICTIONS_DIR}/{PDB_ID}_{chain_id}_predictions.csv') as f:
            prediction = np.array([float(i) for i in f.read().splitlines()])
        predictions[chain_id] = np.where(prediction >= DECISION_THRESHOLD)[0]
        probabilities[chain_id] = prediction
        embedding = np.load(f'{EMBEDDINGS_DIR}/{PDB_ID}_{chain_id}_embeddings.npy')
        distance_matrix = compute_distance_matrix(PDB_PATH, chain_id)
        if distance_matrix is None:
            print(f'  [SKIP] {PDB_ID} chain {chain_id}: no CA atoms found')
            del predictions[chain_id]; del probabilities[chain_id]
            continue
        if distance_matrix.shape[0] != len(prediction):
            print(f'  [SKIP] {PDB_ID} chain {chain_id}: PDB has {distance_matrix.shape[0]} residues but predictions have {len(prediction)}')
            del predictions[chain_id]; del probabilities[chain_id]
            continue

        smoothed_predictions[chain_id] = predictions[chain_id].copy()
        for residue_idx in np.where(np.array(prediction) < DECISION_THRESHOLD)[0]:
            current_residue_embedding = embedding[residue_idx]
            close_residues_indices = np.where(distance_matrix[residue_idx] < POSITIVE_DISTANCE_THRESHOLD)[0]
            close_binding_residues_indices = np.intersect1d(close_residues_indices, np.where(prediction > DECISION_THRESHOLD)[0])
            if len(close_binding_residues_indices) == 0:
                continue
            elif len(close_binding_residues_indices) == 1:
                surrounding_embedding = embedding[close_binding_residues_indices].reshape(-1)
            else:
                surrounding_embedding = np.mean(embedding[close_binding_residues_indices], axis=0).reshape(-1)
            concatenated_embedding = torch.tensor(np.concatenate((current_residue_embedding, surrounding_embedding), axis=0), dtype=torch.float32).to(device)

            test_logits = smoothing_model(concatenated_embedding).squeeze()
            result = (torch.sigmoid(test_logits) > eval_utils.SMOOTHING_DECISION_THRESHOLD).float()
            if result == 1:
                print(f'Smoothing: Chain {chain_id} Residue {residue_idx} set to binding based on surrounding residues')
                predictions[chain_id] = np.append(predictions[chain_id], residue_idx)

    clusters, residue_clusters, cluster_scores, _, _ = execute_atom_clustering(
        pdb_path=PDB_PATH, predictions=smoothed_predictions, probabilities=probabilities)

    if clusters is None:
        print(f'  [SKIP] {PDB_ID}: no surface points found')
        continue

    run_assertions(clusters)
    run_assertions(residue_clusters)
    assert len(cluster_scores) == len(clusters) == len(residue_clusters)

    sanity_check_residues1 = set()
    pocket_residues_to_pocket_number = {}
    for pocket in residue_clusters:
        for residue in residue_clusters[pocket]:
            sanity_check_residues1.add(residue)
            pocket_residues_to_pocket_number[residue] = pocket + 1

    sanity_check_residues2 = output_residues(pocket_residues_to_pocket_number, probabilities, PDB_ID, PDB_PATH)
    output_predictions(clusters, residue_clusters, cluster_scores, PDB_ID)

    for residue in sanity_check_residues1:
        assert residue in sanity_check_residues2, f'Residue {residue} is in sanity_check_residues1 but not in sanity_check_residues2'
    for residue in sanity_check_residues2:
        assert residue in sanity_check_residues1, f'Residue {residue} is in sanity_check_residues2 but not in sanity_check_residues1'