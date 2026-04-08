"""
Clustering utilities for protein binding site prediction.

Provides functions for clustering surface points, managing protein structures,
and propagating cluster labels from atoms to residues.

Key parameters:
- POINTS_DENSITY_PER_ATOM: Number of surface points generated per atom (~50)
- PROBE_RADIUS: Solvent probe radius for SASA calculation (~1.6 Å)
- aal_prot: Set of standard protein residue names
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.SASA import ShrakeRupley
from sklearn.cluster import MeanShift, AgglomerativeClustering, DBSCAN, AffinityPropagation
from sklearn.mixture import BayesianGaussianMixture
from collections import Counter

import sys
sys.path.append('/home/skrhakv/cryptoshow-analysis/src/utils')
import cryptoshow_utils

CIF_FILES = '/home/skrhakv/cryptoshow-analysis/data/cif_files'
POINTS_DENSITY_PER_ATOM = 50
PROBE_RADIUS = 1.6

def execute_atom_clustering(pdb_id, chain_id, predictions, probabilities, eps=10):
    """
    Execute atom-level clustering based on predicted binding residues.

    Process:
    1. Map residue numbering from zero-based (predictions) to auth (PDB) numbering
    2. Extract 3D surface points from binding residues
    3. Apply MeanShift clustering to surface points
    4. Propagate cluster labels to atoms via majority voting
    5. Propagate atom labels to residues via majority voting

    Args:
        pdb_id (str): PDB identifier
        chain_id (str): Chain identifier
        predictions (np.array): Zero-based residue indices marked as binding sites
        probabilities (np.array): Binding probability scores for each residue
        eps (float): MeanShift bandwidth parameter in Angstroms (default: 10)

    Returns:
        Tuple containing:
        - clusters (dict): {cluster_id: [atom_serial_numbers,...]}
        - cluster_residues (list): [cluster_id] -> [residue_ids,...]
        - cluster_scores (list): Average binding score per cluster
        - atom_coords (dict): {atom_id: np.array([x,y,z])}
        - residue_coords (dict): {residue_id: np.array([x,y,z])}

    Returns (None, None, None, None, None) if no surface points found.
    """
    # 1. Map mmCIF numbering to auth numbering
    auth_predictions, scores = cryptoshow_utils.map_mmcif_numbering_to_auth(pdb_id, chain_id, predictions, binding_scores=probabilities)

    # 2. Get surface points and their mapping to atoms, atom coordinates, and atom to residue mapping
    all_points, map_point_to_atom, map_atoms_to_residue_id, atom_coords, residue_coords = get_protein_surface_points(pdb_id, chain_id, auth_predictions)

    if all_points.shape[0] == 0:
        return None, None, None, None, None

    # 3. Cluster surface points and propagate labels to atoms
    atom_labels = cluster_atoms_by_surface(
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
    auth_predictions = np.array(auth_predictions)

    # 4.1 For each atom in each cluster, get its residue and score
    for atom_id, cluster_label in atom_labels.items():
        residue_id = map_atoms_to_residue_id[atom_id] # this is auth residue id
        score = scores[np.where(auth_predictions == int(residue_id))[0][0]]
        cluster_scores[cluster_label].append(score)
        cluster_residues[cluster_label].append(residue_id)

    # 4.2 Vote
    residue_voting = {residue: [0 for _ in range(len(cluster_residues))] for residue in auth_predictions}
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

def cluster_atoms_by_surface(all_points, point_to_atom_map, eps=1.5, gmm=False):
    """
    Cluster surface points using MeanShift or Gaussian Mixture Model.

    Process:
    1. Apply clustering algorithm to 3D surface points
    2. Use majority voting to assign each atom a cluster label
    3. Return atom ID -> cluster label mapping

    Args:
        all_points (np.array): (N, 3) array of 3D surface point coordinates
        point_to_atom_map (np.array): (N,) array mapping each point to atom serial number
        eps (float): MeanShift bandwidth or DBSCAN epsilon in Angstroms (default: 1.5)
        gmm (bool): Use Gaussian Mixture Model instead of MeanShift

    Returns:
        atom_labels (dict): {atom_id: cluster_label} mapping each atom to cluster
    """

    if not gmm:
        # n_jobs=-1 uses all CPU cores.
        # clustering = AffinityPropagation(damping=0.9, preference=-200, max_iter=500, convergence_iter=50)
        clustering = MeanShift(bandwidth=eps, bin_seeding=True, n_jobs=-1) # eps = 9 or 12
        # clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=eps, linkage='ward')
        # clustering = DBSCAN(eps=eps, min_samples=5)
        # clustering = DBSCAN(eps=6, min_samples=1)

        clustering.fit(all_points)
        point_labels = clustering.labels_

    if gmm:
        # get the optimal number of components
        bgmm = BayesianGaussianMixture(
            n_components=max(len(all_points), 1) - 1,
            random_state=42,
            covariance_type='spherical',
        )

        bgmm.fit(all_points)

        active_clusters = sum(bgmm.weights_ > 0.1) # Check how many clusters are actually used - how many are composed of >10% of points
        clustering = BayesianGaussianMixture(
            n_components=max(active_clusters, 1),
            random_state=42,
            covariance_type='spherical',
        )

        point_labels = clustering.fit_predict(all_points)

    # Majority Vote (Propagate to Atoms)
    atom_labels = {}

    # Get unique atom IDs present in the data
    unique_atoms = np.unique(point_to_atom_map)

    for atom_id in unique_atoms:
        # 1. Find indices in the master array belonging to this atom
        indices = np.where(point_to_atom_map == atom_id)[0]

        # 2. Extract the cluster labels for these points
        current_labels = point_labels[indices]

        # 3. Determine the most common label (majority vote)
        counts = Counter(current_labels)
        majority_label = counts.most_common(1)[0][0]
        atom_labels[atom_id] = majority_label

    return atom_labels

aal_prot = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "ASH", "GLH", "HIE", "HID", "HIP", "LYN", "CYX", "CYM", "TYM"
}

def keep_only_standard_residues(structure):
    """
    Remove non-standard residues from protein structure.

    Keeps only standard 20 amino acids plus common protonation variants.
    Removes water, ligands, ions, and other non-protein residues.

    Args:
        structure (PDB.Structure or PDB.Chain): Biopython structure

    Returns:
        PDB.Structure or PDB.Chain: Modified structure (in-place modification)
    """
    for residue in list(structure):
        if residue.get_resname() not in aal_prot:
            structure.detach_child(residue.id)
    return structure

def get_protein_surface_points(pdb_id, chain_id, predicted_binding_sites):
    """
    Extract 3D surface points from predicted binding site residues.

    For each residue, retrieve pre-computed surface points from atom.sasa_points.

    Args:
        pdb_id (str): PDB identifier
        chain_id (str): Chain identifier
        predicted_binding_sites (list): List of residue IDs marked as binding sites

    Returns:
        Tuple containing:
        - surface_points (np.array): (N, 3) array of 3D coordinates
        - map_surface_points_to_atom_id (np.array): (N,) mapping each point to atom serial number
        - map_atoms_to_residue_id (dict): atom_id -> residue_id
        - atom_coords (dict): atom_id -> coordinate vector
        - residue_coords (dict): residue_id -> coordinate vector
    """
    p = MMCIFParser(QUIET=1)
    struct = p.get_structure("protein", f"{CIF_FILES}/{pdb_id}.cif")
    struct = struct[0][chain_id]
    struct = keep_only_standard_residues(struct)

    # compute SASA
    sr = ShrakeRupley(n_points=POINTS_DENSITY_PER_ATOM, probe_radius=PROBE_RADIUS)
    sr.compute(struct, level="A")

    surface_points = []
    map_surface_points_to_atom_id = []
    atom_coords = {}
    residue_coords = {}
    map_atoms_to_residue_id = {}
    for residue in struct.get_residues():
        # consider only residues from predicted binding sites
        residue_id = residue.get_id()[1]

        if 'CA' in residue:
            residue_coords[residue_id] = residue['CA'].get_vector()
        else:
            # if no CA atom, use the first atom's coordinates
            first_atom = next(residue.get_atoms())
            residue_coords[residue_id] = first_atom.get_vector()

        if residue.get_id()[1] not in predicted_binding_sites:
            continue

        # get surface points for each atom in the residue
        for atom in residue.get_atoms():
            atom_id = atom.get_serial_number()
            surface_points.append(atom.sasa_points)
            map_surface_points_to_atom_id.extend([atom_id] * len(atom.sasa_points))
            atom_coords[atom_id] = atom.get_vector()
            map_atoms_to_residue_id[atom_id] = residue_id

    surface_points = np.vstack(surface_points)
    map_surface_points_to_atom_id = np.array(map_surface_points_to_atom_id)
    return surface_points, map_surface_points_to_atom_id, map_atoms_to_residue_id, atom_coords, residue_coords


def print_plots(DCCs, coverages, number_of_pockets, model, dcc_threshold):
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    _, axs = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(DCCs, bins=20, color='skyblue', edgecolor='black', alpha=0.7, ax=axs[0])
    axs[0].set_title(f'{model}: Pocket Center Distance (DCC) distribution\n(median={np.median(DCCs):.2f} Å), DCC ≤ {dcc_threshold} Å: {np.sum(np.array(DCCs) < dcc_threshold)} / {number_of_pockets}', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('DCC (Å)', fontsize=11)
    axs[0].set_ylabel('Count', fontsize=11)
    axs[0].set_xlim(0, 35)

    sns.histplot(coverages, bins=20, color='salmon', edgecolor='black', alpha=0.7, ax=axs[1])
    axs[1].set_title(f'{model}: Residues Covered Percentage\n(median={np.median(coverages):.1f}%)', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('Residues Covered (%)', fontsize=11)
    axs[1].set_ylabel('Count', fontsize=11)

    plt.tight_layout()
    plt.show()
