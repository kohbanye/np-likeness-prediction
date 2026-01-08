import csv
import multiprocessing as mp
import random
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent: list[int] = list(range(n))
        self.rank: list[int] = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def get_clusters(self) -> list[int]:
        """Return cluster labels for all elements."""
        roots: dict[int, int] = {}
        labels: list[int] = []
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in roots:
                roots[root] = len(roots)
            labels.append(roots[root])
        return labels


def calculate_diversity_metrics(
    smiles_list: list[str],
    selected_indices: list[int],
    n_pairs: int = 10000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Calculate diversity metrics for selected compounds.

    Args:
        smiles_list: Full list of SMILES
        selected_indices: Indices of selected compounds
        n_pairs: Number of random pairs for Tanimoto calculation
        random_seed: Random seed

    Returns:
        Dict with diversity metrics
    """
    random.seed(random_seed)

    selected_smiles = [smiles_list[i] for i in selected_indices]

    # 1. Calculate unique scaffolds
    scaffolds = set()
    for smiles in tqdm(selected_smiles, desc="Counting unique scaffolds"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
            scaffolds.add(scaffold_smiles)
        except Exception:
            continue

    # 2. Calculate ECFP4 fingerprints and Tanimoto similarity
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []

    for smiles in tqdm(
        selected_smiles[:50000], desc="Calculating fingerprints"
    ):  # Limit to 50K for speed
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = mfpgen.GetFingerprint(mol)
                fingerprints.append(fp)
        except Exception:
            continue

    # Sample random pairs and calculate Tanimoto
    tanimoto_similarities = []
    if len(fingerprints) > 1:
        n_pairs_actual = min(n_pairs, len(fingerprints) * (len(fingerprints) - 1) // 2)
        for _ in tqdm(range(n_pairs_actual), desc="Calculating Tanimoto similarities"):
            i, j = random.sample(range(len(fingerprints)), 2)
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            tanimoto_similarities.append(sim)

    # 3. Calculate molecular properties
    mol_weights = []
    logps = []

    for smiles in tqdm(selected_smiles, desc="Calculating molecular properties"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol_weights.append(Descriptors.MolWt(mol))
                logps.append(Descriptors.MolLogP(mol))
        except Exception:
            continue

    # Compile metrics
    metrics = {
        "n_selected": len(selected_indices),
        "n_unique_scaffolds": len(scaffolds),
        "scaffold_diversity_ratio": len(scaffolds) / len(selected_indices)
        if selected_indices
        else 0,
        "avg_tanimoto": np.mean(tanimoto_similarities) if tanimoto_similarities else 0,
        "std_tanimoto": np.std(tanimoto_similarities) if tanimoto_similarities else 0,
        "high_similarity_pairs_pct": sum(1 for s in tanimoto_similarities if s > 0.7)
        / len(tanimoto_similarities)
        * 100
        if tanimoto_similarities
        else 0,
        "mol_weight_mean": np.mean(mol_weights) if mol_weights else 0,
        "mol_weight_std": np.std(mol_weights) if mol_weights else 0,
        "mol_weight_min": np.min(mol_weights) if mol_weights else 0,
        "mol_weight_max": np.max(mol_weights) if mol_weights else 0,
        "logp_mean": np.mean(logps) if logps else 0,
        "logp_std": np.std(logps) if logps else 0,
    }

    return metrics


def load_coconut_molecular_weights(
    coconut_csv_path: Path,
    max_mw: float = 1000.0,
) -> np.ndarray:
    """Load molecular weights from COCONUT CSV.

    Args:
        coconut_csv_path: Path to COCONUT CSV file
        max_mw: Maximum molecular weight to consider

    Returns:
        Array of molecular weights
    """
    print(f"Loading molecular weights from {coconut_csv_path}...")
    molecular_weights = []

    # Increase CSV field size limit to handle large fields
    csv.field_size_limit(10000000)  # 10MB

    with open(coconut_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading COCONUT molecular weights"):
            try:
                mw = float(row["molecular_weight"])
                if 0 < mw <= max_mw:
                    molecular_weights.append(mw)
            except (ValueError, KeyError):
                continue

    mw_array = np.array(molecular_weights)
    print(f"Loaded {len(mw_array)} molecular weights from COCONUT")
    print(f"  Mean: {mw_array.mean():.1f}, Std: {mw_array.std():.1f}")
    print(f"  Range: {mw_array.min():.1f} - {mw_array.max():.1f}")

    return mw_array


def _calculate_molecular_weight_chunk(
    args: tuple[list[tuple[int, str]]],
) -> list[tuple[int, float]]:
    """Calculate molecular weights for a chunk of SMILES.

    Args:
        args: Tuple containing (chunk,)
              chunk: List of (index, smiles) tuples

    Returns:
        List of (index, molecular_weight) tuples
    """
    (chunk,) = args
    results = []

    for idx, smiles in chunk:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                results.append((idx, mw))
        except Exception:
            continue

    return results


def calculate_molecular_weights_parallel(
    smiles_list: list[str],
    n_jobs: int = 16,
    chunk_size: int = 100000,
) -> dict[int, float]:
    """Calculate molecular weights for SMILES in parallel.

    Args:
        smiles_list: List of SMILES strings
        n_jobs: Number of parallel processes
        chunk_size: Size of chunks for parallel processing

    Returns:
        Dict mapping index -> molecular weight
    """
    # Create chunks of (index, smiles) tuples
    indexed_smiles = list(enumerate(smiles_list))
    chunks = [
        indexed_smiles[i : i + chunk_size]
        for i in range(0, len(indexed_smiles), chunk_size)
    ]

    # Prepare arguments for parallel processing
    args = [(chunk,) for chunk in chunks]

    # Process in parallel
    mw_dict = {}

    with mp.Pool(processes=n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap(_calculate_molecular_weight_chunk, args),
                total=len(chunks),
                desc="Calculating molecular weights",
            )
        )

    # Merge results
    for result in results:
        for idx, mw in result:
            mw_dict[idx] = mw

    return mw_dict


def distribution_matching_sampling(
    target_distribution: np.ndarray,
    candidate_values: dict[int, float],
    target_count: int = 700000,
    n_bins: int = 50,
    random_seed: int = 42,
) -> list[int]:
    """Sample indices to match a target distribution.

    Args:
        target_distribution: Array of values defining the target distribution
        candidate_values: Dict mapping index -> value (e.g., molecular weight)
        target_count: Target number of samples
        n_bins: Number of histogram bins
        random_seed: Random seed

    Returns:
        List of selected indices
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create histogram of target distribution
    min_val = target_distribution.min()
    max_val = target_distribution.max()
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)

    target_hist, _ = np.histogram(target_distribution, bins=bin_edges)
    target_probs = target_hist / target_hist.sum()

    print("\nTarget distribution statistics:")
    print(f"  Range: {min_val:.1f} - {max_val:.1f}")
    print(f"  Mean: {target_distribution.mean():.1f}")
    print(f"  Std: {target_distribution.std():.1f}")
    print(f"  Number of bins: {n_bins}")

    # Organize candidates by bin
    bin_indices = [[] for _ in range(n_bins)]

    for idx, value in tqdm(
        candidate_values.items(), desc="Organizing candidates by bin"
    ):
        # Find which bin this value belongs to
        bin_idx = np.searchsorted(bin_edges[1:], value, side="right")
        if 0 <= bin_idx < n_bins:
            bin_indices[bin_idx].append(idx)

    # Calculate samples per bin
    samples_per_bin = (target_probs * target_count).astype(int)

    # Adjust to reach exactly target_count
    diff = target_count - samples_per_bin.sum()
    if diff > 0:
        # Add to bins with most available capacity
        available_capacity = np.array([len(bin_indices[i]) for i in range(n_bins)])
        available_capacity = available_capacity - samples_per_bin
        # Add to bins with positive capacity, prioritizing bins with more probability
        for _ in range(diff):
            valid_bins = np.where(available_capacity > 0)[0]
            if len(valid_bins) == 0:
                break
            # Choose bin weighted by target probability
            bin_probs = target_probs[valid_bins]
            bin_probs = bin_probs / bin_probs.sum()
            chosen_bin = np.random.choice(valid_bins, p=bin_probs)
            samples_per_bin[chosen_bin] += 1
            available_capacity[chosen_bin] -= 1

    # Sample from each bin
    selected_indices = []
    bin_stats = []

    for bin_idx in tqdm(range(n_bins), desc="Sampling from bins"):
        n_samples = samples_per_bin[bin_idx]
        available = len(bin_indices[bin_idx])

        if n_samples > 0 and available > 0:
            actual_samples = min(n_samples, available)
            sampled = random.sample(bin_indices[bin_idx], actual_samples)
            selected_indices.extend(sampled)

            bin_stats.append(
                {
                    "bin": bin_idx,
                    "range": f"{bin_edges[bin_idx]:.1f}-{bin_edges[bin_idx + 1]:.1f}",
                    "target_samples": n_samples,
                    "available": available,
                    "actual_samples": actual_samples,
                }
            )

    # Print sampling statistics
    print("\nSampling statistics:")
    print(f"  Total sampled: {len(selected_indices)}")
    print(f"  Target: {target_count}")

    # Show bins with insufficient samples
    insufficient_bins = [
        s for s in bin_stats if s["actual_samples"] < s["target_samples"]
    ]
    if insufficient_bins:
        print(f"\n  Bins with insufficient samples: {len(insufficient_bins)}")
        for stat in insufficient_bins[:5]:  # Show first 5
            print(
                f"    Bin {stat['bin']} ({stat['range']}): "
                f"wanted {stat['target_samples']}, got {stat['actual_samples']} "
                f"(available: {stat['available']})"
            )

    return selected_indices
