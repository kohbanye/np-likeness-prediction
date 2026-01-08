import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    calculate_diversity_metrics,
    calculate_molecular_weights_parallel,
    distribution_matching_sampling,
    load_coconut_molecular_weights,
)


def load_zinc_smiles(input_file: Path) -> tuple[list[str], list[str]]:
    """Load SMILES from ZINC format file.

    Args:
        input_file: Path to ZINC file (tab-separated: tranche, zincid, SMILES)

    Returns:
        Tuple of (smiles_list, original_lines)
    """
    print(f"Loading SMILES from {input_file}...")
    smiles_list = []
    original_lines = []

    with open(input_file, "r") as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading SMILES")):
            line = line.strip()

            # Skip empty lines and header
            if not line or line_num == 0:
                continue

            original_lines.append(line)
            parts = line.split()

            if len(parts) >= 3:
                smiles = parts[2]  # Third column is SMILES
                if smiles and len(smiles) < 500:
                    smiles_list.append(smiles)
            else:
                smiles_list.append("")  # Keep alignment with original_lines

    print(f"Loaded {len(smiles_list)} SMILES")
    return smiles_list, original_lines


def main():
    parser = argparse.ArgumentParser(
        description="Create subset from ZINC dataset matching COCONUT molecular weight distribution"
    )
    parser.add_argument(
        "--coconut_file",
        type=Path,
        default=Path("data/coconut.csv"),
        help="Input COCONUT CSV file path",
    )
    parser.add_argument(
        "--zinc_file",
        type=Path,
        default=Path("data/zinc22_9m.txt"),
        help="Input ZINC file path",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/diverse_synthetic"),
        help="Output directory",
    )
    parser.add_argument(
        "--target_count",
        type=int,
        default=700000,
        help="Target number of compounds",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=16,
        help="Number of parallel processes",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=50,
        help="Number of histogram bins for distribution matching",
    )
    parser.add_argument(
        "--max_mw",
        type=float,
        default=1000.0,
        help="Maximum molecular weight to consider",
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip diversity metrics calculation (faster)",
    )

    args = parser.parse_args()

    # Setup paths
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load COCONUT molecular weight distribution
    print("=" * 60)
    print("STEP 1: Loading COCONUT molecular weight distribution")
    print("=" * 60)
    coconut_mw = load_coconut_molecular_weights(
        args.coconut_file,
        max_mw=args.max_mw,
    )

    # Step 2: Load ZINC SMILES
    print("\n" + "=" * 60)
    print("STEP 2: Loading ZINC SMILES")
    print("=" * 60)
    smiles_list, original_lines = load_zinc_smiles(args.zinc_file)

    # Step 3: Calculate molecular weights for ZINC compounds
    print("\n" + "=" * 60)
    print("STEP 3: Calculating molecular weights for ZINC compounds")
    print("=" * 60)
    zinc_mw = calculate_molecular_weights_parallel(
        smiles_list,
        n_jobs=args.n_jobs,
    )

    print(f"\nCalculated molecular weights for {len(zinc_mw)} compounds")

    # Step 4: Sample to match COCONUT distribution
    print("\n" + "=" * 60)
    print("STEP 4: Sampling to match COCONUT distribution")
    print("=" * 60)
    selected_indices = distribution_matching_sampling(
        target_distribution=coconut_mw,
        candidate_values=zinc_mw,
        target_count=args.target_count,
        n_bins=args.n_bins,
    )

    print(f"\nSelected {len(selected_indices)} compounds")

    # Step 5: Calculate diversity metrics (optional)
    if not args.skip_metrics:
        print("\n" + "=" * 60)
        print("STEP 5: Calculating diversity metrics")
        print("=" * 60)
        metrics = calculate_diversity_metrics(smiles_list, selected_indices)

        print("\n" + "=" * 60)
        print("DIVERSITY METRICS")
        print("=" * 60)
        print(f"Selected compounds: {metrics['n_selected']}")
        print(f"Unique scaffolds: {metrics['n_unique_scaffolds']}")
        print(f"Scaffold diversity ratio: {metrics['scaffold_diversity_ratio']:.3f}")
        print(f"Average Tanimoto similarity: {metrics['avg_tanimoto']:.3f}")
        print(f"Std Tanimoto similarity: {metrics['std_tanimoto']:.3f}")
        print(
            f"High similarity pairs (>0.7): {metrics['high_similarity_pairs_pct']:.2f}%"
        )
        print(
            f"\nMolecular weight: {metrics['mol_weight_mean']:.1f} ± {metrics['mol_weight_std']:.1f}"
        )
        print(
            f"  Range: {metrics['mol_weight_min']:.1f} - {metrics['mol_weight_max']:.1f}"
        )
        print(f"LogP: {metrics['logp_mean']:.2f} ± {metrics['logp_std']:.2f}")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("STEP 5: Skipping diversity metrics calculation (--skip_metrics)")
        print("=" * 60)
        metrics = {"n_selected": len(selected_indices)}

    # Step 6: Save outputs
    print("\n" + "=" * 60)
    print("STEP 6: Saving outputs")
    print("=" * 60)
    print(f"Saving to {output_dir}...")

    # Save in ZINC format (tranche zincid SMILES)
    output_file = output_dir / "zinc22_diverse.txt"
    with open(output_file, "w") as f:
        # Write header
        f.write("tranche zincid SMILES\n")
        # Write selected compounds
        for idx in tqdm(selected_indices, desc="Writing ZINC format"):
            if idx < len(original_lines):
                f.write(original_lines[idx] + "\n")

    print(f"Saved {len(selected_indices)} compounds to {output_file}")

    # Save metadata
    metadata_file = output_dir / "metadata.json"
    metadata = {
        "coconut_file": str(args.coconut_file),
        "zinc_file": str(args.zinc_file),
        "n_coconut_mw": len(coconut_mw),
        "n_zinc_input": len(smiles_list),
        "n_output": len(selected_indices),
        "n_jobs": args.n_jobs,
        "n_bins": args.n_bins,
        "max_mw": args.max_mw,
        "coconut_mw_stats": {
            "mean": float(coconut_mw.mean()),
            "std": float(coconut_mw.std()),
            "min": float(coconut_mw.min()),
            "max": float(coconut_mw.max()),
        },
        "diversity_metrics": metrics,
    }

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"COCONUT molecular weights: {len(coconut_mw)}")
    print(f"  Mean: {coconut_mw.mean():.1f} ± {coconut_mw.std():.1f}")
    print(f"ZINC compounds sampled: {len(selected_indices)}")
    if not args.skip_metrics:
        print(
            f"  Mean MW: {metrics['mol_weight_mean']:.1f} ± {metrics['mol_weight_std']:.1f}"
        )
        print(f"  Unique scaffolds: {metrics['n_unique_scaffolds']}")
    print("=" * 60)

    print("\nDone!")


if __name__ == "__main__":
    main()
