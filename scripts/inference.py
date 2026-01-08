import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import torch

from src.model import NPLikenessScorer, create_model


def load_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model type from checkpoint
    hparams = checkpoint.get("hyper_parameters", {})

    # Determine model type
    if "n_embd" in hparams:
        model_type = "gpt2"
        model_kwargs = {
            "n_embd": hparams.get("n_embd", 768),
            "n_layer": hparams.get("n_layer", 6),
            "n_head": hparams.get("n_head", 12),
        }
    else:
        model_type = "llama"
        model_kwargs = {
            "hidden_size": hparams.get("hidden_size", 1024),
            "num_hidden_layers": hparams.get("num_hidden_layers", 8),
            "num_attention_heads": hparams.get("num_attention_heads", 16),
            "intermediate_size": hparams.get("intermediate_size", 2816),
        }

    # Create model
    model = create_model(
        model_type=model_type,
        learning_rate=hparams.get("learning_rate", 5e-5),
        warmup_steps=hparams.get("warmup_steps", 1000),
        max_length=hparams.get("max_length", 256),
        tokenizer_name=hparams.get(
            "tokenizer_name", "kohbanye/SmilesTokenizer_PubChem_1M"
        ),
        **model_kwargs,
    )

    # Load state dict
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser(description="NP-likeness inference")
    parser.add_argument(
        "--natural-checkpoint",
        type=str,
        required=True,
        help="Path to natural model checkpoint",
    )
    parser.add_argument(
        "--synthetic-checkpoint",
        type=str,
        required=True,
        help="Path to synthetic model checkpoint",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        required=True,
        help="SMILES string to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--sigmoid-k",
        type=float,
        default=1.0,
        help="Sigmoid function slope parameter for normalization (default: 1.0)",
    )
    parser.add_argument(
        "--sigmoid-offset",
        type=float,
        default=0.0,
        help="Sigmoid function offset parameter for normalization (default: 0.0)",
    )
    parser.add_argument(
        "--scaffold_only",
        action="store_true",
        help="Extract and score Bemis-Murcko scaffolds only",
    )
    parser.add_argument(
        "--general-checkpoint",
        type=str,
        default=None,
        help="Path to general model checkpoint (optional)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mixing parameter for weighted scoring: α*P_syn + (1-α)*P_gen (default: 0.5)",
    )
    parser.add_argument(
        "--scoring-mode",
        type=str,
        default=None,
        help="Scoring mode: ratio, natural_only, synthetic_only, or weighted. "
             "If not specified, uses 'weighted' when general model is provided, otherwise 'ratio'",
    )

    args = parser.parse_args()

    # Load models
    print(f"Loading models on {args.device}...")
    natural_model = load_checkpoint(args.natural_checkpoint, args.device)
    synthetic_model = load_checkpoint(args.synthetic_checkpoint, args.device)

    # Load general model if provided
    general_model = None
    if args.general_checkpoint:
        print(f"Loading general model from {args.general_checkpoint}...")
        general_model = load_checkpoint(args.general_checkpoint, args.device)

    # Determine scoring mode
    if args.scoring_mode is None:
        scoring_mode = "weighted" if general_model is not None else "ratio"
    else:
        scoring_mode = args.scoring_mode

    print(f"Using scoring mode: {scoring_mode}")
    if scoring_mode == "weighted":
        print(f"  Alpha (mixing parameter): {args.alpha}")

    # Create scorer
    scorer = NPLikenessScorer(
        natural_model,
        synthetic_model,
        general_model=general_model,
        alpha=args.alpha,
        sigmoid_k=args.sigmoid_k,
        sigmoid_offset=args.sigmoid_offset,
        scaffold_only=args.scaffold_only,
        scoring_mode=scoring_mode,
    )

    # Calculate score
    print(f"\nEvaluating SMILES: {args.smiles}")
    details = scorer.score_with_details(args.smiles)

    # Print results
    print("\n=== NP-Likeness Score ===")
    print(f"Scoring mode: {scoring_mode}")
    print(f"Score: {details['score_normalized']:.4f}")
    print(f"  (sigmoid_k={args.sigmoid_k}, sigmoid_offset={args.sigmoid_offset})")
    print(f"Log P(natural): {details['log_p_natural']:.4f}")
    print(f"Log P(synthetic): {details['log_p_synthetic']:.4f}")
    print(f"Perplexity (natural): {details['perplexity_natural']:.2f}")
    print(f"Perplexity (synthetic): {details['perplexity_synthetic']:.2f}")

    if general_model is not None:
        print(f"Log P(general): {details['log_p_general']:.4f}")
        print(f"Perplexity (general): {details['perplexity_general']:.2f}")

        if scoring_mode == "weighted":
            print(f"Alpha: {args.alpha}")
            print(f"Log mixture: {details.get('log_mixture', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
