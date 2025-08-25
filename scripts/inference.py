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

    args = parser.parse_args()

    # Load models
    print(f"Loading models on {args.device}...")
    natural_model = load_checkpoint(args.natural_checkpoint, args.device)
    synthetic_model = load_checkpoint(args.synthetic_checkpoint, args.device)

    # Create scorer
    scorer = NPLikenessScorer(natural_model, synthetic_model)

    # Calculate score
    print(f"\nEvaluating SMILES: {args.smiles}")
    details = scorer.score_with_details(args.smiles)

    # Print results
    print("\n=== NP-Likeness Score ===")
    print(f"Score: {details['score']:.4f}")
    print(f"Log P(natural): {details['log_p_natural']:.4f}")
    print(f"Log P(synthetic): {details['log_p_synthetic']:.4f}")
    print(f"Perplexity (natural): {details['perplexity_natural']:.2f}")
    print(f"Perplexity (synthetic): {details['perplexity_synthetic']:.2f}")

    # Interpretation
    if details["score"] > 0:
        print("\n→ Natural product-like (score > 0)")
    else:
        print("\n→ Synthetic-like (score < 0)")


if __name__ == "__main__":
    main()
