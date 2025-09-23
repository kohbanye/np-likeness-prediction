import argparse
import sys
from pathlib import Path

# Add the parent directory to the path to import src modules
sys.path.append(str(Path(__file__).parent.parent))

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

import wandb
from src.data import SMILESDataModule
from src.model import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train NP-likeness prediction models")

    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2",
        choices=["gpt2", "llama"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=True,
        choices=["natural", "synthetic"],
        help="Type of dataset to train on",
    )

    # Model architecture arguments
    parser.add_argument(
        "--n_embd", type=int, default=576, help="Embedding dimension (GPT-2)"
    )
    parser.add_argument(
        "--n_layer", type=int, default=6, help="Number of layers (GPT-2)"
    )
    parser.add_argument(
        "--n_head", type=int, default=12, help="Number of attention heads (GPT-2)"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Hidden size (Llama)"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=6, help="Number of layers (Llama)"
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=8,
        help="Number of attention heads (Llama)",
    )
    parser.add_argument(
        "--intermediate_size", type=int, default=1408, help="Intermediate size (Llama)"
    )

    # Training arguments
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--max_epochs", type=int, default=30, help="Maximum number of epochs"
    )
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Maximum number of samples to use"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--test_split", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loader workers"
    )

    # Tokenizer arguments
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="kohbanye/SmilesTokenizer_PubChem_1M",
        help="Tokenizer to use",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--project_name",
        type=str,
        default="np-likeness-prediction",
        help="Wandb project name",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--save_top_k", type=int, default=1, help="Save top k checkpoints"
    )

    # Hardware arguments
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="Accelerator type"
    )
    parser.add_argument("--devices", type=str, default="auto", help="Devices to use")
    parser.add_argument(
        "--precision", type=str, default="16-mixed", help="Training precision"
    )

    # Data augmentation arguments
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize SMILES strings during training for better robustness",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Convert SMILES to canonical form before training",
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )

    return parser.parse_args()


def setup_callbacks(args):
    """Setup training callbacks."""
    callbacks = []

    # Model checkpointing
    if args.val_split > 0:
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(args.checkpoint_dir)
            / f"{args.model_type}_{args.dataset_type}",
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=args.save_top_k,
            save_last=True,
            auto_insert_metric_name=False,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(args.checkpoint_dir)
            / f"{args.model_type}_{args.dataset_type}",
            filename="{epoch:02d}-{train_loss:.4f}",
            monitor="train_loss",
            mode="min",
            save_top_k=args.save_top_k,
            save_last=True,
            auto_insert_metric_name=False,
        )
    callbacks.append(checkpoint_callback)

    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    return callbacks


def setup_logger(args):
    """Setup experiment logger."""
    run_name = args.run_name or f"{args.model_type}_{args.dataset_type}"

    # Create config dict for wandb
    config = vars(args).copy()

    logger = WandbLogger(
        project=args.project_name,
        name=run_name,
        tags=[args.model_type, args.dataset_type],
        config=config,  # Log all hyperparameters
        log_model=True,  # Log model checkpoints to wandb
    )
    return logger


def main():
    args = parse_args()

    # Set seed for reproducibility
    L.seed_everything(args.seed, workers=True)

    # Create output directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print(f"Training {args.model_type} model on {args.dataset_type} dataset")
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Initialize data module
    print("\nInitializing data module...")
    data_module = SMILESDataModule(
        data_dir=args.data_dir,
        tokenizer_name=args.tokenizer_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        dataset_type=args.dataset_type,
        max_samples=args.max_samples,
        randomize=args.randomize,
        canonical=args.canonical,
    )

    # Initialize model
    print(f"\nInitializing {args.model_type} model...")
    if args.model_type == "gpt2":
        model_kwargs = {
            "n_embd": args.n_embd,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
        }
    elif args.model_type == "llama":
        model_kwargs = {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "intermediate_size": args.intermediate_size,
        }
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model = create_model(
        model_type=args.model_type,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        tokenizer_name=args.tokenizer_name,
        **model_kwargs,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup callbacks and logger
    callbacks = setup_callbacks(args)
    logger = setup_logger(args)

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train model
    print("\nStarting training...")
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=args.resume_from_checkpoint,
    )

    # Test model (optional)
    # print("\nTesting model...")
    # trainer.test(model=model, datamodule=data_module, ckpt_path="best")

    # Save final model
    final_model_path = (
        Path(args.checkpoint_dir)
        / f"{args.model_type}_{args.dataset_type}"
        / "final_model.ckpt"
    )
    trainer.save_checkpoint(final_model_path)

    print("\nTraining completed!")
    if hasattr(callbacks[0], "best_model_path") and callbacks[0].best_model_path:
        print(f"Best model saved at: {callbacks[0].best_model_path}")
    print(f"Final model saved at: {final_model_path}")

    # Finish wandb run
    wandb.finish()
    print(f"Wandb run finished. View at: https://wandb.ai/{args.project_name}")


if __name__ == "__main__":
    main()
