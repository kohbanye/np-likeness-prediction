import math

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class BaseLanguageModel(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 5e-5,
        warmup_steps: int = 1000,
        max_length: int = 256,
        tokenizer_name: str = "kohbanye/SmilesTokenizer_PubChem_1M",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_length = max_length

        # Load SMILES tokenizer
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model will be initialized in subclasses
        self.model: PreTrainedModel | None = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        if self.model is None:
            raise ValueError("Model is not initialized.")
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train_perplexity",
            torch.exp(loss),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_perplexity",
            torch.exp(loss),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = self(input_ids, attention_mask, labels)
        loss = outputs.loss

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "test_perplexity",
            torch.exp(loss),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def configure_optimizers(self):
        max_epochs = self.trainer.max_epochs
        if max_epochs is None:
            raise ValueError("Max epochs is not set.")

        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def tokenize_smiles(self, smiles: list[str]) -> BatchEncoding:
        """Tokenize SMILES strings for model input."""
        return self.tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def calculate_perplexity(self, smiles: str) -> float:
        """Calculate perplexity for a single SMILES string."""
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                [smiles],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Labels are the same as input_ids for language modeling
            labels = input_ids.clone()
            # Mask padding tokens in labels
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self(input_ids, attention_mask, labels)
            perplexity = torch.exp(outputs.loss).item()

        return perplexity

    def calculate_log_likelihood(self, smiles: str) -> float:
        """Calculate log likelihood for a single SMILES string."""
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                [smiles],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Labels are the same as input_ids for language modeling
            labels = input_ids.clone()
            # Mask padding tokens in labels
            labels[labels == self.tokenizer.pad_token_id] = -100

            outputs = self(input_ids, attention_mask, labels)
            # Negative log likelihood (cross entropy loss)
            nll = outputs.loss.item()

            # Return log likelihood (negative of NLL)
            return -nll

    def batch_calculate_log_likelihood(self, smiles_list: list[str]) -> list[float]:
        """Calculate log likelihood for a batch of SMILES strings."""
        self.eval()
        log_likelihoods = []

        with torch.no_grad():
            for smiles in smiles_list:
                ll = self.calculate_log_likelihood(smiles)
                log_likelihoods.append(ll)

        return log_likelihoods


class GPT2Model(BaseLanguageModel):
    def __init__(
        self,
        n_embd: int = 576,
        n_layer: int = 6,
        n_head: int = 12,
        learning_rate: float = 5e-4,
        warmup_steps: int = 1000,
        max_length: int = 512,
        tokenizer_name: str = "kohbanye/SmilesTokenizer_PubChem_1M",
    ):
        super().__init__(
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            max_length=max_length,
            tokenizer_name=tokenizer_name,
        )

        # Initialize GPT-2 configuration
        config = GPT2Config(
            vocab_size=len(self.tokenizer),
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_positions=max_length,
            # for kohbanye/SmilesTokenizer_PubChem_1M tokenizer
            bos_token_id=12,
            eos_token_id=13,
            pad_token_id=0,
        )

        # Initialize GPT-2 model
        self.model = GPT2LMHeadModel(config)


class LlamaModel(BaseLanguageModel):
    def __init__(
        self,
        model_name: str = "llama",
        hidden_size: int = 1024,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 16,
        intermediate_size: int = 2816,
        learning_rate: float = 5e-4,
        warmup_steps: int = 1000,
        max_length: int = 512,
        tokenizer_name: str = "kohbanye/SmilesTokenizer_PubChem_1M",
    ):
        super().__init__(
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            max_length=max_length,
            tokenizer_name=tokenizer_name,
        )

        # Initialize Llama configuration
        config = LlamaConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_length,
            # for kohbanye/SmilesTokenizer_PubChem_1M tokenizer
            bos_token_id=12,
            eos_token_id=13,
            pad_token_id=0,
        )

        # Initialize Llama model
        self.model = LlamaForCausalLM(config)


class NPLikenessScorer:
    """Calculate Natural Product Likeness Score using log-likelihood ratio."""

    def __init__(
        self,
        natural_model: BaseLanguageModel,
        synthetic_model: BaseLanguageModel,
        sigmoid_k: float = 1.0,
        sigmoid_offset: float = 0.0,
    ):
        self.natural_model = natural_model
        self.synthetic_model = synthetic_model
        self.sigmoid_k = sigmoid_k
        self.sigmoid_offset = sigmoid_offset

        # Ensure models are in eval mode
        self.natural_model.eval()
        self.synthetic_model.eval()

    def score(self, smiles: str) -> float:
        """
        Calculate NP-likeness score for a SMILES string.

        Score = log P(x|natural) - log P(x|synthetic)

        Higher scores indicate more natural product-like.
        Lower scores indicate more synthetic-like.
        """
        log_p_natural = self.natural_model.calculate_log_likelihood(smiles)
        log_p_synthetic = self.synthetic_model.calculate_log_likelihood(smiles)

        return log_p_natural - log_p_synthetic

    def batch_score(self, smiles_list: list[str]) -> list[float]:
        """Calculate NP-likeness scores for a batch of SMILES strings."""
        log_p_natural_list = self.natural_model.batch_calculate_log_likelihood(
            smiles_list
        )
        log_p_synthetic_list = self.synthetic_model.batch_calculate_log_likelihood(
            smiles_list
        )

        scores = [
            log_p_nat - log_p_syn
            for log_p_nat, log_p_syn in zip(log_p_natural_list, log_p_synthetic_list)
        ]

        return scores

    def _sigmoid_normalize(self, llr: float) -> float:
        """
        Apply sigmoid normalization to log-likelihood ratio.

        Args:
            llr: Log-likelihood ratio (log P(x|natural) - log P(x|synthetic))

        Returns:
            Normalized score in range [0, 1]
        """
        adjusted_llr = llr + self.sigmoid_offset
        return 1.0 / (1.0 + math.exp(-self.sigmoid_k * adjusted_llr))

    def score_normalized(self, smiles: str) -> float:
        """
        Calculate normalized NP-likeness score for a SMILES string.

        Returns score in range [0, 1] where:
        - Values closer to 1 indicate more natural product-like
        - Values closer to 0 indicate more synthetic-like
        """
        raw_score = self.score(smiles)
        return self._sigmoid_normalize(raw_score)

    def batch_score_normalized(self, smiles_list: list[str]) -> list[float]:
        """Calculate normalized NP-likeness scores for a batch of SMILES strings."""
        raw_scores = self.batch_score(smiles_list)
        return [self._sigmoid_normalize(score) for score in raw_scores]

    def score_with_details(self, smiles: str) -> dict[str, float]:
        """
        Calculate NP-likeness score with detailed breakdown.

        Returns dictionary with:
        - score: Raw NP-likeness score (log-likelihood ratio)
        - score_normalized: Normalized score in range [0, 1]
        - log_p_natural: Log likelihood under natural model
        - log_p_synthetic: Log likelihood under synthetic model
        - perplexity_natural: Perplexity under natural model
        - perplexity_synthetic: Perplexity under synthetic model
        """
        log_p_natural = self.natural_model.calculate_log_likelihood(smiles)
        log_p_synthetic = self.synthetic_model.calculate_log_likelihood(smiles)
        perp_natural = self.natural_model.calculate_perplexity(smiles)
        perp_synthetic = self.synthetic_model.calculate_perplexity(smiles)

        raw_score = log_p_natural - log_p_synthetic
        normalized_score = self._sigmoid_normalize(raw_score)

        return {
            "score": raw_score,
            "score_normalized": normalized_score,
            "log_p_natural": log_p_natural,
            "log_p_synthetic": log_p_synthetic,
            "perplexity_natural": perp_natural,
            "perplexity_synthetic": perp_synthetic,
        }


def create_model(model_type: str = "gpt2", **kwargs) -> BaseLanguageModel:
    """Factory function to create language models."""
    if model_type.lower() == "gpt2":
        return GPT2Model(**kwargs)
    elif model_type.lower() == "llama":
        return LlamaModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
