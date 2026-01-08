import math

import lightning as L
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
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

    def _calculate_log_likelihood(self, smiles: str) -> float:
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

    def _calculate_log_likelihoods_batch(self, smiles_list: list[str]) -> list[float]:
        """Calculate log likelihoods for a batch of SMILES (internal helper)."""
        inputs = self.tokenizer(
            smiles_list,
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

        # Get logits from model
        outputs = self(input_ids, attention_mask, labels=None)
        logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Calculate loss per sample
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        # Reshape for loss calculation
        shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels_flat = shift_labels.view(-1)

        # Get per-token losses
        per_token_loss = loss_fct(shift_logits_flat, shift_labels_flat)
        per_token_loss = per_token_loss.view(shift_labels.size())

        # Calculate per-sample loss (average over non-padding tokens)
        # mask out padding tokens (where labels == -100)
        mask = (shift_labels != -100).float()
        per_sample_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)

        # Convert to log likelihoods (negative of loss)
        log_likelihoods = (-per_sample_loss).cpu().tolist()

        return log_likelihoods

    def batch_calculate_log_likelihood(
        self, smiles_list: list[str], batch_size: int = 1000
    ) -> list[float]:
        """
        Calculate log likelihood for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings
            batch_size: Number of SMILES to process at once (default: 1000)

        Returns:
            List of log likelihoods
        """
        self.eval()
        log_likelihoods = []

        with torch.no_grad():
            # Process in chunks to avoid memory issues
            for i in tqdm(
                range(0, len(smiles_list), batch_size),
                desc="Calculating log likelihoods",
            ):
                batch = smiles_list[i : i + batch_size]
                batch_lls = self._calculate_log_likelihoods_batch(batch)
                log_likelihoods.extend(batch_lls)

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
        general_model: BaseLanguageModel | None = None,
        alpha: float = 0.5,
        sigmoid_k: float = 1.0,
        sigmoid_offset: float = 0.0,
        scaffold_only: bool = False,
        scoring_mode: str = "ratio",
    ):
        self.natural_model = natural_model
        self.synthetic_model = synthetic_model
        self.general_model = general_model
        self.alpha = alpha
        self.sigmoid_k = sigmoid_k
        self.sigmoid_offset = sigmoid_offset
        self.scaffold_only = scaffold_only
        self.scoring_mode = scoring_mode

        # Validate alpha
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in range [0, 1], got {alpha}")

        # Validate scoring_mode
        valid_modes = ["ratio", "natural_only", "synthetic_only", "weighted"]
        if scoring_mode not in valid_modes:
            raise ValueError(
                f"Invalid scoring_mode: {scoring_mode}. "
                f"Must be one of {valid_modes}"
            )

        # Validate weighted mode requirements
        if scoring_mode == "weighted" and general_model is None:
            raise ValueError(
                "scoring_mode='weighted' requires general_model to be provided"
            )

        # Ensure models are in eval mode
        self.natural_model.eval()
        self.synthetic_model.eval()
        if self.general_model is not None:
            self.general_model.eval()

    def _extract_scaffold(self, smiles: str) -> str:
        """Extract Bemis-Murcko scaffold from SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles  # Return original if parsing fails
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, isomericSmiles=False)
        except Exception:
            return smiles  # Return original if scaffold extraction fails

    def _process_smiles(self, smiles: str) -> str:
        """Process SMILES with optional scaffold extraction."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Extract scaffold if scaffold_only mode is enabled
        if self.scaffold_only:
            try:
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                mol = scaffold
            except Exception:
                pass  # Keep original molecule if scaffold extraction fails

        return Chem.MolToSmiles(mol, doRandom=True, canonical=False)

    def _log_weighted_mixture(self, log_p_syn: float, log_p_gen: float) -> float:
        """
        Compute log(α * P_syn + (1-α) * P_gen) using log-sum-exp trick.

        Args:
            log_p_syn: Log probability under synthetic model
            log_p_gen: Log probability under general model

        Returns:
            Log of the weighted mixture probability
        """
        import math

        # Handle edge cases
        if self.alpha == 1.0:
            return log_p_syn
        if self.alpha == 0.0:
            return log_p_gen

        # Log-sum-exp trick for numerical stability
        max_log_p = max(log_p_syn, log_p_gen)
        log_alpha = math.log(self.alpha)
        log_one_minus_alpha = math.log(1.0 - self.alpha)

        result = max_log_p + math.log(
            math.exp(log_alpha + log_p_syn - max_log_p) +
            math.exp(log_one_minus_alpha + log_p_gen - max_log_p)
        )

        return result

    def score(self, smiles: str) -> float:
        """
        Calculate NP-likeness score for a SMILES string.

        Scoring modes:
        - "ratio": log P(x|natural) - log P(x|synthetic) (default)
        - "natural_only": log P(x|natural)
        - "synthetic_only": log P(x|synthetic)
        - "weighted": log P(x|natural) - log(α*P(x|synthetic) + (1-α)*P(x|general))

        For "ratio" and "weighted" modes:
        - Higher scores indicate more natural product-like.
        - Lower scores indicate more synthetic/general-like.
        """
        smiles = self._process_smiles(smiles)

        if self.scoring_mode == "ratio":
            log_p_natural = self.natural_model._calculate_log_likelihood(smiles)
            log_p_synthetic = self.synthetic_model._calculate_log_likelihood(smiles)
            return log_p_natural - log_p_synthetic

        elif self.scoring_mode == "weighted":
            log_p_natural = self.natural_model._calculate_log_likelihood(smiles)
            log_p_synthetic = self.synthetic_model._calculate_log_likelihood(smiles)
            log_p_general = self.general_model._calculate_log_likelihood(smiles)
            log_mixture = self._log_weighted_mixture(log_p_synthetic, log_p_general)
            return log_p_natural - log_mixture

        elif self.scoring_mode == "natural_only":
            return self.natural_model._calculate_log_likelihood(smiles)

        elif self.scoring_mode == "synthetic_only":
            return self.synthetic_model._calculate_log_likelihood(smiles)

        else:
            # This should never happen due to validation in __init__
            raise ValueError(f"Invalid scoring_mode: {self.scoring_mode}")

    def batch_score(self, smiles_list: list[str]) -> list[float]:
        """Calculate NP-likeness scores for a batch of SMILES strings."""
        smiles_list = [self._process_smiles(smiles) for smiles in smiles_list]

        if self.scoring_mode == "ratio":
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

        elif self.scoring_mode == "weighted":
            log_p_natural_list = self.natural_model.batch_calculate_log_likelihood(
                smiles_list
            )
            log_p_synthetic_list = self.synthetic_model.batch_calculate_log_likelihood(
                smiles_list
            )
            log_p_general_list = self.general_model.batch_calculate_log_likelihood(
                smiles_list
            )
            scores = [
                log_p_nat - self._log_weighted_mixture(log_p_syn, log_p_gen)
                for log_p_nat, log_p_syn, log_p_gen in zip(
                    log_p_natural_list, log_p_synthetic_list, log_p_general_list
                )
            ]

        elif self.scoring_mode == "natural_only":
            scores = self.natural_model.batch_calculate_log_likelihood(smiles_list)

        elif self.scoring_mode == "synthetic_only":
            scores = self.synthetic_model.batch_calculate_log_likelihood(smiles_list)

        else:
            # This should never happen due to validation in __init__
            raise ValueError(f"Invalid scoring_mode: {self.scoring_mode}")

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
        smiles = self._process_smiles(smiles)

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
        - score: Raw NP-likeness score
        - score_normalized: Normalized score in range [0, 1]
        - log_p_natural: Log likelihood under natural model
        - log_p_synthetic: Log likelihood under synthetic model
        - perplexity_natural: Perplexity under natural model
        - perplexity_synthetic: Perplexity under synthetic model

        If general_model is provided:
        - log_p_general: Log likelihood under general model
        - perplexity_general: Perplexity under general model
        - log_mixture: Log of weighted mixture (only if scoring_mode='weighted')
        """
        smiles = self._process_smiles(smiles)

        log_p_natural = self.natural_model._calculate_log_likelihood(smiles)
        log_p_synthetic = self.synthetic_model._calculate_log_likelihood(smiles)
        perp_natural = self.natural_model.calculate_perplexity(smiles)
        perp_synthetic = self.synthetic_model.calculate_perplexity(smiles)

        result = {
            "log_p_natural": log_p_natural,
            "log_p_synthetic": log_p_synthetic,
            "perplexity_natural": perp_natural,
            "perplexity_synthetic": perp_synthetic,
        }

        # Add general model details if available
        if self.general_model is not None:
            log_p_general = self.general_model._calculate_log_likelihood(smiles)
            perp_general = self.general_model.calculate_perplexity(smiles)
            result["log_p_general"] = log_p_general
            result["perplexity_general"] = perp_general

            if self.scoring_mode == "weighted":
                log_mixture = self._log_weighted_mixture(log_p_synthetic, log_p_general)
                result["log_mixture"] = log_mixture

        # Calculate raw score using current scoring mode
        raw_score = self.score(smiles)
        normalized_score = self._sigmoid_normalize(raw_score)

        result["score"] = raw_score
        result["score_normalized"] = normalized_score

        return result


def create_model(model_type: str = "gpt2", **kwargs) -> BaseLanguageModel:
    """Factory function to create language models."""
    if model_type.lower() == "gpt2":
        return GPT2Model(**kwargs)
    elif model_type.lower() == "llama":
        return LlamaModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
