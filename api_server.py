from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import NPLikenessScorer, create_model


def load_checkpoint(
    checkpoint_path: str,
    model_type: Literal["gpt2", "llama"] = "gpt2",
    device: str = "cuda",
):
    """Load model from checkpoint.

    This duplicates the logic in scripts/inference.py to avoid cross-module import issues.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model type from checkpoint
    hparams = checkpoint.get("hyper_parameters", {})

    # Determine model type
    if model_type == "gpt2":
        model_kwargs = {
            "n_embd": hparams.get("n_embd", 768),
            "n_layer": hparams.get("n_layer", 6),
            "n_head": hparams.get("n_head", 12),
        }
    else:
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


app = FastAPI(title="NP-Likeness Inference API")


class ScoreRequest(BaseModel):
    smiles: str
    natural_checkpoint: str = "checkpoints/20250918_2220/gpt2_natural/last.ckpt"
    synthetic_checkpoint: str = "checkpoints/20250918_2220/gpt2_synthetic/last.ckpt"
    model_type: Literal["gpt2", "llama"] = "gpt2"
    sigmoid_k: float = 1.0
    sigmoid_offset: float = 0.0


class ScoreResponse(BaseModel):
    score: float
    log_p_natural: float
    log_p_synthetic: float
    perplexity_natural: float
    perplexity_synthetic: float
    sigmoid_k: float
    sigmoid_offset: float


class BatchScoreRequest(BaseModel):
    smiles_list: list[str]
    natural_checkpoint: str = "checkpoints/20250918_2220/gpt2_natural/last.ckpt"
    synthetic_checkpoint: str = "checkpoints/20250918_2220/gpt2_synthetic/last.ckpt"
    model_type: Literal["gpt2", "llama"] = "gpt2"
    sigmoid_k: float = 1.0
    sigmoid_offset: float = 0.0


class BatchScoreItem(BaseModel):
    smiles: str
    score: float
    log_p_natural: float
    log_p_synthetic: float
    perplexity_natural: float
    perplexity_synthetic: float


class BatchScoreResponse(BaseModel):
    results: list[BatchScoreItem]
    sigmoid_k: float
    sigmoid_offset: float
    model_type: Literal["gpt2", "llama"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        natural_model = load_checkpoint(
            req.natural_checkpoint, model_type=req.model_type, device=device
        )
        synthetic_model = load_checkpoint(
            req.synthetic_checkpoint, model_type=req.model_type, device=device
        )

        scorer = NPLikenessScorer(
            natural_model,
            synthetic_model,
            sigmoid_k=req.sigmoid_k,
            sigmoid_offset=req.sigmoid_offset,
        )

        details = scorer.score_with_details(req.smiles)

        return ScoreResponse(
            score=float(details["score_normalized"]),
            log_p_natural=float(details["log_p_natural"]),
            log_p_synthetic=float(details["log_p_synthetic"]),
            perplexity_natural=float(details["perplexity_natural"]),
            perplexity_synthetic=float(details["perplexity_synthetic"]),
            sigmoid_k=req.sigmoid_k,
            sigmoid_offset=req.sigmoid_offset,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score-batch", response_model=BatchScoreResponse)
def score_batch(req: BatchScoreRequest):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        natural_model = load_checkpoint(
            req.natural_checkpoint, model_type=req.model_type, device=device
        )
        synthetic_model = load_checkpoint(
            req.synthetic_checkpoint, model_type=req.model_type, device=device
        )

        scorer = NPLikenessScorer(
            natural_model,
            synthetic_model,
            sigmoid_k=req.sigmoid_k,
            sigmoid_offset=req.sigmoid_offset,
        )

        results: list[BatchScoreItem] = []
        for smiles in req.smiles_list:
            details = scorer.score_with_details(smiles)
            results.append(
                BatchScoreItem(
                    smiles=smiles,
                    score=float(details["score_normalized"]),
                    log_p_natural=float(details["log_p_natural"]),
                    log_p_synthetic=float(details["log_p_synthetic"]),
                    perplexity_natural=float(details["perplexity_natural"]),
                    perplexity_synthetic=float(details["perplexity_synthetic"]),
                )
            )

        return BatchScoreResponse(
            results=results,
            sigmoid_k=req.sigmoid_k,
            sigmoid_offset=req.sigmoid_offset,
            model_type=req.model_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
