#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=8:00:00
#$ -N npl_synthetic
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"

uv run python scripts/train.py \
  --dataset_type synthetic \
  --run_name "gpt2_synthetic_$(date +%Y%m%d_%H%M)" \
  --model_type gpt2 \
  --max_samples 700000 \
  --project_name np-likeness-prediction \
  --checkpoint_dir "checkpoints/$(date +%Y%m%d_%H%M)"