#!/bin/sh
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=8:00:00
#$ -N npl_natural
. /etc/profile.d/modules.sh

export PATH="$HOME/.local/bin:$PATH"

uv run python scripts/train.py \
  --dataset_type natural \
  --run_name "gpt2_natural_$(date +%Y%m%d_%H%M)" \
  --model_type gpt2 \
  --project_name np-likeness-prediction \
  --checkpoint_dir "checkpoints/$(date +%Y%m%d_%H%M)"