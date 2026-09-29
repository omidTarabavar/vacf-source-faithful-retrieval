#!/usr/bin/env bash
set -Eeuo pipefail

export HF_HOME=/workspace/.hf_home
export HF_HUB_CACHE=/workspace/.hf_home/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/root/nltk_data
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace

echo "===== HARDWARE ====="
nvidia-smi

echo "===== START XXP ====="
date -Is

/workspace/stage9f_vl_env/bin/python \
  /workspace/benchmark_val152_efficiency.py \
  --method xxp \
  --count 152 \
  --out /workspace/EFF152_XXP.json \
  > /workspace/eff152_xxp.log 2>&1

echo "===== XXP COMPLETE ====="
date -Is

echo "===== START VACF ====="
date -Is

/workspace/stage9f_vl_env/bin/python \
  /workspace/benchmark_val152_efficiency.py \
  --method vacf \
  --count 152 \
  --out /workspace/EFF152_VACF.json \
  > /workspace/eff152_vacf.log 2>&1

echo "===== VACF COMPLETE ====="
date -Is

echo "ALL_DONE"
