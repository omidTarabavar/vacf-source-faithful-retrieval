#!/usr/bin/env bash
set -euo pipefail

source /workspace/stage8b_env/bin/activate

export HF_HOME=/workspace/.hf_home
export HF_HUB_CACHE=/workspace/FINAL_HF_CACHE/hub
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT=/workspace/STAGE8B_HANDOFF_FINAL/scripts/stage7_run_expert.py

STATE=/workspace/TEST352_STATE
META=/workspace/TEST352_META/predictions
IMGROOT=/workspace/AVERIMATEC_TEST_OFFICIAL/extracted/test_data/images

ISTORE=/workspace/AVERIMATEC_TEST_KS/test/converted_datastore/text_related/image_related_store_text_test
TSTORE=/workspace/AVERIMATEC_TEST_KS/test/converted_datastore/text_related/text_related_store_text_test

OUT=/workspace/TEST352_STAGE7

echo "=================================================="
echo "BASE START $(date)"
echo "=================================================="

/workspace/stage8b_env/bin/python \
  "$SCRIPT" \
  --model-key base \
  --batch 64 \
  --v4 "$STATE" \
  --xxp "$META" \
  --image-root "$IMGROOT" \
  --istore "$ISTORE" \
  --tstore "$TSTORE" \
  --out "$OUT"

echo
echo "BASE COMPLETE $(date)"

echo
echo "=================================================="
echo "SIGLIP2 START $(date)"
echo "=================================================="

/workspace/stage8b_env/bin/python \
  "$SCRIPT" \
  --model-key siglip2 \
  --batch 64 \
  --v4 "$STATE" \
  --xxp "$META" \
  --image-root "$IMGROOT" \
  --istore "$ISTORE" \
  --tstore "$TSTORE" \
  --out "$OUT"

echo
echo "SIGLIP2 COMPLETE $(date)"
echo "TEST352_STAGE7_FULL_COMPLETE"
