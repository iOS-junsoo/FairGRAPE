#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FairGRAPE/FairGRAPE
exec /workspace/miniconda/envs/fairgrape_cu128/bin/python -u celeba_layerwise_full.py \
  --checkpoint trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_0.pt \
  --methods A_activation B_activation A_contribution B_contribution \
  --seed 42 --rounds 3 --retrain-epochs 5 \
  --output-dir full_results/celeba_layerwise_seed42_20260927 "$@"
