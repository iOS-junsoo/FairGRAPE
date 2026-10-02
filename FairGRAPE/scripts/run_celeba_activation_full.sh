#!/usr/bin/env bash
set -euo pipefail
cd /workspace/FairGRAPE/FairGRAPE
# Reporting runs separately so changing displayed metrics never restarts training.
task_report_output=full_results/celeba_activation_full_seed42_20260927
task_report_expects_output=false
for task_report_arg in "$@"; do
  if "$task_report_expects_output"; then
    task_report_output="$task_report_arg"
    task_report_expects_output=false
  elif [[ "$task_report_arg" == --output-dir ]]; then
    task_report_expects_output=true
  elif [[ "$task_report_arg" == --output-dir=* ]]; then
    task_report_output="${task_report_arg#--output-dir=}"
  fi
done
mkdir -p "$(dirname -- "$task_report_output")"
/usr/bin/python3 -u scripts/report_celeba_legacy_eo.py \
  --source-output "$task_report_output" --watch --follow-pid "$$" \
  >> "${task_report_output}.report.console.log" 2>&1 &
exec /workspace/miniconda/envs/fairgrape_cu128/bin/python -u celeba_layerwise_full.py \
  --checkpoint trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_0.pt \
  --methods A_activation B_activation --defer-contribution \
  --reuse-initial-stages-from full_results/celeba_layerwise_seed42_20260927 \
  --seed 42 --rounds 3 --retrain-epochs 5 \
  --output-dir full_results/celeba_activation_full_seed42_20260927 "$@"
