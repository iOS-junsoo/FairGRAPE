#!/usr/bin/env bash
# DogsCats: 사전학습 없는 MobileNet-v2 덴스 모델을 β별로 학습해 acc/EO를 비교한다.
# β=0 은 편향 주입 없음(자연 분포 train). val/test 는 모든 β 에서 동일한 4셀 균형(각 309장).
# 사용: bash scripts/run_dogscats_scratch_dense_sweep.sh            # β = 0 5 10 20 50 100 순차 실행
#       bash scripts/run_dogscats_scratch_dense_sweep.sh 5 20        # 지정 β만
# 로그: ~/dogscats_dense_scratch_b<β>_s42.log
# 모델: trained_model/unpruned/DogsCats_unpruned_species_bycolor_mobilenetv2_scratch_<β>.pt
# 요약: python scripts/summarize_dogscats_dense.py
set -euo pipefail
cd "$(dirname "$0")/.."
source /workspace/miniconda/etc/profile.d/conda.sh && conda activate fairgrape
BETAS=("$@"); [ ${#BETAS[@]} -eq 0 ] && BETAS=(0 5 10 20 50 100)
for b in "${BETAS[@]}"; do
  echo "===== beta=$b  $(date '+%F %T') ====="
  PYTHONPATH=/workspace/FairGRAPE/np_compat PYTHONUNBUFFERED=1 python main_test.py \
    --dataset DogsCats --network mobilenetv2 --prune_type Full \
    --loss_type species --sensitive_group color --skew_beta "$b" \
    --batch 256 --no_grl --no_pretrained \
    --skip_readable_check --seed 42 --exp_idx "$b" 2>&1 | tee ~/dogscats_dense_scratch_b"$b"_s42.log
done
echo "===== done $(date '+%F %T') ====="
