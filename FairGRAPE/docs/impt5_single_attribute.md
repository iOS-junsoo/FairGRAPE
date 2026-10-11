# CelebA impt=5: 공식 split 단일 속성 실험

ImageNet MobileNetV2 (`IMAGENET1K_V1`)에서 시작하여 선택 속성 하나로
파인튜닝하고, 그 모델을 프루닝·재학습합니다. 분류기는 2출력입니다.
기여도 정규화·혼합·채널 선택은 impt=2의 구현을 공유합니다.
기존 CelebA impt=2는 무작위 분할·39개 속성을 사용하므로 두 결과는
프루닝 점수 계산 방식만 다른 통제 실험이 아닙니다.

## 데이터 준비

기존 FairGRAPE Python 환경에서 실행합니다. 기본 경로는
`csv/CelebA.csv`, `Images/CelebA/img_align_celeba`입니다.

```bash
python scripts/prepare_celeba_partition.py
```

[CelebA 공식 분할](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)의
`list_eval_partition.txt`를 다운로드하고
[Torchvision에 기록된 체크섬](https://docs.pytorch.org/vision/stable/_modules/torchvision/datasets/celeba.html)
`d32c9cbf5e040fd4025c592c306e6668`로 검증합니다.
파일을 직접 확보했다면 `--celeba_partition_file /path/list_eval_partition.txt`로 지정할 수 있습니다.
공식 Train 162,770 / Validation 19,867 / Test 19,962장의 파일명으로 결합하며,
CSV 순서·실험 시드·속성 선택은 split 소속에 영향을 주지 않습니다.
누락·중복·손상 데이터를 조용히 제거하거나 무작위 분할로 대체하지 않습니다.
`--skip_readable_check`는 이미지 디코딩 사전 검사만 생략하며 파일 존재 검사는 유지합니다.

## 초기 파인튜닝부터 실행

```bash
python main_test.py \
  --dataset CelebA --network mobilenetv2 --prune_type FairGRAPE \
  --impt 5 --target_attr Smiling --sensitive_group gender \
  --batch 64 --prune_rate 0.9 --keep_per_iter 0.975 \
  --retrain_iter 5 --retrain_lr 1e-5 --impt2_alpha 0.3
```

대상 속성은 CSV의 기존 39개 예측 컬럼 이름을 사용합니다
(예: `Smiling`, `Attractive`, `X5_o_Clock_Shadow`). 민감 속성은 `gender`입니다.
기본 초기 스케줄은 8/1/1 epoch, 학습률 1e-4/1e-5/1e-6입니다.
GRL은 기존 기본값대로 활성화되며 `--no_grl`로 끌 수 있습니다.
초기·재학습의 태스크 손실은 선택 속성 하나의 CE이고, GRL 손실은 별도로 기록합니다.
초기 학습에는 마스크를 적용하지 않습니다. 각 프루닝 반복 뒤 재학습과 마지막
8/1/1 스케줄 재학습을 수행합니다. 기존 impt=2와 동일하게 두 옵션은 독립적입니다.

- `--retrain_iter 5`: 매 프루닝 반복 후 5 epoch 재학습합니다.
- `--no_retrain`: 모든 프루닝이 끝난 뒤의 마지막 추가 8/1/1 스케줄만 생략합니다.
- `--retrain_iter 0`: 반복별 재학습만 생략합니다. 마지막 추가 재학습 여부는 `--no_retrain`이 결정합니다.
- 모든 재학습을 생략하려면 `--retrain_iter 0 --no_retrain`을 함께 지정합니다.

따라서 `--retrain_iter 5 --no_retrain`은 반복별 5 epoch 재학습을 수행하고 마지막 추가 재학습만 생략합니다.

`--no_pretrained`와 다중 속성 체크포인트는 허용하지 않습니다.
체크포인트가 없으면 초기 파인튜닝이 필수입니다.

## 데이터 사용과 선택 기준

- Train: 초기 파인튜닝, 성능·공정성 기여도 계산, 재학습.
- Validation: 초기 모델, epoch 및 학습률 구간 사이 최적 모델 선택, 중간 보고.
- Test: 선택이 끝난 최종 모델 평가만 수행.

선택 기준은 Accuracy 최대 → EO 최소 → 태스크 Loss 최소입니다.
EO를 계산할 그룹/정답 셀이 없으면 `null`로 기록하고 동률 선택에서 불리하게 취급합니다.
BA는 기존 방식과 같은 민감 그룹 × 정답 클래스 셀 정확도의 평균입니다.
재학습에서는 공유 함수의 `test`라는 내부 phase 이름에 Validation만 연결합니다.
프루닝 단계와 종료 지점은 CLI 설정으로 고정되며 Test 점수로 선택하지 않습니다.
설정을 비교하는 런은 `--defer_test`로 Test를 보류하고, 선택 후 아래 평가 전용 명령을 사용합니다.

## 체크포인트 재사용과 최종 평가

결과 텍스트는 `retrain_epoch_results/임시 저장소/CelebA_impt5_<속성>_official_seed<seed>_<시각>/`,
모델은 `save_models/<시각>_CelebA_alpha<α>_<norm>_impt5_<속성>_official_seed<seed>/`,
채널 로그는 `channel_pruning_logs/CelebA_impt5_<속성>_official_seed<seed>_<시각>/`에 저장합니다.
`--single_attr_run_dir`를 지정하면 그 디렉터리에 결과를 저장하고, 모델·채널 로그는
그 아래 `models/`, `channel_pruning/`에 모읍니다. 기존 디렉터리는 덮어쓰지 않습니다.

새 프루닝 실험은 `dense/BEST_MODEL_dense_val_acc_…_eo_….pt`를 `--checkpoint`로 지정합니다.
초기 파인튜닝은 자동으로 건너뜁니다. `--no_init_train`도 함께 지정할 수 있지만 필수는 아닙니다.
모델 구조·단일 속성·GRL 설정·ImageNet 출처·공식 split·사용 라벨 해시가 일치해야 합니다.
메타데이터 없는 기존 모델, 다중 속성 모델, 다른 속성 모델, 무작위 초기화 모델,
비공식 분할 모델과 프루닝 후 모델은 새로운 프루닝의 출발점으로 거부합니다.

선택한 최종 모델만 평가하려면:

```bash
python main_test.py \
  --dataset CelebA --network mobilenetv2 --prune_type FairGRAPE \
  --impt 5 --target_attr Smiling --sensitive_group gender \
  --checkpoint /path/pruning/BEST_MODEL_final_val_acc_…_eo_….pt \
  --single_attr_eval_only
```

학습 시 `--no_grl`을 썼다면 로드할 때도 지정합니다.
평가 전용과 `--score_only`는 프루닝 후 체크포인트도 허용하지만 동일한 메타데이터·출처를 검증합니다.
`--score_only`는 Train에서 기여도와 채널 로그만 계산하고 모델·마스크는 저장하지 않습니다.
`--perf_only`, `--phi_gap_only`, `--phi_analysis`, `--impt2_layer_max_remove_ratio`도 지원합니다.

## 기록과 소규모 검증

텍스트 결과와 모델은 각각 `dense/`, `pruning/`으로 분리합니다.

```text
<결과 런>/
  0_run_info.txt, run.json, metrics.csv, train_manifest.csv, val_manifest.csv, test_manifest.csv
  dense/
    Total_info_dense_epoch_01_val.txt ... Total_info_dense_epoch_10_val.txt
    BEST_MODEL_SUMMARY_dense_val.txt
    epochs.jsonl
    val/metrics.json, val/predictions.csv
  pruning/
    iter_01/
      Total_info_prune_01_retrain_01_val.txt ...
      BEST_MODEL_SUMMARY_prune_01_val.txt
      epochs.jsonl
      val/metrics.json, val/predictions.csv
    final/
      BEST_MODEL_SUMMARY_final_val.txt
      FINAL_TEST_SUMMARY.txt
      val/metrics.json, val/predictions.csv
      test/metrics.json, test/predictions.csv
<모델 런>/
  0_run_info.txt
  dense/BEST_MODEL_dense_val_acc_…_eo_….pt
  pruning/BEST_MODEL_prune_01_val_acc_…_eo_….pt ...
  pruning/BEST_MODEL_final_val_acc_…_eo_….pt
```

초기 학습 epoch는 학습률 구간 전체를 통틀어 1~10으로 기록합니다.
`Total_info`는 각 epoch, `BEST_MODEL_SUMMARY`는 학습률 구간 전체에서 선택한 모델을 기록합니다.
재학습이 0 epoch여도 프루닝 iteration의 평가와 요약을 저장합니다.
프루닝이 0회면 최종 평가도 `dense/` 아래에 저장합니다.
체크포인트에는 partition·split·선택 라벨 해시와 출발 dense 체크포인트 해시를 유지합니다.

선택 속성의 ACC, Loss, BA와 함께 다음 **차이 지표(0이 가장 작음)**를 모든 epoch 기록,
선택 모델 요약, `metrics.json`, `metrics.csv`, 체크포인트의 Validation 지표에 기록합니다.

- `eo`: 기존 Equalized Odds gap, `(abs(TPR₀−TPR₁) + abs(FPR₀−FPR₁)) / 2`.
- `eopp`: Equal Opportunity gap, `abs(TPR₀−TPR₁)`.
- `dp`: Demographic Parity gap, `abs(P(pred=1|gender=0) − P(pred=1|gender=1))`.

[공정성 지표 정의 참고(Fairlearn)](https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html).
양성은 선택 속성 라벨 1이며 모델의 두 출력 중 argmax로 예측합니다.
지표는 배치별 차이를 평균하지 않고 평가 데이터 전체의 그룹별 집계를 사용합니다.
필요한 분모가 0이면 해당 지표만 JSON/체크포인트에서는 `null`, CSV에서는 빈칸,
텍스트에서는 `N/A`로 기록합니다. 그룹별 TP/P/FP/N, TPR/FPR 및 양성 예측률도 기록합니다.
모델 선택과 프루닝 점수 계산은 기존 방식 그대로이며 추가 지표는 보고용입니다.
변경 전에 시작한 프로세스와 기존 결과 파일에는 자동 적용되지 않습니다.

```bash
python main_test.py \
  --dataset CelebA --network mobilenetv2 --prune_type FairGRAPE \
  --impt 5 --target_attr Smiling --sensitive_group gender \
  --single_attr_smoke_samples 4 --batch 4 \
  --prune_iter 1 --prune_rate 0.01 --keep_per_iter 0.99 \
  --retrain_iter 1 --stop_batch 1
```

Smoke는 각 공식 split 안에서만 표본을 고르고 각 정답×그룹 셀을 포함합니다.
초기·마지막 학습 스케줄을 1 epoch로 줄이며, metadata로 구분하여
전체 데이터 실험에서 smoke 체크포인트를 사용할 수 없게 합니다.
이 결과는 성능 실험 결과가 아니라 실행 경로 검증용입니다.

```bash
OMP_NUM_THREADS=2 python -m pytest \
  tests/test_celeba_single.py tests/test_celeba_single_runner.py \
  tests/test_channel_cap.py tests/test_phi_group_gap.py tests/test_lfwa.py -q
```

## 구현 검증 기록 (2026-10-02)

- 관련 테스트 39개 통과, 1개 건너뜀: 이전 선택기 소스가 필요한 선택적 비교 테스트.
- 전체 202,599장 CSV의 공식 split 매칭 및 모든 이미지 경로 확인 완료.
- 실제 ImageNet 가중치와 split별 4장으로 GRL 활성/비활성 학습·프루닝·재학습·평가 완료.
- 별도 실행에서 dense 체크포인트 로드 후 재학습 생략, pruned 점수 재계산, 평가 전용 경로 완료.
- 전체 규모 학습·성능 실험과 전체 이미지 디코딩 검사는 수행하지 않았습니다.
- 기계 판독용 기록: `docs/impt5_validation.json`.
