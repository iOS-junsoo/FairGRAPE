# CelebA impt=6: 단일 속성 원본 FairGRAPE

`impt=5`와 동일한 공식 분할·단일 속성 학습·평가 흐름에 원본 FairGRAPE를
연결합니다. ImageNet MobileNetV2 (`IMAGENET1K_V1`)를 선택 속성 하나로
파인튜닝하며 분류기는 2출력입니다. 민감 그룹은 `gender`입니다.

## 실행

기존 `fairgrape_cu128` 환경과 CelebA 데이터·공식 partition 파일을 사용합니다.

```bash
python main_test.py \
  --dataset CelebA --network mobilenetv2 --prune_type FairGRAPE \
  --impt 6 --target_attr Smiling --sensitive_group gender \
  --batch 64 --prune_rate 0.9 --keep_per_iter 0.975 \
  --retrain_iter 5 --retrain_lr 1e-5
```

`--fg_mode original --fg_scope all --delta_p 0 --para_batch 1`이 기본입니다.
다른 모드나 제한된 범위는 거부합니다. 원본 옵션 `--delta_p 1/2`와 양의
`--para_batch`도 지원합니다. 기존 경로의 기본값은 그대로 유지합니다.
`--impt2_alpha`, `--perf_only`, `--phi_analysis`, `--phi_gap_only`,
`--impt2_layer_max_remove_ratio`는 impt=6에서 지원하지 않습니다.

## 점수와 프루닝

- Train 라벨은 `[선택 속성, gender]`, 태스크 출력 범위는 `[(0, 2)]`입니다.
- 각 민감 그룹의 선택 속성 CE 손실만으로 그룹별 가중치 중요도를 구합니다.
  다른 예측 속성이나 GRL 손실을 중요도 계산에 합치지 않습니다.
- `prune_orig_ref.py`의 원본 `fairness_grad(impt_type=0)`를 기존 어댑터를
  통해 직접 실행합니다. 참조 버전은 `b677eb9`입니다.
- 원본의 gradient 누적, 점수 계산 중 Adam 갱신, 마지막 그룹만 배치 수로
  나누는 동작과 `torch.topk`의 동률 선택을 유지합니다. NumPy 선택기를 쓰지 않습니다.
- 첫·마지막 Conv와 분류기를 포함한 모든 Conv2d·Linear, 총 53개 레이어가
  대상입니다. 레이어별 원래 가중치 수가 N이고 누적 제거율이 r이면
  `int(N * (1-r))`개를 greedy로 유지합니다. bias와 BN 파라미터는 대상이 아닙니다.
- 점수 계산용 복사본의 가중치·BN 갱신은 재학습 모델에 전달하지 않습니다.
  첫 scoring 마스크는 원본처럼 학습 가능하고, 프루닝 이후 마스크는 고정됩니다.
  재학습 모델에서도 마스크 forward를 유지하여 제거된 가중치를 복원하지 않습니다.

## 학습·체크포인트·평가

공식 Train/Validation/Test 분할은 각각 162,770/19,867/19,962장입니다.
Train에서 학습·중요도 계산·재학습을 하고, Validation의
ACC 최대 → EO 최소 → 태스크 Loss 최소 순으로 모델을 선택합니다.
Test는 모든 선택이 끝난 뒤 최종 모델에만 사용합니다.

초기 학습 및 마지막 추가 재학습은 기존 8/1/1 epoch,
학습률 1e-4/1e-5/1e-6입니다. GRL은 기본 활성화되고 `--no_grl`로 끌 수 있습니다.
`--retrain_iter`는 단계별 재학습, `--no_retrain`은 마지막 추가 재학습만 제어합니다.

속성·GRL·ImageNet 초기화·공식 분할·라벨 해시가 일치하는 impt=5 또는 impt=6의
dense 체크포인트를 `--checkpoint`로 지정하면 초기 학습을 건너뜁니다.
Smoke 체크포인트는 전체 데이터 실험의 출발점으로 사용할 수 없습니다.
새 프루닝은 dense 체크포인트에서 시작합니다.

```bash
python main_test.py \
  --dataset CelebA --network mobilenetv2 --prune_type FairGRAPE \
  --impt 6 --target_attr Smiling --sensitive_group gender \
  --checkpoint /path/dense/BEST_MODEL_dense_val_acc_..._eo_....pt \
  --prune_rate 0.9 --keep_per_iter 0.975 --retrain_iter 5 --defer_test
```

`--defer_test`는 최종 Test를 보류합니다. `--single_attr_eval_only`는 검증된
선택 체크포인트의 Test 평가만 실행합니다. `--score_only`는 Train에서 원본
점수 계산·greedy·레이어별 로그 생성을 수행하고 제안 마스크를 버립니다.
입력 체크포인트를 수정하거나 새 모델을 저장하지 않습니다. 이미 프루닝된 모델에서는
제안 유지 개수가 현재 활성 가중치 수를 넘지 않도록 제거율을 제한합니다.
프루닝된 체크포인트는 해당 방식과 dense 출처를 확인하며,
impt=5의 프루닝된 모델을 impt=6으로 계산·평가하거나 그 반대로 사용할 수 없습니다.

## 기록

기본 결과·모델·원본 프루닝 로그는 각각
`retrain_epoch_results/임시 저장소/`, `save_models/`, `fg_orig_pruning_logs/`에
저장하며 이름에 `impt6`, 대상 속성, `original_all`을 기록합니다.
`--single_attr_run_dir`를 지정하면 결과와 `models/`, `fg_original/`을 그 아래에 모읍니다.
기존 런 디렉터리를 덮어쓰지 않습니다.

Dense와 pruning 결과를 분리하고 epoch 기록, 선택 모델 요약, 예측 CSV,
체크포인트에 ACC·Loss·BA·EO·EOPP·DP를 유지합니다. 보고서와 metrics는 impt=6을
명시하며 프루닝 대상 통계는 전체 53개 레이어의 실제 마스크를 집계합니다.
원본이 제공하지 않는 최종 선택 점유율과 pick 횟수는 미제공 상태로 기록합니다.

## 소규모 검증

```bash
OMP_NUM_THREADS=2 python scripts/validate_celeba_single_original.py
```

각 공식 split에서 Smiling 정답×gender의 네 셀을 포함한 4장을 사용합니다.
GRL 활성·비활성 각각 초기 학습 1epoch, 프루닝 2회(누적 50%, 75%), 단계별
재학습 1epoch 및 마지막 추가 재학습 1epoch를 실행합니다. 모든 대상 레이어의
예산·이진 마스크·누적 제거·재학습 후 0 가중치를 검사합니다.
실제 impt=5 dense 재사용, 재학습 및 Test 생략, 점수 전용, 평가 전용도 확인합니다.

검증 runner는 실행 로그·명령·체크포인트·`validation.json`을 별도
`smoke_results/impt6_single_<시각>/`에 남깁니다. 원본 직접 호출과의 점수·목표
점유율·마스크 동일성, 다른 속성 라벨의 영향, 모델·BN 분리는 회귀 테스트로 검증합니다.
이 검증은 실행 경로 확인용이며 전체 규모 성능 실험을 수행하지 않습니다.

## 구현 검증 기록 (2026-10-04)

- 관련 회귀 테스트 72개 통과, 선택적 이전 선택기 대조 1개 건너뜀.
- 원본 impt=4 연결과 손상 체크포인트 검사 2개를 추가한 최종 변경 영역
  테스트 39개 통과. 두 실행을 합쳐 서로 다른 테스트 74개가 통과했습니다.
- 실제 GPU에서 GRL 활성·비활성 각각 프루닝 2회와 재학습·최종 Test 완료.
  전체 53개 레이어의 예산, 누적 이진 마스크와 제거 가중치 0 유지 확인.
- 실제 impt=5 dense 체크포인트 재사용, 재학습·Test 생략, 점수 계산 전용,
  평가 전용 완료. 입력 체크포인트 해시 불변 확인.
- 원본 함수와 점수·목표 점유율·마스크 완전 일치 및 다른 속성 라벨의 영향 없음 확인.

기계 판독 기록은 [impt6_validation.json](impt6_validation.json), 원본 실행 로그와
산출물은 `smoke_results/impt6_single_20261004/`에 있습니다.
