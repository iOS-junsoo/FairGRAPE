# 6_layer_activate_dff_corr 실험 폴더 설명

## 개요

fairness proxy와 성능 기여도가 프록시로서 자격이 있는지 검증하기 위한 실험들.
채널별 zeroing으로 실제 ΔEO, ΔAcc를 측정하고, 프록시 값과의 상관관계를 분석한다.

---

## 실험 스크립트

### `celeba_6layer_correlation_experiment.py`
- **역할**: 최초 실험 스크립트. conv1만 zeroing하여 ΔEO를 측정.
- **흐름**: compute_phi_k → compute_importance → 17개 레이어 상관관계 계산 → |ρ| 기준 6개 레이어 자동 선정 → 채널별 conv1 zeroing → ΔEO 측정
- **결과 위치**: `results/`
- **한계**: conv1만 zeroing. ΔAcc 미측정.

### `perf_vs_delta_acc.py`
- **역할**: 기존 6개 레이어 CSV에서 perf 값을 읽고, conv1만 zeroing하여 ΔAcc를 측정.
- **결과 위치**: `results_delta_acc/`
- **한계**: conv1만 zeroing. features.10, 16은 미실행 (시간 부족).

### `unified_channel_zeroing_experiment.py`
- **역할**: conv0+conv1+conv2 통합 zeroing으로 ΔEO + ΔAcc를 동시 측정.
- **배경**: ΔAcc는 conv0+conv1+conv2를 묶어서 제거한 결과인데, 기존 실험은 conv1만 zeroing하여 측정 조건이 불일치. 이를 해결.
- **결과 위치**: `results_unified/`
- **비고**: 기존 `results/` CSV에서 fairness proxy, 성능 기여도 값을 재활용.

### `unified_importance_comparison.py`
- **역할**: 성능 기여도를 conv0+conv1+conv2 통합으로 재계산하여, 기존(conv1-only)과 비교.
- **배경**: 성능 기여도는 ΔAcc의 프록시인데, ΔAcc를 측정할 때 conv0+conv1+conv2를 묶어서 제거하므로 성능 기여도도 동일한 범위로 계산해야 한다. 이 가설을 검증.
- **결과 위치**: `results_unified_importance/`
- **비고**: zeroing을 다시 돌리지 않고 `results_unified/`의 ΔEO, ΔAcc를 재활용. features.1은 conv0(32ch)과 conv1(16ch) 채널 수 불일치로 제외.

---

## 결과 폴더

### `results/` — conv1-only zeroing, ΔEO 측정

| 파일 | 설명 |
|---|---|
| `all_layers_phi_vs_perf.csv` | 17개 레이어 전체의 fairness proxy vs 성능 기여도(conv1-only) Spearman/Pearson. 6개 레이어 선정 근거. |
| `channels_features_N_conv_1_0.csv` | 레이어별 채널 raw 데이터. 컬럼: channel, phi, perf, baseline_eo, pruned_eo, delta_eo, delta_eo_abs |
| `summary_6layer_correlations.csv` | 6개 레이어 요약. fairness proxy→ΔEO, 성능 기여도→ΔEO 상관관계. |
| `plots/` | 레이어별 scatter plot (phi vs ΔEO, perf vs ΔEO) + 요약 bar chart |
| `run_metadata.json` | 실험 조건 (체크포인트, seed, 속성, 소요 시간 등) |

### `results_delta_acc/` — conv1-only zeroing, ΔAcc 측정

| 파일 | 설명 |
|---|---|
| `channels_features_N_conv_1_0.csv` | 레이어별 채널 raw 데이터. 컬럼에 delta_acc, delta_acc_abs 추가. |
| **비고** | features.1, 2, 3, 6만 존재 (features.10, 16은 미실행) |

### `results_unified/` — conv0+conv1+conv2 통합 zeroing, ΔEO + ΔAcc 동시 측정

| 파일 | 설명 |
|---|---|
| `unified_channels_features_N_conv_1_0.csv` | 레이어별 채널 raw 데이터. delta_eo, delta_acc 모두 포함. |
| `summary_unified_correlations.csv` | 6개 레이어 요약. fairness proxy→ΔEO, 성능 기여도(conv1-only)→ΔEO, fairness proxy→ΔAcc, 성능 기여도(conv1-only)→ΔAcc 4가지 상관관계. |
| `plots/` | 레이어별 scatter plot 4개 (phi/perf × ΔEO/ΔAcc) + 요약 bar chart |
| `run_metadata.json` | 실험 조건 |

### `results_unified_importance/` — 통합 성능 기여도 비교

| 파일 | 설명 |
|---|---|
| `comparison_features_N_conv_1_0.csv` | 채널별 raw 데이터. perf_conv1_only와 perf_unified 병기. |
| `summary_importance_comparison.csv` | 5개 레이어 요약. conv1-only vs conv0+conv1+conv2 성능 기여도의 ΔAcc/ΔEO 상관관계 비교. |
| `run_metadata.json` | 실험 조건 |

---

## 실험 흐름 (시간순)

```
[1] celeba_6layer_correlation_experiment.py
    → results/
    결론: 5/6 레이어에서 fairness proxy가 성능 기여도보다 ΔEO를 더 잘 예측

[2] perf_vs_delta_acc.py
    → results_delta_acc/
    결론: 성능 기여도 vs ΔAcc 상관관계 측정 (일부 레이어만)

[3] unified_channel_zeroing_experiment.py
    → results_unified/
    결론: 프록시 자격 검증 4가지 기준 판정. 기준 1(성능 기여도→ΔAcc)이 가장 약함.

[4] unified_importance_comparison.py
    → results_unified_importance/
    결론: 성능 기여도를 conv0+conv1+conv2 통합으로 계산하면 ΔAcc 상관관계가 5/5 개선.
         범위 불일치가 기준 1 실패의 주요 원인이었음 확인.
```

---

## 주요 발견 사항

1. **fairness proxy**는 대부분의 레이어에서 ΔEO를 의미 있게 예측 (4/6 통과)
2. **성능 기여도(conv1-only)**는 ΔAcc 예측이 약함 (Spearman 기준 1/6 통과)
3. 성능 기여도를 **conv0+conv1+conv2 통합**으로 바꾸면 ΔAcc 예측이 5/5 레이어에서 개선
4. 개선 근거: ΔAcc 측정 시 conv0+conv1+conv2를 묶어서 제거하므로, 프록시도 동일 범위로 계산해야 함
5. features.1은 conv0(32ch)과 conv1(16ch) 채널 수 불일치로 통합 불가 → 별도 처리 필요

## 다음 작업

1. `celeba_6layer_correlation_experiment.py`에서 성능 기여도를 통합 방식으로 수정
2. 17개 레이어 상관관계 재계산 → 6개 레이어 재선정
3. 변경된 레이어에 대해 zeroing 추가 실행
4. 프록시 자격 재판정
