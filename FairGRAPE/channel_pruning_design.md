# 채널 단위 Fairness-Aware Pruning 설계서

## 개요

기존 FairGRAPE의 가중치 단위 pruning을 채널 단위로 전환하고,
공정성 기여도 프록시 φ_k를 pruning 기준으로 사용하는 새로운 방식.

**구현 위치**: `prune.py`의 `fairness_grad` 함수 내 `impt_type == 2` 블록  
**기존 코드 수정 금지**: `impt_type == 0`, `impt_type == 1` 블록은 절대 건드리지 않음  
**마스크 반환 형식 유지**: 기존 `build_mask_list`와 동일하게 레이어별 가중치 shape과 동일한 0/1 텐서 리스트로 반환

---

## 1. 코드 구조 이해 (기존)

### 전체 흐름

```
main_test.py → experiment()
  └─ FairGRAPE prunner 생성
  └─ 메인 pruning 루프 (22회)
       └─ prunner.prune(prune_cfgs)
            └─ FairGRAPE.get_mask(prune_cfgs)
                 └─ fairness_grad(..., impt_type=2)  ← 여기에 구현
       └─ train() 재학습 (5 epoch)
```

### prune_cfgs 구성

```python
prune_cfgs = [
    prune_ratio,           # [0] 누적 pruning 비율 (매 iteration 갱신됨)
    frames['train'],       # [1] 데이터프레임
    face_dir,              # [2] 이미지 디렉토리
    sensitive_classes,     # [3] 민감 그룹 수 (impt==2이면 total_classes=78)
    masked_grads,          # [4] True
    output_cols_each_task, # [5] [(0,2),(2,4),...] 39개 태스크
    col_used,              # [6] 39개 속성 + ['gender']
    para_batch,            # [7]
    impt_type,             # [8] 2
    stop_batch,            # [9]
    delta_p,               # [10]
    network,               # [11]
    sensitive_group        # [12] 'gender'
]
```

### 마스크 반환 형식

```python
# 기존 build_mask_list가 반환하는 형식과 동일해야 함
# model의 모든 supported_layers(Linear, Conv2d, Conv1d)를 순서대로 순회하며
# 각 레이어의 weight와 동일한 shape의 0/1 Long 텐서 리스트

mask_list = [
    tensor(shape=[32, 3, 3, 3], dtype=torch.long),   # features.0.0 (Stem)
    tensor(shape=[32, 32, 3, 3], dtype=torch.long),  # features.1.conv.0.0
    ...
]
# Classifier head는 전부 1로 유지 (is_classifier_head 함수 참고)
```

---

## 2. 공정성 기여도 프록시 φ_k

### 수식

```
φ_k = activation_gap_k × mean_gradient_k

activation_gap_k = gap_Y1_k + gap_Y0_k
gap_Y1_k = |E[h_k | Y=1, Male] - E[h_k | Y=1, Female]|
gap_Y0_k = |E[h_k | Y=0, Male] - E[h_k | Y=0, Female]|
mean_gradient_k = E[ |∂L/∂h_k| ]
```

### 구성 요소

| 항목 | 설명 |
|---|---|
| h_k | conv[1] (DW Conv) 출력에 GAP 적용한 채널 k의 스칼라 값 |
| Y | 타겟 레이블 — col_used의 각 속성 (39개 전체 평균) |
| A | 민감 속성 (Male=1, Female=0) — col_used 마지막 컬럼 |
| gap_Y1_k | Y=1 조건에서 성별 그룹 간 활성화 차이 (EO의 TPR 차이에 대응) |
| gap_Y0_k | Y=0 조건에서 성별 그룹 간 활성화 차이 (EO의 FPR 차이에 대응) |
| mean_gradient_k | 채널 k의 활성화값이 태스크 손실에 미치는 영향력 |

### φ_k 계산 대상 레이어

- 전체 Bottleneck 블록의 **conv[1] (DW Conv)** 출력 기준
- MobileNet-v2 기준: `features.1.conv.1`, `features.2.conv.1.0` ~ `features.17.conv.1.0`
- **Stem(`features.0.0`), Head Conv(`features.18.0`)는 제외**
- GAP(Global Average Pooling)로 공간 평균 → 채널당 스칼라 하나

### φ_k 계산 절차 (매 iteration 재계산)

```
1. 데이터 로더에서 배치를 순회하며:
   a. 모델 forward → conv[1] 출력에 hook으로 activation 수집
   b. 배치 내 샘플을 Y=0/Y=1, Male/Female 4개 그룹으로 분리
   c. 각 그룹별 채널 k의 activation에 GAP 적용 → 스칼라 h_k
   d. gap_Y1_k, gap_Y0_k 계산
   e. 태스크 손실 backward → conv[1] 출력의 gradient 수집
   f. |∂L/∂h_k| GAP 적용 → mean_gradient_k

2. 배치 평균으로 집계
3. φ_k = activation_gap_k × mean_gradient_k
```

---

## 3. 성능 기여도

FairGRAPE 기존 방식 `(weight × gradient)²`를 채널 단위로 집계.

```
가중치 단위: (gradient × weight)²
채널 단위: conv[1] 채널 k에 속한 가중치들의 (gradient × weight)² 평균
```

- φ_k와 동일하게 **conv[1] (DW Conv) 기준**으로 계산
- 채널 k에 속한 가중치들(3×3 = 9개)의 `(weight × grad)²`를 **mean** 집계
- 기존 `compute_importance` 함수를 호출하되, conv[1] 레이어만 필터링
- 두 점수 모두 같은 레이어 기준이므로 비교가 공정함

---

## 4. Scoring Formula

```
Score_k = α × 성능기여도_k - (1-α) × φ_k
```

**낮은 점수부터 제거** (성능에 덜 중요하고 불공정성에 많이 기여하는 채널 우선).

두 점수는 비교 가능하도록 각각 최댓값으로 정규화 후 적용:
```
성능기여도_k_scaled = 성능기여도_k / max(성능기여도)
φ_k_scaled = φ_k / max(φ_k)
Score_k = α × 성능기여도_k_scaled - (1-α) × φ_k_scaled
```

### 실험 순서

| 단계 | α | 목적 |
|---|---|---|
| 1단계 | 0.0 | φ_k만으로 pruning 시 EO 개선 여부 확인 |
| 2단계 | 0.5 | 성능 기여도와 φ_k 동등 결합 |
| 3단계 | 0.9 | 성능 기여도 중심, φ_k 보조 |

> α=0.0 결과가 모델 붕괴 없이 EO가 개선되면 α 실험으로 진행.
> 붕괴 발생 시 원인 분석 후 방향 재설정.

---

## 5. 채널 묶음 제거 방식

conv[1]에서 채널 k를 제거 대상으로 선정하면 세 레이어의 마스크를 동시에 0으로 처리.

```
conv[0]: 출력 방향 k번 필터 → weight shape [c_expanded, c_in, 1, 1]에서
         k번째 행(dim=0) 전체를 0으로
         (c_in은 건드리지 않음)

conv[1]: k번 필터 → weight shape [c_expanded, 1, 3, 3]에서
         k번째 행(dim=0) 전체를 0으로

conv[2]: 입력 방향 k번 슬롯 → weight shape [c_out, c_expanded, 1, 1]에서
         k번째 열(dim=1) 전체를 0으로
         (c_out은 건드리지 않음)
```

레이어 이름 매핑 (MobileNet-v2 기준):
```
features.N.conv.0.0  → conv[0] (PW Expand)
features.N.conv.1.0  → conv[1] (DW Conv)  ← 채널 선정 기준
features.N.conv.2    → conv[2] (PW Project)

예외: features.1.conv.0.0, features.1.conv.1 (Stage 1은 구조가 다름)
      features.1.conv.1 이 conv[1]에 해당
      features.1.conv.0.0 이 conv[0]에 해당
      conv[2] 없음 (Stage 1은 2단계 구조)
```

---

## 6. 매 iteration 제거량 결정 (선택 3: 목표에 가장 근접한 쪽)

```
1. 현재 모델의 활성 가중치 총 수 계산
   (mask가 있는 레이어는 mask.sum(), 없으면 weight.numel())

2. 목표 제거량 = 활성 가중치 총 수 × 10%

3. Score_k 낮은 채널부터 정렬 (전체 Bottleneck 레이어 통합 정렬)

4. 채널을 순서대로 선택하며 누적 가중치 수 계산:
   - 채널 k 추가 전 누적량 = prev_accum
   - 채널 k 추가 후 누적량 = prev_accum + (해당 채널 관련 conv[0]+conv[1]+conv[2] 가중치 수 합)
   - |prev_accum - 목표량| vs |new_accum - 목표량| 비교
   - new_accum이 더 가까우면 채널 k 포함하고 중단
   - prev_accum이 더 가까우면 채널 k 미포함하고 중단

5. 선정된 채널들의 conv[0], conv[1], conv[2] 위치를 마스크에서 0으로 설정
```

---

## 7. 마스크 생성 및 반환

기존 `build_mask_list` 반환 형식과 동일하게 맞춰야 함.

```python
# 전체 마스크를 1로 초기화
mask_by_layer = {}
for name, layer in model.named_modules():
    if type(layer).__name__ in supported_layers:
        if is_classifier_head(name):
            mask_by_layer[name] = torch.ones_like(layer.weight, dtype=torch.long)
        else:
            # 기존 마스크가 있으면 그것을 기반으로 시작 (누적 pruning)
            if hasattr(layer, 'mask'):
                mask_by_layer[name] = layer.mask.detach().clone().long()
            else:
                mask_by_layer[name] = torch.ones_like(layer.weight, dtype=torch.long)

# 선정된 채널 k에 대해 conv[0], conv[1], conv[2] 마스크를 0으로
for (block_name, channel_k) in selected_channels:
    conv0_name = block_name + '.conv.0.0'  # or '.conv.0' for stage1
    conv1_name = block_name + '.conv.1.0'  # or '.conv.1' for stage1
    conv2_name = block_name + '.conv.2'

    mask_by_layer[conv0_name][channel_k, :, :, :] = 0  # dim=0 k번째 행
    mask_by_layer[conv1_name][channel_k, :, :, :] = 0  # dim=0 k번째 행
    if conv2_name in mask_by_layer:
        mask_by_layer[conv2_name][:, channel_k, :, :] = 0  # dim=1 k번째 열

# 리스트로 변환 후 반환
mask_list = [mask.to(device) for name, mask in mask_by_layer.items()]
return mask_list
```

---

## 8. 전체 구현 구조

```python
if impt_type == 2:

    # ── Step 1: φ_k 계산 (conv[1] 기준) ──
    phi_by_layer = compute_phi_k(
        model=model,
        test_csv=test_csv,
        new_img_dir=new_img_dir,
        output_cols_each_task=output_cols_each_task,
        col_names=col_names,
        stop_batch=stop_batch,
        masked_grads=masked_grads,
    )
    # 반환: {'features.2.conv.1.0': tensor([C]), ...}  conv[1] 레이어만

    # ── Step 2: 성능 기여도 계산 (conv[1] 기준, mean 집계) ──
    importance_score_all, _ = compute_importance(
        model, None, test_csv,
        new_img_dir=new_img_dir,
        masked_grads=masked_grads,
        output_cols_each_task=output_cols_each_task,
        col_names=col_names,
        stop_batch=stop_batch,
        network=network,
        sensitive_group=sensitive_group,
        sensitive_classes=sensitive_classes,
    )
    # conv[1] 레이어만 필터링 + mean 집계
    perf_by_layer = {
        name: score.mean(dim=(1,2,3))  # [C_out]
        for name, score in importance_score_all.items()
        if '.conv.1' in name  # conv[1] 레이어만
    }

    # ── Step 3: α 설정 ──
    # prune.py 파일 상단의 IMPT_TYPE2_ALPHA 상수를 사용
    alpha = float(IMPT_TYPE2_ALPHA)  # 기본값 0.0

    # ── Step 4: Score_k 계산 ──
    score_by_channel = []  # [(score, block_name, channel_k, weight_count), ...]
    for conv1_name, phi_vec in phi_by_layer.items():
        block_name = conv1_name.rsplit('.conv.', 1)[0]  # 'features.N'
        perf_vec = perf_by_layer.get(conv1_name)
        if perf_vec is None:
            continue

        phi_scaled = phi_vec / (phi_vec.max() + 1e-12)
        perf_scaled = perf_vec / (perf_vec.max() + 1e-12)
        scores = alpha * perf_scaled - (1 - alpha) * phi_scaled

        # 채널별 가중치 수 계산 (conv[0]+conv[1]+conv[2])
        weight_count = _count_channel_weights(model, block_name)

        for k in range(len(scores)):
            score_by_channel.append((scores[k].item(), block_name, k, weight_count))

    # ── Step 5: 목표 제거량 계산 ──
    total_active = _count_active_weights(model)
    target_remove = total_active * keep_ratio  # keep_ratio = 1 - prune_ratio
    # 주의: fairness_grad에서 prune_ratio는 누적 비율이므로
    # 실제 이번 iteration 제거량 = 현재 활성 가중치 × 10%
    # → main_test.py의 keep_per_iter=0.9 기반으로 계산됨
    # → 여기서는 (1 - keep_ratio) 비율만큼 제거
    remove_target = int(total_active * (1.0 - keep_ratio))

    # ── Step 6: Score 낮은 순 정렬 후 채널 선정 (선택 3) ──
    score_by_channel.sort(key=lambda x: x[0])
    selected_channels = []  # [(block_name, channel_k), ...]
    accum = 0
    for score, block_name, k, wcount in score_by_channel:
        new_accum = accum + wcount
        if abs(new_accum - remove_target) <= abs(accum - remove_target):
            selected_channels.append((block_name, k))
            accum = new_accum
        else:
            break  # 더 추가하면 멀어지므로 중단

    # ── Step 7: 마스크 생성 및 반환 ──
    return _build_channel_mask_list(model, selected_channels, device)
```

---

## 9. 필요한 헬퍼 함수

### `compute_phi_k`

```python
def compute_phi_k(model, test_csv, new_img_dir, output_cols_each_task,
                  col_names, stop_batch, masked_grads):
    """
    전체 Bottleneck conv[1] 레이어에 대해 채널별 φ_k를 계산.

    반환: {conv1_layer_name: tensor([C_out])}

    계산 방식:
    - forward hook으로 conv[1] 출력 activation 수집
    - backward hook으로 conv[1] 출력의 gradient 수집
    - 배치 내 샘플을 (Y, gender) 4개 그룹으로 분리
    - 채널별 GAP → h_k 스칼라
    - gap_Y1_k = |mean(h_k|Y=1,male) - mean(h_k|Y=1,female)|
    - gap_Y0_k = |mean(h_k|Y=0,male) - mean(h_k|Y=0,female)|
    - mean_gradient_k = mean(|grad_h_k|) GAP 적용
    - φ_k = (gap_Y1_k + gap_Y0_k) × mean_gradient_k
    - 배치 평균으로 최종 집계

    대상 레이어:
    - 'features.1.conv.1' (Stage 1)
    - 'features.N.conv.1.0' (N = 2~17)
    - Stem, Head Conv 제외
    """
```

### `_count_active_weights`

```python
def _count_active_weights(model):
    """
    현재 모델의 활성 가중치 총 수 반환.
    mask가 있는 레이어는 mask.sum(), 없으면 weight.numel() 사용.
    Classifier head 포함.
    """
```

### `_count_channel_weights`

```python
def _count_channel_weights(model, block_name):
    """
    block_name (예: 'features.5')에 해당하는 Bottleneck 블록에서
    채널 하나를 제거할 때 영향받는 가중치 수 계산.

    conv[0]: weight[k, :, :, :].numel() → c_in × 1 × 1
    conv[1]: weight[k, :, :, :].numel() → 1 × 3 × 3 = 9
    conv[2]: weight[:, k, :, :].numel() → c_out × 1 × 1

    합산해서 반환.
    Stage 1 (features.1)은 conv[2] 없으므로 conv[0]+conv[1]만 계산.
    """
```

### `_build_channel_mask_list`

```python
def _build_channel_mask_list(model, selected_channels, device):
    """
    selected_channels: [(block_name, channel_k), ...]
    기존 마스크를 기반으로 선정된 채널 위치를 0으로 설정한 마스크 리스트 반환.

    반환 형식: 기존 build_mask_list와 동일
    - model의 supported_layers 순서대로
    - 각 레이어의 weight shape과 동일한 Long 텐서
    - Classifier head는 전부 1 유지
    """
```

---

## 10. 상수 추가 (prune.py 상단)

```python
# 기존
IMPT_TYPE1_ALPHA = 0.5

# 추가
IMPT_TYPE2_ALPHA = 0.0  # 1단계: φ_k만 사용. 이후 0.5, 0.9로 변경하며 실험
```

---

## 11. 고정 설정

| 항목 | 값 |
|---|---|
| 총 iteration | 22회 |
| 재학습 | iteration당 5 epoch |
| 매 iteration 제거량 | 활성 가중치 총 수의 약 10% (근사치, 선택 3 방식) |
| φ_k 및 성능기여도 재계산 | 매 iteration 시작 전 |
| pruning 대상 | Bottleneck 블록 전체 (conv[0], conv[1], conv[2]) |
| Stem / Head Conv | pruning 제외, 마스크 전부 1로 유지 |
| GRL | 없음 |
| Seed | 42 |
| 비교 베이스라인 | FairGRAPE impt_type=0 (동일 조건) |

---

## 12. 향후 고려 사항

### Stem / Head Conv pruning 미포함

FairGRAPE는 Stem/Head Conv 포함 전체 레이어를 pruning 대상으로 했음.
완전한 비교를 위해서는 이 레이어들을 어떻게 처리할지 추가 논의 필요.
선택지: ① 계속 제외 ② 가중치 단위로 별도 처리하여 혼합 방식 적용

### α 설정

상관관계 실험 결과에 따라 α 후보값 조정 가능.
φ_k와 성능기여도 간 상관관계가 높게 나오면 formula 자체를 재검토해야 함.

---

## 13. 현재 실험 상태

| 실험 | 상태 | 결과 |
|---|---|---|
| φ_k vs ΔEO (features.17.conv.2) | 완료 | Spearman 0.49 |
| importance_score vs ΔEO | 완료 | Spearman 0.097 (유의미하지 않음) |
| φ_k vs importance_score (mean, conv[1] 전체 레이어) | 완료 | 대부분 낮음, 분리 가능성 있음 |

