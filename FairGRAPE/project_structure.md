# FairGRAPE 프로젝트 구조

**FairGRAPE** (Fairness-aware GRAdient Pruning mEthod)
성별/인종 등 민감 속성에 대한 공정성(Equalized Odds)을 유지하면서 MobileNetV2를 채널 단위로 가지치기(Pruning)하는 프레임워크.

---

## 루트 — 핵심 소스 파일

```
FairGRAPE/
├── config.py                   전역 상태 관리 (prune_iter, EO 점수, GRL 사용 여부 등)
├── dataset.py                  CelebA / FairFace / UTKFace / ImageNet 데이터셋 로드 및 전처리
├── util.py                     공통 유틸: 모델 생성(make_model), 저장, CUDNN 폴백, 정확도 계산 등
├── train_and_val.py            훈련 및 검증 루프 (다중 lr/에폭 탐색, GRL/Debiasing 지원)
├── prune.py                    가지치기 핵심 구현 (FairGRAPE / SNIP / WS / GraSP / Lottery 등)
├── adversarial_debiasing.py    GRL(Gradient Reversal Layer) 기반 공정성 개선 모듈
├── main_test.py                실험 진입점 — 인자 파싱 및 전체 파이프라인 조율
├── experiments.py              실험 보조 스크립트 (이전 버전 / 참고용)
├── environment.yml             Conda 의존성 (Python 3.9, PyTorch 1.11.0 등)
├── LICENSE
├── README.md
├── input.md                    실험 실행 명령어 임시 메모
└── Miniconda3-latest-Linux-x86_64.sh   Miniconda 설치 파일 (삭제 가능)
```

---

## 데이터

```
csv/
├── CelebA.csv                  CelebA 데이터셋 레이블 (속성 40개)
├── FairFace.csv                FairFace 데이터셋 레이블
├── UTKFace_labels.csv          UTKFace 레이블
├── Imagenet.csv                ImageNet 메타정보
├── synsets_safety.csv          ImageNet synset 안전성 분류
└── imageability_scores.csv     ImageNet 이미지 가능성 점수

Images/
├── CelebA/
│   └── img_align_celeba/       CelebA 이미지 약 200K장
└── CelebA.zip                  압축 원본 (이미 압축 해제됨 — 삭제 가능)

fair_dfs/
└── CelebA_FairGRAPE_classes_bygender_mobilenetv2_0.9_{0~5}.csv
                                Pruning iteration별 채널 선택 결과 DataFrame
```

---

## 모델

```
models/                         (예비 디렉토리, 현재 비어 있음)

gender_model/
├── best_gender_model.pth       성별 분류기 체크포인트
└── model_config.txt            모델 설정 정보

trained_model/
├── unpruned/
│   └── CelebA_unpruned_classes_bygender_mobilenetv2_0.pt
│                               가지치기 전 기준 모델
├── FairGRAPE/
│   └── CelebA_FairGRAPE_classes_bygender_mobilenetv2_0.9_{0~7}.pt
│                               FairGRAPE 기법, 90% pruning, iteration별 모델
└── SNIP/
    └── CelebA_SNIP_classes_bygender_mobilenetv2_0.9_0.pt
                                SNIP 기법 비교용 모델

save_models/                    학습 중 저장된 중간/최적 모델 체크포인트
save_models_no_retrain/         재훈련 없이 pruning만 적용한 모델 체크포인트
```

---

## 실험 결과

### 재훈련 결과
```
retrain_epoch_results/
├── Activate_diff/              활성화 값 차이(φ) 기반 pruning 결과
│   ├── 제대로된 방법/          현재 방법론 (uniform 분배 등 수정 적용)
│   │   └── 0.9:0.1/seed42/    alpha=0.9, seed별 epoch 로그 및 BEST 요약
│   └── 잘못된 방법으로 모델 붕괴됨/   global greedy 방식으로 features.17 편중 → 붕괴 기록
├── baseline/                   기본 학습 결과 (pruning 없음)
├── gender_imp/                 성별 중요도 가중치 비율별 실험 (0:1 ~ 9:1)
│   └── {ratio}/seed{n}/
├── GRL_결과/                   GRL 강도(lambda) 별 실험
│   └── {ratio}/lambda_gender_{0.0~1.0}/
├── penalty+GRL/                EO 패널티 + GRL 결합 실험
│   └── beta_{1.0~1000.0}/seed{n}/
├── Gradient_Norm/              gradient norm 기반 pruning
│   ├── GRL_O/                  GRL 적용
│   └── GRL_X/                  GRL 미적용
└── 소수점/                     alpha 소수점 비율 실험 (3.5:6.5 ~ 5.8:4.2)

no_retrain_results/             재훈련 없이 pruning만 적용한 결과
├── 1:9/, 2:8/, ..., 9:1/      성별 가중치 비율별
└── SNIP/                       SNIP 기법 비교
```

### 채널 Pruning 로그
```
channel_pruning_logs/           iteration별 레이어당 채널 제거 비율 및 상세 로그
├── 제대로된 방법/
│   └── {alpha_ratio}/seed{n}/alpha{a}_iter{i:02d}.txt
└── 잘못된 방법/
    └── {alpha_ratio}/seed{n}/alpha{a}_iter{i:02d}.txt
```

### 미정제 모델 평가
```
unpruned_model_evaluation/      pruning 전 기준 모델 평가 결과
```

---

## 탐색 실험 / 분석 스크립트

### 활성화 차이 상관관계 실험
```
activate_experiment/
├── celeba_unpruned_activation_gap_experiment.py
│                               가지치기 전 모델에서 활성화 차이(gap) 측정
├── celeba_importance_phi_correlation_experiment.py
│                               채널 중요도 vs fairness proxy(φ) Spearman 상관도 분석
├── corr_result/                실험 결과 CSV 및 플롯
│   └── Attractive/             속성별 상관 결과
└── 6_layer_activate_dff_corr/  6개 핵심 레이어 집중 분석
    ├── celeba_6layer_correlation_experiment.py    φ vs ΔEO 상관도 측정
    ├── celeba_correlation_experiment_v2.py        V2 개선 버전
    ├── perf_vs_delta_acc.py                       성능 기여도 vs ΔAcc 상관도
    ├── unified_channel_zeroing_experiment.py      채널별 제로잉으로 ΔEO + ΔAcc 동시 측정
    ├── unified_importance_comparison.py           성능 기여도 범위 불일치 검증
    ├── remeasure_delta_eo_all_attrs.py            전체 속성 ΔEO 재측정
    └── results*/               측정 결과 CSV 및 scatter plot
```

### Gradient Norm 분석
```
Gradient_norm/
├── ex1/
│   ├── figure.py               성별 그룹별 gradient 차이 시각화
│   ├── celeba_gender_grad_diff.npz     gradient 차이 압축 numpy 배열
│   └── gradient_analysis.png   시각화 결과
├── ex2/
│   ├── Correlation.py          features.1 채널별 Spearman 상관도 계산
│   └── features1_spearman_scatter.png
├── ex3/
│   ├── spearman_importance_vs_gradgap.py   중요도 vs gradient gap 상관도
│   ├── spearman_per_layer.csv
│   └── scatter_*.png
├── ex4/
│   ├── Correlation_importance.py   중요도 기반 상관도 재분석
│   └── features1_importance_spearman_scatter.png
└── ex5/
    ├── spearman_importance_vs_gender_score.py   중요도 vs 성별 score 상관도
    ├── spearman_per_layer.csv
    └── scatter_*.png
```

### EO 손실 근사 실험
```
experiment4/                    테일러 1차 근사 기반 EO 손실 근사 비교
├── EO_Loss 기반 근사/
├── compute_approximate_eo_by_EO_Loss/
├── compute_approximate_eo_by_CrossEntropy/
├── first_layer_scores/
├── 실제/ 헤시안/ 그래디언트 차이/ 등

experiment4_2nd_order/          2차 테일러 (헤시안) 기반 근사
experiment4_fisher/             Fisher 정보 기반 근사

experiment5/                    성별 그룹별 중요도 분포 분석
experiment8/                    APPROX_EO score vs REAL_EO score 비교
experiment11/                   첫/마지막 레이어의 실제 손실 차이 측정

eo_loss_validation/             EO 손실 함수 검증 결과
score_analysis/                 pruning score 분포 통계 분석

data_quality_logs/              학습 중 제외된 이미지 목록 (품질 필터링 결과)
```

---

## 기타

```
Vscode_to_slack/
└── slack_logger.py             실험 결과를 Slack Webhook으로 실시간 전송

channel_pruning_design.md       채널 pruning 알고리즘 설계 문서
활성화값 차이 프루닝 기존 방법론 개선 정리.md   방법론 개선 과정 정리
```

---

## 핵심 알고리즘 플로우

```
dataset.py → util.py(make_model) → train_and_val.py(초기 훈련)
    ↓
prune.py (FairGRAPE 채널 선택)
  - compute_importance(): Hessian 기반 채널 중요도 계산
  - compute_phi():        성별 그룹 간 활성화 차이(φ) 계산
  - score = α·perf - (1-α)·φ  로 채널 정렬
  - Uniform 분배: 레이어별 비례 할당 후 로컬 선택
    ↓
train_and_val.py (재훈련) + adversarial_debiasing.py (GRL 보정)
    ↓
main_test.py (결과 저장)
  → retrain_epoch_results/ + channel_pruning_logs/ + trained_model/
```
