# ── 타임존을 한국 시간(KST, Asia/Seoul)으로 고정 ──
# 이 모듈은 main_test → train_and_val 체인에서 프로세스 시작 시 import되므로,
# 이후 모든 datetime.now() / time 기반 타임스탬프(로그, 저장 파일명, run_info 등)가 KST로 찍힌다.
import os as _os
import time as _time
_os.environ['TZ'] = 'Asia/Seoul'
if hasattr(_time, 'tzset'):  # Unix 전용 (Windows에는 없음)
    _time.tzset()

# 전역 변수
glo_prune_iter = 0 # 가지치기 반복 횟수
glo_loss = 0
glo_acc = 0
# glo_optimizer = ""
glo_EO = 0 # EO 점수
glo_fairness:dict = {} # 태스크별 TP, FN, FP, TN, TPR, FPR 정보
glo_retrain_epoch = 0 # 에폭 재학습 횟수
glo_is_checked = False # 체크포인트 여부
glo_imp_rate = 0.0 # 중요도 비율
glo_use_grl = True # GRL 기반 debiasing 사용 여부
glo_n_groups = 2 # 민감그룹 수 (gender=2, UTKFace race=4). main_test.experiment()에서 설정됨.
glo_keep_per_iter = None # impt_type=2 채널 프루닝의 iter당 유지율. main_test에서 --keep_per_iter 값으로 설정. None이면 prune.py의 IMPT2_KEEP_PER_ITER 기본값 사용.
glo_impt_type = None # --impt 값. main_test.experiment()에서 설정. save_models run_info.txt에 적용 알파를 기록할 때 사용.
glo_dataset = None # --dataset 값. main_test.experiment()에서 설정. 결과 저장 폴더명에 사용.
glo_seed = None # --seed 값. main_test.experiment()에서 설정. 결과 저장 폴더명에 사용.
glo_results_run_dir = None # retrain_epoch_results/임시 저장소/<dataset>_impt<impt>_seed<seed>_<생성 시각> 런 폴더. train_and_val._get_results_run_dir()이 실험 시작 시 생성.
glo_model_run_dir = None # save_models/<저장 시작 시각> 런 폴더 경로. train_and_val._get_model_run_dir()이 첫 저장 시 생성.
glo_channel_log_run_dir = None # channel_pruning_logs/<dataset>_impt<impt>_seed<seed>_<첫 저장 시각> 런 폴더. prune._save_channel_pruning_log가 첫 저장 시 생성.
glo_weight_log_run_dir = None # weight_pruning_logs/ 아래 동일 규칙 런 폴더 (impt_type=3). prune._save_weight_pruning_log가 첫 저장 시 생성.
glo_perf_only = False # --perf_only: impt_type=2/3에서 φ를 쓰지 않고 성능 기여도 원시값만으로 전역 프루닝 (alpha=1, γ=0, floor=0, cap=None, 정규화 없음). main_test.experiment()에서 설정.
glo_collapse_notes = [] # prune.py가 어떤 층/블록의 활성 채널·가중치가 0이 된 iter를 기록 ("iter N: BLOCK COLLAPSE features.K"). BEST_MODEL_SUMMARY에 그대로 적힌다.
glo_fg_scope = 'blocks' # --fg_scope: impt_type=4(원본 FairGRAPE greedy) 프루닝 대상 범위. 'blocks'=features.1~17 conv만, 'all'=모든 Conv2d+Linear. main_test.experiment()에서 설정.
glo_fg_log_run_dir = None # fg_orig_pruning_logs/ 아래 런 폴더 (impt_type=4). prune._save_fg_orig_pruning_log가 첫 저장 시 생성.
glo_fg_para_batch = None # --para_batch (impt_type=4 run_info 기록용)
glo_fg_delta_p = None # --delta_p (impt_type=4 run_info 기록용)
