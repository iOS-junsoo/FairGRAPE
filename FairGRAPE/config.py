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
glo_model_run_dir = None # save_models/<저장 시작 시각> 런 폴더 경로. train_and_val._get_model_run_dir()이 첫 저장 시 생성.