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