# 변경된 환경에서 활성화값 차이로 실험

/workspace/fairgrape_env_gpu/bin/python main_test.py \
  --checkpoint trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_0.pt \
  --dataset CelebA \
  --network mobilenetv2 \
  --prune_type FairGRAPE \
  --loss_type classes \
  --sensitive_group gender \
  --prune_rate 0.9 \
  --no_init_train \
  --batch 256 \
  --retrain_iter 5 \
  --stop_batch 10000 \
  --impt 2 \
  --no_grl \
  --skip_readable_check