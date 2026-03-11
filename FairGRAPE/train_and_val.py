import os
import torch
import torch.optim as optim
import datetime
import torch.nn as nn
from torchvision import models, transforms
import copy
import pandas as pd
import numpy as np
from util import make_model, save_model, save_output, download_dataset, show_acc_df, setseed, save_unpruned_model, print_acc_scores



############################################################################
# train 함수: 여러 학습률과 에폭 설정으로 모델을 반복 학습하여, 
# 최적의 성능(가장 낮은 손실 또는 높은 정확도)을 내는 모델을 찾습니다.
############################################################################
def train(best_model, criterion, dataloaders, learning_rates = [1e-4,1e-5,1e-6], 
          epoches = [13,3,3], col_used_training=[0,1,2], 
          output_cols_each_task=[(0,7),(7,9),(9,18)], cfgs=None,
          use_debiasing=True,  # 🔥 새로운 파라미터
          lambda_grl=1.0,       # 🔥 GRL lambda
          lambda_gender=0.1):   # 🔥 Gender loss weight
    
    best_acc, best_loss = 0, 100
    best_optimizer = None

    for learning_rate, epoch in zip(learning_rates, epoches):
        print('학습률:', learning_rate)
        print('에폭:', epoch)
        print('현재 시간:', datetime.datetime.now())
        print('학습 또는 재학습 iter:', epoches)
        
        # 학습 가능한 파라미터만 옵티마이저에 추가
        optimizer_conv = optim.Adam(
            filter(lambda p: p.requires_grad, best_model.parameters()), 
            lr=learning_rate
        )
        
        # 🔥 Debiasing 파라미터 전달
        current_model, current_acc, current_loss, current_optimizer = train_model0(
            best_model, dataloaders, criterion, optimizer_conv, epoch, 
            col_used_training, output_cols_each_task, cfgs=cfgs,
            use_debiasing=use_debiasing,
            lambda_grl=lambda_grl,
            lambda_gender=lambda_gender
        )

        if current_acc > best_acc or (current_acc == best_acc and current_loss < best_loss):
            best_acc = current_acc
            best_loss = current_loss
            best_model = current_model
            best_optimizer = current_optimizer
            print('최고 정확도, 손실 갱신:', best_acc, best_loss)
        else:
            print('최고 정확도, 손실 갱신 없음, 현재:', best_acc, best_loss)
    
    torch.cuda.empty_cache()
    return best_model, best_optimizer

############################################################################
# train_model0 함수: 하나의 학습률, 에폭으로 모델을 학습(훈련/검증)하고
# 가장 낮은 손실을 얻은 모델 파라미터를 최종적으로 반환합니다.
# ! epoch_acc: 각 레이블(대머리, 수염, 노란머리)에 따른 정확도와 손실을 계산하고 그 결과 값의 평균 정확도를 저장한 것
# ! best_acc: 여러 에폭 중에서 epoch_loss가 가장 낮았을 때의 epoch_Acc 값을 기록해 둔 것으로, 최소 손실을 달성한 에폭의 정확도
"""

1. **배치별 모델 학습**  
   - train 단계에서만 역전파와 옵티마이저 업데이트가 이뤄짐.

2. **각 라벨별로 정확도(정답 개수)와 손실(loss)를 계산**  
   - `loss_multi_tasks` 함수에서 라벨별 예측 결과를 구해 크로스 엔트로피 등으로 손실을 계산하고,  
   - 동시에 라벨별로 맞춘 정답 수(정확도 계산용)를 합산.

3. **각 라벨별 정확도, 손실을 ‘평균’하여 에폭 단위 정확도/손실을 정의**  
   - 배치별로 누적한 손실(`running_loss`)을 전체 데이터셋 수로 나눠 에폭 손실(`epoch_loss`)을,  
   - 배치별 라벨별 정답 수(`running_corrects`)를 합산 후 전체 (샘플 수 × 라벨 수)로 나눠 평균 정확도(`epoch_acc`)를 구함.

4. **test 단계에서 이전 에폭보다 손실(`epoch_loss`)이 더 낮게 나오면, 그 시점의 정확도(`epoch_acc`)를 `best_acc`로 갱신**  
   - 동시에 모델 파라미터(`best_model_wts`)도 저장.

위 로직에 따라 “에폭별 평균 손실이 가장 낮은 에폭”에서의 정확도가 `best_acc`가 됩니다.
"""

############################################################################
# 기존 import에 추가
from adversarial_debiasing import (
    DebiasedModelWrapper, 
    create_pruning_masks, 
    apply_pruning_masks,
    apply_pruning_masks_to_gradients
)

############################################################################
# train_model0 함수 수정 (Adversarial Debiasing 적용)
############################################################################
def train_model0(model, dataloaders, criterion, optimizer, num_epochs=25,
                 col_used_training=[0,1,2],
                 output_cols_each_task=[(0,7),(7,9),(9,18)],
                 detail_acc=True, show_epoch=200, cfgs=None,
                 use_debiasing=True,  # 🔥 새로운 파라미터
                 lambda_grl=1.0,       # 🔥 GRL lambda
                 lambda_gender=0.1):   # 🔥 Gender loss weight
    
    (dataset, prune_type, loss_type, prune_rate, test_frame, face_dir,
     total_classes, network, col_used, output_cols_each_task,
     sensitive_group, exp_idx) = cfgs

    device = torch.device('cuda:0')
    prev_cudnn_enabled = torch.backends.cudnn.enabled
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    print("\n" + "="*80)
    print(f"🚀 학습 시작 | Debiasing: {use_debiasing} | Pruning Rate: {prune_rate}")
    print("="*80)

    # 1. Pruning Mask 생성 (Debiasing 여부와 상관없이 Pruning 설정이 있다면 수행)
    # config에 따라 prune_rate가 유효할 때만 생성하도록 조건 추가 가능
    if prune_rate > 0:
        pruning_masks = create_pruning_masks(model)
        print(f"✅ Pruning mask 생성 완료: {len(pruning_masks)}개 레이어/파라미터")
    else:
        pruning_masks = None

    # 2. Debiasing 설정 및 모델 래핑
    if use_debiasing:
        print(f"🎯 Adversarial Debiasing 모드 활성화 (lambda_grl={lambda_grl}, lambda_gender={lambda_gender})")
        
        # Wrapper로 감싸기
        debiased_model = DebiasedModelWrapper(model, lambda_grl=lambda_grl).to(device)
        
        # Optimizer 재생성 (Wrapper의 파라미터 = 기본 모델 + 성별 분류기)
        optimizer = type(optimizer)(
            filter(lambda p: p.requires_grad, debiased_model.parameters()),
            lr=optimizer.param_groups[0]['lr']
        )
        
        gender_criterion = nn.BCEWithLogitsLoss()
        training_model = debiased_model
    else:
        training_model = model
        gender_criterion = None

    hook_handle = None 
    
    # GRL 검증 훅 (Debiasing일 때만)
    if use_debiasing:
        def check_grl_hook(module, grad_input, grad_output):
            print("\n🔍 [GRL 검증] Gradient 확인")
            if grad_output[0] is None or grad_input[0] is None:
                print("   ⚠️ Gradient가 None입니다. 아직 계산 안됨.")
                return

            out_grad_mean = grad_output[0].mean().item()
            in_grad_mean = grad_input[0].mean().item()
            
            print(f"   ⬅️ 성별 분류기에서 온 Gradient 평균: {out_grad_mean:.6f}")
            print(f"   ➡️ 특징 추출기로 갈 Gradient 평균:   {in_grad_mean:.6f}")
            
            if out_grad_mean * in_grad_mean < 0:
                print("   ✅ 성공! 부호가 반대로 뒤집혔습니다. (GRL 작동 중)")
            else:
                # 0인 경우는 드물게 있을 수 있으니 경고만 출력
                print("   ⚠️ 부호가 같거나 0입니다. (GRL 확인 필요)")
        if hasattr(training_model, 'grl'):
            hook_handle = training_model.grl.register_full_backward_hook(check_grl_hook)
            print("✨ GRL 검증 훅 등록됨")

    cudnn_fallback_notified = False

    def safe_model_forward(input_batch):
        nonlocal cudnn_fallback_notified
        try:
            return training_model(input_batch)
        except RuntimeError as e:
            err = str(e)
            is_cudnn_runtime_error = (
                "CUDNN_STATUS_EXECUTION_FAILED" in err
                or "Unable to find a valid cuDNN algorithm" in err
                or "CUDNN_STATUS_NOT_SUPPORTED" in err
                or "cuDNN error" in err
            )
            if not is_cudnn_runtime_error:
                raise
            if not cudnn_fallback_notified:
                print(f"[경고] 학습 forward 중 cuDNN 실패로 cuDNN 비활성 재시도: {e}")
                cudnn_fallback_notified = True
            with torch.backends.cudnn.flags(enabled=False):
                return training_model(input_batch)

    best_model_wts = copy.deepcopy(training_model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_acc = 0.0
    best_loss = 100
    best_eo = float('inf')
    best_ba = 0.0

    epoch_logs = []
    import sys
    from io import StringIO
    
    for epoch in range(num_epochs):
        print('에폭 {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)




        for phase in ['train', 'test']:
            output_capture = StringIO()
            original_stdout = sys.stdout
            sys.stdout = output_capture

            if phase == 'train':
                training_model.train()
            else:
                training_model.eval()

            running_loss = 0.0
            running_task_loss = 0.0
            running_gender_loss = 0.0
            running_corrects = [0] * len(col_used_training)
            running_gender_corrects = 0

            # Balanced Accuracy & EO 통계
            balanced_acc_stats = [{(g, c): [0, 0] for g in [0,1] for c in [0,1]} 
                                for _ in range(len(col_used_training))]
            eqodds_stats = [{g: {'TP':0, 'P':0, 'FP':0, 'N':0} for g in [0,1]} 
                           for _ in range(len(col_used_training))]

            for sample_batched in dataloaders[phase]:
                image_batched, label_batched = sample_batched
                image_batched = image_batched.to(device, dtype=torch.float)
                label_batched = label_batched.to(device)
                task_labels = label_batched[:, 0:len(col_used_training)]
                gender_batched = label_batched[:, -1]  # 성별 라벨

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # ----------------------------------------------------------
                    # 🔥 [수정 2] Forward Pass 로직 통합 (Train/Test 모두 성별값 받기)
                    # ----------------------------------------------------------
                    if use_debiasing:
                        # Train, Test 모두 gender_outputs를 받아옵니다.
                        task_outputs, gender_outputs = safe_model_forward(image_batched)
                        outputs = task_outputs # 메인 태스크용 출력
                    else:
                        outputs = torch.squeeze(safe_model_forward(image_batched))
                        gender_outputs = None

                    # ----------------------------------------------------------
                    # Loss 계산
                    # ----------------------------------------------------------
                    task_loss, acc = loss_multi_tasks(outputs, label_batched, criterion, 
                                                    output_cols_each_task, True)
                    
                    total_loss = task_loss # 기본값

                    if use_debiasing:
                        gender_loss = gender_criterion(gender_outputs, gender_batched.float())
                        # Train 단계에서만 Loss 합산 및 역전파
                        if phase == 'train':
                            total_loss = task_loss + lambda_gender * gender_loss
                        else:
                            # Test 때는 기록용으로만 더함 (Backprop 안함)
                            total_loss = task_loss + lambda_gender * gender_loss 

                    # ----------------------------------------------------------
                    # 🔥 [수정 3] 성별 정확도 계산 (Train/Test 모두 수행)
                    # ----------------------------------------------------------
                    if use_debiasing and gender_outputs is not None:
                        # Sigmoid -> 0.5 Thresholding
                        gender_preds = (torch.sigmoid(gender_outputs) > 0.5).float()
                        # 차원 맞추기 후 비교
                        correct_g = (gender_preds.view(-1) == gender_batched.float().view(-1)).sum().item()
                        running_gender_corrects += correct_g

                    # ----------------------------------------------------------
                    # Backward & Optimizer (Train Only)
                    # ----------------------------------------------------------
                    if phase == 'train':
                        total_loss.backward()
                        
                        if pruning_masks is not None:
                            # 🔥 수정: Wrapper인지 확인 후 적절한 모델 전달
                            if use_debiasing:
                                apply_pruning_masks_to_gradients(training_model.base_model, pruning_masks)
                            else:
                                apply_pruning_masks_to_gradients(training_model, pruning_masks)
                        
                        optimizer.step()
                        
                        if pruning_masks is not None:
                            # 🔥 수정: Wrapper인지 확인 후 적절한 모델 전달
                            if use_debiasing:
                                apply_pruning_masks(training_model.base_model, pruning_masks)
                            else:
                                apply_pruning_masks(training_model, pruning_masks)

                    # 통계 누적
                    running_loss += total_loss.item() * image_batched.size(0)
                    if use_debiasing:
                        running_task_loss += task_loss.item() * image_batched.size(0)
                        running_gender_loss += gender_loss.item() * image_batched.size(0)
                    
                    running_corrects = [running_corrects[j] + acc[j] for j in range(len(col_used_training))]

                # 🔥 EO 통계 누적 (기존 로직 유지)
                # Test phase에서만 정확한 예측을 위해 outputs 재계산
                if use_debiasing:
                    task_outputs_for_stats, _ = safe_model_forward(image_batched)
                    outputs_for_stats = task_outputs_for_stats
                else:
                    outputs_for_stats = torch.squeeze(safe_model_forward(image_batched))
                
                for task_idx, (st, ed) in enumerate(output_cols_each_task):
                    task_out = outputs_for_stats[:, st:ed]
                    task_label = task_labels[:, task_idx]
                    _, task_pred = torch.max(task_out, 1)

                    for g in [0,1]:
                        # TPR
                        mask_pos = (gender_batched == g) & (task_label == 1)
                        P = mask_pos.sum().item()
                        TP = ((task_pred == 1) & mask_pos).sum().item()
                        eqodds_stats[task_idx][g]['TP'] += TP
                        eqodds_stats[task_idx][g]['P'] += P

                        # FPR
                        mask_neg = (gender_batched == g) & (task_label == 0)
                        N = mask_neg.sum().item()
                        FP = ((task_pred == 1) & mask_neg).sum().item()
                        eqodds_stats[task_idx][g]['FP'] += FP
                        eqodds_stats[task_idx][g]['N'] += N

                        # Balanced Accuracy
                        for c in [0,1]:
                            mask = (gender_batched == g) & (task_label == c)
                            total = mask.sum().item()
                            if total > 0:
                                correct = ((task_pred == task_label) & mask).sum().item()
                                balanced_acc_stats[task_idx][(g,c)][0] += correct
                                balanced_acc_stats[task_idx][(g,c)][1] += total
            
            # 🔥 Loss 출력 (Debiasing 모드에서 task/gender loss 구분)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = sum(running_corrects).double() / (len(dataloaders[phase].dataset)*len(col_used_training))
            epoch_gender_acc = 0.0
            if use_debiasing:
                epoch_task_loss = running_task_loss / len(dataloaders[phase].dataset)
                epoch_gender_loss = running_gender_loss / len(dataloaders[phase].dataset)
                epoch_gender_acc = running_gender_corrects / len(dataloaders[phase].dataset)

                print(f"\n📊 Loss & Acc 분석 ({phase}):")
                print(f"  Total Loss:  {epoch_loss:.4f}")
                print(f"  Task Loss:   {epoch_task_loss:.4f}")
                print(f"  Gender Loss: {epoch_gender_loss:.4f}")
                print(f"  Gender Acc:  {epoch_gender_acc:.4f} (🔻 목표: 0.5)") # 파일에 저장됨
            else:
                 print(f"\n📊 Loss 분석 ({phase}):")
                 print(f"  Total Loss:  {epoch_loss:.4f}")
            # 🔥 BA & EO 계산 (기존 로직 유지)
            all_ba_means = []
            all_eos = []
            
            for task_idx in range(len(col_used_training)):
                ba_list = []
                for g in [0,1]:
                    for c in [0,1]:
                        correct, total = balanced_acc_stats[task_idx][(g,c)]
                        acc_val = correct/total if total>0 else None
                        ba_list.append(acc_val if acc_val is not None else 0)
                
                valid_ba = [v for v in ba_list if v is not None]
                ba_mean = np.mean(valid_ba) if valid_ba else None
                all_ba_means.append(ba_mean if ba_mean is not None else 0)

                # EO 계산
                tpr_vals, fpr_vals = [], []
                for g in [0,1]:
                    TP = eqodds_stats[task_idx][g]['TP']
                    P = eqodds_stats[task_idx][g]['P']
                    FP = eqodds_stats[task_idx][g]['FP']
                    N = eqodds_stats[task_idx][g]['N']
                    tpr = TP/P if P>0 else None
                    fpr = FP/N if N>0 else None
                    tpr_vals.append(tpr if tpr is not None else 0)
                    fpr_vals.append(fpr if fpr is not None else 0)
                
                if all([v is not None for v in tpr_vals+fpr_vals]):
                    tpr_gap = abs(tpr_vals[0] - tpr_vals[1])
                    fpr_gap = abs(fpr_vals[0] - fpr_vals[1])
                    eo = (tpr_gap + fpr_gap) / 2
                    all_eos.append(eo)
                else:
                    all_eos.append(0)

            # ... (기존 출력 로직 유지) ...
            
            print("="*50)
            print("전체 성능 요약")
            print("="*50)
            if len(running_corrects) > 0:
                total_acc = sum(running_corrects) / (len(dataloaders[phase].dataset)*len(col_used_training))
                print(f"전체 클래스 평균 정확도: {total_acc:.4f}")
            if len(all_ba_means) > 0:
                mean_ba = np.mean(all_ba_means)
                print(f"전체 클래스 평균 Balanced Accuracy: {mean_ba:.4f}")
            if len(all_eos) > 0:
                mean_eo = np.mean(all_eos)
                print(f"전체 클래스 평균 EO: {mean_eo:.4f}")
            print("="*50)

            # 2. 세부 결과 출력
            for task_idx in range(len(col_used_training)):
                class_no = col_used_training[task_idx]
                ba_list = []
                for g in [0,1]:
                    for c in [0,1]:
                        correct, total = balanced_acc_stats[task_idx][(g,c)]
                        acc_val = correct/total if total>0 else None
                        ba_list.append(acc_val if acc_val is not None else 0)
                        # None 체크를 추가한 출력
                        if acc_val is None:
                            print(f"{class_no} (gender={g}, class={c}): {correct}/{total} = N/A")
                        else:
                            print(f"{class_no} (gender={g}, class={c}): {correct}/{total} = {acc_val:.4f}")
                
                valid_ba = [v for v in ba_list if v is not None]
                ba_mean = np.mean(valid_ba) if valid_ba else None
                if ba_mean is None:
                    print(f"{class_no} ⭐⭐⭐ Balanced Accuracy: N/A")
                else:
                    print(f"{class_no} ⭐⭐⭐ Balanced Accuracy: {ba_mean:.4f}\n")

                # Equalized Odds 출력
                tpr_vals, fpr_vals = [], []
                for g in [0,1]:
                    TP = eqodds_stats[task_idx][g]['TP']
                    P = eqodds_stats[task_idx][g]['P']
                    FP = eqodds_stats[task_idx][g]['FP']
                    N = eqodds_stats[task_idx][g]['N']
                    tpr = TP/P if P>0 else None
                    fpr = FP/N if N>0 else None
                    tpr_vals.append(tpr if tpr is not None else 0)
                    fpr_vals.append(fpr if fpr is not None else 0)
                    if tpr is None:
                        print(f"{class_no} TPR (gender={g}): {TP}/{P} = N/A")
                    else:
                        print(f"{class_no} TPR (gender={g}): {TP}/{P} = {tpr:.4f}")
                    if fpr is None:
                        print(f"{class_no} FPR (gender={g}): {FP}/{N} = N/A")
                    else:
                        print(f"{class_no} FPR (gender={g}): {FP}/{N} = {fpr:.4f}")

                if all([v is not None for v in tpr_vals+fpr_vals]):
                    tpr_gap = abs(tpr_vals[0] - tpr_vals[1])
                    fpr_gap = abs(fpr_vals[0] - fpr_vals[1])
                    eo = (tpr_gap + fpr_gap) / 2
                    print(f"{class_no} TPR gap: {tpr_gap:.4f}")
                    print(f"{class_no} FPR gap: {fpr_gap:.4f}")
                    print(f"{class_no} ⭐⭐⭐ EO: {eo:.4f}")
                else:
                    print(f"{class_no} EO: N/A (유효하지 않은 데이터)")
                print("-"*50)

            # 출력 저장
            sys.stdout = original_stdout
            output_text = output_capture.getvalue()
            output_capture.close()

            # 🔥 Best 모델 갱신 (기존 로직 유지)
            if phase == 'test':
                update_best = False
                reason = ""
                
                if epoch_acc > best_acc:
                    update_best = True
                    reason = f"정확도 향상 ({best_acc:.4f} → {epoch_acc:.4f})"
                elif epoch_acc == best_acc:
                    if mean_eo < best_eo:
                        update_best = True
                        reason = f"정확도 동일, EO 개선 ({best_eo:.4f} → {mean_eo:.4f})"
                    elif mean_eo == best_eo and epoch_loss < best_loss:
                        update_best = True
                        reason = f"정확도·EO 동일, 손실 감소 ({best_loss:.4f} → {epoch_loss:.4f})"
                
                if update_best:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_eo = mean_eo
                    best_ba = mean_ba
                    best_model_wts = copy.deepcopy(training_model.state_dict())
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    print(f"\n✅ Best 모델 갱신: {reason}")
                    print(f"   현재 Best - Acc: {best_acc:.4f}, EO: {best_eo:.4f}, Loss: {best_loss:.4f}")
                else:
                    print(f"\n⏸️  Best 모델 유지")
                    print(f"   현재 에폭 - Acc: {epoch_acc:.4f}, EO: {mean_eo:.4f}, Loss: {epoch_loss:.4f}")
                    print(f"   Best 기록 - Acc: {best_acc:.4f}, EO: {best_eo:.4f}, Loss: {best_loss:.4f}")

            import config
           
            if phase == 'test':
                #! 결과 저장
                save_dir = 'retrain_epoch_results'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Total_info_prune: {config.glo_prune_iter + 1}_retrain: {epoch + 1}_output_{timestamp}.txt"
                file_path = os.path.join(save_dir, filename)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(output_text)
                print(f"[출력 저장] {file_path}")

                #! 모델 저장
                model_save_dir = 'save_models'  # 별도의 모델 저장 폴더
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir, exist_ok=True)
                
                model_filename = f"save_model_prune: {config.glo_prune_iter + 1}_retrain: {epoch + 1}_{timestamp}.pt"
                model_path = os.path.join(model_save_dir, model_filename)
                
                torch.save({
                    'model_state_dict': training_model.state_dict(), # 🔥 수정됨 (Wrapper 포함 저장)
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                }, os.path.join(model_save_dir, model_filename))
                
                print(f"[모델 저장] {model_path}")

        if hook_handle is not None:
            hook_handle.remove()
            hook_handle = None
            print("\n✨ [System] GRL 검증 완료. 로그 폭탄 방지를 위해 훅을 제거했습니다.")
            
        print('에폭: {}/{} - 현재 best_acc: {:.4f}'.format(epoch, num_epochs - 1, best_acc))

    # 🔥 Best 모델 로드
    training_model.load_state_dict(best_model_wts)
    print("❗학습 완료, 최고 정확도: ", best_acc, type(best_acc))

    # 🔥 베스트 모델 최종 결과 저장
    import config
    save_dir = 'retrain_epoch_results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 베스트 모델 요약 파일 생성
    best_model_filename = f"BEST_MODEL_SUMMARY_prune_{config.glo_prune_iter + 1}_{timestamp}.txt"
    best_model_path = os.path.join(save_dir, best_model_filename)
    
    with open(best_model_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("🏆 BEST MODEL SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"분석 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Prune Iteration: {config.glo_prune_iter + 1}\n")
        f.write(f"총 재학습 에폭 수: {num_epochs}\n\n")
        f.write("-"*80 + "\n")
        f.write("📊 Best Model Performance\n")
        f.write("-"*80 + "\n")
        f.write(f"Best Accuracy:  {best_acc:.4f}\n")
        f.write(f"Best EO:        {best_eo:.4f}\n")
        f.write(f"Best Loss:      {best_loss:.4f}\n")
        f.write(f"Best BA:        {best_ba:.4f}\n")
        f.write("-"*80 + "\n\n")
    
    print(f"\n🏆 [베스트 모델 요약 저장] {best_model_path}")
    
    # 🔥 베스트 모델 가중치도 별도 저장
    best_model_weights_dir = 'save_models'
    if not os.path.exists(best_model_weights_dir):
        os.makedirs(best_model_weights_dir, exist_ok=True)
    
    best_weights_filename = f"BEST_MODEL_prune_{config.glo_prune_iter + 1}_acc_{best_acc:.4f}_eo_{best_eo:.4f}_{timestamp}.pt"
    best_weights_path = os.path.join(best_model_weights_dir, best_weights_filename)
    
    torch.save({
        'model_state_dict': best_model_wts,
        'optimizer_state_dict': best_optimizer_state
    }, best_weights_path)
    
    print(f"🏆 [베스트 모델 가중치 저장] {best_weights_path}\n")

    # 🔥 Debiasing 모드면 원래 모델 반환
    torch.backends.cudnn.enabled = prev_cudnn_enabled
    torch.backends.cudnn.benchmark = prev_cudnn_benchmark

    if use_debiasing:
        final_model = training_model.get_base_model()
        print("\n✅ Debiased 모델에서 Base 모델 추출 완료")
        return final_model, best_acc, best_loss, best_optimizer_state
    else:
        return training_model, best_acc, best_loss, best_optimizer_state

import config
from pprint import pformat
def save_epoch_logs(epoch_logs, csv_savedir, save_file=True, epoch=None):
        """
        epoch_logs: [{'epoch':..., 'loss':..., 'accuracy':..., ...}, ...]
        csv_savedir: 로그 CSV를 저장할 폴더
        save_file: True면 실제 파일로 저장
        """
        print("save_epoch_logs 실행")

        if not os.path.exists(csv_savedir):
            os.makedirs(csv_savedir, exist_ok=True)

        # 현재 날짜/시간, 예: "20230401"
        timestamp = datetime.datetime.now().strftime("%Y%m%d")


        # 처음에 "train_log_0_YYYYMMDD.csv"로 시도
        base_filename = f"Test_result_prune-iter_{config.glo_prune_iter}_number_{epoch}_{timestamp}.text"
        df_path = os.path.join(csv_savedir, base_filename)

        # 만약 이미 파일이 있으면 "train_log_1_YYYYMMDD.csv", "train_log_2_..." 등으로 변경
        
        #while os.path.exists(df_path):
        #    counter += 1
        #    df_path = os.path.join(csv_savedir, f"Test_result_prune-iter_{config.glo_prune_iter}_number_{num_epochs}_{timestamp}.text")

        # 로그 리스트 -> DataFrame
        #df = pd.DataFrame(epoch_logs)

        if save_file:
            #df.to_csv(df_path, index=False)
            #print(f"학습 로그 CSV 저장 완료: {df_path}")
                    # JSON natively 계층적으로 보이도록 indent=2 설정
            formatted = pformat(epoch_logs, indent=2, width=80, compact=False, sort_dicts=False)

            with open(df_path, 'w', encoding='utf-8') as f:
                f.write(formatted)
            print(f"계층 구조로 저장 완료: {df_path}")

def acc_scores(fair_df, col_used=None, sensitive_group="gender"):
    
        # EO 함수와 동일한 전처리: 중복 컬럼 제거
        df = fair_df.loc[:, ~fair_df.columns.duplicated()]
        
        # col_used가 None이면 모든 가능한 속성 컬럼 찾기
        if col_used is None:
            # _preds_fair로 끝나는 컬럼들 찾기
            pred_cols = [col for col in df.columns if col.endswith('_preds_fair')]
            # 원본 컬럼명 추출 (gender 제외)
            col_used = []
            for pred_col in pred_cols:
                original_col = pred_col.replace('_preds_fair', '')
                if original_col in df.columns and original_col != sensitive_group:
                    col_used.append(original_col)
        
        # col_used가 리스트로 전달된 경우, gender가 마지막에 있다면 제거 (EO 함수 방식)
        elif isinstance(col_used, list) and len(col_used) > 0 and col_used[-1] == sensitive_group:
            col_used = col_used[:-1]  # 마지막 요소(gender) 제거
        
        # gender가 실수로 포함되었다면 제거 (EO 함수처럼 col_used[:-1] 사용)
        if sensitive_group in col_used:
            col_used.remove(sensitive_group)
        
        print(f"사용할 컬럼들: {col_used}")
        
        # 민감 그룹의 고유값들 구하기 (EO 함수와 동일한 방식)
        try:
            unq_groups = sorted(df[sensitive_group].unique())  # EO 함수와 동일
            print(f"민감 그룹들: {unq_groups}")
            
        except Exception as e:
            print(f"Error processing sensitive group: {e}")
            return 0.0
        
        n_row = len(df)
        
        # (A) 각 태스크별 정확도 계산
        acc_each_task = []
        
        for truth_col in col_used:  # EO 함수와 동일한 변수명 사용
            pred_col = truth_col + "_preds_fair"  # EO 함수와 동일한 suffix
            
            # 컬럼이 존재하는지 확인 (EO 함수와 동일한 방식)
            if pred_col not in df.columns:
                print(f"[경고] {pred_col} 컬럼이 없음 → 건너뜀")
                continue
            
            try:
                # 타입 통일: 둘 다 숫자로 변환해서 비교
                truth_values = pd.to_numeric(df[truth_col], errors='coerce')
                pred_values = pd.to_numeric(df[pred_col], errors='coerce')
                
                # NaN 값들은 제외하고 비교
                valid_mask = ~(truth_values.isna() | pred_values.isna())
                valid_truth = truth_values[valid_mask]
                valid_pred = pred_values[valid_mask]
                
                if len(valid_truth) == 0:
                    print(f"{truth_col:<25} 유효한 데이터 없음")
                    continue
                
                correct_predictions = (valid_truth == valid_pred).sum()
                acc_val = correct_predictions / len(valid_truth)
                acc_each_task.append(acc_val)
                
                print(f"{truth_col:<25} Accuracy = {acc_val:.3f} ({correct_predictions}/{len(valid_truth)})")
                
            except Exception as e:
                print(f"Error calculating accuracy for {truth_col}: {e}")
                continue
        
        # (B) 민감 그룹별 정확도 계산 (EO 함수와 동일한 방식)
        acc_each_group = {}
        
        for group in unq_groups:
            try:
                # EO 함수와 동일한 필터링 방식
                sub = df[df[sensitive_group] == group]
                n_group = len(sub)
                
                if n_group == 0:
                    acc_each_group[group] = 0.0
                    continue
                
                group_accs = []
                for truth_col in col_used:
                    pred_col = truth_col + "_preds_fair"
                    
                    # 컬럼 존재 확인
                    if pred_col not in df.columns:
                        continue
                    
                    try:
                        # EO 함수처럼 직접 비교
                        correct_predictions = (sub[truth_col] == sub[pred_col]).sum()
                        group_acc = correct_predictions / n_group
                        group_accs.append(group_acc)
                        
                    except Exception as e:
                        print(f"Error calculating group accuracy for {truth_col}, group {group}: {e}")
                        continue
                
                # 여러 태스크의 평균 정확도
                avg_acc = np.mean(group_accs) if group_accs else 0.0
                acc_each_group[group] = avg_acc
                
            except Exception as e:
                print(f"Error processing group {group}: {e}")
                acc_each_group[group] = 0.0
        
        # 전체 정확도 (모든 태스크의 평균)
        overall_acc = np.mean(acc_each_task) if acc_each_task else 0.0
        
        print("-" * 40)
        print(f"계산된 태스크 수: {len(acc_each_task)}")
        print(f"전체 평균 정확도: {overall_acc:.3f}")
        print(f"그룹별 정확도: {acc_each_group}")
    
        
        return overall_acc, acc_each_group


import sys
from io import StringIO
import os

def analyze_balanced_accuracy(df, selected_cols, epoch, save_text_file=True, text_filename=None, save_folder=None):
    """
    CelebA FairGRAPE 데이터셋에 대한 Balanced Accuracy 분석 함수
    
    Args:
        df: 분석할 데이터프레임
        selected_cols: 분석할 속성 리스트
        save_text_file: 텍스트 파일로 저장할지 여부 (기본값: True)
        text_filename: 텍스트 파일명 (기본값: None이면 자동 생성)
        save_folder: 저장할 폴더 경로 (기본값: None이면 현재 폴더)
    
    Returns:
        dict: {
            'detailed_results': DataFrame,  # 상세 결과
            'summary_results': DataFrame,   # 속성별 평균
            'combined_results': DataFrame,  # 상세+평균 통합
            'analysis_text': str           # 전체 분석 출력 텍스트
        }
    """
    
    def calculate_balanced_accuracy_simple(df, attribute, gender, target_class):
        """
        간단한 balanced accuracy 계산
        
        공식: (정답 개수) / (해당 그룹 전체 개수)
        """

        df = df.loc[:, ~df.columns.duplicated()].copy()

        # 예측 컬럼명은 항상 {attribute}_preds_fair
        pred_col = f"{attribute}_preds_fair"
        
        # 컬럼 존재 확인
        if attribute not in df.columns:
            return None, None, f"실제값 컬럼을 찾을 수 없습니다: {attribute}"
        
        if pred_col not in df.columns:
            return None, None, f"예측값 컬럼을 찾을 수 없습니다: {pred_col}"
        
        # 해당 성별과 클래스에 속하는 데이터 필터링
        mask = (df['gender'] == gender) & (df[attribute] == target_class)
        group_data = df[mask]
        
        total_count = len(group_data)
        
        if total_count == 0:
            return 0, 0, None  # 데이터 없음
        
        # 정답을 맞춘 개수
        correct_count = (group_data[attribute] == group_data[pred_col]).sum()
        
        accuracy = correct_count / total_count
        
        return correct_count, total_count, accuracy

    df = df.loc[:, ~df.columns.duplicated()].copy()

    # 출력을 캡처하기 위한 StringIO 객체
    output_capture = StringIO()
    
    # 원래 stdout 저장
    original_stdout = sys.stdout
    
    try:
        # stdout을 StringIO로 리다이렉트
        sys.stdout = output_capture
        
        # 데이터셋 기본 정보 출력
        print("="*80)
        print("CelebA FairGRAPE 데이터셋 분석")
        print("="*80)
        print(f"분석 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 데이터 개수: {len(df):,}")
        print(f"총 컬럼 개수: {len(df.columns)}")
        print(f"성별 분포:")
        #gender_counts = df['gender'].value_counts()
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            gender_label = "여성" if gender == 0 else "남성"
            print(f"  {gender_label} (gender={gender}): {count:,}개 ({count/len(df)*100:.1f}%)")

        print(f"\n분석할 속성: {len(selected_cols)}개")
        print("-"*80)

        # 모든 결과를 저장할 리스트
        all_results = []
        attribute_averages = []  # 속성별 평균을 저장할 리스트

        # 각 속성별로 분석
        for i, attribute in enumerate(selected_cols, 1):
            print(f"\n[{i:2d}/{len(selected_cols)}] {attribute}")
            
            # 예측 컬럼명
            pred_col = f"{attribute}_preds_fair"
            
            # 컬럼 존재 확인
            if attribute not in df.columns:
                print(f"  오류: '{attribute}' 컬럼이 없습니다.")
                continue
            
            if pred_col not in df.columns:
                print(f"  오류: '{pred_col}' 컬럼이 없습니다.")
                continue
            
            print(f"  예측 컬럼: {pred_col}")
            
            # 클래스 분포 확인
            class_dist = df[attribute].value_counts().sort_index()
            print(f"  클래스 분포: {dict(class_dist)}")
            
            # 4가지 조합에 대해 계산하고 평균 계산을 위한 리스트
            current_attribute_accuracies = []
            current_attribute_results = []  # 현재 속성의 상세 결과들
            
            # 4가지 조합에 대해 계산 (성별 2개 × 클래스 2개)
            for gender in [0, 1]:
                for target_class in [0, 1]:
                    correct, total, accuracy = calculate_balanced_accuracy_simple(
                        df, attribute, gender, target_class
                    )
                    
                    gender_label = "여성" if gender == 0 else "남성"
                    class_label = attribute if target_class == 1 else f"Not_{attribute}"
                    
                    if accuracy is None:
                        if total == 0:
                            result_str = f"{correct}/{total} = N/A (데이터 없음)"
                        else:
                            result_str = f"오류: {correct}"  # correct에 에러 메시지가 들어있음
                    else:
                        result_str = f"{correct}/{total} = {accuracy:.3f}"
                        current_attribute_accuracies.append(accuracy)  # 평균 계산용
                    
                    print(f"    {gender_label}, {class_label}: {result_str}")
                    
                    # 상세 결과 저장
                    detail_result = {
                        'Attribute': attribute,
                        'Gender': gender_label,
                        'Gender_Code': gender,
                        'Target_Class': target_class,
                        'Class_Label': class_label,
                        'Correct_Predictions': correct,
                        'Total_Cases': total,
                        'Balanced_Accuracy': accuracy,
                        'Prediction_Column': pred_col,
                        'Result_Type': 'Detail'  # 상세 결과임을 표시
                    }
                    
                    current_attribute_results.append(detail_result)
                    all_results.append(detail_result)
            
            # 4개 조합의 평균 계산 및 출력
            if current_attribute_accuracies:
                avg_accuracy = np.mean(current_attribute_accuracies)
                min_accuracy = min(current_attribute_accuracies)
                max_accuracy = max(current_attribute_accuracies)
                std_accuracy = np.std(current_attribute_accuracies)
                
                print(f"  ★ {attribute} 평균 Balanced Accuracy: {avg_accuracy:.3f} "
                      f"(min: {min_accuracy:.3f}, max: {max_accuracy:.3f}, std: {std_accuracy:.3f})")
                
                # 속성별 평균 저장
                average_result = {
                    'Attribute': attribute,
                    'Average_Balanced_Accuracy': avg_accuracy,
                    'Num_Valid_Groups': len(current_attribute_accuracies),
                    'Min_Accuracy': min_accuracy,
                    'Max_Accuracy': max_accuracy,
                    'Std_Accuracy': std_accuracy,
                    'Individual_Accuracies': current_attribute_accuracies
                }
                
                # 평균 결과도 통합 데이터에 추가
                combined_average_result = {
                    'Attribute': attribute,
                    'Gender': 'Average',
                    'Gender_Code': -1,
                    'Target_Class': -1,
                    'Class_Label': f'{attribute}_Average',
                    'Correct_Predictions': sum([r['Correct_Predictions'] for r in current_attribute_results if r['Balanced_Accuracy'] is not None]),
                    'Total_Cases': sum([r['Total_Cases'] for r in current_attribute_results if r['Balanced_Accuracy'] is not None]),
                    'Balanced_Accuracy': avg_accuracy,
                    'Prediction_Column': pred_col,
                    'Result_Type': 'Average',  # 평균 결과임을 표시
                    'Min_Accuracy': min_accuracy,
                    'Max_Accuracy': max_accuracy,
                    'Std_Accuracy': std_accuracy,
                    'Num_Valid_Groups': len(current_attribute_accuracies)
                }
                
                all_results.append(combined_average_result)
                attribute_averages.append(average_result)
            else:
                print(f"  ★ {attribute} 평균 Balanced Accuracy: N/A (유효한 데이터 없음)")
                
                # 속성별 평균 저장
                average_result = {
                    'Attribute': attribute,
                    'Average_Balanced_Accuracy': None,
                    'Num_Valid_Groups': 0,
                    'Min_Accuracy': None,
                    'Max_Accuracy': None,
                    'Std_Accuracy': None,
                    'Individual_Accuracies': []
                }
                
                # 평균 결과도 통합 데이터에 추가
                combined_average_result = {
                    'Attribute': attribute,
                    'Gender': 'Average',
                    'Gender_Code': -1,
                    'Target_Class': -1,
                    'Class_Label': f'{attribute}_Average',
                    'Correct_Predictions': 0,
                    'Total_Cases': 0,
                    'Balanced_Accuracy': None,
                    'Prediction_Column': pred_col,
                    'Result_Type': 'Average',
                    'Min_Accuracy': None,
                    'Max_Accuracy': None,
                    'Std_Accuracy': None,
                    'Num_Valid_Groups': 0
                }
                
                all_results.append(combined_average_result)
                attribute_averages.append(average_result)

        # 결과를 DataFrame으로 변환
        combined_df = pd.DataFrame(all_results)
        detailed_df = combined_df[combined_df['Result_Type'] == 'Detail'].copy()
        summary_df = pd.DataFrame(attribute_averages)

        # 전체 요약 통계 출력
        print("\n" + "="*80)
        print("📊 전체 요약 통계")
        print("="*80)
        
        valid_results = detailed_df[detailed_df['Balanced_Accuracy'].notna()]
        valid_averages = summary_df[summary_df['Average_Balanced_Accuracy'].notna()]

        if len(valid_results) > 0:
            print(f"유효한 그룹별 결과: {len(valid_results)}개")
            print(f"전체 그룹별 평균: {valid_results['Balanced_Accuracy'].mean():.3f}")
            print(f"전체 그룹별 최고: {valid_results['Balanced_Accuracy'].max():.3f}")
            print(f"전체 그룹별 최저: {valid_results['Balanced_Accuracy'].min():.3f}")
            
            # 성별별 평균
            gender_avg = valid_results.groupby('Gender')['Balanced_Accuracy'].mean()
            print(f"\n성별별 평균:")
            for gender, avg in gender_avg.items():
                print(f"  {gender}: {avg:.3f}")

        if len(valid_averages) > 0:
            print(f"\n🎯 속성별 평균 분석")
            print(f"유효한 속성: {len(valid_averages)}개")
            print(f"속성별 평균의 평균: {valid_averages['Average_Balanced_Accuracy'].mean():.3f}")
            print(f"속성별 평균 최고: {valid_averages['Average_Balanced_Accuracy'].max():.3f}")
            print(f"속성별 평균 최저: {valid_averages['Average_Balanced_Accuracy'].min():.3f}")
            
            # 상위 10개 속성 (평균 기준)
            print(f"\n🏆 상위 10개 속성 (평균 Balanced Accuracy):")
            top_10_attrs = valid_averages.nlargest(10, 'Average_Balanced_Accuracy')
            for i, (_, row) in enumerate(top_10_attrs.iterrows(), 1):
                print(f"  {i:2d}. {row['Attribute']:<20}: {row['Average_Balanced_Accuracy']:.3f} "
                      f"(min: {row['Min_Accuracy']:.3f}, max: {row['Max_Accuracy']:.3f}, std: {row['Std_Accuracy']:.3f})")
            
            # 하위 10개 속성 (평균 기준)
            print(f"\n⚠️  하위 10개 속성 (평균 Balanced Accuracy):")
            bottom_10_attrs = valid_averages.nsmallest(10, 'Average_Balanced_Accuracy')
            for i, (_, row) in enumerate(bottom_10_attrs.iterrows(), 1):
                print(f"  {i:2d}. {row['Attribute']:<20}: {row['Average_Balanced_Accuracy']:.3f} "
                      f"(min: {row['Min_Accuracy']:.3f}, max: {row['Max_Accuracy']:.3f}, std: {row['Std_Accuracy']:.3f})")

            # 개별 그룹별 상위 5개 결과
            print(f"\n🥇 개별 그룹별 상위 5개 결과:")
            top_5 = valid_results.nlargest(5, 'Balanced_Accuracy')
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                print(f"  {i}. {row['Attribute']} ({row['Gender']}, {row['Class_Label']}): "
                      f"{row['Balanced_Accuracy']:.3f} ({row['Correct_Predictions']}/{row['Total_Cases']})")
            
            # 개별 그룹별 하위 5개 결과
            print(f"\n🥉 개별 그룹별 하위 5개 결과:")
            bottom_5 = valid_results.nsmallest(5, 'Balanced_Accuracy')
            for i, (_, row) in enumerate(bottom_5.iterrows(), 1):
                print(f"  {i}. {row['Attribute']} ({row['Gender']}, {row['Class_Label']}): "
                      f"{row['Balanced_Accuracy']:.3f} ({row['Correct_Predictions']}/{row['Total_Cases']})")

            # 성별간 차이가 큰 속성들
            print(f"\n⚖️  성별간 차이가 큰 속성들 (상위 10개):")
            pivot_data = valid_results.groupby(['Attribute', 'Gender'])['Balanced_Accuracy'].mean().reset_index()
            pivot_table = pivot_data.pivot(index='Attribute', columns='Gender', values='Balanced_Accuracy')
            
            # 두 성별 모두 데이터가 있는 속성만 필터링
            complete_attrs = pivot_table.dropna()
            if len(complete_attrs) > 0:
                complete_attrs['Difference'] = abs(complete_attrs['여성'] - complete_attrs['남성'])
                complete_attrs['Female_Advantage'] = complete_attrs['여성'] - complete_attrs['남성']
                
                # 차이가 큰 순으로 정렬
                top_diff_attrs = complete_attrs.nlargest(10, 'Difference')
                
                for i, (attr, row) in enumerate(top_diff_attrs.iterrows(), 1):
                    advantage = "여성" if row['Female_Advantage'] > 0 else "남성"
                    print(f"  {i:2d}. {attr:<20}: 여성 {row['여성']:.3f} vs 남성 {row['남성']:.3f} "
                          f"(차이: {row['Difference']:.3f}, {advantage} 우세)")

            # 안정적인 성능을 보이는 속성들 (높은 평균, 낮은 분산)
            stable_attrs = valid_averages[
                (valid_averages['Average_Balanced_Accuracy'] > 0.85) & 
                (valid_averages['Std_Accuracy'] < 0.1)
            ].sort_values('Average_Balanced_Accuracy', ascending=False)
            
            if len(stable_attrs) > 0:
                print(f"\n🎯 안정적 고성능 속성들 (평균 > 0.85, 표준편차 < 0.1):")
                for _, row in stable_attrs.iterrows():
                    print(f"  • {row['Attribute']:<20}: {row['Average_Balanced_Accuracy']:.3f} ± {row['Std_Accuracy']:.3f}")

        print("\n" + "="*80)
        print("분석 완료!")
        print("="*80)

    finally:
        # stdout 복원
        sys.stdout = original_stdout
    
    # 캡처된 출력 가져오기
    analysis_text = output_capture.getvalue()
    output_capture.close()
    
    # 콘솔에도 출력
    print(analysis_text)
    
    # 텍스트 파일로 저장
    if save_text_file:
        if text_filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d")
            text_filename = f'balanced_accuracy_analysis_number_{config.glo_prune_iter}_number_{epoch}_{timestamp}.txt'
        
        # 저장 폴더 처리
        if save_folder is not None:
            # 폴더가 없으면 생성
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
                print(f"폴더 '{save_folder}'를 생성했습니다.")
            
            # 전체 파일 경로 생성
            full_path = os.path.join(save_folder, text_filename)
        else:
            full_path = text_filename
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(analysis_text)
        
        print(f"분석 결과가 '{full_path}'에 저장되었습니다.")
    
    # 결과 반환
    return {
        'detailed_results': detailed_df,
        'summary_results': summary_df, 
        'combined_results': combined_df,
        'analysis_text': analysis_text
    }




############################################################################
# loss_multi_tasks 함수: 다중 태스크 예측 출력과 라벨을 분리하여 
# 손실을 계산하고, 정확도도 함께 반환할 수 있습니다.
############################################################################
def loss_multi_tasks(outputs, labels, criterion=None, output_cols_each_task=[(0,7),(7,9),(9,18)], find_acc=False):
    tasks_outputs, tasks_labels, ntasks = [], [], len(output_cols_each_task)
    for i, st_ed in enumerate(output_cols_each_task):
        # 디버깅용 주석: #print(i, st_ed, outputs.shape, labels.shape)
        tasks_outputs.append(outputs[:, st_ed[0]:st_ed[1]])
        tasks_labels.append(labels[:, i])


    if criterion:
        loss = 0
        for i in range(ntasks):
            loss += criterion(tasks_outputs[i], tasks_labels[i])
        loss = loss.double() / ntasks 
    if find_acc:
        acc = [0] * ntasks
        for i in range(ntasks):
            _, task_preds = torch.max(tasks_outputs[i], 1)
            acc[i] = torch.sum(task_preds.cpu() == tasks_labels[i].cpu())

    if find_acc:
        return loss, acc
    else:
        return loss
