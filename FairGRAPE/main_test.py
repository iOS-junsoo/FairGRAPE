
import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
import argparse
import datetime
import pandas as pd
import numpy as np

from dataset import make_frame, make_datasets, prepare_ImageNet
from prune import WS, SNIP, GraSP, Lottery, FairGRAPE, Importance, Random, save_impt_df
from util import make_model, save_model, save_output, download_dataset, show_acc_df, setseed, save_unpruned_model, print_acc_scores
from train_and_val import train, loss_multi_tasks

# 가지치기(Pruning) 기법을 선택하기 위한 매핑
pruner_map = {'WS': WS, "SNIP": SNIP, 'GraSP': GraSP, "Lottery": Lottery, "FairGRAPE": FairGRAPE, "Importance": Importance, "Random": Random}


def experiment(args):
    # -----------------------
    # (1) 명령줄 인자 로드 & 기본 설정
    # -----------------------
    checkpoint = args.checkpoint  # 이전에 이미 가지치기가 적용된 모델(체크포인트) 경로
    dataset = args.dataset  # ['UTKFace', 'FairFace', "CelebA", "Imagenet", "ImbalancedFairFace"]
    network = args.network  # ['resnet34', 'mobilenetv2']
    loss_type = args.loss_type  # ['race', 'raceAndgender', 'gender', 'attrs', 'class']
    sensitive_group = args.sensitive_group  # ['race', 'raceAndgender', 'gender']
    prune_type = args.prune_type  # ['FairGRAPE','SNIP','WS','Lottery', 'GraSP','Full',"readResult"]
    prune_rate = args.prune_rate  # [0.5, 0.7, 0.8, 0.9, 0.99]
    batch_size = args.batch
    init_train = not args.no_init_train
    drop_race = args.drop_race  # util.make_frame() 참고
    retrain = not args.no_retrain
    save_mask = args.save_mask
    delta_p = args.delta_p
    print_acc = args.print_acc
    exp_idx = args.exp_idx
    save_impt = args.save_impt
    para_batch = args.para_batch
    stop_batch = args.stop_batch

    seed = args.seed
    setseed(seed)

    # 결과 저장용 디렉토리 생성
    save_dir = "trained_model/{}".format(prune_type)
    csv_savedir = "fair_dfs"
    dirs = [csv_savedir, 'models', save_dir, "Images"]
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)

    # 실행 정보 출력
    print("유형:{}, 네트워크:{}, 희소도:{}, 데이터셋:{}".format(prune_type, network, prune_rate, dataset))

    # -----------------------
    # (2) 데이터셋 설정 & DataLoader 생성
    # -----------------------
    if dataset == 'FairFace' or dataset == "ImbalancedFairFace":
        csv = 'csv/FairFace.csv'
        face_dir = 'Images/FairFace'
        download_dataset(dataset, face_dir)
        # 학습에 사용될 변수 설정
        if loss_type == 'race':
            total_classes, output_cols_each_task, col_used_training = 7, [(0, 7)], [loss_type]
        elif loss_type == 'gender':
            total_classes, output_cols_each_task, col_used_training = 2, [(0, 2)], [loss_type]
        else:
            total_classes, output_cols_each_task, col_used_training = 14, [(0, 14)], ['raceAndgender']
        # col_used 안에는 민감속성(sensitive group)도 포함
        # FairGRAPE 가지치기 시 사용하지만, 학습에는 포함되지 않을 수 있음
        col_used = col_used_training + [sensitive_group]
        epoches = [13, 3, 3]
        imbalance = dataset == "ImbalancedFairFace"
        frames = make_frame(csv, face_dir, imbalance=imbalance)
        if drop_race:
            # drop_race가 설정되면 특정 인종 제외
            frames_minority = make_frame(csv, face_dir, drop_race=drop_race, imbalance=imbalance)
            train_loader_minority, _ = make_datasets(frames_minority['train'], frames_minority['val'], True, batch_size, col_used)
    elif dataset == 'UTKFace':
        csv = 'csv/UTKFace_labels.csv'
        face_dir = 'Images/UTKFace'
        download_dataset(dataset, face_dir)
        # 학습에 사용될 변수 설정
        if loss_type == 'race':
            total_classes, output_cols_each_task, col_used_training = 4, [(0, 4)], [loss_type]
        elif loss_type == 'gender':
            total_classes, output_cols_each_task, col_used_training = 2, [(0, 2)], [loss_type]
        else:
            total_classes, output_cols_each_task, col_used_training = 6, [(0, 4), (4, 6)], ['race', 'gender']
        col_used = col_used_training + [sensitive_group]
        epoches = [13, 3, 3]
        frames = make_frame(csv, face_dir, seven_races=False)
        if drop_race:
            frames_minority = make_frame(csv, face_dir, seven_races=False, drop_race=drop_race)
            train_loader_minority, _ = make_datasets(frames_minority['train'], frames_minority['val'], True, batch_size, col_used)
    elif dataset == "CelebA":
        csv = 'csv/CelebA.csv'
        face_dir = 'Images/CelebA/img_align_celeba'
        download_dataset(dataset, face_dir)
        frames = make_frame(csv, face_dir, seven_races=False)
        

        # CelebA의 경우 39개 속성(각 2클래스) → total 78개
        output_cols_each_task = [(i * 2, i * 2 + 2) for i in range(39)] # 예측해야 할 각 속성(attribute)을 2차원(1/-1)으로 인덱싱해 저장한 리스트
        col_used_training = [frames['train'].columns[i] for i in range(2, 41)] # frames['train'] 데이터프레임에서 인덱스 2~40번 열을 뽑아, 학습에 쓰는 39개 속성으로 지정(male 속성제외)
        total_classes = 39 * 2 # 최종 모델 클래스 수를 78로 설정
        col_used = col_used_training + [sensitive_group] # 앞서 선정한 학습용 39개 속성(col_used_training)에 민감 속성(예: 'Male')을 추가
        print(f"사용할 CelebA 속성: {col_used}")
        
        epoches = [8, 1, 1]
    elif dataset == "Imagenet":
        csv = 'csv/Imagenet.csv'
        face_dir = 'Images/Imagenet'
        prepare_ImageNet(csv)
        download_dataset(dataset, face_dir)
        frames = make_frame(csv, face_dir, seven_races=False)
        output_cols_each_task = [(0, 104)]
        col_used_training = ['classes']
        total_classes = 104
        col_used = col_used_training + [sensitive_group]
        epoches = [8, 5, 5]
    else:
        raise NotImplementedError("{} is not implemented!".format(dataset))

    lr_schedule = [1e-4, 1e-5, 1e-6]
    # 여기서 make_datasets 호출 시 shuffle=True로 되어 있음
    train_loader, test_loader = make_datasets(frames['train'], frames['val'], True, batch_size, col_used)
    dataloaders = {'train': train_loader, 'test': test_loader}

    save_model_iter = args.save_model_iter
    save_model_iter = [] if isinstance(save_model_iter, int) else save_model_iter
    print(save_model_iter)

    device = torch.device('cuda:0')
    criterion = nn.CrossEntropyLoss()

    torch.cuda.empty_cache()
    best_model = make_model(network=network, n_classes=total_classes).to(device)

    # -----------------------
    # (3) 각 가지치기 기법별 파라미터 설정
    # -----------------------
    if prune_type == 'WS':
        prune_cfgs = [prune_rate]
    elif prune_type == 'SNIP':
        num_batch_sampling, var_scaling = 1, True
        prune_cfgs = [prune_rate, num_batch_sampling, var_scaling]
    elif prune_type == 'Lottery':
        prune_cfgs = [prune_rate]
    elif prune_type == 'GraSP':
        # GraSP: 출력 클래스별 균형을 잡아 신호 계산
        if len(col_used_training) > 1:
            target_col, samples_per_class, num_classes = [i for i in range(len(col_used_training))], 1, len(col_used_training)
        else:
            target_col, num_classes = len(col_used_training) - 1, total_classes
            samples_per_class = 10 if dataset != "Imagenet" else 2
        if drop_race and loss_type == 'race':
            num_classes = num_classes - 1 if drop_race < 10 else 1
        prune_cfgs = [prune_rate, target_col, num_classes, samples_per_class]
    elif prune_type == "FairGRAPE" or prune_type == "Importance":
        sensitive_classes = len(set(frames['train'][sensitive_group]))
        masked_grads = True
        impt = args.impt
        if impt == 2:
            sensitive_classes = total_classes
        prune_cfgs = [prune_rate, frames['train'], face_dir, sensitive_classes, masked_grads, output_cols_each_task, col_used, para_batch, impt, stop_batch, delta_p, network, sensitive_group]
        #prune_cfgs = [prune_rate, frames['train'], face_dir, sensitive_classes, masked_grads, output_cols_each_task, col_used, para_batch, impt, stop_batch, delta_p]
    elif prune_type == "Random":
        prune_cfgs = [prune_rate, True]
    elif prune_type == 'Full':
        prune_rate = 0
    else:
        raise NotImplementedError("Prune method {} is not implemented!".format(prune_type))

    # -----------------------
    # (4) 가지치기 반복 횟수 설정 (single shot or iterative)
    # -----------------------
    pct_remain_after_this_iter = 1 - args.init_pruned
    if prune_type in ['WS', 'SNIP', 'GraSP', 'Full', 'Random']:
        prune_iters = 1 if prune_type != 'Full' else 0
        retrain_lr = 0
        retrain_iters = 0
        keep_per_iter = 1 - prune_rate
        lr_decay_iter = 1
    elif prune_type in ['FairGRAPE', 'Lottery', 'Importance']:
        prune_iters = args.prune_iter # 가지치기 반복횟수
        retrain_lr = args.retrain_lr # 재학습 학습률
        retrain_iters = args.retrain_iter # 재학습 반복 횟수
        keep_per_iter = args.keep_per_iter # 매 횟수마다 남기는 가중치 비율 (90%라면 10%만 삭제)
        lr_decay_iter = args.lr_decay_iter # 가지치기 과정중 학습률을 감소시키는 시점점
        prune_iters = math.ceil(math.log((1 - prune_rate) / pct_remain_after_this_iter, keep_per_iter)) if prune_iters is None else prune_iters
        lr_decay_iter = int(prune_iters * 0.7) if lr_decay_iter is None else lr_decay_iter

    # -----------------------
    # (5) 가지치기 전 학습 or 초기 성능 평가
    # -----------------------

    
    if init_train and prune_type in ['WS', 'Full', 'FairGRAPE', 'Lottery', 'Importance'] or print_acc:
        print("Pruning 전 모델을 학습합니다!" if prune_type != 'Full' else "Pruning 없음, 전체 모델 학습!")
        best_model, best_optimizer = train(best_model, criterion, dataloaders, lr_schedule, epoches, col_used_training, output_cols_each_task)
        
        

         # 가지치기 전 모델 성능을 CSV에 저장(false로 인해서 저장 안함)
        full_fair_df = save_output(best_model, [dataset, prune_type, loss_type, prune_rate, frames['test'], face_dir, total_classes, network, col_used, output_cols_each_task, sensitive_group, exp_idx], csv_savedir, False)

        # 가지치기 전 모델도 저장(원한다면) --> 여기 추가
        save_unpruned_model(
            model=best_model,
            optimizer=best_optimizer,
            cfgs=(dataset, loss_type, sensitive_group, network, exp_idx)
        )
        # import pdb; pdb.set_trace()  # 여기서 코드 실행이 중단됨



    # 체크포인트 로드
    """
    if checkpoint is not None:
        print("체크포인트 로드:", checkpoint)
        best_model.load_state_dict(torch.load(checkpoint))
    best_model = best_model.to(device)
    """
    # 체크포인트 로드
    if checkpoint is not None:
        print("체크포인트 로드:", checkpoint)
        checkpoint_data = torch.load(checkpoint)
        
        # 새로운 저장 형식 처리
        if isinstance(checkpoint_data, dict) and 'model_state' in checkpoint_data:
            best_model.load_state_dict(checkpoint_data['model_state'])
            
            # 옵티마이저 상태 복원
            if checkpoint_data.get('optimizer_state') is not None and best_optimizer is not None:
                best_optimizer.load_state_dict(checkpoint_data['optimizer_state'])
                print("옵티마이저 상태가 복원되었습니다.")
            
            # 스케줄러 상태 복원 (스케줄러가 존재하는 경우)
            if checkpoint_data.get('scheduler_state') is not None and 'scheduler' in locals():
                scheduler.load_state_dict(checkpoint_data['scheduler_state'])
                print("스케줄러 상태가 복원되었습니다.")
        else:
            # 기존 체크포인트 형식과의 호환성 유지
            best_model.load_state_dict(checkpoint_data)
            print("기존 형식의 체크포인트에서 모델 상태만 복원되었습니다.")

    best_model = best_model.to(device)
    
    # 임시로 제한
    #save_output(best_model, [dataset, prune_type, loss_type, prune_rate, frames['test'], face_dir, total_classes, network, col_used, output_cols_each_task, sensitive_group, exp_idx], csv_savedir, False)

    # Pruning 관련 설정
    if prune_type in pruner_map:
        print("Pruning 반복 횟수:", prune_iters)
        prune_loader = train_loader if not drop_race else train_loader_minority
        
        # prunner 생성
        prunner = pruner_map[prune_type](best_model, criterion, prune_loader, 
                                        output_cols_each_task, save_mask)
        
        # 가지치기(마스크) 초기화
        prunner.init_mask()
        
        # 필요하면 prunner에서 모델 가져오기
        best_model = prunner.get_model()  

        # update_model 등 추가 작업이 필요하다면
        prunner.update_model(best_model)




    # 전역변수 가져오기
    import config
    from pprint import pformat



    # -----------------------
    # (6) 메인 가지치기 반복
    # -----------------------

    for i in range(prune_iters):

        print('현재 시각:', datetime.datetime.now())
        pct_remain_after_this_iter *= keep_per_iter
        accumulated_pruned = min(1 - pct_remain_after_this_iter, prune_rate)
        print("\nPrune iter:{}, 이번 반복 후 남는 비율:{}".format(i, 1 - accumulated_pruned))

        config.glo_prune_iter = i # 프룬 횟수 
        

        prune_cfgs[0] = accumulated_pruned
        best_model = prunner.prune(prune_cfgs, True)

        # 🔥 retrain_iters == 0일 때도 평가 및 저장

    
        if retrain_iters == 0:
            print("\n" + "="*80)
            print(f"🔍 재학습 없음 - Prune Iteration {i+1} 모델 평가 및 저장")
            print("="*80)
            
            # 1. 모델 평가
            best_model.eval()
            device = torch.device('cuda:0')
            
            # 성능 지표 계산을 위한 변수 초기화
            running_loss = 0.0
            running_corrects = [0] * len(col_used_training)
            
            # Balanced Accuracy용
            balanced_acc_stats = [{(g, c): [0, 0] for g in [0,1] for c in [0,1]} 
                                for _ in range(len(col_used_training))]
            # Equalized Odds용
            eqodds_stats = [{g: {'TP':0, 'P':0, 'FP':0, 'N':0} for g in [0,1]} 
                        for _ in range(len(col_used_training))]
            
            with torch.no_grad():
                for sample_batched in dataloaders['test']:
                    image_batched, label_batched = sample_batched
                    image_batched = image_batched.to(device, dtype=torch.float)
                    label_batched = label_batched.to(device)
                    task_labels = label_batched[:, 0:len(col_used_training)]
                    gender_batched = label_batched[:, -1]
                    
                    outputs = torch.squeeze(best_model(image_batched))
                    loss, acc = loss_multi_tasks(outputs, label_batched, criterion, 
                                            output_cols_each_task, True)
                    
                    running_corrects = [running_corrects[j] + acc[j] for j in range(len(col_used_training))]
                    running_loss += loss.item() * image_batched.size(0)
                    
                    # Balanced Accuracy와 EO 통계 누적 (train_and_val.py와 동일)
                    for task_idx, (st, ed) in enumerate(output_cols_each_task):
                        task_out = outputs[:, st:ed]
                        task_label = task_labels[:, task_idx]
                        _, task_pred = torch.max(task_out, 1)
                        
                        for g in [0,1]:
                            mask_pos = (gender_batched == g) & (task_label == 1)
                            P = mask_pos.sum().item()
                            TP = ((task_pred == 1) & mask_pos).sum().item()
                            eqodds_stats[task_idx][g]['TP'] += TP
                            eqodds_stats[task_idx][g]['P'] += P
                            
                            mask_neg = (gender_batched == g) & (task_label == 0)
                            N = mask_neg.sum().item()
                            FP = ((task_pred == 1) & mask_neg).sum().item()
                            eqodds_stats[task_idx][g]['FP'] += FP
                            eqodds_stats[task_idx][g]['N'] += N
                            
                            for c in [0,1]:
                                mask = (gender_batched == g) & (task_label == c)
                                total = mask.sum().item()
                                if total > 0:
                                    correct = ((task_pred == task_label) & mask).sum().item()
                                    balanced_acc_stats[task_idx][(g,c)][0] += correct
                                    balanced_acc_stats[task_idx][(g,c)][1] += total
            
            # 성능 지표 계산
            epoch_loss = running_loss / len(dataloaders['test'].dataset)
            epoch_acc = sum(running_corrects).double() / (len(dataloaders['test'].dataset)*len(col_used_training))
            
            all_ba_means = []
            all_eos = []
            
            for task_idx in range(len(col_used_training)):
                ba_list = []
                for g in [0,1]:
                    for c in [0,1]:
                        correct, total = balanced_acc_stats[task_idx][(g,c)]
                        acc_val = correct/total if total>0 else 0
                        ba_list.append(acc_val)
                
                ba_mean = np.mean(ba_list)
                all_ba_means.append(ba_mean)
                
                # EO 계산
                tpr_vals, fpr_vals = [], []
                for g in [0,1]:
                    TP = eqodds_stats[task_idx][g]['TP']
                    P = eqodds_stats[task_idx][g]['P']
                    FP = eqodds_stats[task_idx][g]['FP']
                    N = eqodds_stats[task_idx][g]['N']
                    tpr = TP/P if P>0 else 0
                    fpr = FP/N if N>0 else 0
                    tpr_vals.append(tpr)
                    fpr_vals.append(fpr)
                
                tpr_gap = abs(tpr_vals[0] - tpr_vals[1])
                fpr_gap = abs(fpr_vals[0] - fpr_vals[1])
                eo = (tpr_gap + fpr_gap) / 2
                all_eos.append(eo)
            
            mean_ba = np.mean(all_ba_means)
            mean_eo = np.mean(all_eos)
            
            # 2. 결과 출력 및 저장
            print(f"\n📊 Prune Iteration {i+1} 성능:")
            print(f"  Accuracy: {epoch_acc:.4f}")
            print(f"  Loss:     {epoch_loss:.4f}")
            print(f"  BA:       {mean_ba:.4f}")
            print(f"  EO:       {mean_eo:.4f}")
            
            # 3. 결과 파일 저장
            save_dir = 'no_retrain_results'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"NO_RETRAIN_prune_{i+1}_acc_{epoch_acc:.4f}_eo_{mean_eo:.4f}_{timestamp}.txt"
            result_path = os.path.join(save_dir, result_filename)
            
            with open(result_path, "w", encoding="utf-8") as f:
                f.write("="*80 + "\n")
                f.write(f"🔍 재학습 없음 - Prune Iteration {i+1} 결과\n")
                f.write("="*80 + "\n\n")
                f.write(f"분석 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Prune Rate: {accumulated_pruned:.2%}\n")
                f.write(f"Importance/Gender Rate: {config.glo_imp_rate:.1f}:{1-config.glo_imp_rate:.1f}\n\n")
                f.write("-"*80 + "\n")
                f.write("📊 Performance Metrics\n")
                f.write("-"*80 + "\n")
                f.write(f"Accuracy:  {epoch_acc:.4f}\n")
                f.write(f"Loss:      {epoch_loss:.4f}\n")
                f.write(f"BA:        {mean_ba:.4f}\n")
                f.write(f"EO:        {mean_eo:.4f}\n")
                f.write("-"*80 + "\n\n")
                
                # 세부 결과도 저장
                f.write("📋 Detailed Results by Task\n")
                f.write("-"*80 + "\n")
                for task_idx in range(len(col_used_training)):
                    f.write(f"\nTask {task_idx} ({col_used_training[task_idx]}):\n")
                    f.write(f"  BA: {all_ba_means[task_idx]:.4f}\n")
                    f.write(f"  EO: {all_eos[task_idx]:.4f}\n")
            
            print(f"✅ [결과 저장] {result_path}")
            
            # 4. 모델 저장
            model_save_dir = 'save_models_no_retrain'
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir, exist_ok=True)
            
            model_filename = f"NO_RETRAIN_prune_{i+1}_acc_{epoch_acc:.4f}_eo_{mean_eo:.4f}_{timestamp}.pt"
            model_path = os.path.join(model_save_dir, model_filename)
            
            torch.save({
                'model_state_dict': best_model.state_dict(),
                'prune_iteration': i+1,
                'prune_rate': accumulated_pruned,
                'accuracy': epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc,
                'loss': epoch_loss,
                'balanced_accuracy': mean_ba,
                'equalized_odds': mean_eo,
            }, model_path)
            
            print(f"✅ [모델 저장] {model_path}")
            print("="*80 + "\n")

        
        classifier_params_before = {name: param.clone() 
                           for name, param in best_model.named_parameters() 
                           if 'classifier' in name}

        if retrain_iters > 0:
            best_model, _ = train(best_model, criterion, dataloaders, [retrain_lr], [retrain_iters], col_used_training, output_cols_each_task, cfgs=[dataset, prune_type, loss_type, prune_rate, frames['test'], face_dir, total_classes, network, col_used, output_cols_each_task, sensitive_group, exp_idx])
            #df = save_output(best_model, [dataset, prune_type, loss_type, prune_rate, frames['test'], face_dir, total_classes, network, col_used, output_cols_each_task, sensitive_group, exp_idx], csv_savedir, False) # EO 측정용
            #acc, group_acc = acc_scores(df, col_used, sensitive_group)
            #print("retrain_iters 진행중")
            classifier_params_after = {name: param 
                            for name, param in best_model.named_parameters() 
                            if 'classifier' in name}

            # 파라미터가 변하지 않았는지 확인
            print("\n" + "="*50)
            print("분류기 레이어 고정 여부 확인:")
            print("="*50)
            for name in classifier_params_before:
                if torch.equal(classifier_params_before[name], classifier_params_after[name]):
                    print(f"✅ {name}: 고정됨")
                else:
                    # 변경된 정도도 출력
                    diff = torch.abs(classifier_params_before[name] - classifier_params_after[name]).max().item()
                    print(f"❌ {name}: 변경됨 (최대 차이: {diff:.6f})")
            print("="*50 + "\n")
        
        print("\n⭐⭐⭐[가지치기 준비] 모든 레이어 활성화")
        for name, param in best_model.named_parameters():
            if param.dtype in [torch.float32, torch.float64, torch.float16]:
                param.requires_grad = True


        """
        epoch_log = {
                    'time': str(datetime.datetime.now()),
                    #'loss': config.glo_loss,
                    #'accuracy': config.glo_acc,
                    'Accuracy': acc,
                    'Accuracy(group)': group_acc,
                    'EO': config.glo_EO,
                    'Total_fairness': config.glo_fairness
                    #'optimizer': config.glo_optimizer                   
        }

        # print(f"epoch_log: {epoch_log}")
        
        save_epoch_logs([epoch_log], "epoch-by-epoch_results")
        """
        prunner.update_model(best_model)
    
        if i in save_model_iter:
            save_model(best_model, [prune_type, dataset, prune_type, loss_type, sensitive_group, network, accumulated_pruned, exp_idx])
    


    # -----------------------
    # (7) 가지치기 후 재학습
    # -----------------------

    
    if prune_iters > 0 and retrain:
        print("가지치기 후 재학습을 진행합니다.")
        #best_model, _ = train(best_model, criterion, dataloaders, lr_schedule, epoches, col_used_training, output_cols_each_task)
        best_model, _ = train(best_model, criterion, dataloaders, lr_schedule, epoches, col_used_training, output_cols_each_task, cfgs=[dataset, prune_type, loss_type, prune_rate, frames['test'], face_dir, total_classes, network, col_used, output_cols_each_task, sensitive_group, exp_idx])

    
    
    # 최종 모델 저장
    save_model(best_model, [prune_type, dataset, prune_type, loss_type, sensitive_group, network, prune_rate, exp_idx])

    # 결과(추론) 저장
    fair_df = save_output(best_model, [dataset, prune_type, loss_type, prune_rate, frames['test'], face_dir, total_classes, network, col_used, output_cols_each_task, sensitive_group, exp_idx], csv_savedir)

    # 최종 정확도 및 그룹별 지표 출력
    if print_acc:
        show_acc_df(fair_df, full_fair_df, col_used, sensitive_group)

    """
    # 정확도 저장 및 EO 측정
    df = save_output(best_model, [dataset, prune_type, loss_type, prune_rate, frames['test'], face_dir, total_classes, network, col_used, output_cols_each_task, sensitive_group, exp_idx], csv_savedir, False) # EO 측정용
    acc, group_acc = acc_scores(df, col_used, sensitive_group)
    epoch_log = {
                    'time': str(datetime.datetime.now()),
                    #'loss': config.glo_loss,
                    #'accuracy': config.glo_acc,
                    'Accuracy': acc,
                    'Accuracy(group)': group_acc,
                    'EO': config.glo_EO,
                    'Total_fairness': config.glo_fairness
                    #'optimizer': config.glo_optimizer                   
        }

    print(f"epoch_log: {epoch_log}")

    save_epoch_logs([epoch_log], "epoch-by-epoch_results")
    """
    # (옵션) 최종 모델 중요도 점수 계산
    if save_impt:
        sensitive_classes = len(set(frames['train'][sensitive_group])) if args.impt != 2 else total_classes
        cfgs = [best_model, frames['val'], face_dir, True, output_cols_each_task, col_used, 1]
        save_impt_df(cfgs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parameters for pruning experiments')

    parser.add_argument('--checkpoint', type=str, default=None, help='학습된 모델 파일(.pth)의 경로')  # 이미 학습된 모델이 있다면 해당 경로를 지정
    parser.add_argument('--dataset', type=str, default='FairFace', help='학습 및 평가에 사용할 데이터셋 이름')
    parser.add_argument('--network', type=str, default='resnet34', help='모델 아키텍처(네트워크) 명')
    parser.add_argument('--prune_type', type=str, default='FairGRAPE', help='가지치기(Pruning) 기법의 유형')
    parser.add_argument('--loss_type', type=str, default='gender', help='분류할 목표(타겟) 태스크')
    parser.add_argument('--sensitive_group', type=str, default='gender', help='Gradient를 컨트롤할 민감 정보 그룹')
    parser.add_argument('--init_pruned', type=float, default=0, help='이미 가지치기가 되어 있는 비율(%)')
    parser.add_argument('--prune_rate', type=float, default=0.9, help='최종적으로 목표하는 가지치기 비율')
    parser.add_argument('--prune_iter', type=int, default=None, help='반복적 가지치기를 수행할 횟수')
    parser.add_argument('--retrain_iter', type=int, default=3, help='각 가지치기 단계 후 재학습을 진행할 횟수')
    parser.add_argument('--retrain_lr', type=float, default=1e-5, help='재학습 시 사용할 학습률(learning rate)')
    parser.add_argument('--keep_per_iter', type=float, default=0.9, help='반복적 가지치기에서 단계별로 남길(유지할) 파라미터 비율')
    parser.add_argument('--lr_decay_iter', type=int, default=15, help='학습률을 감소시킬 주기(반복 횟수 또는 에폭)')
    parser.add_argument('--batch', type=int, default=64, help='데이터 로더에서 사용할 미니배치 크기')
    parser.add_argument('--impt', type=int, default=0, help='가지치기 중요도(importance) 점수를 계산하는 방식 (0, 1, 2 등)')
    parser.add_argument('--save_impt', action='store_true', help='출력 모델의 중요도 점수를 파일로 저장할지 여부')
    parser.add_argument('--para_batch', type=int, default=1, help='FairGRAPE 방식에서 race group 업데이트 전 선택할 파라미터 묶음의 크기')
    parser.add_argument('--stop_batch', type=int, default=10000, help='중요도 계산에 사용할 미니배치의 최대 개수')
    parser.add_argument('--exp_idx', type=int, default=0, help='여러 실험을 구분하기 위한 인덱스')
    parser.add_argument('--no_init_train', action='store_true', help='초기 학습 과정을 생략할지 여부')
    parser.add_argument('--drop_race', type=int, default=0, help='특정 인종(race) 데이터를 드롭할지 여부 (1이면 드롭, 0이면 미사용)')
    parser.add_argument('--no_retrain', action='store_true', help='가지치기 후 재학습 과정을 생략할지 여부')
    parser.add_argument('--save_mask', action='store_true', help='가지치기 마스크를 .npy 파일 형태로 저장할지 여부')
    parser.add_argument('--print_acc', action='store_true', help='가지치기 및 파인튜닝 후 테스트 정확도를 표시할지 여부')
    parser.add_argument('--delta_p', type=int, default=0, help='FairGRAPE 방식에서 중요도 계산 시 고려할 요소 (0: i, 1: p, 2: i*p)')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드 설정 (재현성 확보를 위함)')
    parser.add_argument('--save_model_iter', nargs='+', type=int, default=-1, help='특정 가지치기 iteration(또는 epoch)에 모델을 저장할지 여부')



    args = parser.parse_args()

    experiment(args)
