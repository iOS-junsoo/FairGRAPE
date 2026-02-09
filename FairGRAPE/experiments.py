import math
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import copy
import types
import pandas as pd
from collections import defaultdict
import os
import torch.optim as optim
from joblib import Parallel, delayed


#! 중요도는 높고 EO는 낮은 가중치를 선택하여 마스크 리스트 생성
def select_high_imp_low_eo_weights(model, importance_score, eo_score, prune_ratio):

    mask_list = []
    
    for name, layer in model.named_modules():
        if type(layer).__name__ not in supported_layers:
            continue
            
        # 현재 레이어의 importance와 EO score
        imp = importance_score[name]
        eo = eo_score[name]
        
        # 점수 정규화
        imp_norm = imp / (imp.abs().max() + 1e-12)
        eo_norm = eo / (eo.abs().max() + 1e-12)
        
        # 마스크 생성 (모두 0으로 초기화)
        mask = torch.zeros_like(layer.weight, dtype=torch.uint8, device=layer.weight.device)  # device 지정
        
        # 남길 가중치 수 계산
        total_params = mask.numel()
        num_to_keep = int(total_params * (1 - prune_ratio))
        
        # importance는 높고 EO는 낮은 가중치 선택
        criteria = imp_norm.to(device) - eo_norm.to(device)  # 텐서를 GPU로 이동
        
        # 상위 k개 선택
        flat_scores = criteria.flatten()
        threshold = torch.topk(flat_scores, num_to_keep, sorted=True)[0][-1]
        
        # 마스크 생성
        mask = (criteria >= threshold).to(torch.uint8).to(device)  # 마스크를 GPU로 이동
        mask_list.append(mask)
    
    return mask_list




#! 실험1: 편향되었지만 중요도가 높은 노드가 있는지 확인

import matplotlib.pyplot as plt
import config

def analyze_eo_importance_distribution(importance_score, eo_score):
    """
    EO와 importance score의 분포를 분석하고 시각화하는 함수
    """
    # 1. 모든 레이어의 가중치를 하나의 배열로 합치기
    all_eo = []
    all_imp = []
    
    for name in eo_score.keys():
        eo = eo_score[name].flatten().cpu().numpy()
        imp = importance_score[name].flatten().cpu().numpy()
        
        all_eo.extend(eo)
        all_imp.extend(imp)
    
    all_eo = np.array(all_eo)
    all_imp = np.array(all_imp)
    
    # 2. 실제 EO 점수 범위 계산
    min_eo = all_eo.min()
    max_eo = all_eo.max()
    print(f"\nEO 점수 범위: {min_eo:.2e} ~ {max_eo:.2e}")
    
    # 점수를 0-100 범위로 정규화
    normalized_eo = (all_eo - min_eo) * (100 / (max_eo - min_eo))
    
    # 각 구간의 실제 EO 점수 범위 계산
    score_intervals = []
    for i in range(20):
        start_normalized = i * 5
        end_normalized = (i + 1) * 5
        start_score = min_eo + (start_normalized / 100) * (max_eo - min_eo)
        end_score = min_eo + (end_normalized / 100) * (max_eo - min_eo)
        score_intervals.append(f"{start_score:.2e}~{end_score:.2e}")
    
    # 3. 구간별 importance score 평균 계산
    bin_imp_means = []
    bin_counts = []
    
    for i in range(20):
        mask = (normalized_eo >= i*5) & (normalized_eo < (i+1)*5)
        bin_imp_means.append(np.mean(all_imp[mask]) if np.sum(mask) > 0 else 0)
        bin_counts.append(np.sum(mask))
    
    # 4. 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(range(20), bin_imp_means, alpha=0.7)
    plt.title('Average Importance Score per EO Score Range')
    plt.xlabel('EO Score Range')
    plt.ylabel('Average Importance Score')
    
    # x축 레이블에 백분율과 실제 점수 범위 함께 표시
    plt.xticks(range(20), 
               [f'{i*5}~{(i+1)*5-1}%\n({score_intervals[i]})' for i in range(20)], 
               rotation=45, ha='right')
    
    # 실험 결과 저장을 위한 디렉토리 생성
    save_dir = 'experiment1'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'편향되었지만 중요한 노드_prune_{config.glo_prune_iter+1}.png'))
    plt.close()
    
    # 5. 결과 출력
    print("\nEO 구간별 분석:")
    for i in range(20):
        print(f"EO 점수 구간 {i*5}~{(i+1)*5-1}% ({score_intervals[i]}):")
        print(f"  가중치 개수: {bin_counts[i]}")
        print(f"  평균 importance score: {bin_imp_means[i]:.2e}")





#! 실험2: 프루닝된 가중치들의 importance score 분포 분석
def analyze_pruned_weights_distribution(mask_list, importance_score):
    """
    프루닝된 가중치들의 importance score를 실제 점수 범위로 분석
    비율(%) 기준으로 시각화
    """
    pruned_importance = []
    
    # 1. 프루닝된 가중치들의 importance score 수집
    for (name, imp), mask in zip(importance_score.items(), mask_list):
        imp = imp.flatten().cpu().numpy()
        mask = mask.flatten().cpu().numpy()
        pruned_importance.extend(imp[mask == 0])
    
    pruned_importance = np.array(pruned_importance)
    total_pruned = len(pruned_importance)  # 전체 프루닝된 가중치 수
    
    # 2. 실제 점수 범위 계산
    min_score = pruned_importance.min()
    max_score = pruned_importance.max()
    print(f"\n중요도 점수 범위: {min_score:.2e} ~ {max_score:.2e}")
    
    # 3. 실제 점수 범위를 30개 구간으로 나누기
    score_intervals = np.linspace(min_score, max_score, 31)  # 30개 구간을 위해 31개 경계점
    bin_percentages = []  # 비율을 저장할 리스트
    
    # 4. 각 구간별 가중치 비율(%) 계산
    for i in range(30):
        mask = (pruned_importance >= score_intervals[i]) & (pruned_importance < score_intervals[i+1])
        percentage = (np.sum(mask) / total_pruned) * 100  # 비율을 퍼센트로 변환
        bin_percentages.append(percentage)
        
    # 5. 시각화
    plt.figure(figsize=(15, 6))
    bars = plt.bar(range(30), bin_percentages, alpha=0.7)
    plt.title('Distribution of Pruned Weights\' Importance Scores')
    plt.xlabel('Importance Score Range')
    plt.ylabel('Percentage of Pruned Weights (%)')
    
    # 각 막대 위에 퍼센트 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{bin_percentages[i]:.1f}%',
                ha='center', va='bottom')
    
    # x축 레이블에 과학적 표기법 사용
    plt.xticks(range(30), 
               [f'{score_intervals[i]:.2e}~\n{score_intervals[i+1]:.2e}' for i in range(30)], 
               rotation=45)
    
    # 실험 결과 저장을 위한 디렉토리 생성
    save_dir = 'experiment2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'프루닝된 가중치 중요도 분석_prune_{config.glo_prune_iter+1}.png'))
    plt.close()
    
    # 6. 결과 출력
    print("\n프루닝된 가중치의 importance score 분포:")
    for i in range(30):
        score_range = f"{score_intervals[i]:.2e}~{score_intervals[i+1]:.2e}"
        print(f"점수 구간 {score_range}: {bin_percentages[i]:.2f}%")
        
    # 7. 추가 통계
    print(f"\n총 프루닝된 가중치 수: {total_pruned}")
    print(f"평균 점수: {np.mean(pruned_importance):.2e}")
    print(f"중앙값 점수: {np.median(pruned_importance):.2e}")







#! 실험3: 프루닝된 가중치와 유지된 가중치의 점수 분석
def analyze_weights_scores(mask_list, importance_score, eo_score, alpha):
    """
    프루닝된 가중치와 프루닝되지 않은 가중치의 평균 점수를 분석하고 결과를 저장
    """
    pruned_importance = []
    pruned_eo = []
    kept_importance = []
    kept_eo = []
    
    # 1. 가중치 점수 수집
    for (name, imp), (_, eo), mask in zip(importance_score.items(), eo_score.items(), mask_list):
        imp = imp.flatten().cpu().numpy()
        eo = eo.flatten().cpu().numpy()
        mask = mask.flatten().cpu().numpy()
        
        # 프루닝된 가중치 (mask == 0)
        pruned_importance.extend(imp[mask == 0])
        pruned_eo.extend(eo[mask == 0])
        
        # 유지된 가중치 (mask == 1)
        kept_importance.extend(imp[mask == 1])
        kept_eo.extend(eo[mask == 1])
    
    # 2. 배열로 변환
    pruned_importance = np.array(pruned_importance)
    pruned_eo = np.array(pruned_eo)
    kept_importance = np.array(kept_importance)
    kept_eo = np.array(kept_eo)
    
    # 3. 통계 계산
    stats = {
        "프루닝된 가중치": {
            "개수": len(pruned_importance),
            "평균 importance": f"{np.mean(pruned_importance):.2e}",
            "평균 EO": f"{np.mean(pruned_eo):.2e}",
            "중앙값 importance": f"{np.median(pruned_importance):.2e}",
            "중앙값 EO": f"{np.median(pruned_eo):.2e}"
        },
        "유지된 가중치": {
            "개수": len(kept_importance),
            "평균 importance": f"{np.mean(kept_importance):.2e}",
            "평균 EO": f"{np.mean(kept_eo):.2e}",
            "중앙값 importance": f"{np.median(kept_importance):.2e}",
            "중앙값 EO": f"{np.median(kept_eo):.2e}"
        }
    }
    
    # 4. 결과를 텍스트 파일로 저장
    save_dir = 'experiment3'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with open(os.path.join(save_dir, f'가중치_분석_prune_{config.glo_prune_iter+1}_비율_{alpha}:{1-alpha}.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 프루닝된 가중치와 유지된 가중치의 점수 분석 ===\n\n")
        
        for weight_type, metrics in stats.items():
            f.write(f"# {weight_type}\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value}\n")
            f.write("\n")
            
        # 점수 차이 계산 및 저장
        imp_diff = float(stats["유지된 가중치"]["평균 importance"]) - float(stats["프루닝된 가중치"]["평균 importance"])
        eo_diff = float(stats["유지된 가중치"]["평균 EO"]) - float(stats["프루닝된 가중치"]["평균 EO"])
        
        f.write("# 점수 차이 (유지된 가중치 - 프루닝된 가중치)\n")
        f.write(f"importance 점수 차이: {imp_diff:.2e}\n")
        f.write(f"EO 점수 차이: {eo_diff:.2e}\n")
    
    return stats





#! 실험4: 첫번쨰 레이어의 가중치 하나씩 비활성화하며 실제 EO 계산
def compute_actual_eo_scores(model, test_loader, sensitive_idx, output_cols_each_task):
    """첫 번째 레이어의 각 가중치에 대한 EO 점수를 계산"""
    import time
    start_time = time.time()
    
    # 첫 번째 레이어 찾기
    first_layer = None
    first_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            first_layer = layer
            first_layer_name = name
            break
    
    if first_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None

    # 원본 가중치 저장
    original_weight = first_layer.weight.data.clone()
    weight_shape = first_layer.weight.shape
    total_weights = first_layer.weight.numel()

    # 원본 모델의 EO 점수 계산
    print("\n원본 모델의 EO 점수 계산 중...")
    original_eo = compute_batch_eo(model, test_loader, sensitive_idx, output_cols_each_task, device, stop_batch=10)
    print(f"원본 EO 점수: {original_eo:.4f}")

    # EO 점수 저장용 텐서
    eo_scores = torch.zeros_like(first_layer.weight)
    
    # 진행 상황 추적
    processed = 0
    skipped = 0
    layer_start_time = time.time()

    print(f"\n각 가중치별 EO 점수 계산 중...")
    print(f"총 가중치 수: {total_weights}")

    for idx in np.ndindex(weight_shape):
        # 진행률과 예상 시간 출력
        processed += 1
        if processed % 100 == 0:  # 100개마다 진행상황 출력
            elapsed = time.time() - layer_start_time
            weights_per_sec = processed / elapsed
            remaining_weights = total_weights - processed
            eta = remaining_weights / weights_per_sec
            
            hours = int(eta // 3600)
            minutes = int((eta % 3600) // 60)
            seconds = int(eta % 60)
            
            print(f"\r진행률: {processed}/{total_weights} "
                  f"({processed/total_weights*100:.2f}%) "
                  f"- 예상 남은 시간: {hours}시간 {minutes}분 {seconds}초", end="")

        # 특정 가중치만 0으로 설정
        temp_weight = first_layer.weight.data.clone()
        temp_weight[idx] = 0
        first_layer.weight.data = temp_weight
        
        # 변경된 모델의 EO 점수 계산
        weight_eo = compute_batch_eo(model, test_loader, sensitive_idx, output_cols_each_task, device, stop_batch=10)
        
        # EO 변화량을 점수로 사용
        eo_change = abs(weight_eo - original_eo)
        eo_scores[idx] = eo_change
        
        # 원래 가중치 복구
        first_layer.weight.data = original_weight.clone()

    # 결과 저장
    save_dir = 'experiment4'
    os.makedirs(save_dir, exist_ok=True)

    # 1. EO 점수 NPY 파일로 저장
    np.save(os.path.join(save_dir, f'실제_eo_scores_{first_layer_name}.npy'), 
            eo_scores.cpu().numpy())

    # 2. EO 분포 히스토그램
    # EO 점수 분포도 시각화 코드 추가
    plt.figure(figsize=(15, 5))

    # 첫 번째 서브플롯: EO 점수 히스토그램
    plt.subplot(1, 2, 1)
    plt.hist(eo_scores.cpu().numpy().flatten(), bins=50)
    plt.title('Distribution of EO Scores')
    plt.xlabel('EO Score')
    plt.ylabel('Count')
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)

    # 두 번째 서브플롯: EO 점수 히트맵
    plt.subplot(1, 2, 2)
    # 가중치 shape에 따라 적절히 reshape
    if len(weight_shape) == 2:  # Linear 레이어인 경우
        reshaped_scores = eo_scores.cpu().numpy().reshape(weight_shape)
    else:  # Conv 레이어인 경우
        # 첫 두 차원(out_channels, in_channels)을 유지하고 나머지는 평균
        flattened = eo_scores.cpu().numpy()
        reshaped_scores = flattened.reshape(weight_shape[0], -1).mean(axis=1)
        reshaped_scores = reshaped_scores.reshape(-1, 1)  # 히트맵을 위해 2D로 만듦

    plt.imshow(reshaped_scores, cmap='coolwarm')
    plt.colorbar(label='EO Score')
    plt.title('EO Scores Heatmap')
    plt.xlabel('Input Channel')
    plt.ylabel('Output Channel')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'실제_eo_score_distribution_{first_layer_name}.png'))
    plt.close()



    # 4. 분석 보고서 저장
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    with open(os.path.join(save_dir, '실제_eo_analysis_report.txt'), 'w') as f:
        f.write("=== EO 점수 분석 보고서 ===\n\n")
        f.write(f"처리 시간: {hours}시간 {minutes}분 {seconds}초\n")
        f.write(f"총 가중치 수: {total_weights}\n")
        f.write(f"원본 EO 점수: {original_eo:.6f}\n\n")
        f.write("EO 변화량 통계:\n")
        f.write(f"최소값: {eo_scores.min().item():.6f}\n")
        f.write(f"최대값: {eo_scores.max().item():.6f}\n")
        f.write(f"평균: {eo_scores.mean().item():.6f}\n")
        f.write(f"표준편차: {eo_scores.std().item():.6f}\n")

    print(f"\n\n분석 완료!")
    print(f"소요 시간: {hours}시간 {minutes}분 {seconds}초")
    print(f"결과가 '{save_dir}' 디렉토리에 저장되었습니다.")

    return eo_scores

def compute_batch_eo(model, test_loader, sensitive_idx, output_cols_each_task, device, stop_batch):
    """모든 태스크의 출력을 처리하여 평균 EO 점수를 계산합니다."""
    
    n_tasks = len(output_cols_each_task)
    task_metrics = {i: {'tp_0': 0, 'tp_1': 0, 'fp_0': 0, 'fp_1': 0,
                       'total_p_0': 0, 'total_p_1': 0, 'total_n_0': 0, 'total_n_1': 0}
                   for i in range(n_tasks)}

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            if batch_idx >= stop_batch:
                break
            
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            g = labels[:, sensitive_idx].float()

            # 각 태스크별로 처리
            for task_idx, (start, end) in enumerate(output_cols_each_task):
                # 현재 태스크의 로짓 추출
                task_logits = outputs[:, start:end]
                
                # Softmax를 적용하여 확률 계산
                probs = torch.softmax(task_logits, dim=1)
                
                # 현재 태스크의 정답 레이블과 예측값
                y_true = labels[:, task_idx]
                y_pred = (probs[:, 1] > 0.5).float()
                
                # 각 그룹별로 처리
                for g_val in [0, 1]:
                    mask = (g == g_val) & (y_true != -1)
                    if not mask.any():
                        continue
                    
                    y_true_group = y_true[mask]
                    y_pred_group = y_pred[mask]
                    
                    # 메트릭 계산
                    tp = ((y_pred_group == 1) & (y_true_group == 1)).sum().item()
                    fp = ((y_pred_group == 1) & (y_true_group == 0)).sum().item()
                    p = (y_true_group == 1).sum().item()
                    n = (y_true_group == 0).sum().item()
                    
                    # 그룹별 메트릭 누적
                    metrics = task_metrics[task_idx]
                    if g_val == 0:
                        metrics['tp_0'] += tp
                        metrics['fp_0'] += fp
                        metrics['total_p_0'] += p
                        metrics['total_n_0'] += n
                    else:
                        metrics['tp_1'] += tp
                        metrics['fp_1'] += fp
                        metrics['total_p_1'] += p
                        metrics['total_n_1'] += n
    
    # 각 태스크별 EO 점수 계산
    task_eo_scores = []
    for task_idx in range(n_tasks):
        metrics = task_metrics[task_idx]
        
        # TPR과 FPR 계산
        tpr_0 = metrics['tp_0'] / metrics['total_p_0'] if metrics['total_p_0'] > 0 else 0
        tpr_1 = metrics['tp_1'] / metrics['total_p_1'] if metrics['total_p_1'] > 0 else 0
        fpr_0 = metrics['fp_0'] / metrics['total_n_0'] if metrics['total_n_0'] > 0 else 0
        fpr_1 = metrics['fp_1'] / metrics['total_n_1'] if metrics['total_n_1'] > 0 else 0
        
        # 태스크별 EO 점수 계산
        task_eo = (abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1)) / 2.0
        task_eo_scores.append(task_eo)
        
        # 태스크별 결과 출력 (디버깅용)
        print(f"\nTask {task_idx} metrics:")
        print(f"TPR - Group 0: {tpr_0:.4f}, Group 1: {tpr_1:.4f}, |차이|: {abs(tpr_0 - tpr_1):.4f}")
        print(f"FPR - Group 0: {fpr_0:.4f}, Group 1: {fpr_1:.4f}, |차이|: {abs(fpr_0 - fpr_1):.4f}")
        print(f"Task {task_idx} EO Score: {task_eo:.4f}")
    
    # 전체 태스크의 평균 EO 점수 계산
    mean_eo_score = sum(task_eo_scores) / len(task_eo_scores)
    print(f"\n평균 EO 점수: {mean_eo_score:.4f}")
    
    return mean_eo_score



#! 실험4: 첫 레이어의 가중치에 대해 그래디언트 기반 EO 근사 계산
def compute_approximate_eo_scores(model, test_loader, sensitive_idx, output_cols_each_task):

    """
    첫 번째 레이어의 가중치에 대해 그래디언트 기반 EO 근사 계산
    """
    import time
    start_time = time.time()
    
    IGNORE_TASK_INDICES = {0, 4, 16, 21, 29, 35} # 무시할 작업 인덱스 집합
    stop_batch = 10
    eo_scores = {}
    layer_stats = {}
    
    # 첫 번째 레이어 찾기
    first_layer = None
    first_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            first_layer = layer
            first_layer_name = name
            break  # 첫 번째 레이어를 찾으면 즉시 중단
    
    if first_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None, None
    
    # 통계 정보 초기화
    layer_stats[first_layer_name] = {
        'shape': first_layer.weight.shape,
        'total_weights': first_layer.weight.numel(),
        'eo_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0,
            'std': 0
        }
    }
    
    # EO 그래디언트 누적을 위한 변수 초기화
    eo_grad_accum = torch.zeros_like(first_layer.weight).to(device)
    eo_score = torch.zeros_like(first_layer.weight).to(device)
    batches = 0
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"\n첫 번째 레이어 {first_layer_name} EO 근사 계산 중...")
    print(f"형태: {layer_stats[first_layer_name]['shape']}")
    print(f"가중치 수: {layer_stats[first_layer_name]['total_weights']}")
    
    # 배치별 처리
    for batch_idx, (data, labels) in enumerate(test_loader):
        if batch_idx >= stop_batch:
            break
            
        if batch_idx % 2 == 0:
            print(f"배치 {batch_idx}/{stop_batch} 처리 중...")
            
        data = data.to(device)
        labels = labels.to(device)
        batches += 1
        
        # EO Loss 계산
        optimizer.zero_grad()
        outputs = model(data)

        g = labels[:, sensitive_idx].float()

        eo_loss = eo_loss_all(outputs, labels, g, 
                              output_cols_each_task=output_cols_each_task,
                              ignore_tasks=IGNORE_TASK_INDICES)
        eo_loss.backward()
        
        # 그래디언트 누적
        grads = first_layer.weight.grad.clone()
        weights = first_layer.weight.data.clone()
        #hess = -(weights * grads)**2 * torch.sign(weights * grads)
        hess = grads
        
        eo_grad_accum += grads
        eo_score += hess.abs()
    
    # 배치 수로 평균
    eo_grad_accum /= batches
    eo_score /= batches
    
    # 결과 저장
    save_dir = 'experiment4'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # EO 점수를 CPU로 이동하고 NumPy 배열로 변환
    eo_array = eo_score.cpu().numpy()
    
    # 통계 계산
    layer_stats[first_layer_name]['eo_stats'].update({
        'min': float(np.min(eo_array)),
        'max': float(np.max(eo_array)),
        'mean': float(np.mean(eo_array)),
        'std': float(np.std(eo_array))
    })
    
    # NumPy 파일로 저장
    np.save(os.path.join(save_dir, f'approximated_eo_scores_{first_layer_name}.npy'), eo_array)
    
    # 분포 시각화
    plt.figure(figsize=(10, 5))
    plt.hist(eo_array.flatten(), bins=50)
    plt.title(f'Layer: {first_layer_name}\nApproximated EO Score Distribution')
    plt.xlabel('EO Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, f'approximated_eo_distribution_{first_layer_name}.png'))
    plt.close()
    
    # 통계 정보를 텍스트 파일로 저장
    with open(os.path.join(save_dir, 'approximated_eo_scores_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 첫 번째 레이어의 근사 EO 점수 분석 ===\n\n")
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        f.write(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초\n\n")
        
        f.write(f"\n레이어: {first_layer_name}\n")
        f.write(f"형태: {layer_stats[first_layer_name]['shape']}\n")
        f.write(f"가중치 수: {layer_stats[first_layer_name]['total_weights']}\n")
        f.write("EO 점수 통계:\n")
        f.write(f"  최소값: {layer_stats[first_layer_name]['eo_stats']['min']:.2e}\n")
        f.write(f"  최대값: {layer_stats[first_layer_name]['eo_stats']['max']:.2e}\n")
        f.write(f"  평균: {layer_stats[first_layer_name]['eo_stats']['mean']:.2e}\n")
        f.write(f"  표준편차: {layer_stats[first_layer_name]['eo_stats']['std']:.2e}\n")
    
    return eo_score, layer_stats


# 배치 데이터의 문제가 아닌지




def soft_tpr(p, y, g, g_val, tau=20., eps=1e-6):
    s = torch.sigmoid(tau * (p - 0.5))
    num = (s * y * (g == g_val)).sum()
    den = (y * (g == g_val)).sum() + eps
    return num / den

def soft_fpr(p, y, g, g_val, tau=20., eps=1e-6):
    s = torch.sigmoid(tau * (p - 0.5))
    num = (s * (1-y) * (g == g_val)).sum()
    den = ((1-y) * (g == g_val)).sum() + eps
    return num / den

def eo_loss_fuc(p, y, g, tau=20.):
    tpr_gap = torch.abs(soft_tpr(p, y, g, 0, tau) -
                        soft_tpr(p, y, g, 1, tau))
    fpr_gap = torch.abs(soft_fpr(p, y, g, 0, tau) -
                        soft_fpr(p, y, g, 1, tau))
    return 0.5 * (tpr_gap + fpr_gap)

def eo_loss_all(p_mat, y_mat, g_vec, tau=20., output_cols_each_task=[(0,7),(7,9),(9,18)], ignore_tasks=None):
    """
    p_mat: 모델의 출력값 모음
    y_mat: 각 태스크의 정답 레이블
    g_vec: 그룹 표시 벡터(0 또는 1)
    ignore_tasks: EO Loss 계산에서 제외할 태스크 인덱스의 set 또는 list
    """
    if ignore_tasks is None:
        ignore_tasks = set() # 기본값으로 빈 세트 사용

    eo_sum = 0.0
    tasks_outputs, tasks_labels, ntasks = [], [], len(output_cols_each_task)
    for i, st_ed in enumerate(output_cols_each_task):
        tasks_outputs.append(p_mat[:, st_ed[0]:st_ed[1]])
        tasks_labels.append(y_mat[:, i])

    eo_sum = []
    # enumerate를 통해 태스크 인덱스(i)를 함께 가져옵니다.
    for i, (p_logits, y_true) in enumerate(zip(tasks_outputs, tasks_labels)):
        # ==================== 수정된 부분 시작 ====================
        # 현재 태스크 인덱스가 ignore_tasks에 있으면 계산을 건너뜁니다.
        if i in ignore_tasks:
            continue
        # ==================== 수정된 부분 종료 ====================
            
        p_prob = torch.softmax(p_logits, dim=1)[:, 1]
        eo_sum.append(eo_loss_fuc(p_prob, y_true, g_vec, tau=tau))

    # 만약 유효한 태스크가 하나도 없어서 eo_sum이 비어있을 경우 에러가 나지 않도록 처리
    if not eo_sum:
        return torch.tensor(0.0, device=p_mat.device, requires_grad=True)

    return torch.stack(eo_sum).mean()










#! 레이어별 가중치 확인

supported_layers = ['Linear', 'Conv2d', 'Conv1d']

def print_model_weights(model):
    """
    모델의 레이어별 가중치 정보를 출력
    Args:
        model: PyTorch 모델
    """
    print("\n=== 모델 레이어별 가중치 정보 ===")
    
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            weights = layer.weight.data
            
            print(f"\n레이어: {name}")
            print(f"타입: {type(layer).__name__}")
            print(f"형태: {weights.shape}")
            print(f"가중치 수: {weights.numel()}")
            print(f"통계:")
            print(f"  최소값: {weights.min().item():.4f}")
            print(f"  최대값: {weights.max().item():.4f}")
            print(f"  평균: {weights.mean().item():.4f}")
            print(f"  표준편차: {weights.std().item():.4f}")
            
            # 마스크가 있는 경우 마스크 정보도 출력
            if hasattr(layer, 'mask'):
                mask = layer.mask
                active_weights = (mask > 0).sum().item()
                print(f"활성 가중치 비율: {active_weights/weights.numel()*100:.2f}%")




#! 실험5: 처음 레이어의 가중치에 대해 실제 중요도 계산
def compute_actual_importance_scores(model, test_loader, output_cols_each_task):
    """
    첫 번째 레이어의 가중치만 하나씩 비활성화하면서 실제 중요도를 계산하고 결과를 파일로 저장
    """
    import time
    start_time = time.time()
    
    stop_batch = 10
    importance_scores = {}
    layer_stats = {}
    criterion = nn.CrossEntropyLoss()
    
    # 첫 번째 레이어 찾기
    first_layer = None
    first_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            first_layer = layer
            first_layer_name = name
            # 첫 번째 레이어 정보 출력
            print("\n=== 첫 번째 레이어 정보 ===")
            print(f"레이어 이름: {first_layer_name}")
            print(f"레이어 타입: {type(first_layer).__name__}")
            print(f"형태: {first_layer.weight.shape}")
            print(f"가중치 수: {first_layer.weight.numel()}")
            break  # 첫 번째 레이어를 찾으면 중단
    
    if first_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None, None
    
    # 원본 가중치 저장
    original_weight = first_layer.weight.data.clone()
    
    # 통계 정보 초기화
    layer_stats[first_layer_name] = {
        'shape': first_layer.weight.shape,
        'total_weights': first_layer.weight.numel(),
        'importance_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0,
            'std': 0
        }
    }
    
    print(f"\n레이어 {first_layer_name}:")
    print(f"형태: {layer_stats[first_layer_name]['shape']}")
    print(f"가중치 수: {layer_stats[first_layer_name]['total_weights']}")
    
    # 각 가중치별 중요도 점수 계산
    importance_scores[first_layer_name] = torch.zeros_like(first_layer.weight)
    weight_shape = first_layer.weight.shape
    total_weights = first_layer.weight.numel()
    processed = 0
    layer_start_time = time.time()
    
    for idx in np.ndindex(weight_shape):
        # 진행률과 예상 시간 표시
        processed += 1
        if processed % 1000 == 0:
            elapsed_time = time.time() - layer_start_time
            weights_per_second = processed / elapsed_time
            remaining_weights = total_weights - processed
            estimated_remaining_time = remaining_weights / weights_per_second
            
            hours = int(estimated_remaining_time // 3600)
            minutes = int((estimated_remaining_time % 3600) // 60)
            seconds = int(estimated_remaining_time % 60)
            
            print(f"진행률: {processed}/{total_weights} ({processed/total_weights*100:.2f}%) "
                  f"- 예상 남은 시간: {hours}시간 {minutes}분 {seconds}초")
        
        # 1. 해당 가중치만 0으로 설정
        temp_weight = first_layer.weight.data.clone()
        temp_weight[idx] = 0
        first_layer.weight.data = temp_weight
        
        # 2. 손실 누적값 초기화
        total_loss = 0
        
        # 3. stop_batch 개수만큼의 배치에 대해 예측
        for batch_idx, (data, labels) in enumerate(test_loader):
            if batch_idx >= stop_batch:
                break
                
            data = data.to(device)
            labels = labels.to(device)
            
            # 모델 예측
            outputs = model(data)
            
            # 손실 계산
            loss = loss_multi_tasks(outputs, labels, criterion, output_cols_each_task)
            total_loss += loss.item()
        
        # 4. 평균 손실을 중요도 점수로 사용
        importance_scores[first_layer_name][idx] = total_loss / stop_batch
        
        # 5. 원래 가중치 복구
        first_layer.weight.data = original_weight.clone()
    
    # 결과 저장
    save_dir = 'experiment5'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 중요도 점수를 CPU로 이동하고 NumPy 배열로 변환
    importance_array = importance_scores[first_layer_name].cpu().numpy()
    
    # 통계 계산
    layer_stats[first_layer_name]['importance_stats'].update({
        'min': float(np.min(importance_array)),
        'max': float(np.max(importance_array)),
        'mean': float(np.mean(importance_array)),
        'std': float(np.std(importance_array))
    })
    
    # NumPy 파일로 저장
    np.save(os.path.join(save_dir, f'actual_importance_scores_{first_layer_name}.npy'), importance_array)
    
    # 분포 시각화
    plt.figure(figsize=(10, 5))
    plt.hist(importance_array.flatten(), bins=50)
    plt.title(f'Layer: {first_layer_name}\nActual Importance Score Distribution')
    plt.xlabel('Importance Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, f'importance_distribution_{first_layer_name}.png'))
    plt.close()
    
    # 통계 정보를 텍스트 파일로 저장
    with open(os.path.join(save_dir, 'actual_importance_scores_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 첫 번째 레이어의 실제 중요도 점수 분석 ===\n\n")
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        f.write(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초\n\n")
        
        f.write(f"\n레이어: {first_layer_name}\n")
        f.write(f"형태: {layer_stats[first_layer_name]['shape']}\n")
        f.write(f"가중치 수: {layer_stats[first_layer_name]['total_weights']}\n")
        f.write("중요도 점수 통계:\n")
        f.write(f"  최소값: {layer_stats[first_layer_name]['importance_stats']['min']:.2e}\n")
        f.write(f"  최대값: {layer_stats[first_layer_name]['importance_stats']['max']:.2e}\n")
        f.write(f"  평균: {layer_stats[first_layer_name]['importance_stats']['mean']:.2e}\n")
        f.write(f"  표준편차: {layer_stats[first_layer_name]['importance_stats']['std']:.2e}\n")
    
    return importance_scores, layer_stats


def compute_approximate_importance_scores(model, test_loader, output_cols_each_task):
    """
    첫 번째 레이어의 가중치에 대해 그래디언트 기반 중요도를 근사 계산
    """
    import time
    start_time = time.time()
    
    stop_batch = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 첫 번째 레이어 찾기
    first_layer = None
    first_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            first_layer = layer
            first_layer_name = name
            # 첫 번째 레이어 정보 출력
            print("\n=== 첫 번째 레이어 정보 ===")
            print(f"레이어 이름: {first_layer_name}")
            print(f"레이어 타입: {type(first_layer).__name__}")
            print(f"형태: {first_layer.weight.shape}")
            print(f"가중치 수: {first_layer.weight.numel()}")
            break
    
    if first_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None, None
    
    # 누적 변수 초기화
    imp_grad_accum = torch.zeros_like(first_layer.weight)
    importance_score = torch.zeros_like(first_layer.weight)
    batches = 0
    
    print(f"\n{stop_batch}개 배치에 대해 근사 중요도 계산 중...")
    
    for batch_idx, sample_batched in enumerate(test_loader):
        if batch_idx >= stop_batch:
            break
        
        batches += 1
        if batch_idx % 2 == 0:
            print(f"{batch_idx}/{stop_batch} 배치 처리 중...")
            
        # 배치 데이터 준비
        image_batched, label_batched = sample_batched
        image_batched = image_batched.to(device, dtype=torch.float)
        label_batched = label_batched.to(device)
        
        # 그래디언트 초기화
        optimizer.zero_grad()
        
        # 순전파 → 손실 계산 → 역전파
        outputs = model(image_batched)
        loss = loss_multi_tasks(outputs, label_batched, criterion, output_cols_each_task)
        loss.backward()
        
        # 그래디언트와 중요도 점수 누적
        grads = first_layer.weight.grad.clone().detach()
        weights = first_layer.weight.data.clone().detach()
        
        # 테일러 급수를 이용한 근사
        hess = (weights.abs() * grads.abs())**2
        
        imp_grad_accum += grads
        importance_score += hess
    
    # 배치 수로 평균
    imp_grad_accum /= batches
    importance_score /= batches
    
    # 결과 저장
    save_dir = 'experiment5'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 중요도 점수를 CPU로 이동하고 NumPy 배열로 변환
    importance_array = importance_score.cpu().numpy()
    
    # 통계 계산
    stats = {
        'min': float(np.min(importance_array)),
        'max': float(np.max(importance_array)),
        'mean': float(np.mean(importance_array)),
        'std': float(np.std(importance_array))
    }
    
    # NumPy 파일로 저장
    np.save(os.path.join(save_dir, f'approximated_importance_scores_{first_layer_name}.npy'), 
            importance_array)
    
    # 분포 시각화
    plt.figure(figsize=(10, 5))
    plt.hist(importance_array.flatten(), bins=50)
    plt.title(f'Layer: {first_layer_name}\nApproximated Importance Score Distribution')
    plt.xlabel('Importance Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, f'approximated_importance_distribution_{first_layer_name}.png'))
    plt.close()
    
    # 통계 정보를 텍스트 파일로 저장
    with open(os.path.join(save_dir, 'approximated_importance_scores_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 첫 번째 레이어의 근사 중요도 점수 분석 ===\n\n")
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        f.write(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초\n\n")
        
        f.write(f"\n레이어: {first_layer_name}\n")
        f.write(f"형태: {first_layer.weight.shape}\n")
        f.write(f"가중치 수: {first_layer.weight.numel()}\n")
        f.write("중요도 점수 통계:\n")
        f.write(f"  최소값: {stats['min']:.2e}\n")
        f.write(f"  최대값: {stats['max']:.2e}\n")
        f.write(f"  평균: {stats['mean']:.2e}\n")
        f.write(f"  표준편차: {stats['std']:.2e}\n")
    
    return importance_score.cpu()




import time
def compute_monte_carlo_eo_scores_absolute(model, test_loader, sensitive_idx, stop_batch=10, n_samples=100):
    """
    몬테카를로 방식으로 마지막 레이어의 EO 점수를 절대적 변화량으로 근사 계산
    """
    start_time = time.time()
    
    # 마지막 레이어 찾기
    last_layer = None
    last_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            last_layer = layer
            last_layer_name = name
    
    if last_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None, None

    # 통계 정보 초기화
    layer_stats = {last_layer_name: {
        'shape': last_layer.weight.shape,
        'total_weights': last_layer.weight.numel(),
        'eo_stats': {'min': float('inf'), 'max': float('-inf'), 'mean': 0, 'std': 0}
    }}
    
    print(f"\n=== 마지막 레이어 정보 ===")
    print(f"레이어 이름: {last_layer_name}")
    print(f"레이어 타입: {type(last_layer).__name__}")
    print(f"형태: {last_layer.weight.shape}")
    print(f"가중치 수: {last_layer.weight.numel()}")

    # 원본 가중치 저장
    original_weight = last_layer.weight.data.clone()
    
    # 원본 모델의 EO 점수 계산 (변화 없는 기준값)
    print("\n원본 모델의 EO 점수 계산 중...")
    original_eo = compute_batch_eo(model, test_loader, sensitive_idx, stop_batch)
    print(f"원본 EO 점수: {original_eo:.4f}")
    
    # EO 점수 누적용 텐서
    eo_accum = torch.zeros_like(last_layer.weight).to(device)
    
    print(f"\n몬테카를로 샘플링으로 EO 변화량 계산 중 (n_samples={n_samples})...")
    
    # 몬테카를로 샘플링
    for sample_idx in range(n_samples):
        if sample_idx % 10 == 0:
            print(f"샘플 {sample_idx}/{n_samples} 처리 중...")
            
        # 전체 가중치 행렬에 대한 랜덤 마스크 생성
        # 0 또는 1로 구성된 랜덤 벡터를 생성 (가중치를 켜거나 끄는 역할)
        random_mask = torch.randint(0, 2, last_layer.weight.shape, device=device).float()
        
        # 가중치에 마스크를 적용하여 랜덤하게 제거
        last_layer.weight.data = original_weight * random_mask
        
        # 변화된 가중치로 EO 계산
        eo_changed = compute_batch_eo(model, test_loader, sensitive_idx, stop_batch)
        
        # EO 변화량 계산
        # abs(eo_changed - original_eo)를 가중치 마스크와 곱하여 중요도를 누적
        eo_accum += abs(eo_changed - original_eo) * random_mask
        
        # 원래 가중치 복구
        last_layer.weight.data = original_weight.clone()
    
    # 샘플 수로 평균하여 최종 EO 점수 계산
    # 마스크가 0인 경우 가중치 중요도가 0이므로, 전체 누적값을 샘플 수로 나눔
    eo_score = eo_accum / n_samples
    
    # 결과 저장 (이하 동일)
    save_dir = 'experiment4'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    eo_array = eo_score.cpu().numpy()
    
    # 통계 계산
    layer_stats[last_layer_name]['eo_stats'].update({
        'min': float(np.min(eo_array)),
        'max': float(np.max(eo_array)),
        'mean': float(np.mean(eo_array)),
        'std': float(np.std(eo_array))
    })
    
    # NumPy 파일로 저장
    np.save(os.path.join(save_dir, f'monte_carlo_eo_scores_{last_layer_name}.npy'), eo_array)
    
    # 분포 시각화
    plt.figure(figsize=(10, 5))
    plt.hist(eo_array.flatten(), bins=50)
    plt.title(f'Layer: {last_layer_name}\nMonte Carlo EO Score Distribution (Absolute)')
    plt.xlabel('EO Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, f'monte_carlo_eo_distribution_{last_layer_name}_absolute.png'))
    plt.close()
    
    # 통계 정보를 텍스트 파일로 저장
    with open(os.path.join(save_dir, 'monte_carlo_eo_scores_analysis.txt'), 'a', encoding='utf-8') as f:
        f.write("\n\n=== 마지막 레이어의 몬테카를로 EO 점수 분석 (절대값 변화) ===\n\n")
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        f.write(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초\n\n")
        
        f.write(f"\n레이어: {last_layer_name}\n")
        f.write(f"형태: {layer_stats[last_layer_name]['shape']}\n")
        f.write(f"가중치 수: {layer_stats[last_layer_name]['total_weights']}\n")
        f.write("EO 점수 통계:\n")
        f.write(f"  최소값: {layer_stats[last_layer_name]['eo_stats']['min']:.2e}\n")
        f.write(f"  최대값: {layer_stats[last_layer_name]['eo_stats']['max']:.2e}\n")
        f.write(f"  평균: {layer_stats[last_layer_name]['eo_stats']['mean']:.2e}\n")
        f.write(f"  표준편차: {layer_stats[last_layer_name]['eo_stats']['std']:.2e}\n")
        f.write(f"  몬테카를로 샘플 수: {n_samples}\n")
    
    return eo_score, layer_stats


def compute_batch_eo(model, test_loader, sensitive_idx, stop_batch):
    """전체 배치에 대한 EO 점수 계산 (이전 코드와 동일)"""
    tpr_0, tpr_1 = 0, 0
    fpr_0, fpr_1 = 0, 0
    total_p_0, total_p_1 = 0, 0
    total_n_0, total_n_1 = 0, 0
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            if batch_idx >= stop_batch:
                break
                
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            probs = torch.sigmoid(outputs)
            g = labels[:, sensitive_idx].float()
            
            for g_val in [0, 1]:
                mask = (g == g_val)
                if not mask.any():
                    continue
                    
                y_true = labels[mask, 0]
                y_pred = (probs[mask, 0] > 0.5).float()
                
                tp = ((y_pred == 1) & (y_true == 1)).sum().item()
                fp = ((y_pred == 1) & (y_true == 0)).sum().item()
                
                p = (y_true == 1).sum().item()
                n = (y_true == 0).sum().item()
                
                if g_val == 0:
                    tpr_0 += tp
                    fpr_0 += fp
                    total_p_0 += p
                    total_n_0 += n
                else:
                    tpr_1 += tp
                    fpr_1 += fp
                    total_p_1 += p
                    total_n_1 += n
    
    tpr_0 = tpr_0 / total_p_0 if total_p_0 > 0 else 0
    tpr_1 = tpr_1 / total_p_1 if total_p_1 > 0 else 0
    fpr_0 = fpr_0 / total_n_0 if total_n_0 > 0 else 0
    fpr_1 = fpr_1 / total_n_1 if total_n_1 > 0 else 0
    
    eo_score = (abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1)) / 2.0
    
    return eo_score



def analyze_eo_correlation(model, test_loader, sensitive_idx, output_cols_each_task):
    """실제 EO와 근사 EO의 피어슨 상관관계 분석"""
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    
    def compute_real_eo(outputs, labels, g):
        """실제 EO 계산"""
        st, ed = output_cols_each_task[0]
        p = torch.softmax(outputs[:, st:ed], dim=1)[:, 1]
        y = labels[:, 0]

        # TPR과 FPR 계산 (epsilon 추가)
        eps = 1e-7
        tpr_0 = (((p[g == 0] > 0.5).float() * (y[g == 0] == 1).float()).sum() / 
                 ((y[g == 0] == 1).float().sum() + eps))
        tpr_1 = (((p[g == 1] > 0.5).float() * (y[g == 1] == 1).float()).sum() / 
                 ((y[g == 1] == 1).float().sum() + eps))
        
        fpr_0 = (((p[g == 0] > 0.5).float() * (y[g == 0] == 0).float()).sum() / 
                 ((y[g == 0] == 0).float().sum() + eps))
        fpr_1 = (((p[g == 1] > 0.5).float() * (y[g == 1] == 0).float()).sum() / 
                 ((y[g == 1] == 0).float().sum() + eps))

        print(f"TPR_0: {tpr_0:.4f}, TPR_1: {tpr_1:.4f}, |차이|: {abs(tpr_0 - tpr_1):.4f}")
        print(f"FPR_0: {fpr_0:.4f}, FPR_1: {fpr_1:.4f}, |차이|: {abs(fpr_0 - fpr_1):.4f}")

        return 0.5 * (abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1))

    def compute_soft_eo(outputs, labels, g, tau):
        """근사 EO 계산"""
        st, ed = output_cols_each_task[0]
        p = torch.softmax(outputs[:, st:ed], dim=1)[:, 1]
        y = labels[:, 0]
        
        # Soft TPR과 FPR 계산
        tpr_0 = soft_tpr(p, y, g, 0, tau)
        tpr_1 = soft_tpr(p, y, g, 1, tau)
        fpr_0 = soft_fpr(p, y, g, 0, tau)
        fpr_1 = soft_fpr(p, y, g, 1, tau)
        
        print(f"[tau={tau}] Soft TPR_0: {tpr_0:.4f}, Soft TPR_1: {tpr_1:.4f}, |차이|: {abs(tpr_0 - tpr_1):.4f}")
        print(f"[tau={tau}] Soft FPR_0: {fpr_0:.4f}, Soft FPR_1: {fpr_1:.4f}, |차이|: {abs(fpr_0 - fpr_1):.4f}")
        
        return eo_loss_fuc(p, y, g, tau)

    # 2. 데이터 수집
    tau_values = [1, 5, 10, 20, 50, 100]
    real_eos = []
    approx_eos = {tau: [] for tau in tau_values}
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            if batch_idx >= 50:  # 50개 배치만 사용
                break
            
            # 데이터 준비    
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            g = labels[:, sensitive_idx].float()
            
            # 실제 EO와 근사 EO 계산
            try:
                real_eo = compute_real_eo(outputs, labels, g).cpu().item()
                real_eos.append(real_eo)
                
                for tau in tau_values:
                    approx_eo = compute_soft_eo(outputs, labels, g, tau).cpu().item()
                    approx_eos[tau].append(approx_eo)
            except Exception as e:
                print(f"배치 {batch_idx}에서 오류 발생: {str(e)}")
                continue

    # 3. 결과 분석
    correlations = {}
    
    try:
        plt.figure(figsize=(15, 10))
        for idx, tau in enumerate(tau_values, 1):
            plt.subplot(2, 3, idx)
            
            x = np.array(real_eos)
            y = np.array(approx_eos[tau])
            
            # 산점도
            plt.scatter(x, y, alpha=0.5)
            
            # 상관계수 계산
            if len(x) > 1:
                correlation, p_value = stats.pearsonr(x, y)
                correlations[f'tau_{tau}'] = correlation
                
                # 추세선 (단순 선형 회귀)
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", alpha=0.8)
                
                plt.title(f'tau={tau}\nr={correlation:.3f}, p={p_value:.3e}')
            else:
                plt.title(f'tau={tau}\nNot enough data')
            
            plt.xlabel('Real EO')
            plt.ylabel('Approximated EO')
        
        plt.tight_layout()
        plt.savefig('eo_correlation_analysis.png')
        plt.close()
        
        # 히트맵
        if correlations:
            plt.figure(figsize=(10, 4))
            correlation_values = [correlations[f'tau_{tau}'] for tau in tau_values]
            sns.heatmap([correlation_values], 
                       xticklabels=[f'tau={tau}' for tau in tau_values],
                       yticklabels=['Correlation'],
                       annot=True, 
                       cmap='RdYlBu_r',
                       center=0)
            plt.title('Correlation between Real and Approximated EO')
            plt.tight_layout()
            plt.savefig('eo_correlation_heatmap.png')
            plt.close()
            
    except Exception as e:
        print(f"시각화 중 오류 발생: {str(e)}")
        
    return correlations



#! 그래디언트 차이 기반 근사
def compute_approximate_eo_by_loss(model, test_loader, sensitive_idx, output_cols_each_task, criterion, stop_batch=100):
    print("EO 근사 계산 시작...")
    
    MIN_SAMPLES_PER_GROUP = 10  # 최소 샘플 수 기준

    # 첫 번째 레이어 찾기
    first_layer = None
    first_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            first_layer = layer
            first_layer_name = name
            break
    
    if first_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None

    # 결과 저장을 위한 딕셔너리 초기화
    eo_grad_male = {first_layer_name: None}    # 남성 그룹의 그래디언트 누적
    eo_grad_female = {first_layer_name: None}  # 여성 그룹의 그래디언트 누적
    eo_scores = {}       # 최종 EO 점수
    batches_male = 0     # 남성 데이터가 있는 배치 수
    batches_female = 0   # 여성 데이터가 있는 배치 수

    model.train()  # 그래디언트 계산을 위해 학습 모드로 설정
    optimizer = torch.optim.Adam([first_layer.weight], lr=1e-4)  # 첫 번째 레이어만 최적화

    

    for batch_idx, (data, labels) in enumerate(test_loader):
        if batch_idx >= stop_batch:
            break
            
        if batch_idx % 10 == 0:
            print(f"배치 {batch_idx}/{stop_batch} 처리 중...")

        data = data.to(device)
        labels = labels.to(device)
        
        gender = labels[:, sensitive_idx]
        male_mask = (gender == 0)
        female_mask = (gender == 1)
        
        # 각 그룹의 샘플 수 확인
        n_male = male_mask.sum().item()
        n_female = female_mask.sum().item()
        
        # print(f"배치 {batch_idx}: (남성: {n_male}, 여성: {n_female})")

        # 최소 샘플 수 검증
        if n_male < MIN_SAMPLES_PER_GROUP or n_female < MIN_SAMPLES_PER_GROUP:
            print(f"배치 {batch_idx}: 샘플 부족 (남성: {n_male}, 여성: {n_female} < {MIN_SAMPLES_PER_GROUP})")
            skipped_batches += 1
            continue

        def normalize_gradient(grad):
            norm = torch.norm(grad, p=2)
            return grad / (norm + 1e-8)
     
        # 각 그룹에 대해 처리
        for group_idx, (group_mask, grad_accum, n_batches) in enumerate([
            (male_mask, eo_grad_male, batches_male),
            (female_mask, eo_grad_female, batches_female)
        ]):
            if group_mask.sum() > 0:
                optimizer.zero_grad()
                
                group_data = data[group_mask]
                group_labels = labels[group_mask]
                group_outputs = model(group_data)
                
                group_loss = loss_multi_tasks(
                    group_outputs, 
                    group_labels, 
                    criterion, 
                    output_cols_each_task
                )
                
                group_loss.backward()
                
                # 첫 번째 레이어의 그래디언트만 누적
                grad = first_layer.weight.grad
                if grad is not None:
                    grad = normalize_gradient(grad.clone().detach().cpu())
                    
                    if grad_accum[first_layer_name] is None:
                        grad_accum[first_layer_name] = grad
                    else:
                        # 지수 이동 평균 사용
                        alpha = 0.9
                        grad_accum[first_layer_name] = alpha * grad_accum[first_layer_name] + (1 - alpha) * grad
                
                # 배치 카운터 업데이트
                if group_idx == 0:
                    batches_male += 1
                else:
                    batches_female += 1

    # 그룹별 평균 그래디언트 계산
    if batches_male > 0:
        eo_grad_male[first_layer_name] /= batches_male
    if batches_female > 0:
        eo_grad_female[first_layer_name] /= batches_female
        
    # EO 점수 계산 (그래디언트 차이의 절대값)
    eo_scores[first_layer_name] = torch.abs(eo_grad_male[first_layer_name] - eo_grad_female[first_layer_name])


    print(f"처리된 배치 수: 남성 {batches_male}, 여성 {batches_female}")

     # 결과 저장을 위한 디렉토리 생성
    save_dir = 'experiment4'
    os.makedirs(save_dir, exist_ok=True)

    # 1. 가중치별 근사 값 저장 (npy)
    for name, score in eo_scores.items():
        np.save(
            os.path.join(save_dir, f'eo_score_{name}.npy'),
            score.cpu().numpy()
        )

    # 2. 분석 보고서 작성 (txt)
    with open(os.path.join(save_dir, 'eo_analysis_report.txt'), 'w') as f:
        f.write("=== EO 근사 분석 보고서 ===\n\n")
        f.write(f"처리된 총 배치 수: {stop_batch}\n")
        f.write(f"남성 데이터 배치 수: {batches_male}\n")
        f.write(f"여성 데이터 배치 수: {batches_female}\n\n")

        # 레이어별 통계
        for name, score in eo_scores.items():
            f.write(f"\n레이어: {name}\n")
            f.write(f"shape: {score.shape}\n")
            f.write(f"평균: {score.mean().item():.6f}\n")
            f.write(f"표준편차: {score.std().item():.6f}\n")
            f.write(f"최소값: {score.min().item():.6f}\n")
            f.write(f"최대값: {score.max().item():.6f}\n")
            f.write(f"0에 가까운 가중치 비율: {(score.abs() < 1e-6).float().mean().item():.2%}\n")
            f.write("-" * 50 + "\n")

    # 3. EO 점수 분포도 시각화 (png)
    import matplotlib.pyplot as plt
    for name, score in eo_scores.items():
        plt.figure(figsize=(10, 6))
        plt.hist(score.cpu().numpy().flatten(), bins=50)
        plt.title(f'Distribution of EO Scores - {name}')
        plt.xlabel('EO Score')
        plt.ylabel('Count')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(save_dir, f'eo_distribution_{name}.png'))
        plt.close()

    print(f"\n분석 결과가 '{save_dir}' 디렉토리에 저장되었습니다.")
    print(f"- npy 파일: 각 레이어의 EO 점수")
    print(f"- txt 파일: 상세 분석 보고서")
    print(f"- png 파일: EO 점수 분포도")
    # 3. EO 점수 분포도 시각화 (png)
    plt.figure(figsize=(15, 5))
    
    # 첫 번째 서브플롯: 그룹별 그래디언트 점수 비교
    plt.subplot(1, 2, 1)
    weights_indices = range(eo_grad_male[first_layer_name].numel())
    
    # 남성 그룹 그래디언트
    male_grads = eo_grad_male[first_layer_name].flatten().cpu().numpy()
    plt.plot(weights_indices, male_grads, 'b-', alpha=0.6, label='Male Group Gradients')
    
    # 여성 그룹 그래디언트
    female_grads = eo_grad_female[first_layer_name].flatten().cpu().numpy()
    plt.plot(weights_indices, female_grads, 'r-', alpha=0.6, label='Female Group Gradients')
    
    plt.title(f'Group-wise Gradients Comparison - {first_layer_name}')
    plt.xlabel('Weight Index')
    plt.ylabel('Gradient Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 두 번째 서브플롯: 그래디언트 차이
    plt.subplot(1, 2, 2)
    grad_diff = (male_grads - female_grads)
    plt.plot(weights_indices, grad_diff, 'g-', alpha=0.7, label='Gradient Difference')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.title(f'Gradient Differences - {first_layer_name}')
    plt.xlabel('Weight Index')
    plt.ylabel('Gradient Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'gradient_analysis_{first_layer_name}.png'))
    plt.close()
    
    print(f"- 그래디언트 분석 그래프가 '{save_dir}' 디렉토리에 저장되었습니다.")
    
    return eo_scores





import time 

def compute_differentiable_eo_scores(model, test_loader, sensitive_idx, output_cols_each_task, stop_batch=100):
    """미분 가능한 방식으로 EO 점수를 근사 계산하는 함수"""
    print("\n=== 미분 가능한 EO 점수 근사 계산 시작 ===")
    
    # 1. 첫 번째 레이어 찾기
    first_layer = None
    first_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            first_layer = layer
            first_layer_name = name
            first_layer.weight.requires_grad_(True)  # 그래디언트 계산 활성화
            break
    
    if first_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None

    model.eval()  # 모델을 평가 모드로 설정

    def soft_eo_loss(outputs, labels, g):
        """미분 가능한 EO 손실 계산"""
        # 첫 번째 태스크의 출력 추출
        task_logits = outputs[:, output_cols_each_task[0][0]:output_cols_each_task[0][1]]
        probs = torch.softmax(task_logits, dim=1)[:, 1]  # 양성 클래스의 확률
        y_true = labels[:, 0]
        
        # 그룹별 TPR과 FPR 계산 (미분 가능한 형태)
        mask_pos = (y_true == 1).float()
        mask_neg = (y_true == 0).float()
        
        # 그룹 0
        g0_mask = (g == 0).float()
        tpr_0 = (probs * mask_pos * g0_mask).sum() / (mask_pos * g0_mask).sum().clamp(min=1e-7)
        fpr_0 = (probs * mask_neg * g0_mask).sum() / (mask_neg * g0_mask).sum().clamp(min=1e-7)
        
        # 그룹 1
        g1_mask = (g == 1).float()
        tpr_1 = (probs * mask_pos * g1_mask).sum() / (mask_pos * g1_mask).sum().clamp(min=1e-7)
        fpr_1 = (probs * mask_neg * g1_mask).sum() / (mask_neg * g1_mask).sum().clamp(min=1e-7)
        
        # EO 점수 계산
        return 0.5 * (torch.abs(tpr_0 - tpr_1) + torch.abs(fpr_0 - fpr_1))

    # 3. 배치별 처리 및 그래디언트 누적
    eo_grads_accum = torch.zeros_like(first_layer.weight)
    batches = 0
    start_time = time.time()
    total_weights = first_layer.weight.numel()
    
    for batch_idx, (data, labels) in enumerate(test_loader):
        if batch_idx >= stop_batch:
            break
            
        # 진행상황 및 예상 시간 계산
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            # 초당 처리하는 배치 수 계산
            batches_per_sec = (batch_idx + 1) / elapsed
            # 남은 배치 수
            remaining_batches = stop_batch - (batch_idx + 1)
            # 예상 남은 시간 (초)
            if batches_per_sec > 0:
                eta_seconds = remaining_batches / batches_per_sec
                # 보기 좋게 변환
                eta_h = int(eta_seconds // 3600)
                eta_m = int((eta_seconds % 3600) // 60)
                eta_s = int(eta_seconds % 60)
                
                print(f"\r배치 {batch_idx + 1}/{stop_batch} 처리 중... "
                      f"예상 남은 시간: {eta_h}시간 {eta_m}분 {eta_s}초", end="")
            else:
                print(f"\r배치 {batch_idx + 1}/{stop_batch} 처리 중...", end="")
        
        data = data.to(device)
        labels = labels.to(device)
        g = labels[:, sensitive_idx].float()
        
        # 그래디언트 초기화
        model.zero_grad()
        
        # 순전파 및 EO 손실 계산
        with torch.set_grad_enabled(True):
            outputs = model(data)
            eo_loss = soft_eo_loss(outputs, labels, g)
            
            # 역전파
            eo_loss.backward()
        
        # 그래디언트 누적
        if first_layer.weight.grad is not None:
            eo_grads_accum += first_layer.weight.grad.abs().detach()
            batches += 1

    print()  # 진행 표시줄 다음 줄로 이동
    
    # 4. 평균 그래디언트 계산
    if batches > 0:
        eo_scores = eo_grads_accum / batches
    else:
        print("처리된 배치가 없습니다.")
        return None
    
    # 5. 결과 저장 및 시각화
    save_dir = 'experiment4'
    os.makedirs(save_dir, exist_ok=True)
    
    # EO 점수 저장
    np.save(
        os.path.join(save_dir, f'differentiable_eo_scores_{first_layer_name}.npy'), 
        eo_scores.cpu().numpy()
    )
    
    # 분포 시각화
    plt.figure(figsize=(15, 5))
    
    # EO 점수 히스토그램
    plt.subplot(1, 2, 1)
    plt.hist(eo_scores.cpu().numpy().flatten(), bins=50)
    plt.title('Distribution of Differentiable EO Scores')
    plt.xlabel('EO Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # EO 점수 히트맵
    plt.subplot(1, 2, 2)
    weight_shape = first_layer.weight.shape
    if len(weight_shape) == 2:  # Linear 레이어
        reshaped_scores = eo_scores.cpu().numpy().reshape(weight_shape)
    else:  # Conv 레이어
        reshaped_scores = eo_scores.cpu().numpy().reshape(weight_shape[0], -1).mean(axis=1)
        reshaped_scores = reshaped_scores.reshape(-1, 1)
    
    plt.imshow(reshaped_scores, cmap='coolwarm')
    plt.colorbar(label='EO Score')
    plt.title('EO Scores Heatmap')
    plt.xlabel('Input Channel')
    plt.ylabel('Output Channel')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'differentiable_eo_analysis_{first_layer_name}.png'))
    plt.close()
    
    # 분석 리포트 저장
    total_time = time.time() - start_time
    with open(os.path.join(save_dir, 'differentiable_eo_analysis.txt'), 'w') as f:
        f.write("=== 미분 가능한 EO 점수 분석 ===\n\n")
        f.write(f"총 소요 시간: {int(total_time//3600)}시간 {int((total_time%3600)//60)}분 {int(total_time%60)}초\n")
        f.write(f"처리된 배치 수: {batches}\n\n")
        f.write(f"레이어: {first_layer_name}\n")
        f.write(f"가중치 수: {total_weights}\n")
        f.write("\nEO 점수 통계:\n")
        f.write(f"최소값: {eo_scores.min().item():.6f}\n")
        f.write(f"최대값: {eo_scores.max().item():.6f}\n")
        f.write(f"평균: {eo_scores.mean().item():.6f}\n")
        f.write(f"표준편차: {eo_scores.std().item():.6f}\n")
    
    print(f"\n분석 완료! 결과가 '{save_dir}' 디렉토리에 저장되었습니다.")
    return eo_scores




def compute_approximate_eo_by_surrogate2(model, train_loader, sensitive_idx, output_cols_each_task, stop_batch=100):
    """로그-시그모이드 합 방법으로 EO를 근사하는 함수"""
    print("\n=== 로그-시그모이드 합 기반 EO 점수 근사 계산 시작 ===")
    
    # 1. 모델 설정
    model.eval()  # 평가 모드로 설정
    
    # 첫 번째 레이어 찾기
    first_layer = None
    first_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            first_layer = layer
            first_layer_name = name
            first_layer.weight.requires_grad_(True)
            break
    
    if first_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None

    def surrogate_g(r, c=math.log(2)):
        """개선된 로그-시그모이드 합 대리 함수"""
        softplus = torch.log(1 + torch.exp(r))
        linear = (r - 1) * c / math.log(2)
        return softplus - linear

    def compute_group_metrics(outputs, labels, g):
        """각 그룹의 TPR, FPR 평균 계산"""
        logits = outputs[:, output_cols_each_task[0][0]:output_cols_each_task[0][1]]
        y_true = labels[:, 0]
        
        # 그룹별 마스크
        g0_mask = (g == 0)
        g1_mask = (g == 1)
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)
        
        # 각 그룹별 TPR, FPR 계산
        eps = 1e-7
        
        # Group 0
        g0_tp = surrogate_g(logits[g0_mask & pos_mask]).mean() if (g0_mask & pos_mask).any() else torch.tensor(0.).to(device)
        g0_fp = surrogate_g(logits[g0_mask & neg_mask]).mean() if (g0_mask & neg_mask).any() else torch.tensor(0.).to(device)
        
        # Group 1
        g1_tp = surrogate_g(logits[g1_mask & pos_mask]).mean() if (g1_mask & pos_mask).any() else torch.tensor(0.).to(device)
        g1_fp = surrogate_g(logits[g1_mask & neg_mask]).mean() if (g1_mask & neg_mask).any() else torch.tensor(0.).to(device)
        
        return g0_tp, g1_tp, g0_fp, g1_fp

    def compute_eo_loss(outputs, labels, g):
        """EO 대리 손실 계산"""
        g0_tp, g1_tp, g0_fp, g1_fp = compute_group_metrics(outputs, labels, g)
        
        # TPR, FPR 차이의 제곱
        tpr_gap = (g0_tp - g1_tp) ** 2
        fpr_gap = (g0_fp - g1_fp) ** 2
        
        return tpr_gap + fpr_gap

    # 2. 그래디언트 누적을 위한 변수 초기화
    eo_grads_accum = torch.zeros_like(first_layer.weight)
    batches = 0
    start_time = time.time()

    # 3. 배치 처리
    for batch_idx, (data, labels) in enumerate(train_loader):
        if batch_idx >= stop_batch:
            break
            
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            remaining = (stop_batch - batch_idx) * (elapsed / (batch_idx + 1)) if batch_idx > 0 else 0
            print(f"\r배치 {batch_idx}/{stop_batch} 처리 중... "
                  f"예상 남은 시간: {int(remaining//3600)}시간 "
                  f"{int((remaining%3600)//60)}분 {int(remaining%60)}초", end="")

        data = data.to(device)
        labels = labels.to(device)
        g = labels[:, sensitive_idx]
        
        # 순전파 및 손실 계산
        model.zero_grad()
        outputs = model(data)
        eo_loss = compute_eo_loss(outputs, labels, g)
        
        # 역전파
        eo_loss.backward()
        
        # 그래디언트 누적
        if first_layer.weight.grad is not None:
            eo_grads_accum += first_layer.weight.grad.abs().detach()
            batches += 1

    # 4. 평균 계산
    if batches > 0:
        eo_scores = eo_grads_accum / batches
    else:
        print("처리된 배치가 없습니다.")
        return None

    # 5. 결과 저장
    save_dir = 'experiment4'
    os.makedirs(save_dir, exist_ok=True)
    
    # numpy 파일로 저장
    np.save(
        os.path.join(save_dir, f'surrogate_eo_scores2_{first_layer_name}.npy'), 
        eo_scores.cpu().numpy()
    )
    
    # 시각화 및 분석 리포트 저장
    plt.figure(figsize=(10, 5))
    
    # EO 점수 히스토그램
    plt.subplot(1, 2, 1)
    plt.hist(eo_scores.cpu().numpy().flatten(), bins=50)
    plt.title('Distribution of Surrogate EO Scores')
    plt.xlabel('EO Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # EO 점수 히트맵
    plt.subplot(1, 2, 2)
    weight_shape = first_layer.weight.shape
    if len(weight_shape) == 2:  # Linear 레이어
        reshaped_scores = eo_scores.cpu().numpy().reshape(weight_shape)
    else:  # Conv 레이어
        reshaped_scores = eo_scores.cpu().numpy().reshape(weight_shape[0], -1).mean(axis=1)
        reshaped_scores = reshaped_scores.reshape(-1, 1)
    
    plt.imshow(reshaped_scores, cmap='coolwarm')
    plt.colorbar(label='EO Score')
    plt.title('EO Scores Heatmap')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'surrogate_eo2_analysis_{first_layer_name}.png'))
    plt.close()

    # 분석 리포트 작성
    total_time = time.time() - start_time
    with open(os.path.join(save_dir, 'surrogate_eo2_analysis.txt'), 'w') as f:
        f.write("=== 로그-시그모이드 합 기반 EO 점수 분석 ===\n\n")
        f.write(f"총 소요 시간: {int(total_time//3600)}시간 {int((total_time%3600)//60)}분 {int(total_time%60)}초\n")
        f.write(f"처리된 배치 수: {batches}\n\n")
        f.write(f"레이어: {first_layer_name}\n")
        f.write(f"가중치 수: {first_layer.weight.numel()}\n\n")
        f.write("EO 점수 통계:\n")
        f.write(f"  최소값: {eo_scores.min().item():.6f}\n")
        f.write(f"  최대값: {eo_scores.max().item():.6f}\n")
        f.write(f"  평균: {eo_scores.mean().item():.6f}\n")
        f.write(f"  표준편차: {eo_scores.std().item():.6f}\n")
    
    print(f"\n분석 완료! 결과가 '{save_dir}' 디렉토리에 저장되었습니다.")
    return eo_scores





def compute_approximate_eo_by_surrogate(model, train_loader, sensitive_idx, output_cols_each_task, stop_batch=100):
    """미분 가능한 대리 함수를 사용하여 EO를 근사하는 함수"""
    print("\n=== 대리 함수 기반 EO 점수 근사 계산 시작 ===")
    
    # 1. 모델 설정
    model.eval()  # 평가 모드로 설정
    
    # 첫 번째 레이어 찾기
    first_layer = None
    first_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            first_layer = layer
            first_layer_name = name
            first_layer.weight.requires_grad_(True)
            break
    
    if first_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None

    def surrogate_g(r):
        """로그-시그모이드 차이 대리 함수 (Softplus)"""
        return torch.log(1 + torch.exp(r))

    def compute_group_metrics(outputs, labels, gender):
        """각 그룹의 TP, FP 평균 계산"""
        logits = outputs[:, output_cols_each_task[0][0]:output_cols_each_task[0][1]]
        y_true = labels[:, 0]
        
        # 각 그룹별 마스크
        group_0_mask = (gender == 0)
        group_1_mask = (gender == 1)
        positive_mask = (y_true == 1)
        negative_mask = (y_true == 0)
        
        # Group A (gender=0) metrics
        group_a_tp = surrogate_g(logits[group_0_mask & positive_mask]).mean() if (group_0_mask & positive_mask).any() else torch.tensor(0.).to(device)
        group_a_fp = surrogate_g(logits[group_0_mask & negative_mask]).mean() if (group_0_mask & negative_mask).any() else torch.tensor(0.).to(device)
        
        # Group B (gender=1) metrics
        group_b_tp = surrogate_g(logits[group_1_mask & positive_mask]).mean() if (group_1_mask & positive_mask).any() else torch.tensor(0.).to(device)
        group_b_fp = surrogate_g(logits[group_1_mask & negative_mask]).mean() if (group_1_mask & negative_mask).any() else torch.tensor(0.).to(device)
        
        return group_a_tp, group_b_tp, group_a_fp, group_b_fp

    def compute_eo_loss(outputs, labels, gender):
        """EO 대리 손실 계산"""
        group_a_tp, group_b_tp, group_a_fp, group_b_fp = compute_group_metrics(outputs, labels, gender)
        
        # TPR과 FPR 차이의 제곱
        tpr_loss = (group_a_tp - group_b_tp) ** 2
        fpr_loss = (group_a_fp - group_b_fp) ** 2
        
        # 전체 EO 손실
        return tpr_loss + fpr_loss

    # 2. 그래디언트 누적을 위한 변수 초기화
    eo_grads_accum = torch.zeros_like(first_layer.weight)
    batches = 0
    start_time = time.time()

    # 3. 배치 처리
    for batch_idx, (data, labels) in enumerate(train_loader):
        if batch_idx >= stop_batch:
            break
            
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            remaining = (stop_batch - batch_idx) * (elapsed / (batch_idx + 1)) if batch_idx > 0 else 0
            print(f"\r배치 {batch_idx}/{stop_batch} 처리 중... "
                  f"예상 남은 시간: {int(remaining//3600)}시간 "
                  f"{int((remaining%3600)//60)}분 {int(remaining%60)}초", end="")

        data = data.to(device)
        labels = labels.to(device)
        gender = labels[:, sensitive_idx]
        
        # 순전파 및 손실 계산
        model.zero_grad()
        outputs = model(data)
        eo_loss = compute_eo_loss(outputs, labels, gender)
        
        # 역전파
        eo_loss.backward()
        
        # 그래디언트 누적
        if first_layer.weight.grad is not None:
            eo_grads_accum += first_layer.weight.grad.detach()
            batches += 1

    # 4. 평균 계산
    if batches > 0:
        eo_scores = eo_grads_accum / batches
    else:
        print("처리된 배치가 없습니다.")
        return None

    # 5. 결과 저장
    save_dir = 'experiment4'
    os.makedirs(save_dir, exist_ok=True)
    
    # numpy 파일로 저장
    np.save(
        os.path.join(save_dir, f'surrogate_eo_scores_{first_layer_name}.npy'), 
        eo_scores.cpu().numpy()
    )
    
    # 시각화
    plt.figure(figsize=(10, 5))
    
    # EO 점수 히스토그램
    plt.subplot(1, 2, 1)
    plt.hist(eo_scores.cpu().numpy().flatten(), bins=50)
    plt.title('Distribution of Surrogate EO Scores')
    plt.xlabel('EO Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # EO 점수 히트맵
    plt.subplot(1, 2, 2)
    weight_shape = first_layer.weight.shape
    if len(weight_shape) == 2:  # Linear 레이어
        reshaped_scores = eo_scores.cpu().numpy().reshape(weight_shape)
    else:  # Conv 레이어
        reshaped_scores = eo_scores.cpu().numpy().reshape(weight_shape[0], -1).mean(axis=1)
        reshaped_scores = reshaped_scores.reshape(-1, 1)
    
    plt.imshow(reshaped_scores, cmap='coolwarm')
    plt.colorbar(label='EO Score')
    plt.title('EO Scores Heatmap')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'surrogate_eo_analysis_{first_layer_name}.png'))
    plt.close()

    # 분석 리포트 작성
    total_time = time.time() - start_time
    with open(os.path.join(save_dir, 'surrogate_eo_analysis.txt'), 'w') as f:
        f.write("=== 대리 함수 기반 EO 점수 분석 ===\n\n")
        f.write(f"총 소요 시간: {int(total_time//3600)}시간 {int((total_time%3600)//60)}분 {int(total_time%60)}초\n")
        f.write(f"처리된 배치 수: {batches}\n\n")
        f.write(f"레이어: {first_layer_name}\n")
        f.write(f"가중치 수: {first_layer.weight.numel()}\n\n")
        f.write("EO 점수 통계:\n")
        f.write(f"  최소값: {eo_scores.min().item():.6f}\n")
        f.write(f"  최대값: {eo_scores.max().item():.6f}\n")
        f.write(f"  평균: {eo_scores.mean().item():.6f}\n")
        f.write(f"  표준편차: {eo_scores.std().item():.6f}\n")
    
    print(f"\n분석 완료! 결과가 '{save_dir}' 디렉토리에 저장되었습니다.")
    return eo_scores


def validate_eo_loss_all(model, test_loader, sensitive_idx, output_cols_each_task, tau_values=[1, 5, 10, 20, 50, 100]):
    """
    여러 tau 값에 대해 eo_loss_all 함수의 정확도를 실제 EO 값과 비교하여 검증
    """
    print("\n=== eo_loss_all 함수 검증 시작 ===")
    
    # 결과 저장을 위한 딕셔너리
    results = {tau: {
        'real_eos': [],
        'approx_eos': [],
        'correlation': None,
        'p_value': None,
        'mae': None,
        'rmse': None
    } for tau in tau_values}
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            if batch_idx >= 50:  # 50개 배치만 사용
                break
                
            if batch_idx % 10 == 0:
                print(f"배치 {batch_idx}/50 처리 중...")
                
            # 데이터 준비    
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            g = labels[:, sensitive_idx].float()
            
            # 1. 실제 EO 계산
            task_real_eos = []
            for task_idx, (st, ed) in enumerate(output_cols_each_task):
                p = torch.softmax(outputs[:, st:ed], dim=1)[:, 1]
                y = labels[:, task_idx]
                
                eps = 1e-7
                # TPR 계산
                tpr_0 = (((p[g == 0] > 0.5).float() * (y[g == 0] == 1).float()).sum() / 
                         ((y[g == 0] == 1).float().sum() + eps))
                tpr_1 = (((p[g == 1] > 0.5).float() * (y[g == 1] == 1).float()).sum() / 
                         ((y[g == 1] == 1).float().sum() + eps))
                
                # FPR 계산
                fpr_0 = (((p[g == 0] > 0.5).float() * (y[g == 0] == 0).float()).sum() / 
                         ((y[g == 0] == 0).float().sum() + eps))
                fpr_1 = (((p[g == 1] > 0.5).float() * (y[g == 1] == 0).float()).sum() / 
                         ((y[g == 1] == 0).float().sum() + eps))
                
                task_eo = 0.5 * (abs(tpr_0 - tpr_1) + abs(fpr_0 - fpr_1))
                task_real_eos.append(task_eo)
            
            real_eo = sum(task_real_eos) / len(task_real_eos)
            
            # 2. 각 tau 값에 대해 근사 EO 계산
            for tau in tau_values:
                approx_eo = eo_loss_all(outputs, labels, g, tau=tau, 
                                      output_cols_each_task=output_cols_each_task)
                
                results[tau]['real_eos'].append(real_eo.cpu().item())
                results[tau]['approx_eos'].append(approx_eo.cpu().item())
    
    # 결과 분석 및 시각화
    save_dir = 'eo_loss_validation'
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 모든 tau에 대한 산점도를 하나의 그림에 표시
    plt.figure(figsize=(15, 10))
    for idx, tau in enumerate(tau_values, 1):
        plt.subplot(2, 3, idx)
        
        x = np.array(results[tau]['real_eos'])
        y = np.array(results[tau]['approx_eos'])
        
        plt.scatter(x, y, alpha=0.5)
        
        if len(x) > 1:
            # 상관관계 분석
            correlation, p_value = stats.pearsonr(x, y)
            results[tau]['correlation'] = correlation
            results[tau]['p_value'] = p_value
            
            # MAE와 RMSE 계산
            results[tau]['mae'] = np.mean(np.abs(x - y))
            results[tau]['rmse'] = np.sqrt(np.mean((x - y)**2))
            
            # 추세선
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.8)
            
            plt.title(f'tau={tau}\nr={correlation:.3f}, p={p_value:.3e}')
        else:
            plt.title(f'tau={tau}\nNot enough data')
        
        plt.xlabel('Real EO')
        plt.ylabel('Approximated EO')
        
        # 이상적인 y=x 선 추가
        ideal = np.linspace(min(min(x), min(y)), max(max(x), max(y)), 100)
        plt.plot(ideal, ideal, 'g--', alpha=0.3, label='Ideal')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eo_loss_all_correlation.png'))
    plt.close()
    
    # 2. 상관계수 히트맵
    plt.figure(figsize=(10, 4))
    correlation_values = [results[tau]['correlation'] for tau in tau_values]
    sns.heatmap([correlation_values], 
               xticklabels=[f'tau={tau}' for tau in tau_values],
               yticklabels=['Correlation'],
               annot=True, 
               cmap='RdYlBu_r',
               center=0)
    plt.title('Correlation between Real and Approximated EO')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'eo_loss_all_correlation_heatmap.png'))
    plt.close()
    
    # 3. 상세 분석 리포트 저장
    with open(os.path.join(save_dir, 'eo_loss_all_analysis.txt'), 'w') as f:
        f.write("=== eo_loss_all 함수 검증 결과 ===\n\n")
        
        for tau in tau_values:
            f.write(f"\nTau = {tau}:\n")
            f.write(f"  상관계수: {results[tau]['correlation']:.6f}\n")
            f.write(f"  p-value: {results[tau]['p_value']:.6e}\n")
            f.write(f"  MAE: {results[tau]['mae']:.6f}\n")
            f.write(f"  RMSE: {results[tau]['rmse']:.6f}\n")
            
            x = np.array(results[tau]['real_eos'])
            y = np.array(results[tau]['approx_eos'])
            
            f.write("\n  통계 정보:\n")
            f.write(f"    실제 EO - 평균: {np.mean(x):.6f}, 표준편차: {np.std(x):.6f}\n")
            f.write(f"    근사 EO - 평균: {np.mean(y):.6f}, 표준편차: {np.std(y):.6f}\n")
    
    print(f"\n검증 결과가 '{save_dir}' 디렉토리에 저장되었습니다.")
    
    return results


# 활성화 값 차이 기반 근사
def compute_first_layer_eo_importance(model, train_loader, sensitive_idx, target_label_idx, 
                                    num_samples=1000, tau=0.5, fallback_activation=nn.ReLU()):
    """
    첫 번째 레이어의 EO 중요도를 계산 (수정된 버전)
    
    Args:
        model: 평가할 모델
        train_loader: 학습 데이터 로더
        sensitive_idx: 민감 속성(예: gender) 인덱스
        target_label_idx: 분석할 타겟 라벨 인덱스
        num_samples: 각 그룹당 수집할 샘플 수
        tau: 활성화 통계량 계산시 사용할 분위수
        fallback_activation: 활성화 함수가 없을 때 사용할 기본 활성화 함수
    """
    device = next(model.parameters()).device
    model.eval()
    
    # 1. 첫 번째 레이어와 다음 활성화 함수 찾기
    first_layer = None
    activation_fn = None
    
    print("\n=== 모델 구조 분석 ===")
    for name, module in model.named_modules():
        print(f"레이어: {name} (타입: {type(module).__name__})")
        
        if first_layer is None and isinstance(module, (nn.Conv2d, nn.Linear)):
            first_layer = module
            first_layer_name = name
            print(f"\n첫 번째 레이어 발견:")
            print(f"- 이름: {name}")
            print(f"- 타입: {type(module).__name__}")
            print(f"- 형태: {module.weight.shape}")
            break
    
    if first_layer is None:
        raise ValueError("지원되는 첫 번째 레이어를 찾을 수 없습니다.")
    
    # 2. 활성화 함수 결정
    next_activation = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.SiLU, nn.LeakyReLU)):
            next_activation = module
            print(f"\n활성화 함수 발견: {type(module).__name__}")
            break
    
    activation_fn = next_activation if next_activation is not None else fallback_activation
    print(f"사용할 활성화 함수: {type(activation_fn).__name__}")
    
    # 3. pre-activation 수집을 위한 데이터 구조
    activations = {
        'pos': {0: [], 1: []},  # Y=1인 경우의 그룹별 활성화값 
        'neg': {0: [], 1: []}   # Y=0인 경우의 그룹별 활성화값
    }
    
    # 4. pre-activation 수집 훅
    pre_activation = None
    def get_activation(name):
        def hook(module, input, output):
            nonlocal pre_activation
            pre_activation = output.clone()
        return hook
    
    # 5. 첫 번째 레이어에 훅 등록
    hook = first_layer.register_forward_hook(get_activation('first_layer'))
    
    try:
        # 6. 균형잡힌 샘플 수집
        group_counts = {
            'pos': {0: 0, 1: 0},
            'neg': {0: 0, 1: 0}
        }
        
        with torch.no_grad():
            for data, labels in train_loader:
                min_count = min(min(group_counts['pos'].values()), 
                              min(group_counts['neg'].values()))
                
                if min_count >= num_samples:
                    break
                    
                data = data.to(device)
                labels = labels.to(device)
                
                # 성별(그룹)과 타겟 라벨 추출
                g = labels[:, sensitive_idx]
                if g.dtype != torch.long:
                    g = g.round().long()
                
                # 모든 타겟 라벨에 대해 반복
                for target_idx in target_label_idx:
                    # 타겟 라벨 추출 및 타입 처리
                    y_true = labels[:, target_idx]
                    if y_true.dtype != torch.long:
                        y_true = y_true.round().long()
                    
                    # forward pass로 pre-activation 얻기
                    _ = model(data)
                    
                    # pre-activation에 활성화 함수 적용
                    acts = pre_activation
                    
                    # Conv 레이어인 경우 GAP 적용
                    if isinstance(first_layer, nn.Conv2d):
                        acts = acts.mean(dim=(2, 3))  # (N, C, H, W) → (N, C)
                    
                    # 활성화 함수 적용
                    acts = activation_fn(acts)
                    
                    # CPU로 이동
                    acts = acts.cpu()
                    g_cpu = g.cpu()
                    y_true_cpu = y_true.cpu()
                    
                    # 그룹별 수집
                    for g_val in [0, 1]:
                        for y_val in [0, 1]:
                            key = 'pos' if y_val == 1 else 'neg'
                            mask = (g_cpu == g_val) & (y_true_cpu == y_val)
                            
                            if mask.any() and group_counts[key][g_val] < num_samples:
                                current_acts = acts[mask]
                                needed = num_samples - group_counts[key][g_val]
                                current_acts = current_acts[:needed]
                                
                                activations[key][g_val].append(current_acts)
                                group_counts[key][g_val] += len(current_acts)
        
        # 7. 그룹별 통계량 계산
        def compute_group_diff(key):
            try:
                acts_g1 = torch.cat(activations[key][1], dim=0)
                acts_g0 = torch.cat(activations[key][0], dim=0)
            except:
                print(f"경고: {key} 버킷이 비어 있습니다!")
                return torch.zeros(first_layer.out_channels if isinstance(first_layer, nn.Conv2d) 
                                else first_layer.out_features)
            
            # tau가 0~1 사이 값임을 보장
            quantile_level = min(max(tau, 0.0), 1.0)
            
            q1 = torch.quantile(acts_g1, quantile_level, dim=0)
            q0 = torch.quantile(acts_g0, quantile_level, dim=0)
            return q1 - q0
        
        delta_pos = compute_group_diff('pos')
        delta_neg = compute_group_diff('neg')
        
        # 8. 가중치와 결합하여 중요도 계산
        weights = first_layer.weight.data.cpu()

        if isinstance(first_layer, nn.Conv2d):
            # Conv 레이어: (out_channels, in_channels, kernel_h, kernel_w)
            weights_shape = weights.shape
            weights_flat = weights.view(weights_shape[0], -1)  # (out_c, in_c * k * k)
            weights_norm = weights_flat.abs()
        else:
            # Linear 레이어: (out_features, in_features)
            weights_norm = weights.abs()

        # EO 중요도 계산 (채널별 차이를 각 가중치에 전파)
        delta_pos_expanded = delta_pos.unsqueeze(1)  # (out_c, 1)
        delta_neg_expanded = delta_neg.unsqueeze(1)  # (out_c, 1)

        # 각 출력 채널의 모든 가중치에 대해 EO 영향 계산
        I_pos = weights_norm * delta_pos_expanded  # (out_c, in_c * k * k)
        I_neg = weights_norm * delta_neg_expanded  # (out_c, in_c * k * k)

        # MAD로 정규화
        def normalize_by_mad(x):
            median = torch.median(x)
            mad = torch.median((x - median).abs())
            return x / (mad + 1e-8)

        I_pos = normalize_by_mad(I_pos)
        I_neg = normalize_by_mad(I_neg)

        # 최종 EO 중요도
        importance_scores = 0.5 * (I_pos + I_neg)

        # 원래 가중치 형태로 복원
        if isinstance(first_layer, nn.Conv2d):
            importance_scores = importance_scores.view(weights_shape)

        # 9. 결과 저장 및 시각화 부분을 다음과 같이 수정
        save_dir = 'experiment4/first_layer_scores'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 각 가중치별 EO 중요도 저장 (원래 가중치와 동일한 형태)
        scores_np = importance_scores.detach().cpu().numpy()
        scores_flat = scores_np.flatten()  # 분포 시각화를 위해 1차원으로 펼침

        # npy 파일로 저장
        np.save(os.path.join(save_dir, f'first_layer_eo_scores_{first_layer_name}.npy'), scores_np)

        # 분포 시각화
        plt.figure(figsize=(10, 6))
        plt.hist(scores_flat, bins=50, edgecolor='black')
        plt.title(f'Distribution of EO Importance Scores\nLayer: {first_layer_name}')
        plt.xlabel('EO Importance Score')
        plt.ylabel('Count')

        # 주요 통계값 표시
        plt.axvline(np.median(scores_flat), color='r', linestyle='--', label=f'Median: {np.median(scores_flat):.2e}')
        plt.axvline(scores_flat.mean(), color='g', linestyle='--', label=f'Mean: {scores_flat.mean():.2e}')

        # 상위 10% 임계값 표시
        threshold_90 = np.percentile(scores_flat, 90)
        plt.axvline(threshold_90, color='b', linestyle='--', label=f'90th percentile: {threshold_90:.2e}')

        plt.legend()
        plt.grid(True, alpha=0.3)

        # 그래프 저장
        plt.savefig(os.path.join(save_dir, f'eo_importance_dist_{first_layer_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 통계 정보 출력 및 저장
        stats_info = {
            'layer_name': first_layer_name,
            'weight_shape': scores_np.shape,
            'total_weights': scores_np.size,
            'mean': float(scores_flat.mean()),
            'median': float(np.median(scores_flat)),
            'std': float(scores_flat.std()),
            'min': float(scores_flat.min()),
            'max': float(scores_flat.max()),
            '90th_percentile': float(threshold_90)
        }

        # 통계 정보를 JSON 파일로 저장
        import json
        with open(os.path.join(save_dir, f'eo_importance_stats_{first_layer_name}.json'), 'w') as f:
            json.dump(stats_info, f, indent=4)

        print("\n=== EO 중요도 계산 완료 ===")
        print(f"저장된 파일들:")
        print(f"- 중요도 점수: {os.path.join(save_dir, f'first_layer_eo_scores_{first_layer_name}.npy')}")
        print(f"- 분포 그래프: {os.path.join(save_dir, f'eo_importance_dist_{first_layer_name}.png')}")
        print(f"- 통계 정보: {os.path.join(save_dir, f'eo_importance_stats_{first_layer_name}.json')}")

        print("\n주요 통계:")
        print(f"- 형태: {scores_np.shape}")
        print(f"- 평균: {scores_flat.mean():.2e}")
        print(f"- 중앙값: {np.median(scores_flat):.2e}")
        print(f"- 표준편차: {scores_flat.std():.2e}")
        print(f"- 상위 10% 임계값: {threshold_90:.2e}")
        
    finally:
        hook.remove()
            
    return importance_scores, first_layer_name




def compute_actual_importance_scores(model, test_loader, output_cols_each_task):
    """
    첫 번째 레이어의 각 가중치의 실제 중요도를 계산
    중요도 = |원본 모델의 정확도 - 가중치 비활성화 시의 정확도|
    """
    import time
    start_time = time.time()
    
    stop_batch = 10
    importance_scores = {}
    layer_stats = {}
    
    # 첫 번째 레이어 찾기
    first_layer = None
    first_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            first_layer = layer
            first_layer_name = name
            # 첫 번째 레이어 정보 출력
            print("\n=== 첫 번째 레이어 정보 ===")
            print(f"레이어 이름: {first_layer_name}")
            print(f"레이어 타입: {type(first_layer).__name__}")
            print(f"형태: {first_layer.weight.shape}")
            print(f"가중치 수: {first_layer.weight.numel()}")
            break  # 첫 번째 레이어를 찾으면 중단
    
    if first_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None, None
    
    # 원본 가중치 저장
    original_weight = first_layer.weight.data.clone()
    
    # 통계 정보 초기화
    layer_stats[first_layer_name] = {
        'shape': first_layer.weight.shape,
        'total_weights': first_layer.weight.numel(),
        'importance_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0,
            'std': 0
        }
    }
    
    print(f"\n레이어 {first_layer_name}:")
    print(f"형태: {layer_stats[first_layer_name]['shape']}")
    print(f"가중치 수: {layer_stats[first_layer_name]['total_weights']}")
    
    def compute_accuracy(model, data_loader, stop_batch):
        """배치들에 대한 평균 정확도 계산"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(data_loader):
                if batch_idx >= stop_batch:
                    break
                
                data = data.to(device)
                labels = labels.to(device)
                
                outputs = model(data)
                
                # 각 태스크별 정확도 계산
                for task_idx, (start, end) in enumerate(output_cols_each_task):
                    task_outputs = outputs[:, start:end]
                    task_labels = labels[:, task_idx]
                    
                    # 예측값 계산
                    _, predicted = torch.max(task_outputs.data, 1)
                    
                    # 유효한 레이블에 대해서만 정확도 계산 (-1은 무시)
                    mask = task_labels != -1
                    task_total = mask.sum().item()
                    if task_total > 0:
                        task_correct = (predicted[mask] == task_labels[mask]).sum().item()
                        correct += task_correct
                        total += task_total
        
        return correct / total if total > 0 else 0
    
    # 원본 모델의 정확도 계산
    print("\n원본 모델의 정확도 계산 중...")
    original_accuracy = compute_accuracy(model, test_loader, stop_batch)
    print(f"원본 모델의 정확도: {original_accuracy:.4f}")
    
    # 각 가중치별 중요도 점수 계산
    importance_scores[first_layer_name] = torch.zeros_like(first_layer.weight)
    weight_shape = first_layer.weight.shape
    total_weights = first_layer.weight.numel()
    processed = 0
    layer_start_time = time.time()
    
    for idx in np.ndindex(weight_shape):
        # 진행률 표시
        processed += 1
        if processed % 1000 == 0:
            elapsed_time = time.time() - layer_start_time
            weights_per_second = processed / elapsed_time
            remaining_weights = total_weights - processed
            estimated_remaining_time = remaining_weights / weights_per_second
            
            hours = int(estimated_remaining_time // 3600)
            minutes = int((estimated_remaining_time % 3600) // 60)
            seconds = int(estimated_remaining_time % 60)
            
            print(f"진행률: {processed}/{total_weights} ({processed/total_weights*100:.2f}%) "
                  f"- 예상 남은 시간: {hours}시간 {minutes}분 {seconds}초")
        
        # 1. 해당 가중치만 0으로 설정
        temp_weight = first_layer.weight.data.clone()
        temp_weight[idx] = 0
        first_layer.weight.data = temp_weight
        
        # 2. 정확도 계산
        perturbed_accuracy = compute_accuracy(model, test_loader, stop_batch)
        
        # 3. 정확도 차이를 중요도 점수로 사용
        importance_scores[first_layer_name][idx] = abs(original_accuracy - perturbed_accuracy)
        
        # 4. 원래 가중치 복구
        first_layer.weight.data = original_weight.clone()
    
    # 결과 저장
    save_dir = 'experiment5'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 중요도 점수를 CPU로 이동하고 NumPy 배열로 변환
    importance_array = importance_scores[first_layer_name].cpu().numpy()
    
    # 통계 계산 및 저장 (이전과 동일)
    layer_stats[first_layer_name]['importance_stats'].update({
        'min': float(np.min(importance_array)),
        'max': float(np.max(importance_array)),
        'mean': float(np.mean(importance_array)),
        'std': float(np.std(importance_array))
    })
    
    # 파일 저장 및 시각화 (이전과 동일)
    np.save(os.path.join(save_dir, f'actual_importance_scores_{first_layer_name}.npy'), importance_array)
    
    plt.figure(figsize=(10, 5))
    plt.hist(importance_array.flatten(), bins=50)
    plt.title(f'Layer: {first_layer_name}\nActual Importance Score Distribution')
    plt.xlabel('Importance Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, f'importance_distribution_{first_layer_name}.png'))
    plt.close()
    
    with open(os.path.join(save_dir, 'actual_importance_scores_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 첫 번째 레이어의 실제 중요도 점수 분석 ===\n\n")
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        f.write(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초\n\n")
        
        f.write(f"원본 모델 정확도: {original_accuracy:.4f}\n\n")
        f.write(f"\n레이어: {first_layer_name}\n")
        f.write(f"형태: {layer_stats[first_layer_name]['shape']}\n")
        f.write(f"가중치 수: {layer_stats[first_layer_name]['total_weights']}\n")
        f.write("중요도 점수 통계:\n")
        f.write(f"  최소값: {layer_stats[first_layer_name]['importance_stats']['min']:.2e}\n")
        f.write(f"  최대값: {layer_stats[first_layer_name]['importance_stats']['max']:.2e}\n")
        f.write(f"  평균: {layer_stats[first_layer_name]['importance_stats']['mean']:.2e}\n")
        f.write(f"  표준편차: {layer_stats[first_layer_name]['importance_stats']['std']:.2e}\n")
    
    return importance_scores, layer_stats



def compute_actual_eo_scores(model, test_loader, output_cols_each_task, sensitive_idx, tau=50.):
    """
    첫 번째 레이어의 각 가중치에 대한 실제 EO 중요도를 계산
    중요도 = |원본 모델의 EO Loss - 가중치 비활성화 시의 EO Loss|
    """
    import time
    start_time = time.time()
    
    stop_batch = 10
    eo_scores = {}
    layer_stats = {}
    
    # 첫 번째 레이어 찾기
    first_layer = None
    first_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            first_layer = layer
            first_layer_name = name
            print("\n=== 첫 번째 레이어 정보 ===")
            print(f"레이어 이름: {first_layer_name}")
            print(f"레이어 타입: {type(first_layer).__name__}")
            print(f"형태: {first_layer.weight.shape}")
            print(f"가중치 수: {first_layer.weight.numel()}")
            break
    
    if first_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None, None
    
    # 원본 가중치 저장
    original_weight = first_layer.weight.data.clone()
    
    # 통계 정보 초기화
    layer_stats[first_layer_name] = {
        'shape': first_layer.weight.shape,
        'total_weights': first_layer.weight.numel(),
        'eo_stats': {
            'min': float('inf'),
            'max': float('-inf'),
            'mean': 0,
            'std': 0
        }
    }
    
    print(f"\n레이어 {first_layer_name}:")
    print(f"형태: {layer_stats[first_layer_name]['shape']}")
    print(f"가중치 수: {layer_stats[first_layer_name]['total_weights']}")
    
    # 원본 모델의 EO Loss 계산
    print("\n원본 모델의 EO Loss 계산 중...")
    original_eo_loss = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        if batch_idx >= stop_batch:
            break
            
        data = data.to(device)
        labels = labels.to(device)
        
        outputs = model(data)
        p = torch.sigmoid(outputs)
        g = labels[:, sensitive_idx].float()
        eo_loss = eo_loss_all(p, labels, g, tau=tau, output_cols_each_task=output_cols_each_task)
        original_eo_loss += eo_loss.item()
    
    original_eo_loss /= stop_batch
    print(f"원본 모델의 평균 EO Loss: {original_eo_loss:.4f}")
    
    # 각 가중치별 EO 중요도 점수 계산
    eo_scores[first_layer_name] = torch.zeros_like(first_layer.weight)
    weight_shape = first_layer.weight.shape
    total_weights = first_layer.weight.numel()
    processed = 0
    layer_start_time = time.time()
    
    for idx in np.ndindex(weight_shape):
        # 진행률 표시
        processed += 1
        if processed % 1000 == 0:
            elapsed_time = time.time() - layer_start_time
            weights_per_second = processed / elapsed_time
            remaining_weights = total_weights - processed
            estimated_remaining_time = remaining_weights / weights_per_second
            
            hours = int(estimated_remaining_time // 3600)
            minutes = int((estimated_remaining_time % 3600) // 60)
            seconds = int(estimated_remaining_time % 60)
            
            print(f"진행률: {processed}/{total_weights} ({processed/total_weights*100:.2f}%) "
                  f"- 예상 남은 시간: {hours}시간 {minutes}분 {seconds}초")
        
        # 1. 해당 가중치만 0으로 설정
        temp_weight = first_layer.weight.data.clone()
        temp_weight[idx] = 0
        first_layer.weight.data = temp_weight
        
        # 2. EO Loss 누적값 초기화
        total_eo_loss = 0
        
        # 3. stop_batch 개수만큼의 배치에 대해 예측
        for batch_idx, (data, labels) in enumerate(test_loader):
            if batch_idx >= stop_batch:
                break
                
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            p = torch.sigmoid(outputs)
            g = labels[:, sensitive_idx].float()
            eo_loss = eo_loss_all(p, labels, g, tau=tau, output_cols_each_task=output_cols_each_task)
            total_eo_loss += eo_loss.item()
        
        # 4. 평균 EO Loss와 원본 EO Loss의 차이를 중요도 점수로 사용
        perturbed_eo_loss = total_eo_loss / stop_batch
        eo_scores[first_layer_name][idx] = abs(perturbed_eo_loss - original_eo_loss)
        
        # 5. 원래 가중치 복구
        first_layer.weight.data = original_weight.clone()
    
    # 결과 저장
    save_dir = 'experiment8/REAL_EO_scores'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # EO 중요도 점수를 CPU로 이동하고 NumPy 배열로 변환
    eo_array = eo_scores[first_layer_name].cpu().numpy()
    
    # 통계 계산
    layer_stats[first_layer_name]['eo_stats'].update({
        'min': float(np.min(eo_array)),
        'max': float(np.max(eo_array)),
        'mean': float(np.mean(eo_array)),
        'std': float(np.std(eo_array))
    })
    
    # NumPy 파일로 저장
    np.save(os.path.join(save_dir, f'actual_eo_scores_{first_layer_name}.npy'), eo_array)
    
    # 분포 시각화
    plt.figure(figsize=(10, 5))
    plt.hist(eo_array.flatten(), bins=50)
    plt.title(f'Layer: {first_layer_name}\nActual EO Score Distribution')
    plt.xlabel('EO Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, f'eo_distribution_{first_layer_name}.png'))
    plt.close()
    
    # 통계 정보를 텍스트 파일로 저장
    with open(os.path.join(save_dir, 'actual_eo_scores_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 첫 번째 레이어의 실제 EO 중요도 점수 분석 ===\n\n")
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        f.write(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초\n\n")
        
        f.write(f"원본 모델의 EO Loss: {original_eo_loss:.4f}\n\n")
        f.write(f"\n레이어: {first_layer_name}\n")
        f.write(f"형태: {layer_stats[first_layer_name]['shape']}\n")
        f.write(f"가중치 수: {layer_stats[first_layer_name]['total_weights']}\n")
        f.write("EO 중요도 점수 통계:\n")
        f.write(f"  최소값: {layer_stats[first_layer_name]['eo_stats']['min']:.2e}\n")
        f.write(f"  최대값: {layer_stats[first_layer_name]['eo_stats']['max']:.2e}\n")
        f.write(f"  평균: {layer_stats[first_layer_name]['eo_stats']['mean']:.2e}\n")
        f.write(f"  표준편차: {layer_stats[first_layer_name]['eo_stats']['std']:.2e}\n")
    
    return eo_scores, layer_stats


def compute_approximate_gender_importance_scores_lastLayer(model, test_loader, output_cols_each_task):
    """
    마지막 레이어의 가중치에 대해 그래디언트 기반 중요도를 근사 계산
    (성별 분류 모델용)
    """
    import time
    start_time = time.time()
    
    stop_batch = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 마지막 레이어 찾기
    last_layer = None
    last_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            last_layer = layer
            last_layer_name = name
        # 마지막 레이어를 찾을 때까지 계속 업데이트
    
    print("\n=== 마지막 레이어 정보 ===")
    print(f"레이어 이름: {last_layer_name}")
    print(f"레이어 타입: {type(last_layer).__name__}")
    print(f"형태: {last_layer.weight.shape}")
    print(f"가중치 수: {last_layer.weight.numel()}")

    if last_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None, None
    
    # 누적 변수 초기화
    imp_grad_accum = torch.zeros_like(last_layer.weight)
    importance_score = torch.zeros_like(last_layer.weight)
    batches = 0
    
    print(f"\n{stop_batch}개 배치에 대해 근사 중요도 계산 중...")
    
    model.eval()  # 평가 모드로 설정
    
    for batch_idx, sample_batched in enumerate(test_loader):
        if batch_idx >= stop_batch:
            break
        
        batches += 1
        if batch_idx % 2 == 0:
            print(f"{batch_idx}/{stop_batch} 배치 처리 중...")
            
        image_batched, label_batched = sample_batched
        image_batched = image_batched.to(device, dtype=torch.float)
        gender_labels = label_batched[:, -1].long().to(device)
        
        optimizer.zero_grad()
        
        outputs = model(image_batched)
        loss = criterion(outputs, gender_labels)
        loss.backward()
        
        grads = last_layer.weight.grad.clone().detach()
        weights = last_layer.weight.data.clone().detach()
        
        hess = (grads*weights)**2
        
        imp_grad_accum += grads
        importance_score += hess
    
    imp_grad_accum /= batches
    importance_score /= batches
    
    # 파일 저장 경로 변경
    save_dir = 'experiment11/gender_last_layer_importance_grad'  # 경로 이름 변경
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    importance_array = importance_score.cpu().numpy()
    
    stats = {
        'min': float(np.min(importance_array)),
        'max': float(np.max(importance_array)),
        'mean': float(np.mean(importance_array)),
        'std': float(np.std(importance_array))
    }
    
    np.save(os.path.join(save_dir, f'gender_importance_scores_{last_layer_name}.npy'), 
            importance_array)
    
    plt.figure(figsize=(10, 5))
    plt.hist(importance_array.flatten(), bins=50)
    plt.title(f'Layer: {last_layer_name}\nGender Classification Importance Score Distribution')
    plt.xlabel('Importance Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, f'gender_importance_distribution_{last_layer_name}.png'))
    plt.close()
    
    with open(os.path.join(save_dir, 'gender_importance_scores_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 마지막 레이어의 성별 분류 중요도 점수 분석 ===\n\n")
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        f.write(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초\n\n")
        
        f.write(f"\n레이어: {last_layer_name}\n")
        f.write(f"형태: {last_layer.weight.shape}\n")
        f.write(f"가중치 수: {last_layer.weight.numel()}\n")
        f.write("중요도 점수 통계:\n")
        f.write(f"  최소값: {stats['min']:.2e}\n")
        f.write(f"  최대값: {stats['max']:.2e}\n")
        f.write(f"  평균: {stats['mean']:.2e}\n")
        f.write(f"  표준편차: {stats['std']:.2e}\n")
    
    return importance_score.cpu()


def compute_actual_gender_importance_scores_lastLayer(model, test_loader, output_cols_each_task):
    """
    마지막 레이어의 가중치를 하나씩 비활성화하면서 실제 중요도를 계산
    중요도 = 원본 손실 - 가중치 제거 후 손실
    """
    import time
    start_time = time.time()
    
    stop_batch = 10
    criterion = nn.CrossEntropyLoss()
    
    # 마지막 레이어 찾기
    last_layer = None
    last_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            last_layer = layer
            last_layer_name = name
            
    print("\n=== 마지막 레이어 정보 ===")
    print(f"레이어 이름: {last_layer_name}")
    print(f"레이어 타입: {type(last_layer).__name__}")
    print(f"형태: {last_layer.weight.shape}")
    print(f"가중치 수: {last_layer.weight.numel()}")
    
    if last_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None, None
    
    # 원본 가중치 저장
    original_weight = last_layer.weight.data.clone()
    
    # 원본 모델의 손실 계산
    original_total_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            if batch_idx >= stop_batch:
                break
            data = data.to(device)
            gender_labels = labels[:, -1].long().to(device)
            outputs = model(data)
            loss = criterion(outputs, gender_labels)
            original_total_loss += loss.item()
    original_loss = original_total_loss / stop_batch
    
    # 각 가중치별 중요도 점수 계산
    importance_score = torch.zeros_like(last_layer.weight)
    weight_shape = last_layer.weight.shape
    total_weights = last_layer.weight.numel()
    processed = 0
    layer_start_time = time.time()
    
    for idx in np.ndindex(weight_shape):
        processed += 1
        if processed % 1000 == 0:
            elapsed_time = time.time() - layer_start_time
            weights_per_second = processed / elapsed_time
            remaining_weights = total_weights - processed
            estimated_remaining_time = remaining_weights / weights_per_second
            
            hours = int(estimated_remaining_time // 3600)
            minutes = int((estimated_remaining_time % 3600) // 60)
            seconds = int(estimated_remaining_time % 60)
            
            print(f"진행률: {processed}/{total_weights} ({processed/total_weights*100:.2f}%) "
                  f"- 예상 남은 시간: {hours}시간 {minutes}분 {seconds}초")
        
        # 1. 해당 가중치만 0으로 설정
        temp_weight = last_layer.weight.data.clone()
        temp_weight[idx] = 0
        last_layer.weight.data = temp_weight
        
        # 2. 손실 누적값 초기화
        total_loss = 0
        
        # 3. stop_batch 개수만큼의 배치에 대해 예측
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(test_loader):
                if batch_idx >= stop_batch:
                    break
                data = data.to(device)
                gender_labels = labels[:, -1].long().to(device)
                outputs = model(data)
                loss = criterion(outputs, gender_labels)
                total_loss += loss.item()
        
        # 4. 원본 손실과의 차이를 중요도 점수로 사용
        modified_loss = total_loss / stop_batch
        importance_score[idx] = original_loss - modified_loss
        
        # 5. 원래 가중치 복구
        last_layer.weight.data = original_weight.clone()
    
    # 파일 저장 경로 변경
    save_dir = 'experiment11/gender_last_layer_actual_importance'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    importance_array = importance_score.cpu().numpy()
    
    stats = {
        'min': float(np.min(importance_array)),
        'max': float(np.max(importance_array)),
        'mean': float(np.mean(importance_array)),
        'std': float(np.std(importance_array))
    }
    
    np.save(os.path.join(save_dir, f'gender_actual_importance_scores_{last_layer_name}.npy'), 
            importance_array)
    
    plt.figure(figsize=(10, 5))
    plt.hist(importance_array.flatten(), bins=50)
    plt.title(f'Layer: {last_layer_name}\nActual Gender Classification Importance Score Distribution')
    plt.xlabel('Importance Score (Original Loss - Modified Loss)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, f'gender_actual_importance_distribution_{last_layer_name}.png'))
    plt.close()
    
    with open(os.path.join(save_dir, 'gender_actual_importance_scores_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 마지막 레이어의 실제 성별 분류 중요도 점수 분석 ===\n\n")
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        f.write(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초\n\n")
        
        f.write(f"\n레이어: {last_layer_name}\n")
        f.write(f"형태: {last_layer.weight.shape}\n")
        f.write(f"가중치 수: {last_layer.weight.numel()}\n")
        f.write("중요도 점수 통계:\n")
        f.write(f"  최소값: {stats['min']:.2e}\n")
        f.write(f"  최대값: {stats['max']:.2e}\n")
        f.write(f"  평균: {stats['mean']:.2e}\n")
        f.write(f"  표준편차: {stats['std']:.2e}\n")
    
    return importance_score.cpu()



def compute_approximate_gender_importance_scores_firstLayer(model, test_loader, output_cols_each_task):
    """
    첫 번째 레이어의 가중치에 대해 그래디언트 기반 중요도를 근사 계산
    (성별 분류 모델용)
    """
    import time
    start_time = time.time()
    
    stop_batch = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 첫 번째 레이어 찾기
    first_layer = None
    first_layer_name = None
    for name, layer in model.named_modules():
        if type(layer).__name__ in supported_layers:
            first_layer = layer
            first_layer_name = name
            break  # 첫 번째 레이어를 찾으면 종료
    
    print("\n=== 첫 번째 레이어 정보 ===")
    print(f"레이어 이름: {first_layer_name}")
    print(f"레이어 타입: {type(first_layer).__name__}")
    print(f"형태: {first_layer.weight.shape}")
    print(f"가중치 수: {first_layer.weight.numel()}")

    if first_layer is None:
        print("지원되는 레이어를 찾을 수 없습니다.")
        return None, None
    
    # 누적 변수 초기화
    imp_grad_accum = torch.zeros_like(first_layer.weight)
    importance_score = torch.zeros_like(first_layer.weight)
    batches = 0
    
    print(f"\n{stop_batch}개 배치에 대해 근사 중요도 계산 중...")
    
    model.eval()  # 평가 모드로 설정
    
    for batch_idx, sample_batched in enumerate(test_loader):
        if batch_idx >= stop_batch:
            break
        
        batches += 1
        if batch_idx % 2 == 0:
            print(f"{batch_idx}/{stop_batch} 배치 처리 중...")
            
        image_batched, label_batched = sample_batched
        image_batched = image_batched.to(device, dtype=torch.float)
        gender_labels = label_batched[:, -1].long().to(device)
        
        optimizer.zero_grad()
        
        outputs = model(image_batched)
        loss = criterion(outputs, gender_labels)
        loss.backward()
        
        grads = first_layer.weight.grad.clone().detach()
        weights = first_layer.weight.data.clone().detach()
        
        hess = (grads*weights)**2
        
        imp_grad_accum += grads
        importance_score += hess
    
    imp_grad_accum /= batches
    importance_score /= batches
    
    # 파일 저장 경로 변경
    save_dir = 'experiment11/gender_first_layer_importance_grad'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    importance_array = importance_score.cpu().numpy()
    
    stats = {
        'min': float(np.min(importance_array)),
        'max': float(np.max(importance_array)),
        'mean': float(np.mean(importance_array)),
        'std': float(np.std(importance_array))
    }
    
    np.save(os.path.join(save_dir, f'gender_importance_scores_{first_layer_name}.npy'), 
            importance_array)
    
    plt.figure(figsize=(10, 5))
    plt.hist(importance_array.flatten(), bins=50)
    plt.title(f'Layer: {first_layer_name}\nGender Classification Importance Score Distribution')
    plt.xlabel('Importance Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, f'gender_importance_distribution_{first_layer_name}.png'))
    plt.close()
    
    with open(os.path.join(save_dir, 'gender_importance_scores_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("=== 첫 번째 레이어의 성별 분류 중요도 점수 분석 ===\n\n")
        
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        f.write(f"총 소요 시간: {hours}시간 {minutes}분 {seconds}초\n\n")
        
        f.write(f"\n레이어: {first_layer_name}\n")
        f.write(f"형태: {first_layer.weight.shape}\n")
        f.write(f"가중치 수: {first_layer.weight.numel()}\n")
        f.write("중요도 점수 통계:\n")
        f.write(f"  최소값: {stats['min']:.2e}\n")
        f.write(f"  최대값: {stats['max']:.2e}\n")
        f.write(f"  평균: {stats['mean']:.2e}\n")
        f.write(f"  표준편차: {stats['std']:.2e}\n")
    
    return importance_score.cpu()
