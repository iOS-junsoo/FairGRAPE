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
import torchvision.models as models



# 사용자 정의 코드들
from train_and_val import loss_multi_tasks
from util import make_model, custom_forward_conv2d, custom_forward_conv1d, custom_forward_linear, safe_forward_with_cudnn_fallback
from dataset import split_image_name, make_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

supported_layers = ['Linear', 'Conv2d', 'Conv1d']

# impt_type == 1에서 사용할 성능-공정성 혼합 가중치.
# 사용자가 파일을 직접 열어 여기 값을 수정하면 됩니다.
IMPT_TYPE1_ALPHA = 0.0
IMPT_TYPE2_ALPHA = 0.9
IMPT_TYPE2_IMPORTANCE_BATCH_SIZE = 128
IMPT2_KEEP_PER_ITER = 0.9  # impt_type=2: 매 iteration마다 유지할 채널 가중치 비율
IMPT2_MIN_KEEP_RATIO_PER_LAYER = 0.1  # impt_type=2: 각 레이어가 원본 채널의 최소 10%는 유지


forward_mapping_dict = {
    'Linear': custom_forward_linear,
    'Conv2d': custom_forward_conv2d,
    'Conv1d': custom_forward_conv1d
}

#!###############
#* [Prune]
# - 역할: 모델을 실제로 프루닝하는 핵심 메서드
# - 입력
#   - prune_cfgs: 각종 프루닝 설정(예: 프루닝 비율, 민감 클래스 정보, 중요도 계산 방식 등)
#   - show_pruned_details: bool: True면 프루닝 요약 정보를 콘솔에 출력
# - 출력
#   - self.prun_model: 프루닝이 적용된 (가지치기 완료된) 모델 객체
#!###############
class Prunner:

    def __init__(self, model, criterion, dataloader, output_cols_each_task=None, save_mask=False):
        self.update_model(model)
        self.criterion = criterion.to(device)
        self.dataloader = dataloader
        self.output_cols_each_task=output_cols_each_task
        self.update_forward_pass()
        self.save_mask = save_mask

    def update_model(self, model):
        self.model = copy.deepcopy(model).to(device)
        self.prun_model = copy.deepcopy(model).to(device)

    def get_model(self):
        return self.model

    def init_mask(self):
        # 마스크 초기화
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))        

    # 모델 가중치에 마스크 적용 함수
    def apply_hook(self, masks):
        layers = filter(lambda l: type(l).__name__ in forward_mapping_dict, self.prun_model.modules()) # 모델 안의 모든 레이어 중에서 forward_mapping_dict에 해당하는 타입만 골라서 뽑아내는 역할
        def apply_masking(mask): # 마스크 적용
            def hook(weight):
                return weight * mask
            return hook
        
        # shape 검증 / 가중치에 마스크 적용 / 역전파 시에도 마스크 적용하기 위한 hook 등록
        for layer, mask in zip(layers, masks):
            assert layer.weight.shape == mask.shape
            layer.weight.data = layer.weight.data * mask
            layer.weight.register_hook(apply_masking(mask))
    # 가지치기 함수
    def prune(self, prune_cfgs, show_pruned_details=False):
        # 실제 마스크를 생성하는 함수 (get_mask)는 자식 클래스에서 구현
        masks = self.get_mask(prune_cfgs) # 마스크 생성
        if self.save_mask: # 마스크 저장(옵션)
            print("마스크 타입:", type(masks), "첫 번째 마스크 타입:", type(masks[0]))
            mask_np = np.array([m.cpu().numpy() for m in masks])
            np.save("mask.npy",mask_np)
            del mask_np
        self.apply_hook(masks) # 마스크 레이어에 적용
        mask_by_layer = {}

        """
        특정 타입의 레이어를 순회하며, 각각의 레이어에 대응하는 마스크를 nn.Parameter 형태로 등록하고, 추후 사용이나 확인을 위해 mask_by_layer 딕셔너리에 같이 저장
        """
        for layer in self.prun_model.modules(): # 모델에 있는 모든 레이어 순회
            if type(layer).__name__ in forward_mapping_dict: # type(layer).__name__이 forward_mapping_dict에 있는 타입만 선별
                layer.mask = nn.Parameter(masks.pop(0), requires_grad=False) # 해당 레이어에 대응되는 마스크를 꺼내기
                mask_by_layer[layer] = layer.mask # 어떤 레이어에 어떤 마스크가 사용되었는지 저장
        if show_pruned_details:
            self.print_remain()
        return self.prun_model

    def update_forward_pass(self):
        # forward 함수를 사용자 정의 함수로 교체
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.forward = types.MethodType(forward_mapping_dict[type(layer).__name__], layer)

    def variance_scaling_init(self):
        # Xavier 초기화
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False
    #!###############
    #* [print_remain]
    # - 역할: 프루닝이 끝난 후, 모델에 남아 있는(마스킹되지 않은) 파라미터 수를 출력.
    # - 입력
    #   - 없음(내부적으로 self.prun_model의 레이어를 순회)
    # - 출력
    #   - 콘솔에 “(레이어 이름, 남은 파라미터 수, 레이어 shape, 전체 대비 남은 비율)” 등을 출력하여 보여줌.
    #!###############
    def print_remain(self):
        # 남은 파라미터 정보 출력
        remain, total = 0, 0
        for name, layer in self.prun_model.named_modules():
            if type(layer).__name__ in forward_mapping_dict:
                remain += torch.sum(layer.mask)
                total += torch.prod(torch.tensor(layer.weight.shape))
                print(name, torch.sum(layer.mask), layer.weight.shape)
        print("남은 가중치 파라미터:", remain, "총 파라미터:", total, "비율:", remain/total)
#"""
class Random(Prunner):
    def __init__(self, model, criterion, dataloader, output_cols_each_task, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)

    def get_mask(self, prune_cfgs):
        compression_rate, by_layer = prune_cfgs
        masks = []
        if by_layer:
            for layer in self.prun_model.modules():
                mask = np.random.rand(layer.weight.shape)
                keep_params = int((1 - compression_rate) * math.prod(mask.shape))
                values, _ = torch.topk(mask, keep_params, sorted=True)
                threshold = values[-1]
                masks.append((mask > threshold).int())
        else:
            total_params = 0
            for layer in self.prun_model.modules():
                masks.append(np.random.rand(layer.weight.shape))
                total_params += math.prod(layer.weight.shape)
            keep_params = int((1 - compression_rate) * total_params)
            values, _ = torch.topk(masks, keep_params, sorted=True)
            threshold = values[-1]
            masks = [(mask > threshold).int() for mask in masks]
        return masks

class SNIP(Prunner): 
    def __init__(self, model, criterion, dataloader, output_cols_each_task, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)
         
    def get_mask(self, prune_cfgs):
        compression_factor, num_batch_sampling,init = prune_cfgs
        if init:
            self.variance_scaling_init()
        grads, grads_list = self.compute_grads(num_batch_sampling)
        keep_params = int((1 - compression_factor) * len(grads))
        values, idxs = torch.topk(grads / grads.sum(), keep_params, sorted=True)
        threshold = values[-1]
        masks = [(grad / grads.sum() > threshold).int() for grad in grads_list]
        return masks

    def compute_grads(self, num_batch_sampling=1):
        # SNIP에서 GRAD 계산
        moving_average_grads = 0
        for i, (data, labels) in enumerate(self.dataloader):
            if i == num_batch_sampling:
                break
            data, labels = data.to(device), labels.to(device)
            out = safe_forward_with_cudnn_fallback(self.model, data)
            loss = loss_multi_tasks(out, labels, self.criterion, self.output_cols_each_task, False)
            self.model.zero_grad()
            loss.backward()
            grads_list = []
            for layer in self.model.modules():
                if type(layer).__name__ in forward_mapping_dict:
                    grads_list.append(torch.abs(layer.mask.grad))
            grads = torch.cat([torch.flatten(grad) for grad in grads_list])
            if i == 0:
                moving_average_grads = grads
                moving_average_grad_list = grads_list
            else:
                moving_average_grads = ((moving_average_grads * i) + grads) / (i + 1)
                moving_average_grad_list = [((mv_avg_grad * i) + grad) / (i + 1)
                                            for mv_avg_grad, grad in zip(moving_average_grad_list, grads_list)]
        return moving_average_grads, moving_average_grad_list


############
# GraSP 코드 (github 참고)
############
class GraSP(Prunner): 
    def __init__(self, model, criterion, dataloader, output_cols_each_task=None, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)

    def count_total_parameters(self,net):
        # 전체 파라미터 수 계산
        total = 0
        for m in net.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                total += m.weight.numel()
        return total

    def count_fc_parameters(self,net):
        # FC 레이어 파라미터 수 계산
        total = 0
        for m in net.modules():
            if isinstance(m, (nn.Linear)):
                total += m.weight.numel()
        return total

    def GraSP_fetch_data(self, dataloader, num_classes, samples_per_class, target_col=0):
        # 클래스별로 일정 개수의 샘플을 수집
        datas = [[] for _ in range(num_classes)]
        labels = [[] for _ in range(num_classes)]
        mark = dict()
        dataloader_iter = iter(dataloader)
        
        while True:
            inputs, targets = next(dataloader_iter)
            for idx in range(inputs.shape[0]):
                x, y = inputs[idx:idx+1], targets[idx:idx+1]
                if isinstance(target_col, int):
                    category = y[0,target_col] 
                else:
                    # CelebA 같은 경우, 여러 컬럼 중 양성이면 그 클래스로 간주
                    category = -1
                    for target_i in target_col:
                        label = y[0,target_i].item()
                        if label == 1 and target_i not in mark:
                            category = target_i
                            break
                category = category.item() if not isinstance(category, int) else category
                if category == -1:
                    continue
                if len(datas[category]) == samples_per_class:
                    mark[category] = True
                    continue
                datas[category].append(x)
                labels[category].append(y)
            print("수집 완료된 클래스 수:", len(mark))
            if len(mark) == num_classes:
                break

        X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels])
        return X, y

    def get_mask(self, prune_cfgs, num_iters=1, T=200, reinit=True, fair_grad = False):
        # GraSP 알고리즘으로 마스크 계산
        ratio,target_col, num_classes, samples_per_class= prune_cfgs
        net = self.model
        train_dataloader = self.dataloader
        output_cols_each_task= self.output_cols_each_task
        eps = 1e-10
        keep_ratio = 1-ratio
        old_net = net
        criterion = F.cross_entropy

        net = copy.deepcopy(net)  
        net.zero_grad()

        weights = []
        total_parameters = self.count_total_parameters(net)
        fc_parameters = self.count_fc_parameters(net)

        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if isinstance(layer, nn.Linear) and reinit:
                    nn.init.xavier_normal(layer.weight)
                weights.append(layer.weight)
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))
                

        inputs_one = []
        targets_one = []

        grad_w = None
        for w in weights:
            w.requires_grad_(True)

        print_once = False
        for it in range(num_iters):
            print("(1): %d/%d 번째 반복." % (it, num_iters))
            inputs, targets = self.GraSP_fetch_data(train_dataloader, num_classes, samples_per_class, target_col)
            N = inputs.shape[0]
            din = copy.deepcopy(inputs)
            dtarget = copy.deepcopy(targets)
            inputs_one.append(din[:N//2])
            targets_one.append(dtarget[:N//2])
            inputs_one.append(din[N // 2:])
            targets_one.append(dtarget[N // 2:])
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = net.forward(inputs[:N//2])/T
            if print_once:
                x = F.softmax(outputs)
                print(x)
                print("최댓값:", x.max(), "최솟값:", x.min())
                print_once = False
            loss = loss_multi_tasks(outputs, targets[:N//2], criterion, output_cols_each_task)
            grad_w_p = autograd.grad(loss, weights)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

            outputs = net.forward(inputs[N // 2:])/T
            loss = loss_multi_tasks(outputs, targets[N//2:], criterion, output_cols_each_task)
            grad_w_p = autograd.grad(loss, weights, create_graph=False)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]

        ret_inputs = []
        ret_targets = []

        for it in range(len(inputs_one)):
            print("(2): %d/%d 번째 반복." % (it, num_iters))
            inputs = inputs_one.pop(0).to(device)
            targets = targets_one.pop(0).to(device)
            ret_inputs.append(inputs)
            ret_targets.append(targets)
            outputs = net.forward(inputs)/T
            loss = loss_multi_tasks(outputs, targets, criterion, output_cols_each_task)
            grad_f = autograd.grad(loss, weights, create_graph=True)
            z = 0
            count = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    z += (grad_w[count].data * grad_f[count]).sum()
                    count += 1
            z.backward()

        grads = dict()
        old_modules = list(old_net.modules())
        selected_layers = []
        for idx, (name, layer) in enumerate(net.named_modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grad = -layer.weight.data * layer.weight.grad
                if fair_grad:
                    grad += 0
                grads[old_modules[idx]] = grad
                selected_layers.append(idx)

        # 모든 점수(gradient importance)를 하나로 모아 정규화
        all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
        norm_factor = torch.abs(torch.sum(all_scores)) + eps
        print("** 정규화 계수:", norm_factor)
        all_scores.div_(norm_factor)

        num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
        threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
        acceptable_score = threshold[-1]
        print('** 허용 임계값: ', acceptable_score)
        keep_masks = dict()    
        for m, g in grads.items():
            keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()
        
        # 위 코드는 GraSP 레포에서 가져온 것입니다. 아래에는 mask를 prunner를 위한 리스트로 변환합니다.
        mask_list = []
        for idx, (name, layer) in enumerate(net.named_modules()):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                mask_list.append(keep_masks[old_modules[idx]])
            elif type(layer).__name__ in forward_mapping_dict:
                mask_list.append(layer.mask)

        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))
        return mask_list

######################
# Deep compression 코드
# 만약 s 가 숫자면 모든 레이어에 동일 민감도 사용
# s 가 dict 면 각 레이어마다 다르게 사용
#######################
class WS(Prunner): 
    def __init__(self, model, criterion, dataloader, output_cols_each_task=None, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)

    def get_mask(self, prune_cfgs):
        # Deep compression 스타일 프루닝
        net = self.model
        pruning_rate=prune_cfgs[0]
        masks = []
        for name, module in net.named_modules():
            if type(module).__name__ not in forward_mapping_dict:
                continue
            s = pruning_rate[name] if isinstance(pruning_rate, dict) else pruning_rate
            n_to_kepp = int(torch.prod(torch.tensor(module.weight.shape)) * (1-s))
            threshold = torch.topk(module.weight.data.cpu().view(-1).abs(), n_to_kepp, sorted=True)[0][-1]
            mask = np.where(abs(module.weight.cpu().abs()) <= threshold, 0, 1)
            mask = torch.tensor(mask).to(device)
            masks.append(mask)

        return masks

######################
# Lottery selection
# 초기 상태를 저장하고, 프루닝 후 이를 다시 세팅
######################
class Lottery(Prunner): 
    def __init__(self, model, criterion, dataloader, output_cols_each_task=None, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)
        self.initial_model = copy.deepcopy(model)

    def get_mask(self, prune_cfgs):
        net = self.model
        pruning_rate=prune_cfgs[0]
        masks = []
        for name, module in net.named_modules():
            if type(module).__name__ not in forward_mapping_dict:
                continue
            s = pruning_rate[name] if isinstance(pruning_rate, dict) else pruning_rate
            n_to_kepp = int(torch.prod(torch.tensor(module.weight.shape)) * (1-s))
            threshold = torch.topk(module.weight.data.cpu().view(-1).abs(), n_to_kepp, sorted=True)[0][-1]
            print(f'레이어 {name}에 대한 가중치 프루닝 임계값: {threshold}')
            mask = np.where(abs(module.weight.cpu().abs()) <= threshold, 0, 1)
            mask = torch.tensor(mask).to(device)
            masks.append(mask)
        # Lottery 특성상 모델을 초기 상태로 리셋
        self.update_model(self.initial_model)

        return masks

############################
# 중요도 기반 추정 (민감 집단 고려 X)
############################
class Importance(Prunner): 
    def __init__(self, model, criterion, dataloader, output_cols_each_task, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)
        self.init_mask()

    def init_mask(self):
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))

    def get_mask(self, prune_cfgs):
        prune_ratio, test_csv, new_img_dir, _, masked_grads, output_cols_each_task ,col_used, _,_, stop_batch, _ = prune_cfgs
        masks = []
        _,impts = importance_by_class0(self.model, test_csv, new_img_dir, masked_grads,output_cols_each_task,col_used,stop_batch=stop_batch)
        for name,layer in self.model.named_modules():
            if name not in impts[0]:
                continue
            impt = impts[0][name]
            keep_params = int((1 - prune_ratio) * math.prod(impt.shape))
            print(name, impt.shape, prune_ratio, keep_params)
            values, _ = torch.topk(impt.view(-1), keep_params, sorted=True)
            threshold = values[-1]
            masks.append((impt  > threshold).int().to(device))
        return masks
#"""
############################
# Fairness selection
############################
class FairGRAPE(Prunner): 
    def __init__(self, model, criterion, dataloader, output_cols_each_task, save_mask=False):
        super().__init__(model, criterion, dataloader, output_cols_each_task, save_mask)
        self.init_mask()

    def init_mask(self):
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))
    
    #!###############
    #* [get_mask]
    # - 역할: 프루닝 규칙에 따라 레이어별 마스크를 생성하는 함수(FairGRAPE: 민감 그룹 사이의 fairness를 고려한 방식)
    # - 입력
    #   - prune_cfgs: 각종 프루닝 설정(예: 프루닝 비율, 민감 클래스 정보, 중요도 계산 방식 등)
    # - 출력
    #   - mask: “프루닝 가능한 레이어 개수”만큼의 텐서 리스트를 반환
    #!###############
    def get_mask(self, prune_cfgs):
        prune_ratio, test_csv, new_img_dir, sensitive_classes, masked_grads, output_cols_each_task ,col_used, para_batch, impt_type, stop_batch, delta_p, network, sensitive_group = prune_cfgs
        print("-------------get_mask-------------")
        print("prune_ratio:", prune_ratio)
        print("sensitive_classes:", sensitive_classes)
        print("impt_type:", impt_type)
        print("sensitive_group:", sensitive_group)
        print("stop_batch:", stop_batch)
        
        if impt_type == 2:
            print("alpha:", IMPT_TYPE2_ALPHA)
        else:
            print("alpha:", IMPT_TYPE1_ALPHA)

        mask = fairness_grad(self.model, prune_ratio, test_csv, new_img_dir, sensitive_classes, masked_grads, output_cols_each_task ,col_names=col_used, para_batch=para_batch, impt_type=impt_type, stop_batch=stop_batch, delta_p=delta_p, network=network, sensitive_group=sensitive_group)
        return mask

#!###############
#* [fairness_grad]
# - 역할: 민감 클래스(예: 성별, 인종) 간에 gradient 분포가 어느 한쪽에 치우치지 않도록 프루닝 마스크를 만드는 함수.
# - 입력
#   - model: 프루닝 대상 모델
#   - prune_ratio: 프루닝 비율
#   - test_csv, new_img_dir: 데이터/이미지 경로
#   - sensitive_classes: 민감 그룹(예: 2 → 남/여, 7 → 7개 인종 등)
# - 출력
#   - mask: “프루닝 가능한 레이어 개수”만큼의 텐서 리스트를 반환
#!###############
def unravel_index(indices: torch.Tensor, shape):
    """
    NumPy의 unravel_index를 PyTorch <2.2 환경에서 흉내 낸 버전.
    indices : 1-D 혹은 N-D 정수 텐서
    shape   : 원본 텐서 모양(튜플/리스트)
    반환    : 각 차원 좌표를 담은 튜플
    """
    coords = []
    for dim in reversed(shape):
        coords.append(indices % dim)
        indices = indices // dim
    return tuple(reversed(coords))

import torch
import torch.nn as nn

def load_gender_model_from_ckpt(base_model, ckpt_path, device='cuda'):
    """
    base_model: 원래 39-way 모델 인스턴스(같은 아키텍처)
    ckpt_path : 'gender_model/best_gender_model.pth'
    반환: gender_model (2-way 헤드), load_info(누락/예상치못한 키 로그)
    """
    # 1) 백본 동일하게 시작: deepcopy 대신 '동일 아키텍처 새 인스턴스'를 권장 (메모리 절약)
    #    이미 base_model이 있으니 복사 없이 그대로 재사용해서 '성별용 모델'을 만듭니다.
    gender_model = type(base_model)() if hasattr(type(base_model), '__call__') else None
    if gender_model is None:
        gender_model = models.mobilenet_v2()  # base_model이 mobilenet_v2인 경우
    # 2) 헤드 2-way로 교체 (MobileNetV2 기준: classifier[1])
    in_f = gender_model.classifier[1].in_features
    gender_model.classifier[1] = nn.Linear(in_f, 2)

    # 3) 체크포인트 로드 (mask/형상 불일치 키 정리)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['model_state_dict']

    # 3-1) .mask 버퍼 제거
    sd = {k: v for k, v in sd.items() if not k.endswith('.mask')}

    # 3-2) 현재 모델 헤드와 충돌하는 키 제거 (2-way 보장)
    for k in ['classifier.1.weight', 'classifier.1.bias']:
        if k in sd and sd[k].shape != gender_model.state_dict()[k].shape:
            sd.pop(k)

    # 3-3) state_dict 로드
    load_info = gender_model.load_state_dict(sd, strict=False)

    # 4) 디바이스 & 모드
    gender_model.to(device)
    gender_model.eval()  # 추론/점수 계산 용도면 eval, 학습 재개면 train()

    # 5) 최종 헤드 차원 확인
    assert tuple(gender_model.classifier[1].weight.shape)[0] == 2, "헤드가 2-way가 아닙니다."

    return gender_model, load_info



from sklearn.cluster import KMeans
import numpy as np

def is_classifier_head(name: str) -> bool:
        return name.startswith('classifier.')  # mobilenetv2 최종 헤드 포함

def cluster_weights_by_gender_imp(model, gender_score):
    """
    레이어별로 gender importance 기준 클러스터링
    """
    clusters_by_layer = {}
    
    for name, layer in model.named_modules():
        if type(layer).__name__ not in supported_layers or is_classifier_head(name):
            continue
            
        if name not in gender_score:
            continue
            
        # gender_imp를 1차원으로 펼치기
        gender_imp = gender_score[name].flatten().numpy()
        n_weights = len(gender_imp)
        
        # 클러스터 수 동적 결정 (예: 가중치 수에 따라)
        n_clusters = min(int(np.sqrt(n_weights)), 10)  # 최대 10개 
        
        # KMeans 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(gender_imp.reshape(-1, 1))
        
        clusters_by_layer[name] = {
            'labels': cluster_labels,
            'scores': gender_imp,
            'n_clusters': n_clusters
        }
    
    return clusters_by_layer



def select_representatives(clusters_by_layer):
    """
    각 클러스터에서 gender_imp가 가장 큰 가중치를 대표로 선정
    """
    representatives = {}
    
    for name, cluster_info in clusters_by_layer.items():
        layer_reps = []
        scores = cluster_info['scores']
        labels = cluster_info['labels']
        
        for i in range(cluster_info['n_clusters']):
            cluster_mask = (labels == i)
            cluster_scores = scores[cluster_mask]
            if len(cluster_scores) == 0:
                continue
                
            # 가장 큰 gradient를 가진 가중치의 인덱스 찾기
            rep_idx = np.where(cluster_mask)[0][cluster_scores.argmax()]
            rep_score = scores[rep_idx]
            layer_reps.append((i, rep_idx, rep_score))
            
        representatives[name] = layer_reps
        
    return representatives

def compute_eo_impact(model, test_loader, layer_name, weight_idx):
    """
    특정 가중치를 0으로 만들었을 때의 EO 변화량 계산
    """
    original_eo = compute_model_eo(model, test_loader)
    
    # 임시로 가중치를 0으로 만들기
    for name, layer in model.named_modules():
        if name == layer_name:
            original_weight = layer.weight.data.clone()
            temp_weight = original_weight.clone()
            temp_weight.flatten()[weight_idx] = 0
            layer.weight.data = temp_weight
            
            # EO 변화량 계산
            new_eo = compute_model_eo(model, test_loader)
            eo_impact = abs(original_eo - new_eo)
            
            # 가중치 복구
            layer.weight.data = original_weight
            return eo_impact
            
    return 0.0

def compute_model_eo(model, test_loader, sensitive_idx, stop_batch=10):
    """
    모델의 Equal Opportunity (EO) 메트릭을 계산하는 함수
    
    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터 로더
        sensitive_idx: 민감 속성(예: gender) 인덱스 (기본값: 7)
        stop_batch: 평가할 배치 수 (기본값: 10)
    
    Returns:
        float: EO 점수 (TPR 차이와 FPR 차이의 평균)
    """
    model.eval()  # 평가 모드로 설정
    
    # 그룹별 통계 초기화
    stats = {
        0: {'tp': 0, 'fp': 0, 'total_p': 0, 'total_n': 0},  # 그룹 0 (예: 남성)
        1: {'tp': 0, 'fp': 0, 'total_p': 0, 'total_n': 0}   # 그룹 1 (예: 여성)
    }
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(test_loader):
            if batch_idx >= stop_batch:
                break
                
            data = data.to(device)
            labels = labels.to(device)
            
            # 예측 수행
            outputs = safe_forward_with_cudnn_fallback(model, data)
            probs = torch.sigmoid(outputs)  # sigmoid 적용
            preds = (probs > 0.5).float()  # 임계값 0.5로 이진 분류
            
            # 민감 속성(그룹) 및 실제 레이블
            g = labels[:, sensitive_idx]
            y_true = labels[:, 0]  # 첫 번째 태스크만 고려
            
            # 각 그룹별로 통계 수집
            for g_val in [0, 1]:
                group_mask = (g == g_val)
                if not group_mask.any():
                    continue
                    
                y_true_group = y_true[group_mask]
                y_pred_group = preds[group_mask, 0]  # 첫 번째 태스크의 예측값
                
                # True Positives
                tp = ((y_pred_group == 1) & (y_true_group == 1)).sum().item()
                # False Positives
                fp = ((y_pred_group == 1) & (y_true_group == 0)).sum().item()
                # 전체 Positive 샘플 수
                total_p = (y_true_group == 1).sum().item()
                # 전체 Negative 샘플 수
                total_n = (y_true_group == 0).sum().item()
                
                # 통계 누적
                stats[g_val]['tp'] += tp
                stats[g_val]['fp'] += fp
                stats[g_val]['total_p'] += total_p
                stats[g_val]['total_n'] += total_n
    
    # 각 그룹의 TPR과 FPR 계산
    tpr = {}
    fpr = {}
    for g_val in [0, 1]:
        # 0으로 나누기 방지를 위한 epsilon
        eps = 1e-7
        
        # True Positive Rate (TPR) = TP / (TP + FN) = TP / total_positives
        tpr[g_val] = stats[g_val]['tp'] / (stats[g_val]['total_p'] + eps)
        
        # False Positive Rate (FPR) = FP / (FP + TN) = FP / total_negatives
        fpr[g_val] = stats[g_val]['fp'] / (stats[g_val]['total_n'] + eps)
    
    # EO 점수 계산: TPR 차이와 FPR 차이의 평균
    eo_score = (abs(tpr[0] - tpr[1]) + abs(fpr[0] - fpr[1])) / 2.0
    
    # 상세 정보 출력 (디버깅용)
    if False:  # 필요시 True로 변경
        print("\nEqual Opportunity 분석:")
        print(f"그룹 0 - TPR: {tpr[0]:.4f}, FPR: {fpr[0]:.4f}")
        print(f"그룹 1 - TPR: {tpr[1]:.4f}, FPR: {fpr[1]:.4f}")
        print(f"|TPR 차이|: {abs(tpr[0] - tpr[1]):.4f}")
        print(f"|FPR 차이|: {abs(fpr[0] - fpr[1]):.4f}")
        print(f"EO 점수: {eo_score:.4f}")
    
    return eo_score


def distribute_eo_impact(clusters_by_layer, representatives, gender_score):
    """
    대표 가중치의 EO 영향도를 클러스터 내 다른 가중치들에게 분배
    """
    cluster_eo_score = {}
    
    for name in clusters_by_layer.keys():
        scores = gender_score[name].flatten()
        labels = clusters_by_layer[name]['labels']
        layer_eo = torch.zeros_like(scores)
        
        for cluster_id, rep_idx, rep_score in representatives[name]:
            cluster_mask = (labels == cluster_id)
            cluster_weights = scores[cluster_mask]
            
            # 대표 가중치 대비 상대적인 비율로 EO 영향도 분배
            ratios = cluster_weights / rep_score
            distributed_eo = ratios * representatives[name][cluster_id][2]  # rep의 EO 값
            layer_eo[cluster_mask] = torch.tensor(distributed_eo)
            
        cluster_eo_score[name] = layer_eo.view_as(gender_score[name])
        
    return cluster_eo_score


import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_and_save_scores(imp_scaled_dict, gen_scaled_dict, output_dir='score_analysis'):
    """
    imp_scaled와 gen_scaled의 통계 분석 및 히스토그램 저장
    
    Args:
        imp_scaled_dict: {layer_name: imp_scaled_tensor}
        gen_scaled_dict: {layer_name: gen_scaled_tensor}
        output_dir: 저장 경로
    """
    # 출력 디렉토리 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # 1️⃣ 통계 분석 (텍스트 파일 저장)
    # ========================================
    stats_file = Path(output_dir) / 'score_statistics.txt'
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Score Statistics Analysis\n")
        f.write("="*80 + "\n\n")
        
        # Performance Score (imp_scaled) 통계
        f.write("📊 Performance Score (imp_scaled) Statistics\n")
        f.write("-"*80 + "\n")
        
        all_imp_values = []
        for name, tensor in imp_scaled_dict.items():
            values = tensor.flatten().cpu().numpy()
            all_imp_values.extend(values)
            
            f.write(f"\n[Layer: {name}]\n")
            f.write(f"  Shape:        {list(tensor.shape)}\n")
            f.write(f"  Min:          {np.min(values):.6f}\n")
            f.write(f"  Max:          {np.max(values):.6f}\n")
            f.write(f"  Mean:         {np.mean(values):.6f}\n")
            f.write(f"  Std:          {np.std(values):.6f}\n")
            f.write(f"  25% (Q1):     {np.percentile(values, 25):.6f}\n")
            f.write(f"  50% (Median): {np.percentile(values, 50):.6f}\n")
            f.write(f"  75% (Q3):     {np.percentile(values, 75):.6f}\n")
        
        # 전체 통계
        all_imp_values = np.array(all_imp_values)
        f.write("\n" + "="*80 + "\n")
        f.write("[Overall Performance Score Statistics]\n")
        f.write(f"  Total values: {len(all_imp_values)}\n")
        f.write(f"  Min:          {np.min(all_imp_values):.6f}\n")
        f.write(f"  Max:          {np.max(all_imp_values):.6f}\n")
        f.write(f"  Mean:         {np.mean(all_imp_values):.6f}\n")
        f.write(f"  Std:          {np.std(all_imp_values):.6f}\n")
        f.write(f"  25% (Q1):     {np.percentile(all_imp_values, 25):.6f}\n")
        f.write(f"  50% (Median): {np.percentile(all_imp_values, 50):.6f}\n")
        f.write(f"  75% (Q3):     {np.percentile(all_imp_values, 75):.6f}\n")
        
        # Fairness Score (gen_scaled) 통계
        f.write("\n\n" + "="*80 + "\n")
        f.write("📊 Fairness Score (gen_scaled) Statistics\n")
        f.write("-"*80 + "\n")
        
        all_gen_values = []
        for name, tensor in gen_scaled_dict.items():
            values = tensor.flatten().cpu().numpy()
            all_gen_values.extend(values)
            
            f.write(f"\n[Layer: {name}]\n")
            f.write(f"  Shape:        {list(tensor.shape)}\n")
            f.write(f"  Min:          {np.min(values):.6f}\n")
            f.write(f"  Max:          {np.max(values):.6f}\n")
            f.write(f"  Mean:         {np.mean(values):.6f}\n")
            f.write(f"  Std:          {np.std(values):.6f}\n")
            f.write(f"  25% (Q1):     {np.percentile(values, 25):.6f}\n")
            f.write(f"  50% (Median): {np.percentile(values, 50):.6f}\n")
            f.write(f"  75% (Q3):     {np.percentile(values, 75):.6f}\n")
        
        # 전체 통계
        all_gen_values = np.array(all_gen_values)
        f.write("\n" + "="*80 + "\n")
        f.write("[Overall Fairness Score Statistics]\n")
        f.write(f"  Total values: {len(all_gen_values)}\n")
        f.write(f"  Min:          {np.min(all_gen_values):.6f}\n")
        f.write(f"  Max:          {np.max(all_gen_values):.6f}\n")
        f.write(f"  Mean:         {np.mean(all_gen_values):.6f}\n")
        f.write(f"  Std:          {np.std(all_gen_values):.6f}\n")
        f.write(f"  25% (Q1):     {np.percentile(all_gen_values, 25):.6f}\n")
        f.write(f"  50% (Median): {np.percentile(all_gen_values, 50):.6f}\n")
        f.write(f"  75% (Q3):     {np.percentile(all_gen_values, 75):.6f}\n")
    
    print(f"✅ 통계 분석 완료: {stats_file}")
    
    # ========================================
    # 2️⃣ 히스토그램 생성 및 저장
    # ========================================
    
    # Performance Score 히스토그램
    plt.figure(figsize=(12, 6))
    plt.hist(all_imp_values, bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Performance Score (imp_scaled)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Performance Score (imp_scaled)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 통계 정보 텍스트 추가
    stats_text = f'Mean: {np.mean(all_imp_values):.4f}\n'
    stats_text += f'Std: {np.std(all_imp_values):.4f}\n'
    stats_text += f'Median: {np.median(all_imp_values):.4f}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    imp_hist_path = Path(output_dir) / 'performance_score_histogram.png'
    plt.tight_layout()
    plt.savefig(imp_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Performance Score 히스토그램 저장: {imp_hist_path}")
    
    # Fairness Score 히스토그램
    plt.figure(figsize=(12, 6))
    plt.hist(all_gen_values, bins=100, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Fairness Score (gen_scaled)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Fairness Score (gen_scaled)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 통계 정보 텍스트 추가
    stats_text = f'Mean: {np.mean(all_gen_values):.4f}\n'
    stats_text += f'Std: {np.std(all_gen_values):.4f}\n'
    stats_text += f'Median: {np.median(all_gen_values):.4f}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    gen_hist_path = Path(output_dir) / 'fairness_score_histogram.png'
    plt.tight_layout()
    plt.savefig(gen_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Fairness Score 히스토그램 저장: {gen_hist_path}")
    
    # 두 분포를 겹쳐서 비교하는 히스토그램
    plt.figure(figsize=(14, 6))
    plt.hist(all_imp_values, bins=100, alpha=0.5, color='blue', label='Performance Score', edgecolor='black')
    plt.hist(all_gen_values, bins=100, alpha=0.5, color='red', label='Fairness Score', edgecolor='black')
    plt.xlabel('Score Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Performance vs Fairness Score Distribution', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    combined_hist_path = Path(output_dir) / 'combined_score_histogram.png'
    plt.tight_layout()
    plt.savefig(combined_hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 결합 히스토그램 저장: {combined_hist_path}")
    
    return stats_file, imp_hist_path, gen_hist_path, combined_hist_path


def compute_gender_gradient_gap_importance(model, test_csv, new_img_dir=None, masked_grads=True, output_cols_each_task=[(0,7),(7,9),(9,18)], col_names=['race','gender'], stop_batch=10000, sensitive_group=None):
    print("-------------compute_gender_gradient_gap_importance-------------")

    if sensitive_group not in (None, 'gender'):
        raise NotImplementedError("impt_type == 1 은 현재 gender 기준 그룹 분리만 지원합니다.")

    if col_names[-1] != 'gender':
        raise ValueError(f"impt_type == 1 은 마지막 라벨 컬럼이 gender 여야 합니다. 현재: {col_names[-1]}")

    if isinstance(test_csv, str):
        test_frame = pd.read_csv(test_csv)
    else:
        test_frame = test_csv.copy()

    if new_img_dir:
        faces = set(os.listdir(new_img_dir))
        new_face_name = []
        face_found_mask = []
        for i in range(test_frame.shape[0]):
            face_name_align = split_image_name(test_frame['face_name_align'][i])
            face_found_mask.append(face_name_align in faces)
            if face_name_align in faces:
                new_face_name.append(os.path.join(new_img_dir, face_name_align))
        test_frame = test_frame[face_found_mask].reset_index(drop=True)
        test_frame['face_name_align'] = new_face_name

    batch_size = 384
    test_loader, _ = make_datasets(test_frame, test_frame, True, batch_size, col_used=col_names)
    criterion = nn.CrossEntropyLoss()

    print(f"gender gap 설정: batch_size={batch_size}, stop_batch={stop_batch}, total_batches={len(test_loader)}")
    print(f"gender gap 라벨 컬럼: {col_names}")

    grad_abs_sums = {0: {}, 1: {}}
    group_sample_counts = {0: 0, 1: 0}

    model.eval()

    prev_cudnn_enabled = torch.backends.cudnn.enabled
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    try:
        for batch_idx, sample_batched in enumerate(test_loader):
            if batch_idx >= stop_batch:
                break

            if batch_idx % 100 == 0:
                print(f"{batch_idx}번째 mini-batch gender grad gap 계산 중! 현재 group counts={group_sample_counts}")

            image_batched, label_batched = sample_batched
            image_batched = image_batched.to(device, dtype=torch.float, non_blocking=True).contiguous()
            label_batched = label_batched.to(device, non_blocking=True)
            gender_labels = label_batched[:, -1].long()

            for group_value in [0, 1]:
                group_mask = gender_labels == group_value
                group_size = int(group_mask.sum().item())
                if group_size == 0:
                    continue

                if batch_idx < 3 or batch_idx % 100 == 0:
                    print(f"  - batch {batch_idx}, gender={group_value}, samples={group_size}")

                model.zero_grad(set_to_none=True)
                with torch.backends.cudnn.flags(enabled=False):
                    outputs = safe_forward_with_cudnn_fallback(model, image_batched)

                group_outputs = outputs[group_mask]
                group_targets = label_batched[group_mask]
                group_loss = loss_multi_tasks(group_outputs, group_targets, criterion, output_cols_each_task)
                group_loss.backward()

                group_sample_counts[group_value] += group_size

                for name, layer in model.named_modules():
                    if type(layer).__name__ not in supported_layers:
                        continue
                    if not hasattr(layer, 'weight') or layer.weight.grad is None:
                        continue

                    grad_abs = layer.weight.grad.detach().abs().clone()
                    if masked_grads and hasattr(layer, 'mask'):
                        mask = layer.mask.detach().clone()
                        if mask.device != grad_abs.device:
                            mask = mask.to(grad_abs.device)
                        grad_abs = grad_abs * mask

                    weighted_grad_abs = grad_abs * group_size
                    if name not in grad_abs_sums[group_value]:
                        grad_abs_sums[group_value][name] = weighted_grad_abs
                    else:
                        grad_abs_sums[group_value][name] += weighted_grad_abs
    finally:
        torch.backends.cudnn.enabled = prev_cudnn_enabled
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark

    if group_sample_counts[0] == 0 or group_sample_counts[1] == 0:
        raise RuntimeError(f"gender group 샘플이 부족합니다. group counts: {group_sample_counts}")

    fairness_score = {}
    common_names = set(grad_abs_sums[0].keys()) & set(grad_abs_sums[1].keys())
    print(f"gender grad gap 계산 완료: group counts={group_sample_counts}, common layers={len(common_names)}")

    for name in common_names:
        if is_classifier_head(name):
            continue
        avg_group0 = grad_abs_sums[0][name] / group_sample_counts[0]
        avg_group1 = grad_abs_sums[1][name] / group_sample_counts[1]
        fairness_score[name] = torch.abs(avg_group0 - avg_group1)

    print("gender grad gap 레이어별 요약:")
    for name in sorted(fairness_score.keys()):
        score = fairness_score[name]
        print(
            f"  ✓ {name}: shape={tuple(score.shape)}, "
            f"mean={score.mean().item():.6e}, max={score.max().item():.6e}, min={score.min().item():.6e}"
        )

    return fairness_score


def _is_impt_type2_target_layer(name: str) -> bool:
    if name == 'features.1.conv.0.0':  # features.1 DW(32ch) 기준으로 변경
        return True
    if name == 'features.18.0':
        return True
    if name.startswith('features.') and name.endswith('.conv.1.0'):
        return not name.startswith('features.18.')
    return False


def _get_impt_type2_target_layers(model):
    target_layers = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and _is_impt_type2_target_layer(name):
            target_layers[name] = layer
    return target_layers


def _get_impt_type2_block_layer_names(block_name: str):
    if block_name == 'features.1':
        # DW(32ch)를 채널 기준으로, PW project는 입력 채널(dim 1)로 묶음 제거
        return f'{block_name}.conv.0.0', None, f'{block_name}.conv.1'
    if block_name == 'features.18':
        # classifier.1은 별도 weight-level pruning으로 분리
        return 'features.18.0', None, None
    return f'{block_name}.conv.0.0', f'{block_name}.conv.1.0', f'{block_name}.conv.2'


def _pool_channel_values(tensor: torch.Tensor, use_abs: bool = False) -> torch.Tensor:
    pooled = tensor.abs() if use_abs else tensor
    if pooled.ndim == 4:
        return pooled.mean(dim=(2, 3))
    if pooled.ndim == 3:
        return pooled.mean(dim=2)
    if pooled.ndim == 2:
        return pooled
    if pooled.ndim == 1:
        return pooled.unsqueeze(0)
    return pooled.reshape(pooled.shape[0], pooled.shape[1], -1).mean(dim=2)


def _compute_binary_group_gap(channel_values: torch.Tensor, target_y: torch.Tensor, sensitive_a: torch.Tensor, y_value: int):
    group0_mask = (target_y == y_value) & (sensitive_a == 0)
    group1_mask = (target_y == y_value) & (sensitive_a == 1)
    if not group0_mask.any() or not group1_mask.any():
        return None

    group0_mean = channel_values[group0_mask].mean(dim=0).to(torch.float64)
    group1_mean = channel_values[group1_mask].mean(dim=0).to(torch.float64)
    return torch.abs(group0_mean - group1_mean)


def _aggregate_channel_importance(score_tensor: torch.Tensor) -> torch.Tensor:
    if score_tensor.ndim == 4:
        return score_tensor.mean(dim=(1, 2, 3))
    if score_tensor.ndim == 3:
        return score_tensor.mean(dim=(1, 2))
    if score_tensor.ndim == 2:
        return score_tensor.mean(dim=1)
    if score_tensor.ndim == 1:
        return score_tensor
    return score_tensor.reshape(score_tensor.shape[0], -1).mean(dim=1)


def _count_total_weights(model):
    total = 0
    for _, layer in model.named_modules():
        if type(layer).__name__ in supported_layers and hasattr(layer, 'weight'):
            total += int(layer.weight.numel())
    return total


def _count_active_weights(model):
    total_active = 0
    for _, layer in model.named_modules():
        if type(layer).__name__ not in supported_layers or not hasattr(layer, 'weight'):
            continue
        if hasattr(layer, 'mask') and layer.mask.shape == layer.weight.shape:
            total_active += int(layer.mask.detach().sum().item())
        else:
            total_active += int(layer.weight.numel())
    return total_active


def _count_all_channel_weights(model):
    """채널 pruning 대상 블록(features.1~17)의 conv0+conv1+conv2 전체 weight 수 (마스크 무관).
    features.18은 프루닝 대상에서 제외."""
    modules = dict(model.named_modules())
    total = 0
    for block_num in range(1, 18):
        block_name = f'features.{block_num}'
        conv0_name, conv1_name, conv2_name = _get_impt_type2_block_layer_names(block_name)
        if conv0_name in modules and hasattr(modules[conv0_name], 'weight'):
            total += modules[conv0_name].weight.numel()
        if conv1_name is not None and conv1_name in modules and hasattr(modules[conv1_name], 'weight'):
            total += modules[conv1_name].weight.numel()
        if conv2_name is not None and conv2_name in modules and hasattr(modules[conv2_name], 'weight'):
            total += modules[conv2_name].weight.numel()
    return total


def _count_channel_weights(model, block_name: str, channel_k: int):
    modules = dict(model.named_modules())
    conv0_name, conv1_name, conv2_name = _get_impt_type2_block_layer_names(block_name)
    total = 0

    def current_mask(layer):
        if hasattr(layer, 'mask') and layer.mask.shape == layer.weight.shape:
            return layer.mask.detach()
        return torch.ones_like(layer.weight)

    if conv0_name in modules and hasattr(modules[conv0_name], 'weight'):
        conv0_mask = current_mask(modules[conv0_name])
        if channel_k < conv0_mask.shape[0]:
            total += int(conv0_mask[channel_k].sum().item())

    if conv1_name in modules and hasattr(modules[conv1_name], 'weight'):
        conv1_mask = current_mask(modules[conv1_name])
        if channel_k < conv1_mask.shape[0]:
            total += int(conv1_mask[channel_k].sum().item())

    if conv2_name is not None and conv2_name in modules and hasattr(modules[conv2_name], 'weight'):
        conv2_mask = current_mask(modules[conv2_name])
        if channel_k < conv2_mask.shape[1]:
            total += int(conv2_mask[:, channel_k].sum().item())

    return total

def _save_channel_pruning_log(
    selected_channels,
    score_by_channel_dict,
    phi_by_layer,
    perf_by_layer,
    accum_removed,
    remove_target,
    prune_iter,
    alpha,
    model=None,
    log_dir='/workspace/FairGRAPE/FairGRAPE/channel_pruning_logs',
    total_model_params=None,
    total_active_after=None,
    model_sparsity=None,
):
    import datetime
    os.makedirs(log_dir, exist_ok=True)

    filename = f"alpha{alpha:.1f}_iter{prune_iter + 1:02d}.txt"
    filepath = os.path.join(log_dir, filename)

    # 블록별 선택된 채널 수 집계
    block_count = defaultdict(int)
    for block_name, _ in selected_channels:
        block_count[block_name] += 1

    # 레이어별 총 채널 수 및 누적 제거 채널 수 계산 (features.1~17만)
    layer_channel_stats = {}  # block_name -> (total_ch, cumul_removed)
    if model is not None:
        modules = dict(model.named_modules())
        for block_num in range(1, 18):
            block_name = f'features.{block_num}'
            conv0_name, conv1_name, _ = _get_impt_type2_block_layer_names(block_name)
            ref_name = conv0_name
            if ref_name in modules and hasattr(modules[ref_name], 'weight'):
                w = modules[ref_name].weight
                total_ch = w.shape[0]
                if hasattr(modules[ref_name], 'mask') and modules[ref_name].mask.shape == w.shape:
                    mask = modules[ref_name].mask
                    cumul_removed = int((mask.view(total_ch, -1).sum(dim=1) == 0).sum().item())
                else:
                    cumul_removed = 0
                layer_channel_stats[block_name] = (total_ch, cumul_removed)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Channel Pruning Log\n")
        f.write(f"  iteration  : {prune_iter + 1}\n")
        f.write(f"  alpha      : {alpha:.4f}\n")
        f.write(f"  timestamp  : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  remove_target  : {remove_target}\n")
        f.write(f"  accum_removed  : {accum_removed}\n")
        f.write(f"  selected_channels : {len(selected_channels)}\n")
        if model_sparsity is not None:
            f.write(f"  total_params   : {total_model_params}\n")
            f.write(f"  active_after   : {total_active_after}\n")
            f.write(f"  model_sparsity : {model_sparsity*100:.1f}%\n")
        f.write("=" * 80 + "\n\n")

        # 레이어별 채널 제거 비율 요약 (features.1~17)
        f.write("[ 레이어별 채널 제거 비율 ]\n")
        f.write(f"  {'layer':<20} {'total_ch':>8} {'this_iter':>10} {'cumul':>8} {'cumul_%':>8}\n")
        f.write("  " + "-" * 58 + "\n")
        for block_num in range(1, 18):
            block_name = f'features.{block_num}'
            this_iter = block_count.get(block_name, 0)
            if block_name in layer_channel_stats:
                total_ch, cumul_removed = layer_channel_stats[block_name]
                cumul_after = cumul_removed + this_iter
                ratio = cumul_after / total_ch * 100 if total_ch > 0 else 0.0
                f.write(f"  {block_name:<20} {total_ch:>8} {this_iter:>10} {cumul_after:>8} {ratio:>7.1f}%\n")
            else:
                f.write(f"  {block_name:<20} {'N/A':>8} {this_iter:>10} {'N/A':>8} {'N/A':>8}\n")
        f.write("\n")

        # 블록별 요약
        f.write("[ 블록별 제거 채널 수 ]\n")
        for block_name in sorted(block_count.keys()):
            f.write(f"  {block_name}: {block_count[block_name]}채널 제거\n")
        f.write("\n")

        # 채널별 상세
        f.write("[ 선택된 채널 상세 ]\n")
        f.write(f"{'block':<30} {'ch':>5} {'score':>12} {'phi':>12} {'perf':>12} {'weights':>8}\n")
        f.write("-" * 80 + "\n")

        for block_name, channel_k in selected_channels:
            key = (block_name, channel_k)
            score, weight_count = score_by_channel_dict.get(key, (float('nan'), 0))

            # phi/perf_by_layer 키 복원 (ref_name: DW 레이어 또는 conv0)
            if block_name == 'features.1':
                conv1_name = 'features.1.conv.0.0'  # DW(32ch) 기준
            else:
                conv1_name = block_name + '.conv.1.0'

            phi_val = float('nan')
            perf_val = float('nan')
            if conv1_name in phi_by_layer and channel_k < len(phi_by_layer[conv1_name]):
                phi_val = phi_by_layer[conv1_name][channel_k].item()
            if conv1_name in perf_by_layer and channel_k < len(perf_by_layer[conv1_name]):
                perf_val = perf_by_layer[conv1_name][channel_k].item()

            f.write(
                f"{block_name:<30} {channel_k:>5} {score:>12.6f} "
                f"{phi_val:>12.6e} {perf_val:>12.6e} {weight_count:>8}\n"
            )

    print(f"[채널 pruning 로그 저장] {filepath}")



def _build_channel_mask_list(model, selected_channels, device):
    mask_by_layer = {}
    for name, layer in model.named_modules():
        if type(layer).__name__ not in supported_layers or not hasattr(layer, 'weight'):
            continue

        if is_classifier_head(name):
            # 기존 weight-level 마스크가 있으면 누적 유지 (features.18 채널 pruning 시 열 단위로 추가 적용됨)
            if hasattr(layer, 'mask') and layer.mask.shape == layer.weight.shape:
                mask_by_layer[name] = layer.mask.detach().clone().long()
            else:
                mask_by_layer[name] = torch.ones_like(layer.weight, dtype=torch.long)
        elif hasattr(layer, 'mask') and layer.mask.shape == layer.weight.shape:
            mask_by_layer[name] = layer.mask.detach().clone().long()
        else:
            mask_by_layer[name] = torch.ones_like(layer.weight, dtype=torch.long)

    for block_name, channel_k in selected_channels:
        conv0_name, conv1_name, conv2_name = _get_impt_type2_block_layer_names(block_name)

        if conv0_name in mask_by_layer and channel_k < mask_by_layer[conv0_name].shape[0]:
            mask_by_layer[conv0_name][channel_k] = 0

        if conv1_name in mask_by_layer and channel_k < mask_by_layer[conv1_name].shape[0]:
            mask_by_layer[conv1_name][channel_k] = 0

        if conv2_name is not None and conv2_name in mask_by_layer and channel_k < mask_by_layer[conv2_name].shape[1]:
            mask_by_layer[conv2_name][:, channel_k] = 0

    mask_list = [mask.to(device) for _, mask in mask_by_layer.items()]
    print(f"channel mask 생성 완료: 선택 채널={len(selected_channels)}, 총 레이어={len(mask_list)}")
    return mask_list


def compute_phi_k(model, test_csv, new_img_dir=None, output_cols_each_task=[(0,7),(7,9),(9,18)], col_names=['race','gender'], stop_batch=10000, masked_grads=True, sensitive_group=None):
    del masked_grads
    print("-------------compute_phi_k-------------")

    if sensitive_group not in (None, 'gender'):
        raise NotImplementedError("impt_type == 2 는 현재 gender 기준 민감 그룹만 지원합니다.")

    if not col_names or col_names[-1] != 'gender':
        raise ValueError(f"impt_type == 2 는 마지막 라벨 컬럼이 gender 여야 합니다. 현재: {col_names[-1] if col_names else None}")

    target_layers = _get_impt_type2_target_layers(model)
    if not target_layers:
        raise RuntimeError("impt_type == 2 대상 conv[1] 레이어를 찾지 못했습니다.")

    # features.0.0: weight-level phi 계산을 위한 별도 hook 대상
    modules_dict = dict(model.named_modules())
    weight_phi_target_names = ['features.0.0']  # classifier.1은 features.18 phi 재사용
    weight_grad_sums = {}
    weight_grad_counts = defaultdict(int)
    # features.0.0의 activation gap 누적 (channel-level과 동일 방식)
    f0_gap_y0_sums = {}
    f0_gap_y0_counts = defaultdict(int)
    f0_gap_y1_sums = {}
    f0_gap_y1_counts = defaultdict(int)
    f0_activations = {}

    if isinstance(test_csv, str):
        test_frame = pd.read_csv(test_csv)
    else:
        test_frame = test_csv.copy()

    if new_img_dir:
        faces = set(os.listdir(new_img_dir))
        new_face_name = []
        face_found_mask = []
        for i in range(test_frame.shape[0]):
            face_name_align = split_image_name(test_frame['face_name_align'][i])
            face_found_mask.append(face_name_align in faces)
            if face_name_align in faces:
                new_face_name.append(os.path.join(new_img_dir, face_name_align))
        test_frame = test_frame[face_found_mask].reset_index(drop=True)
        test_frame['face_name_align'] = new_face_name

    batch_size = 64
    test_loader, _ = make_datasets(test_frame, test_frame, True, batch_size, col_used=col_names)
    criterion = nn.CrossEntropyLoss()

    gap_y0_sums = {}
    gap_y1_sums = {}
    gap_y0_counts = defaultdict(int)
    gap_y1_counts = defaultdict(int)
    mean_grad_sums = {}
    mean_grad_counts = defaultdict(int)
    activations = {}
    handles = []

    def register_activation_hook(layer_name):
        def hook(_, __, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return
            if output.requires_grad:
                output.retain_grad()
            activations[layer_name] = output
        return hook

    for layer_name, layer in target_layers.items():
        handles.append(layer.register_forward_hook(register_activation_hook(layer_name)))

    # features.0.0 hook 등록 (weight-level phi용 activation 수집)
    def register_f0_hook(layer_name):
        def hook(_, __, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return
            if output.requires_grad:
                output.retain_grad()
            f0_activations[layer_name] = output
        return hook

    if 'features.0.0' in modules_dict:
        handles.append(modules_dict['features.0.0'].register_forward_hook(register_f0_hook('features.0.0')))

    model.eval()

    prev_cudnn_enabled = torch.backends.cudnn.enabled
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    try:
        for batch_idx, sample_batched in enumerate(test_loader):
            if batch_idx >= stop_batch:
                break

            if batch_idx % 50 == 0:
                print(f"{batch_idx}번째 mini-batch phi 계산 중!")

            image_batched, label_batched = sample_batched
            image_batched = image_batched.to(device, dtype=torch.float, non_blocking=True).contiguous()
            label_batched = label_batched.to(device, non_blocking=True).long()

            activations.clear()
            model.zero_grad(set_to_none=True)

            with torch.backends.cudnn.flags(enabled=False):
                outputs = safe_forward_with_cudnn_fallback(model, image_batched)

            task_loss = loss_multi_tasks(outputs, label_batched, criterion, output_cols_each_task)
            task_loss.backward()

            pooled_activations = {}
            sensitive_a = label_batched[:, -1].long()
            num_attrs = min(len(output_cols_each_task), label_batched.shape[1] - 1)

            for layer_name, activation in activations.items():
                pooled_act = _pool_channel_values(activation.detach(), use_abs=False)
                pooled_activations[layer_name] = pooled_act

                if activation.grad is None:
                    continue

                pooled_grad = _pool_channel_values(activation.grad.detach(), use_abs=True)
                batch_mean_grad = pooled_grad.mean(dim=0).to(torch.float64)
                if layer_name not in mean_grad_sums:
                    mean_grad_sums[layer_name] = batch_mean_grad
                else:
                    mean_grad_sums[layer_name] += batch_mean_grad
                mean_grad_counts[layer_name] += 1

            for attr_idx in range(num_attrs):
                target_y = label_batched[:, attr_idx].long()
                for layer_name, pooled_act in pooled_activations.items():
                    gap_y0 = _compute_binary_group_gap(pooled_act, target_y, sensitive_a, y_value=0)
                    if gap_y0 is not None:
                        if layer_name not in gap_y0_sums:
                            gap_y0_sums[layer_name] = gap_y0
                        else:
                            gap_y0_sums[layer_name] += gap_y0
                        gap_y0_counts[layer_name] += 1

                    gap_y1 = _compute_binary_group_gap(pooled_act, target_y, sensitive_a, y_value=1)
                    if gap_y1 is not None:
                        if layer_name not in gap_y1_sums:
                            gap_y1_sums[layer_name] = gap_y1
                        else:
                            gap_y1_sums[layer_name] += gap_y1
                        gap_y1_counts[layer_name] += 1

            # features.0.0 activation gap 누적 (weight-level phi용)
            if 'features.0.0' in f0_activations:
                f0_act = f0_activations['features.0.0']
                pooled_f0 = _pool_channel_values(f0_act.detach(), use_abs=False)
                for attr_idx in range(num_attrs):
                    target_y = label_batched[:, attr_idx].long()
                    gap_y0 = _compute_binary_group_gap(pooled_f0, target_y, sensitive_a, y_value=0)
                    if gap_y0 is not None:
                        f0_gap_y0_sums['features.0.0'] = f0_gap_y0_sums.get('features.0.0', torch.zeros_like(gap_y0)) + gap_y0
                        f0_gap_y0_counts['features.0.0'] += 1
                    gap_y1 = _compute_binary_group_gap(pooled_f0, target_y, sensitive_a, y_value=1)
                    if gap_y1 is not None:
                        f0_gap_y1_sums['features.0.0'] = f0_gap_y1_sums.get('features.0.0', torch.zeros_like(gap_y1)) + gap_y1
                        f0_gap_y1_counts['features.0.0'] += 1

            # features.0.0, classifier.1 weight gradient 누적
            for wname in ['features.0.0', 'classifier.1']:
                if wname not in modules_dict:
                    continue
                wlayer = modules_dict[wname]
                if hasattr(wlayer, 'weight') and wlayer.weight.grad is not None:
                    grad_abs = wlayer.weight.grad.detach().abs().to(torch.float64)
                    weight_grad_sums[wname] = weight_grad_sums.get(wname, torch.zeros_like(grad_abs)) + grad_abs
                    weight_grad_counts[wname] += 1
    finally:
        for handle in handles:
            handle.remove()
        torch.backends.cudnn.enabled = prev_cudnn_enabled
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark

    phi_by_layer = {}
    print("phi 레이어별 요약:")
    for layer_name, layer in target_layers.items():
        num_channels = int(layer.weight.shape[0])
        zero_vec = torch.zeros(num_channels, dtype=torch.float64, device=device)

        gap_y0 = gap_y0_sums.get(layer_name, zero_vec)
        if gap_y0_counts[layer_name] > 0:
            gap_y0 = gap_y0 / gap_y0_counts[layer_name]
        else:
            gap_y0 = zero_vec

        gap_y1 = gap_y1_sums.get(layer_name, zero_vec)
        if gap_y1_counts[layer_name] > 0:
            gap_y1 = gap_y1 / gap_y1_counts[layer_name]
        else:
            gap_y1 = zero_vec

        activation_gap = gap_y0 + gap_y1
        mean_gradient = mean_grad_sums.get(layer_name, zero_vec)
        if mean_grad_counts[layer_name] > 0:
            mean_gradient = mean_gradient / mean_grad_counts[layer_name]
        else:
            mean_gradient = zero_vec

        phi = torch.nan_to_num(activation_gap * mean_gradient, nan=0.0, posinf=0.0, neginf=0.0)
        phi_by_layer[layer_name] = phi.to(torch.float32)
        print(
            f"  ✓ {layer_name}: channels={num_channels}, "
            f"phi_mean={phi.mean().item():.6e}, phi_max={phi.max().item():.6e}, "
            f"gap_y0_terms={gap_y0_counts[layer_name]}, gap_y1_terms={gap_y1_counts[layer_name]}, "
            f"grad_batches={mean_grad_counts[layer_name]}"
        )

    # weight-level phi 계산 (features.0.0, classifier.1)
    phi_weight_by_layer = {}
    print("weight-level phi 요약:")

    # features.0.0: activation_gap_k × mean_|weight.grad|[k, :, :, :]
    if 'features.0.0' in modules_dict and 'features.0.0' in weight_grad_sums:
        f0_layer = modules_dict['features.0.0']
        n_ch = f0_layer.weight.shape[0]
        zero_vec = torch.zeros(n_ch, dtype=torch.float64, device=device)
        gap_y0 = f0_gap_y0_sums.get('features.0.0', zero_vec)
        if f0_gap_y0_counts['features.0.0'] > 0:
            gap_y0 = gap_y0 / f0_gap_y0_counts['features.0.0']
        gap_y1 = f0_gap_y1_sums.get('features.0.0', zero_vec)
        if f0_gap_y1_counts['features.0.0'] > 0:
            gap_y1 = gap_y1 / f0_gap_y1_counts['features.0.0']
        activation_gap_f0 = (gap_y0 + gap_y1).to(torch.float64)  # shape [32]
        mean_wgrad_f0 = weight_grad_sums['features.0.0'] / weight_grad_counts['features.0.0']  # [32,3,3,3]
        # broadcast activation_gap per output channel
        gap_broadcast = activation_gap_f0.view(-1, *([1] * (mean_wgrad_f0.ndim - 1)))
        phi_w_f0 = torch.nan_to_num(gap_broadcast * mean_wgrad_f0, nan=0.0, posinf=0.0, neginf=0.0)
        phi_weight_by_layer['features.0.0'] = phi_w_f0.to(torch.float32)
        print(f"  ✓ features.0.0: shape={tuple(phi_w_f0.shape)}, phi_mean={phi_w_f0.mean().item():.6e}")

    # classifier.1: phi_features18[k] × mean_|weight.grad|[i, k]
    if 'classifier.1' in weight_grad_sums and 'features.18.0' in phi_by_layer:
        phi_f18 = phi_by_layer['features.18.0'].to(torch.float64).to(device)  # [1280]
        mean_wgrad_cls = weight_grad_sums['classifier.1'] / weight_grad_counts['classifier.1']  # [num_cls, 1280]
        phi_w_cls = torch.nan_to_num(phi_f18.unsqueeze(0) * mean_wgrad_cls, nan=0.0, posinf=0.0, neginf=0.0)
        phi_weight_by_layer['classifier.1'] = phi_w_cls.to(torch.float32)
        print(f"  ✓ classifier.1: shape={tuple(phi_w_cls.shape)}, phi_mean={phi_w_cls.mean().item():.6e}")

    return phi_by_layer, phi_weight_by_layer


def fairness_grad(model, prune_ratio, test_csv, new_img_dir=None, sensitive_classes = 2, masked_grads=True, output_cols_each_task=[(0,7),(7,9),(9,18)],col_names=['race','gender'], para_batch=1, impt_type = 0, stop_batch=10000, delta_p=False,n_jobs=1, network=None, sensitive_group=None):
    def safe_topk_mask(score_tensor, keep_ratio, largest=True):
        flat = score_tensor.flatten().cpu() # Top-k는 CPU에서 계산 추천 (메모리 절약)
        N = flat.numel()
        k = int(N * keep_ratio)

        if k <= 0: return torch.zeros_like(flat, dtype=torch.long)
        if k >= N: return torch.ones_like(flat, dtype=torch.long)

        _, topk_indices = torch.topk(flat, k, largest=largest, sorted=False)
        
        mask = torch.zeros_like(flat, dtype=torch.long)
        mask[topk_indices] = 1
        return mask.view_as(score_tensor).to(score_tensor.device) # 다시 GPU로

    def print_score_summary(score_by_layer, tag):
        print(f"{tag} 점수 요약: 총 {len(score_by_layer)}개 레이어")
        for name in sorted(score_by_layer.keys()):
            score = score_by_layer[name]
            print(
                f"  ✓ {name}: shape={tuple(score.shape)}, "
                f"mean={score.mean().item():.6e}, std={score.std().item():.6e}, "
                f"min={score.min().item():.6e}, max={score.max().item():.6e}"
            )

    def scale_score_tensor(score_tensor):
        scaled = score_tensor / (score_tensor.max() + 1e-12)
        return torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)

    def combine_weighted_scores(importance_by_layer, fairness_by_layer, alpha):
        combined_score = {}
        common_names = set(importance_by_layer.keys()) & set(fairness_by_layer.keys())

        print("\n🔍 Importance Score 레이어 수:", len(importance_by_layer))
        print("🔍 Fairness Score 레이어 수:", len(fairness_by_layer))
        print("🔍 공통 레이어 수:", len(common_names))

        for name in common_names:
            if is_classifier_head(name):
                continue

            imp = importance_by_layer[name]
            fair = fairness_by_layer[name]

            if imp.shape != fair.shape:
                print(f"⚠️ Shape mismatch: {name} - imp:{imp.shape} vs fair:{fair.shape}")
                continue

            imp_scaled = scale_score_tensor(imp)
            fair_scaled = scale_score_tensor(fair)
            combined_score[name] = (alpha * imp_scaled) - ((1.0 - alpha) * fair_scaled)

        return combined_score

    def build_mask_list(score_by_layer, keep_ratio, keep_largest_scores=True):
        mask_by_layernames = {}

        for name, layer in model.named_modules():
            if type(layer).__name__ not in supported_layers:
                continue

            if not hasattr(layer, 'weight'):
                continue

            if is_classifier_head(name):
                mask = torch.ones_like(layer.weight, dtype=torch.long)
            elif name in score_by_layer:
                mask = safe_topk_mask(score_by_layer[name], keep_ratio, largest=keep_largest_scores)
                kept = int(mask.sum().item())
                total = mask.numel()
                print(f"  mask[{name}]: keep={kept}/{total} ({kept / max(total, 1):.4f}), keep_largest_scores={keep_largest_scores}")
            else:
                mask = torch.ones_like(layer.weight, dtype=torch.long)
                print(f"  mask[{name}]: 점수 없음, 전체 유지")

            mask_by_layernames[name] = mask

        mask_list = [mask.to(device) for name, mask in mask_by_layernames.items()]
        print(f"마스크 추출 완료: 총 {len(mask_list)}개 레이어 적용됨.")
        return mask_list

    keep_ratio = 1.0 - prune_ratio
    print("-------------fairness_grad-------------")
    print(f"prune_ratio={prune_ratio}, keep_ratio={keep_ratio}, impt_type={impt_type}, sensitive_group={sensitive_group}")

    # 민감 집단에 대한 gradient 분포를 고려한 프루닝
    if impt_type == 0:
        print("imp 0")

        gender_model, info = load_gender_model_from_ckpt(model, 'gender_model/best_gender_model.pth', device)
        print('missing:', info.missing_keys)
        print('unexpected:', info.unexpected_keys)
        print('gender head shape:', tuple(gender_model.classifier[1].weight.shape))

        importance_score, gender_score = compute_importance(
            model,
            gender_model,
            test_csv,
            new_img_dir=new_img_dir,
            masked_grads=masked_grads,
            output_cols_each_task=output_cols_each_task,
            col_names=col_names,
            stop_batch=stop_batch,
            network=network,
            sensitive_group=sensitive_group,
            sensitive_classes=sensitive_classes,
        )

        beta = 1.0
        hybrid_importance = {}
        common_names = set(importance_score.keys()) & set(gender_score.keys())

        print("\n🔍 Importance Score 레이어 수:", len(importance_score))
        print("🔍 Gender Score 레이어 수:", len(gender_score))
        print("🔍 공통 레이어 수:", len(common_names))

        for name in common_names:
            if is_classifier_head(name):
                continue

            imp = importance_score[name]
            gen = gender_score[name]

            if imp.shape != gen.shape:
                print(f"⚠️ Shape mismatch: {name} - imp:{imp.shape} vs gen:{gen.shape}")
                continue

            imp_scaled = imp / (imp.max() + 1e-12)
            gen_scaled = gen / (gen.max() + 1e-12)

            imp_scaled = torch.nan_to_num(imp_scaled, nan=0.0)
            gen_scaled = torch.nan_to_num(gen_scaled, nan=0.0)

            denominator = 1.0 + (beta * gen_scaled)
            hybrid_score = imp_scaled / denominator
            hybrid_importance[name] = hybrid_score

        print_score_summary(hybrid_importance, "hybrid importance")
        return build_mask_list(hybrid_importance, keep_ratio)

    if impt_type == 1:
        if sensitive_classes != 2:
            raise NotImplementedError("impt_type == 1 은 현재 2개 gender 그룹만 지원합니다.")

        alpha = float(IMPT_TYPE1_ALPHA)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"impt_type == 1 의 alpha는 0과 1 사이여야 합니다. 현재 값: {alpha}")

        import config
        config.glo_imp_rate = alpha

        print(f"impt_type == 1: alpha={alpha:.4f}로 성능/공정성 혼합 pruning 시작")

        importance_score, _ = compute_importance(
            model,
            None,
            test_csv,
            new_img_dir=new_img_dir,
            masked_grads=masked_grads,
            output_cols_each_task=output_cols_each_task,
            col_names=col_names,
            stop_batch=stop_batch,
            network=network,
            sensitive_group=sensitive_group,
            sensitive_classes=sensitive_classes,
        )

        fairness_score = compute_gender_gradient_gap_importance(
            model,
            test_csv,
            new_img_dir=new_img_dir,
            masked_grads=masked_grads,
            output_cols_each_task=output_cols_each_task,
            col_names=col_names,
            stop_batch=stop_batch,
            sensitive_group=sensitive_group,
        )

        print_score_summary(importance_score, "task importance")
        print_score_summary(fairness_score, "gender gradient gap fairness")

        combined_score = combine_weighted_scores(importance_score, fairness_score, alpha)
        print_score_summary(combined_score, "alpha-weighted combined")
        return build_mask_list(combined_score, keep_ratio, keep_largest_scores=True)

    if impt_type == 2:
        if sensitive_classes != 78:
            print(f"impt_type == 2 참고: 현재 sensitive_classes={sensitive_classes}, CelebA multi-task 기준 78로 전달되는 구성을 가정합니다.")
        
        alpha = float(IMPT_TYPE2_ALPHA)
        import config
        config.glo_imp_rate = alpha  # ← 추가된 줄
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"impt_type == 2 의 alpha는 0과 1 사이여야 합니다. 현재 값: {alpha}")

        print(f"impt_type == 2: alpha={alpha:.4f}로 채널 단위 fairness-aware pruning 시작")

        phi_by_layer, phi_weight_by_layer = compute_phi_k(
            model,
            test_csv,
            new_img_dir=new_img_dir,
            output_cols_each_task=output_cols_each_task,
            col_names=col_names,
            stop_batch=stop_batch,
            masked_grads=masked_grads,
            sensitive_group=sensitive_group,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        importance_score_all, _ = compute_importance(
            model,
            None,
            test_csv,
            new_img_dir=new_img_dir,
            masked_grads=masked_grads,
            output_cols_each_task=output_cols_each_task,
            col_names=col_names,
            stop_batch=stop_batch,
            network=network,
            sensitive_group=sensitive_group,
            sensitive_classes=sensitive_classes,
            imp_batch_size=IMPT_TYPE2_IMPORTANCE_BATCH_SIZE,
        )

        # ── 채널 단위 성능 기여도 (conv0+conv1+conv2 합산) ──
        # conv0/conv1: 출력 채널(dim 0) 기준 집계
        # conv2(PW project): 입력 채널(dim 1) 기준 집계 → mean(dim=(0,2,3))
        perf_by_layer = {}
        for block_num in range(1, 19):
            block_name = f'features.{block_num}'
            conv0_name, conv1_name, conv2_name = _get_impt_type2_block_layer_names(block_name)
            # ref_name: phi_by_layer와 동일한 키 (DW 레이어 or conv0)
            ref_name = conv1_name if conv1_name is not None else conv0_name
            combined = None

            if conv0_name in importance_score_all:
                score = _aggregate_channel_importance(importance_score_all[conv0_name]).to(torch.float32)
                combined = score

            if conv1_name is not None and conv1_name in importance_score_all:
                score = _aggregate_channel_importance(importance_score_all[conv1_name]).to(torch.float32)
                combined = score if combined is None else combined + score

            if conv2_name is not None and conv2_name in importance_score_all:
                s = importance_score_all[conv2_name].to(torch.float32)
                # 입력 채널(dim 1) 기준: mean over (out_ch, h, w)
                if s.ndim == 4:
                    score = s.mean(dim=(0, 2, 3))
                elif s.ndim == 2:
                    score = s.mean(dim=0)
                else:
                    score = s.mean(dim=0)
                if combined is not None and combined.shape == score.shape:
                    combined = combined + score
                elif combined is None:
                    combined = score

            if combined is not None:
                perf_by_layer[ref_name] = combined

        score_by_channel = []
        for conv1_name, phi_vec in phi_by_layer.items():
            perf_vec = perf_by_layer.get(conv1_name)
            if perf_vec is None:
                print(f"  - {conv1_name}: 성능 중요도 없음, 스킵")
                continue

            if perf_vec.shape != phi_vec.shape:
                print(f"  - {conv1_name}: shape mismatch perf={tuple(perf_vec.shape)} phi={tuple(phi_vec.shape)}, 스킵")
                continue

            # features.18은 프루닝 대상에서 제외
            if conv1_name == 'features.18.0':
                continue
            block_name = conv1_name.rsplit('.conv.', 1)[0]

            phi_scaled = scale_score_tensor(phi_vec)
            perf_scaled = scale_score_tensor(perf_vec)
            combined_scores = (alpha * perf_scaled) - ((1.0 - alpha) * phi_scaled)

            for channel_k in range(len(combined_scores)):
                weight_count = _count_channel_weights(model, block_name, channel_k)
                if weight_count <= 0:
                    continue
                score_by_channel.append((combined_scores[channel_k].item(), block_name, channel_k, weight_count))

        if not score_by_channel:
            print("impt_type == 2: 제거 가능한 채널 점수를 만들지 못해 기존 마스크를 그대로 반환합니다.")
            return _build_channel_mask_list(model, [], device)

        # ── remove_target: 현재 살아있는 채널 가중치의 일정 비율 제거 ──
        active_channel_weights = sum(w for _, _, _, w in score_by_channel)
        remove_target = int(round(active_channel_weights * (1.0 - IMPT2_KEEP_PER_ITER)))

        print(
            f"impt_type == 2 제거량 계산: active_channel_weights={active_channel_weights}, "
            f"keep_per_iter={IMPT2_KEEP_PER_ITER}, remove_target={remove_target}, "
            f"candidates={len(score_by_channel)}"
        )

        # ── 레이어별 최소 유지량 계산 (원본 채널 × IMPT2_MIN_KEEP_RATIO_PER_LAYER) ──
        import math as _math
        modules = dict(model.named_modules())
        per_layer_original_ch = {}            # block_name -> 원본 채널 수
        per_layer_already_removed = defaultdict(int)  # block_name -> 이미 완전 제거된 채널 수
        for block_num in range(1, 18):
            bn = f'features.{block_num}'
            conv0_name, _, _ = _get_impt_type2_block_layer_names(bn)
            if conv0_name in modules and hasattr(modules[conv0_name], 'weight'):
                w = modules[conv0_name].weight
                total_ch = int(w.shape[0])
                per_layer_original_ch[bn] = total_ch
                if hasattr(modules[conv0_name], 'mask') and modules[conv0_name].mask.shape == w.shape:
                    mask = modules[conv0_name].mask
                    per_layer_already_removed[bn] = int(
                        (mask.view(total_ch, -1).sum(dim=1) == 0).sum().item()
                    )

        per_layer_min_keep = {
            bn: int(_math.ceil(total_ch * IMPT2_MIN_KEEP_RATIO_PER_LAYER))
            for bn, total_ch in per_layer_original_ch.items()
        }

        # ── 전역(global) 점수 기반 채널 선택 (레이어별 최소 유지 제약 반영) ──
        selected_channels = []
        accum_removed = 0
        per_layer_select_count = defaultdict(int)
        skipped_by_floor = defaultdict(int)

        if remove_target > 0:
            sorted_all = sorted(score_by_channel, key=lambda x: x[0])
            for score, block_name, channel_k, weight_count in sorted_all:
                if accum_removed >= remove_target:
                    break

                total_ch = per_layer_original_ch.get(block_name, 0)
                min_keep = per_layer_min_keep.get(block_name, 0)
                already_removed = per_layer_already_removed[block_name]
                being_removed = per_layer_select_count[block_name]
                remaining_after = total_ch - already_removed - being_removed - 1

                if remaining_after < min_keep:
                    # 이 채널을 제거하면 해당 레이어 최소 유지량 미만 → 스킵
                    skipped_by_floor[block_name] += 1
                    continue

                selected_channels.append((block_name, channel_k))
                accum_removed += weight_count
                per_layer_select_count[block_name] += 1

        # 제약으로 스킵된 레이어 로그
        for bn in sorted(skipped_by_floor.keys(), key=lambda x: int(x.split('.')[1])):
            print(
                f"  [최소 유지 제약] layer={bn}, skipped={skipped_by_floor[bn]}, "
                f"min_keep={per_layer_min_keep[bn]}/{per_layer_original_ch[bn]}"
            )

        # 레이어별 선택 현황 출력
        layer_select_count = defaultdict(int)
        for bn, _ in selected_channels:
            layer_select_count[bn] += 1
        for bn in sorted(layer_select_count.keys(), key=lambda x: int(x.split('.')[1])):
            print(f"  [Global 선택] layer={bn}, selected={layer_select_count[bn]}")

        print(
            f"impt_type == 2 채널 선택 완료: selected={len(selected_channels)}, "
            f"estimated_removed={accum_removed}, remove_target={remove_target}"
        )

        # 전체 모델 sparsity 계산 (Conv2d + Linear 기준)
        total_model_params = _count_total_weights(model)
        total_active_after = _count_active_weights(model) - accum_removed
        model_sparsity = 1.0 - (total_active_after / total_model_params)
        print(f"모델 sparsity (Conv2d+Linear 기준, 이번 iter 후): {model_sparsity*100:.1f}%")

        # ── 채널 마스크 생성 (features.1~17) ──
        channel_mask_list = _build_channel_mask_list(model, selected_channels, device)

        # 채널 마스크 리스트를 layer 순서대로 dict로 변환
        layer_order = [name for name, layer in model.named_modules()
                       if type(layer).__name__ in supported_layers and hasattr(layer, 'weight')]
        mask_by_name = dict(zip(layer_order, channel_mask_list))

        # 로그 저장
        import config as _config
        score_by_channel_dict = {
            (b, k): (s, w)
            for s, b, k, w in score_by_channel
        }
        _save_channel_pruning_log(
            selected_channels=selected_channels,
            score_by_channel_dict=score_by_channel_dict,
            phi_by_layer=phi_by_layer,
            perf_by_layer=perf_by_layer,
            accum_removed=accum_removed,
            remove_target=remove_target,
            prune_iter=_config.glo_prune_iter,
            alpha=alpha,
            model=model,
            log_dir='/workspace/FairGRAPE/FairGRAPE/channel_pruning_logs',
            total_model_params=total_model_params,
            total_active_after=total_active_after,
            model_sparsity=model_sparsity,
        )

        # 최종 마스크 리스트 반환 (layer 순서 유지)
        final_mask_list = [mask_by_name[name].to(device) for name in layer_order if name in mask_by_name]
        return final_mask_list

    raise ValueError(f"지원하지 않는 impt_type 입니다: {impt_type}")




def compute_importance(model, gender_model, test_csv, new_img_dir=None, masked_grads=True, output_cols_each_task=[(0,7),(7,9),(9,18)], col_names=['race','gender'],network=None,optimizer=None, lr=1e-4, stop_batch=10000, sensitive_group=None, sensitive_classes=None, imp_batch_size=384):
    
    print("-------------compute_importance-------------")
    torch.autograd.set_detect_anomaly(False)  # [CHANGED] True -> False (안정성/속도)

    # [CHANGED] 로그/배치 고정값
    log_every = 100
    print(f"log_every={log_every}")
    print(f"compute_importance batch_size={imp_batch_size}")

    # 0) model의 mask를 모아둠
    def collect_masks_from(model):
        masks = {}
        for n, m in model.named_modules():
            if hasattr(m, 'mask'):
                masks[n] = m.mask.detach().clone()  # [CHANGED] cpu 강제 제거
        return masks

    model_masks = collect_masks_from(model)

    def is_classifier_head(name):  # MobileNetV2 기준
        return name.startswith('classifier.')

    supported_layers = ['Linear', 'Conv2d', 'Conv1d']

    #! 1) 모델 준비
    model.eval()        # [CHANGED] train() -> eval() (BN 안정화)
    if gender_model is not None:
        gender_model.eval() # [CHANGED] train() -> eval()

    #! 2) 데이터 준비
    if isinstance(test_csv, str):
        test_frame = pd.read_csv(test_csv)
    else:
        test_frame = test_csv.copy()

    if new_img_dir:
        faces = set(os.listdir(new_img_dir))
        new_face_name = []
        face_found_mask = []
        for i in range(test_frame.shape[0]):
            face_name_align = split_image_name(test_frame['face_name_align'][i])
            face_found_mask.append(face_name_align in faces)
            if face_name_align in faces:
                new_face_name.append(os.path.join(new_img_dir, face_name_align))
        test_frame = test_frame[face_found_mask].reset_index(drop=True)
        test_frame['face_name_align'] = new_face_name

    # [CHANGED] 배치 크기 고정 64
    test_loader, _ = make_datasets(test_frame, test_frame, True, imp_batch_size, col_used=col_names)
    print(f"compute_importance 배치 수: {min(len(test_loader), stop_batch)}/{len(test_loader)} (batch_size={imp_batch_size})")

    #! 3) 손실 함수
    criterion = nn.CrossEntropyLoss()

    #! 4) 누적 딕셔너리 초기화
    imp_grad_accum = {}
    gender_grad_accum = {}
    importance_score = {}
    gender_score = {}
    mask_at_each_layer = {}
    batches_imp = 0

    # [CHANGED] cuDNN 전역 상태 저장/비활성 (함수 범위)
    prev_cudnn_enabled = torch.backends.cudnn.enabled
    prev_cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

    try:
        #! 5) Task importance
        for batch_idx, sample_batched in enumerate(test_loader):
            if batch_idx >= stop_batch:
                break
            batches_imp += 1
            if batch_idx % log_every == 0:  # [CHANGED]
                print(f"{batch_idx}번째 mini-batch acc_imp 계산 중!")

            image_batched, label_batched = sample_batched
            image_batched = image_batched.to(device, dtype=torch.float, non_blocking=True).contiguous()  # [CHANGED]
            label_batched = label_batched.to(device, non_blocking=True)

            model.zero_grad(set_to_none=True)  # [CHANGED]
            with torch.backends.cudnn.flags(enabled=False):  # [CHANGED]
                outputs = safe_forward_with_cudnn_fallback(model, image_batched)

            imp_loss = loss_multi_tasks(outputs, label_batched, criterion, output_cols_each_task)
            imp_loss.backward()

            for name, layer in model.named_modules():
                if type(layer).__name__ not in supported_layers:
                    continue
                if not hasattr(layer, 'weight') or layer.weight.grad is None:
                    continue

                grads = layer.weight.grad.detach().clone()
                weights = layer.weight.detach().clone()

                if masked_grads and hasattr(layer, 'mask'):
                    masks = layer.mask.detach().clone()
                    if masks.device != grads.device:  # [CHANGED] 디바이스 정렬
                        masks = masks.to(grads.device)
                    mask_at_each_layer[name] = [int(masks.sum().item()), grads.shape]
                    grads = grads * masks

                hess = (weights.abs() * grads.abs()) ** 2  # [CHANGED] masked_grads와 무관하게 항상 계산

                if name not in imp_grad_accum:
                    imp_grad_accum[name] = grads
                    importance_score[name] = hess
                else:
                    imp_grad_accum[name] += grads
                    importance_score[name] += hess

        #! 6) Gender importance
        batches_gender = 0
        if gender_model is not None:
            for batch_idx, sample_batched in enumerate(test_loader):
                if batch_idx >= stop_batch:
                    break
                batches_gender += 1
                if batch_idx % log_every == 0:  # [CHANGED]
                    print(f"{batch_idx}번째 mini-batch gender_imp 중요도 계산 중!")

                image_batched, label_batched = sample_batched
                image_batched = image_batched.to(device, dtype=torch.float, non_blocking=True).contiguous()  # [CHANGED]
                gender_labels = label_batched[:, -1].long().to(device, non_blocking=True)

                gender_model.zero_grad(set_to_none=True)  # [CHANGED]
                with torch.backends.cudnn.flags(enabled=False):  # [CHANGED]
                    outputs = safe_forward_with_cudnn_fallback(gender_model, image_batched)

                gender_loss = criterion(outputs, gender_labels)
                gender_loss.backward()

                for name, layer in gender_model.named_modules():
                    if type(layer).__name__ not in supported_layers:
                        continue
                    if not hasattr(layer, 'weight') or layer.weight.grad is None:
                        continue

                    grads = layer.weight.grad.detach().clone()
                    weights = layer.weight.detach().clone()

                    mask = None if is_classifier_head(name) else model_masks.get(name, None)
                    if masked_grads and mask is not None and mask.shape == grads.shape:
                        if mask.device != grads.device:  # [CHANGED] 디바이스 정렬
                            mask = mask.to(grads.device)
                        grads = grads * mask

                    hess = (weights * grads).pow(2)

                    if masked_grads and mask is not None and mask.shape == hess.shape:
                        if mask.device != hess.device:
                            mask = mask.to(hess.device)
                        hess = hess * mask

                    if name not in gender_grad_accum:
                        gender_grad_accum[name] = grads
                        gender_score[name] = hess
                    else:
                        gender_grad_accum[name] += grads
                        gender_score[name] += hess

    finally:
        # [CHANGED] cuDNN 상태 복구
        torch.backends.cudnn.enabled = prev_cudnn_enabled
        torch.backends.cudnn.benchmark = prev_cudnn_benchmark

    #! 7) 평균
    if batches_imp > 0:
        for name in imp_grad_accum:
            imp_grad_accum[name] /= batches_imp
            importance_score[name] /= batches_imp

    if batches_gender > 0:
        for name in gender_grad_accum:
            gender_grad_accum[name] /= batches_gender
            gender_score[name] /= batches_gender

    # 디버깅 출력
    print("\n🔍 Importance Score 계산된 레이어:")
    for name in importance_score.keys():
        print(f"  ✓ {name}: shape={importance_score[name].shape}")
    
    print("\n🔍 Gender Score 계산된 레이어:")
    for name in gender_score.keys():
        print(f"  ✓ {name}: shape={gender_score[name].shape}")

    return importance_score, gender_score


#!###############
#* [importance_by_class0]
# - 역할: 입력된 모델과 데이터셋(test_csv 등)을 이용해, 민감 그룹별(또는 클래스별) gradient 크기를 측정.
# - 입력
#   - model_path: 모델(혹은 모델 경로)
#   - test_csv, new_img_dir: 평가/분석용 데이터셋 정보
#   - masked_grads: 마스킹된 가중치만 gradient를 계산할지 여부
#   - output_cols_each_task, col_names: 여러 속성(라벨) 관련 설정
#   - stop_batch: 몇 개 배치까지만 처리할지 등 (성능·시간 제약)
# - 출력
#   - grad_each_group: { 그룹ID: { 레이어이름: gradient 텐서 }, ... }
#   - H_each_group: { 그룹ID: { 레이어이름: Hessian-like 텐서 }, ... }
#!###############
"""
def importance_by_class0(model_path, test_csv, new_img_dir=None, masked_grads=True, output_cols_each_task=[(0,7),(7,9),(9,18)], col_names=['race','gender'],network=None,optimizer=None, lr=1e-4, stop_batch=10000):
    supported_layers = ['Linear', 'Conv2d', 'Conv1d']

    # 프루닝 후 재학습된 모델 로드
    model = model_path 

    model.train()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    test_frame = pd.read_csv(test_csv) if isinstance(test_csv, str) else test_csv
    criterion = nn.CrossEntropyLoss()
    criterion_sensitive = nn.BCELoss()
    activation = nn.Sigmoid()s

    # 테스트 프레임 내 이미지들이 실제로 존재하는지 확인
    if new_img_dir:
        initial_rows = test_frame.shape[0]
        faces = set(os.listdir(new_img_dir))s
        faces_found = 0
        new_face_name = []
        face_found_mask = []
        for i in range(test_frame.shape[0]):
            face_name_align = split_image_name(test_frame['face_name_align'][i])
            face_found_mask.append(face_name_align in faces)
            if face_name_align in faces:
                faces_found += 1
                new_face_name.append(os.path.join(new_img_dir, face_name_align))
        test_frame = test_frame[face_found_mask].reset_index(drop=True)
        test_frame['face_name_align'] = new_face_name
    test_loader,_ =  make_datasets(test_frame,test_frame,True,64,col_used=col_names)

    model.train()
    sensitive_cols_in_target = len(output_cols_each_task)
    sensitive_groups = sorted(set(test_frame[col_names[-1]]))

    grad_each_group = {}
    H_each_group = {}
    mask_at_each_layer = {}
    batches = 0
    for batch_idx, sample_batched in enumerate(test_loader):
        if batch_idx >= stop_batch:
            break
        batches += 1
        if batch_idx % 200 == 0:
            print("{}번째 mini-batch 중요도 계산 중!".format(batch_idx))
        image_batched, label_batched = sample_batched
        image_batched = image_batched.to(device, dtype=torch.float)
        label_batched = label_batched.to(device)
        for group_idx, group in enumerate(sensitive_groups):
            gradients = {}
            hessians = {}
            obs_this_group = torch.squeeze((label_batched[:, sensitive_cols_in_target] == group).nonzero())
            outputs = safe_forward_with_cudnn_fallback(model, image_batched)
            output_cols_for_non_protected = output_cols_each_task[:(len(output_cols_each_task))]
            outputs_this_group = outputs[obs_this_group,:].view(-1,outputs.shape[1])
            if outputs_this_group.shape[0] < 1 or len(outputs_this_group.shape) < 2:
                continue
            targets_this_group = label_batched[obs_this_group,:].view(-1, label_batched.shape[1])
            loss_non_protected = loss_multi_tasks(outputs_this_group,targets_this_group,criterion,output_cols_for_non_protected)
            loss = loss_non_protected

            loss.backward()
            optimizer.step()

            for name, layer in model.named_modules():
                if type(layer).__name__ in supported_layers:
                    grads = layer.weight.grad.clone().detach().cpu()
                    weights = layer.weight.data.clone().detach().cpu()
                    if masked_grads:
                        masks = layer.mask.clone().detach().cpu()
                        mask_at_each_layer[name] = [torch.sum(masks), grads.shape]
                        grads *= masks
                    hessians[name] = (weights.abs() * grads.abs())**2
                    gradients[name] = grads
            if group_idx not in grad_each_group:
                grad_each_group[group_idx] = copy.deepcopy(gradients)
                H_each_group[group_idx] = copy.deepcopy(hessians)
            else:
                for name, layer in model.named_modules():
                    if type(layer).__name__ in supported_layers:
                        grad_each_group[group_idx][name] += gradients[name]
                        H_each_group[group_idx][name] += hessians[name]

    for name, layer in model.named_modules():
       if type(layer).__name__ in supported_layers:
           grad_each_group[group_idx][name] /= batches
           H_each_group[group_idx][name] /= batches
                     
    return grad_each_group, H_each_group
"""
    


def make_mask_by_grad(grad_each_group, n_classes=7):
    # 각 그룹별 gradient 정보를 레이어별로 합치는 함수
    groups = [i for i in range(n_classes)]
    layer_names = list(grad_each_group[0].keys())
    grad_at_each_layer = {}
    for layer in layer_names:
        layer_shape = tuple(list(grad_each_group[groups[0]][layer].shape)+[1])
        grad_merged = torch.cat([grad_each_group[group][layer].view(layer_shape) for group in groups], dim=len(layer_shape)-1)
        grad_at_each_layer[layer] = grad_merged
    return grad_at_each_layer

def fetch_a_fair_batch(dataloader, num_classes, samples_per_class, target_col):
    # GraSP github에서 가져온 함수: 페어한 배치 선택
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    combination_idx = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y[0,target_col].item()
            combination_idx[str(category)] = category
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels])
    return X, y, combination_idx


def save_impt_df(cfgs):
    # 중요도 정보를 CSV로 저장하는 함수
    best_model, test_csv, new_img_dir, masked_grads,output_cols_each_task,col_used,stop_batch = cfgs
    _,impts = importance_by_class0(best_model, test_csv, new_img_dir, masked_grads,output_cols_each_task,col_used,stop_batch=stop_batch)
    impt_df = {}
    n_groups = len(impts)
    for i in range(n_groups):
        impt_df["".join(['group', str(i)])] = []
        
    for name, layer in best_model.named_modules():
        for i in range(n_groups):
            impt_df["".join(['group', str(i)])].append(impts[i][name].sum())
            
    impt_df = pd.DataFrame(impt_df)
    impt_df.to_csv("importance_by_layer.csv")
#"""