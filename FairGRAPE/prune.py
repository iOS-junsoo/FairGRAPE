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
from util import make_model, custom_forward_conv2d, custom_forward_conv1d, custom_forward_linear
from dataset import split_image_name, make_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

supported_layers = ['Linear', 'Conv2d', 'Conv1d']

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
            out = self.model(data)
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
        print("sensitive_classes:",sensitive_classes)
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
            outputs = model(data)
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


def fairness_grad(model, prune_ratio, test_csv, new_img_dir=None, sensitive_classes = 2, masked_grads=True, output_cols_each_task=[(0,7),(7,9),(9,18)],col_names=['race','gender'], para_batch=1, impt_type = 0, stop_batch=10000, delta_p=False,n_jobs=1, network=None, sensitive_group=None):
    

    gender_model, info = load_gender_model_from_ckpt(model, 'gender_model/best_gender_model.pth', device)
    print('missing:', info.missing_keys)
    print('unexpected:', info.unexpected_keys)
    print('gender head shape:', tuple(gender_model.classifier[1].weight.shape))



    
    # 민감 집단에 대한 gradient 분포를 고려한 프루닝
    if impt_type == 0:
        importance_score, gender_score = compute_importance(model, gender_model, test_csv, new_img_dir=new_img_dir, masked_grads=masked_grads, output_cols_each_task=output_cols_each_task,col_names=col_names,stop_batch=stop_batch, network=network, sensitive_group=sensitive_group, sensitive_classes=sensitive_classes)
    elif impt_type == 1:
        _,grad_mag_by_race = importance_by_class1(model, test_csv, new_img_dir=new_img_dir, masked_grads=masked_grads, output_cols_each_task=output_cols_each_task,col_names=col_names, n_classes=sensitive_classes)    
    elif impt_type == 2:
        _,grad_mag_by_race = importance_by_class2(model, test_csv, new_img_dir, output_cols_each_task,col_names)

    def show_stats(name, tensor, k=5):
        flat = tensor.flatten()
        topk_vals, topk_idx = torch.topk(flat.abs(), k)

        shape_str = str(list(tensor.shape))       # ← ★ 리스트를 문자열로 변환
        # 폭 지정(:>12)도 이제 정상 작동
        #print(f"[{name:<30}] shape={shape_str:>12} "
        #    f"mean={flat.mean():.3e}  std={flat.std():.3e}  "
        #    f"max={flat.max():.3e}")

        #print(f"top{topk_vals.tolist()}  (indices {topk_idx.tolist()})")

    
    #print("\n── Importance(테일러) 요약 ──")
    #for n, t in importance_score.items():
    #    show_stats(n, t, k=3)

    #print("\n── Gender-Importance 요약 ──")
    #for n, t in gender_score.items():
    #    show_stats(n, t, k=3)

    import config
    from scipy.stats import rankdata
    
    beta = 50.0

    hybrid_importance = {}

     # 🔥 분석용 딕셔너리 추가
    #imp_scaled_dict = {}
    #gen_scaled_dict = {}

    common_names = set(importance_score.keys()) & set(gender_score.keys())

    # 🔍 디버깅 출력
    print("\n🔍 Importance Score 레이어 수:", len(importance_score))
    print("🔍 Gender Score 레이어 수:", len(gender_score))
    print("🔍 공통 레이어 수:", len(common_names))

    # ---------------------------------------------------------
    # 2. 점수 계산 (Layer-wise)
    # ---------------------------------------------------------
    for name in common_names:
        if is_classifier_head(name): 
            continue

        imp = importance_score[name] # Performance Score
        gen = gender_score[name]     # Fairness Score (Bias)

        if imp.shape != gen.shape:
            print(f"⚠️ Shape mismatch: {name} - imp:{imp.shape} vs gen:{gen.shape}")
            continue

        # [중요] Min-Max Scaling 제거! 
        # 대신, 절대적인 크기를 비교하기 위해 값의 범위를 '비율'로 맞춥니다.
        # 단순히 max 값으로 나누어 0~1 스케일(상대 크기 보존)만 맞춥니다.
        # 이렇게 하면 분포(Distribution)가 찌그러지지 않습니다.
        
        # 1) Performance Score 정규화 (최대값 기준)
        # 0으로 나누는 것 방지 (+ 1e-12)
        imp_scaled = imp / (imp.max() + 1e-12) 
        
        # 2) Fairness Score 정규화 (최대값 기준)
        # gen 값이 클수록 '편향된' 것이라고 가정합니다.
        gen_scaled = gen / (gen.max() + 1e-12)

        # NaN / Inf 방지 (안전장치)
        imp_scaled = torch.nan_to_num(imp_scaled, nan=0.0)
        gen_scaled = torch.nan_to_num(gen_scaled, nan=0.0)

        # 🔥 분석용 저장
        #imp_scaled_dict[name] = imp_scaled.clone()
        #gen_scaled_dict[name] = gen_scaled.clone()

        # -----------------------------------------------------
        # 🔥 핵심 변경: 벌점(Penalty) 수식 적용
        # Score = imp_score / (1 + beta * gender_score)
        # 원리: 성능이 아무리 좋아도(분자 ↑), 편향이 심하면(분모 ↑) 점수가 확 깎임.
        # -----------------------------------------------------
        denominator = 1.0 + (beta * gen_scaled)
        hybrid_score = imp_scaled / denominator
        
        hybrid_importance[name] = hybrid_score

    # ---------------------------------------------------------
    # 3. 통계 확인 (디버깅용)
    # ---------------------------------------------------------
    # 🔥 통계 분석 및 히스토그램 생성
    #print("\n" + "="*80)
    #print("📊 Score 분석 시작...")
    #print("="*80)
    
    # analyze_and_save_scores(imp_scaled_dict, gen_scaled_dict, 
    #                       output_dir='score_analysis')
    
    #print("="*80)
    #print("📊 Score 분석 완료!")
    #print("="*80 + "\n")


    # ---------------------------------------------------------
    # 4. 마스크 생성 (Top-K 방식 유지)
    # ---------------------------------------------------------
    keep_ratio = 1.0 - prune_ratio
    mask_by_layernames = {}

    def safe_topk_mask(score_tensor, keep_ratio):
        flat = score_tensor.flatten().cpu() # Top-k는 CPU에서 계산 추천 (메모리 절약)
        N = flat.numel()
        k = int(N * keep_ratio)

        if k <= 0: return torch.zeros_like(flat, dtype=torch.long)
        if k >= N: return torch.ones_like(flat, dtype=torch.long)

        # 상위 k개 선택
        _, topk_indices = torch.topk(flat, k, largest=True, sorted=False)
        
        mask = torch.zeros_like(flat, dtype=torch.long)
        mask[topk_indices] = 1
        return mask.view_as(score_tensor).to(score_tensor.device) # 다시 GPU로

    for name, layer in model.named_modules():
        # ... (기존 코드와 동일하게 레이어 필터링) ...
        # 여기서는 예시로 supported_layers 체크 생략 (작성하신 코드엔 포함되어야 함)
        if type(layer).__name__ not in supported_layers:
            continue

        if hasattr(layer, 'weight'): # 가중치 있는 레이어만
            
            if is_classifier_head(name):
                # 분류기는 무조건 보존
                mask = torch.ones_like(layer.weight, dtype=torch.long)
            elif name in hybrid_importance:
                # 계산된 점수로 마스킹
                score_tensor = hybrid_importance[name]
                mask = safe_topk_mask(score_tensor, keep_ratio)
            else:
                # 점수가 없으면(예: Conv가 아닌 레이어 등) 보존 or 삭제 정책 결정
                mask = torch.ones_like(layer.weight, dtype=torch.long)
            
            mask_by_layernames[name] = mask

    # 리스트로 변환 (GPU 이동)
    mask_list = [mask.to(device) for name, mask in mask_by_layernames.items()]
    
    print(f"마스크 추출 완료: 총 {len(mask_list)}개 레이어 적용됨.")
    return mask_list




def compute_importance(model, gender_model, test_csv, new_img_dir=None, masked_grads=True, output_cols_each_task=[(0,7),(7,9),(9,18)], col_names=['race','gender'],network=None,optimizer=None, lr=1e-4, stop_batch=10000, sensitive_group=None, sensitive_classes=None):
    
    print("-------------compute_importance-------------")
    torch.autograd.set_detect_anomaly(True)

    # 0) model의 mask를 모아둠 (compute_importance 시작 부분에)
    def collect_masks_from(model):
        masks = {}
        for n, m in model.named_modules():
            if hasattr(m, 'mask'):
                masks[n] = m.mask.detach().cpu().clone()
        return masks

    model_masks = collect_masks_from(model)

    def is_classifier_head(name):  # MobileNetV2 기준
        return name.startswith('classifier.')

    
    
    supported_layers = ['Linear', 'Conv2d', 'Conv1d'] # 점수를 계산할 레이어 선정

    #! 1) 모델 로드 & 준비
    model.train() # 학습 모드로 전환
    gender_model.train()

    #! 2) 옵티마이저 준비
    #if optimizer is None:
    #    optimizer = optim.Adam(model.parameters(), lr=lr)
    #    gender_optimizer = optim.Adam(gender_model.parameters(), lr=lr)
    
    #! 3) 데이터 준비
    if isinstance(test_csv, str):
        test_frame = pd.read_csv(test_csv)
    else:
        test_frame = test_csv.copy()

    if new_img_dir: # 새로운 디렉토리 경로가 주어졌다면, 해당 디렉토리에 이미 이미 존재하는 이미지만 필터링
        faces = set(os.listdir(new_img_dir))
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

    # 데이터 로더 준비
    test_loader,_ =  make_datasets(test_frame,test_frame,True, 448, col_used=col_names)

    #! 4) 손실 함수
    criterion = nn.CrossEntropyLoss()

    #! 5) 누적 딕셔너리 초기화
    imp_grad_accum = {}
    gender_grad_accum = {}
    importance_score = {}
    gender_score = {}
    mask_at_each_layer = {}
    batches_imp = 0


    #! 6) 배치(이미지)별 순환
        #! 1. 미니-배치마다 순전파 → 손실 → 역전파로 모든 가중치의 그래디언트 계산
        #! 2. 레이별 가중치의 그래디언트 복사(clone().detach().cpu())
        #! 3. 가지치기된 가중치의 그래디언트를 0으로 
        #! 4. 가중치와 가중치의 그래디언트을 곱한 결과의 제곱으로 중요도를 근사
        #! 5. 배치마다 구해진 해당 가중치의 근사 값을 누적하고 배치 개수로 나눠서 최종 가중치의 중요도 값의 평균 도출

    for batch_idx, sample_batched in enumerate(test_loader):
        if batch_idx >= stop_batch:
            break
        batches_imp += 1
        if batch_idx % 200 == 0:
            print(f"{batch_idx}번째 mini-batch acc_imp 계산 중!")

        # (1) 배치 데이터 준비
        image_batched, label_batched = sample_batched 
        image_batched = image_batched.to(device, dtype=torch.float)
        label_batched = label_batched.to(device)

        # (2) 이전 배치 그래디언트 초기화
       

        # (3) 순전파 → 손실 계산 → 역전파
        outputs = model(image_batched)
        imp_loss = loss_multi_tasks(outputs, label_batched, criterion, output_cols_each_task)
        model.zero_grad()
        imp_loss.backward() # 전체 가중치의 그래디언트 구하기

        # (4) 레이어별 그래디언트 읽어서 누적
        for name, layer in model.named_modules(): 
            if type(layer).__name__ not in supported_layers:
                continue
            # (4-1) gradient
            grads = layer.weight.grad.clone().detach().cpu() # 현재 배치의 그래디언트 복사
            
            # (4-2) mask 적용 (pruned weight 제외)
            if masked_grads and hasattr(layer, 'mask'):
                masks = layer.mask.clone().detach().cpu() # 이전에 저장되었던 마스크 값 가져오기
                mask_at_each_layer[name] = [int(masks.sum()), grads.shape]
                grads *= masks # 마스크에 비활성화 처리된 가중치는 모두 0으로 처리
                    
                # (4-3) 테일러 급수를 이용한 근사 - (그래디언트*가중치)^2
                weights = layer.weight.data.clone().detach().cpu()
                hess = (weights.abs() * grads.abs())**2
            
            # (4-4) 미니 배치에 대한 레이어별 각 가중치 그래디언트/테일러 급수 근사 결과 누적합
            if name not in imp_grad_accum:
                imp_grad_accum[name] = grads
                importance_score[name] = hess
            else:
                imp_grad_accum[name] += grads # 배치마다 grads 텐서(모든 weight별 gradient)를 더하고
                importance_score[name] += hess
        
        #optimizer.step()


        # TODO: gender_imp 근사--------------------------------------------------------------------------------
    batches_gender = 0    
    for batch_idx, sample_batched in enumerate(test_loader):
        if batch_idx >= stop_batch:
            break
        batches_gender += 1
        if batch_idx % 200 == 0:
            print(f"{batch_idx}번째 mini-batch gender_imp 중요도 계산 중!")

        image_batched, label_batched = sample_batched 
        image_batched = image_batched.to(device, dtype=torch.float)
        gender_labels = label_batched[:, -1].long().to(device)
        image_batched_detached = image_batched.detach().requires_grad_(True) # gender_model의 계산이 model의 계산 상태에 절대 영향을 받지 않도록

        
        outputs = gender_model(image_batched_detached)
        gender_loss = criterion(outputs, gender_labels)
        gender_model.zero_grad()
        gender_loss.backward()

        for name, layer in gender_model.named_modules():
            if type(layer).__name__ not in supported_layers:
                continue
            if not hasattr(layer, 'weight') or layer.weight.grad is None:
                continue

            grads   = layer.weight.grad.detach().cpu().clone()
            weights = layer.weight.detach().cpu().clone()

            # --- 마스크 적용 규칙 ---
            # 1) 분류기 헤드는 마스킹 "제외"
            # 2) 그 외 레이어는 model의 동일 레이어 마스크가 있으면 적용
            mask = None if is_classifier_head(name) else model_masks.get(name, None)
            if masked_grads and mask is not None and mask.shape == grads.shape:
                grads *= mask              # hess 계산 "직전"에 grad 마스킹

            # hess 계산 (항상)
            hess = (weights * grads).pow(2)

            # (선택) 한 번 더 보증 차원에서 hess에도 마스킹
            if masked_grads and mask is not None and mask.shape == hess.shape:
                hess *= mask

            # 누적
            if name not in gender_grad_accum:
                gender_grad_accum[name] = grads
                gender_score[name] = hess
            else:
                gender_grad_accum[name] += grads
                gender_score[name] += hess
        #gender_optimizer.step()


    #! 7) 모든 배치 처리 후에 평균 내기
    for name in imp_grad_accum:
        imp_grad_accum[name] /= batches_imp
        importance_score[name] /= batches_imp
        # 미니 배치에 대한 레이어별 각 가중치 그래디언트 누적합을 배치 수로 전체 배치에 대한 특정 가중치의 그래디언트 평균 구하기
    #print(f"**{name}레이어의 importance_score: {importance_score[name]}**")

    for name in gender_grad_accum:
        gender_grad_accum[name] /= batches_gender
        gender_score[name] /= batches_gender
        # 미니 배치에 대한 레이어별 각 가중치 그래디언트 누적합을 배치 수로 전체 배치에 대한 특정 가중치의 그래디언트 평균 구하기
    #print(f"**{name}레이어의 eo_score: {eo_score[name]}**")

    
    # 🔍 디버깅: 계산된 레이어 확인
    print("\n🔍 Importance Score 계산된 레이어:")
    for name in importance_score.keys():
        print(f"  ✓ {name}: shape={importance_score[name].shape}")
    
    print("\n🔍 Gender Score 계산된 레이어:")
    for name in gender_score.keys():
        print(f"  ✓ {name}: shape={gender_score[name].shape}")
    
    # (8-5) 최종 return 시 gender_score도 함께 반환하도록 변경
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
            outputs = model(image_batched)
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
    
def importance_by_class1(model_path, test_csv, new_img_dir=None, masked_grads=True, output_cols_each_task=[(0,7),(7,9),(9,18)], col_names=['race','gender'],network=None,sample_per_class=32,optimizer=None, lr=1e-4, n_classes=2):
    # 그룹별 gradient 정보 계산 (두 번째 버전)
    supported_layers = ['Linear', 'Conv2d', 'Conv1d']
    model = model_path 
    
    test_frame = pd.read_csv(test_csv) if isinstance(test_csv, str) else test_csv
    criterion = nn.CrossEntropyLoss()
    criterion_sensitive = nn.BCELoss()
    activation = nn.Sigmoid()

    if new_img_dir:
        initial_rows = test_frame.shape[0]
        faces = set(os.listdir(new_img_dir))
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
    test_loader = iter(test_loader)
    sensitive_cols_in_target = len(output_cols_each_task)
    images, targets, comb_idx = fetch_a_fair_batch(test_loader, n_classes, sample_per_class, sensitive_cols_in_target)
    targets = targets.to(device)
    sensitive_groups = sorted([[int(i) for i in comb.split('_')] for comb in comb_idx])
    sensitive_group_idx_in_output = [i for i in range(len(sensitive_groups))]

    outputs = torch.squeeze(model(images.to(device)))

    grad_each_group = {}
    H_each_group = {}
    mask_at_each_layer = {}
    for group_idx, group in enumerate(sensitive_groups):
       gradients = {}
       hessians = {}
       obs_this_group = torch.squeeze((targets[:, sensitive_cols_in_target] == group[0]).nonzero())
       output_cols_for_non_protected = output_cols_each_task[:(len(output_cols_each_task))]
       outputs_this_group = outputs[obs_this_group,:]
       targets_this_group = targets[obs_this_group,:]
       loss_non_protected = loss_multi_tasks(outputs_this_group,targets_this_group,criterion,output_cols_for_non_protected)
       cur_sensitive_group_output = outputs[obs_this_group, sensitive_group_idx_in_output[group_idx]].to(device)
       sensitive_target_this_group = torch.squeeze(targets[obs_this_group, sensitive_cols_in_target] == group[0]).clone().float().to(device)
       loss = loss_non_protected + criterion_sensitive(activation(cur_sensitive_group_output).view(-1), sensitive_target_this_group).cuda()
       loss = loss_non_protected

       try:
           loss.backward(retain_graph=True)
       except:
           print(loss)

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
       grad_each_group[group_idx] = copy.deepcopy(gradients)
       H_each_group[group_idx] = copy.deepcopy(hessians)
                     
    return grad_each_group, H_each_group

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

def importance_by_class2(model_path, test_csv, new_img_dir=None, output_cols = [(0,7)], col_names=['race'],masked_grads=True,sample_per_class=10,lr=1e-5):
    # 세 번째 버전의 그룹별 gradient 계산
    supported_layers = ['Linear', 'Conv2d', 'Conv1d']

    model = model_path 

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    test_frame = pd.read_csv(test_csv) if isinstance(test_csv, str) else test_csv
    criterion = nn.BCELoss()
    criterion_sensitive = nn.BCELoss()
    activation = nn.Sigmoid()

    if new_img_dir:
        initial_rows = test_frame.shape[0]
        faces = set(os.listdir(new_img_dir))
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
    test_loader = iter(test_loader)
    target_col = 0
    num_classes = output_cols[0][1] - output_cols[0][0]
    images, targets, comb_idx = fetch_a_fair_batch(test_loader, num_classes, sample_per_class,target_col)

    outputs = torch.squeeze(model(images.to(device)))

    group_outputs = outputs
    group_targets = torch.squeeze(targets[:, target_col]).to(device)

    groups = sorted([[int(i) for i in comb.split('_')] for comb in comb_idx])
    grad_each_group = {}
    H_each_group = {}
    mask_at_each_layer = {}

    for group_idx, group in enumerate(groups):
       gradients = {}
       hessians = {}
       output_this_group = group_outputs[:, group]
       target_this_group = (group_targets == group[0]).clone().detach().float()
       loss = criterion(activation(output_this_group).view(-1), target_this_group)
       try:
           loss.backward(retain_graph=True)
       except:
           print(loss)
       for name, layer in model.named_modules():
            if type(layer).__name__ in supported_layers:
                grads = layer.weight.grad.clone().detach().cpu()
                weights = layer.weight.data.clone().detach().cpu()
                if masked_grads:
                    masks = layer.mask.clone().detach().cpu()
                    mask_at_each_layer[name] = [torch.sum(masks), grads.shape]
                    grads *= masks
                hessians[name] = weights.abs() * grads.abs()
                gradients[name] = grads
       grad_each_group[group_idx] = copy.deepcopy(gradients)
       H_each_group[group_idx] = copy.deepcopy(hessians)

    return grad_each_group, H_each_group

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