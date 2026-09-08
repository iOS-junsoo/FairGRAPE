"""원본 FairGRAPE prune.py (Bernardo1998/FairGRAPE, commit b677eb9) 대조용 사본.

Activate_diff 의 impt_type=4 (FG_ORIG_MODE='faithful') 가 원본과 같은 마스크를 내는지 tests/compare_fg_orig.py 에서
같은 환경으로 돌려 비교하기 위해 둔다. main_test 에서는 import 하지 않는다.

원본 대비 변경점 (모두 `# [REF-HOOK]` 주석으로 표시):
  - fairness_grad 가 계산한 레이어별 grad_target 을 모듈 전역 LAST_GRAD_TARGET 에 남긴다 (비교용, 동작 불변).
  - 원본의 탭 들여쓰기를 공백으로 통일 (동작 불변).
  - fairness_grad 의 nonzero() 인덱스를 CPU 로 옮김 (torch 2.x 에서 CPU 텐서를 CUDA 인덱스로 색인 불가; 값·순서 불변).
그 외 코드(버그 Q1·Q2·Q3 포함)는 원본 그대로다.
"""
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


# custom codes
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

LAST_GRAD_TARGET = {}  # [REF-HOOK] fairness_grad 가 마지막으로 계산한 {layer_name: np.array(n_classes)} (비교용)

################
# Based on SNIP code from github
################
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
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))        

    # Expected mask should be a list of arrays, len = number of prunable layers, same shapes as weights
    def apply_hook(self, masks):
        layers = filter(lambda l: type(l).__name__ in forward_mapping_dict, self.prun_model.modules())
        def apply_masking(mask):
            def hook(weight):
                return weight * mask
            return hook
        for layer, mask in zip(layers, masks):
            assert layer.weight.shape == mask.shape
            layer.weight.data = layer.weight.data * mask
            layer.weight.register_hook(apply_masking(mask))

    def prune(self, prune_cfgs, show_pruned_details=False):  
        masks = self.get_mask(prune_cfgs) # get_mask() need to be implemented by child classes
        if self.save_mask:
            print(type(masks), type(masks[0]))
            mask_np = np.array([m.cpu().numpy() for m in masks])
            np.save("mask.npy",mask_np)
            del mask_np
        self.apply_hook(masks)
        mask_by_layer = {}
        for layer in self.prun_model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(masks.pop(0), requires_grad=False)
                mask_by_layer[layer] = layer.mask
        if show_pruned_details:
            self.print_remain()
        return self.prun_model

    def update_forward_pass(self):
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.forward = types.MethodType(forward_mapping_dict[type(layer).__name__], layer)

    def variance_scaling_init(self):
        for layer in self.model.modules():
            if type(layer).__name__ in forward_mapping_dict:
                layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))
                nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False

    def print_remain(self):
        remain, total = 0, 0
        for name, layer in self.prun_model.named_modules():
            if type(layer).__name__ in forward_mapping_dict:
                remain += torch.sum(layer.mask)
                total += torch.prod(torch.tensor(layer.weight.shape))
                print(name, torch.sum(layer.mask), layer.weight.shape)
        print(remain, total, remain/total)


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

    def get_mask(self, prune_cfgs):
        prune_ratio, test_csv, new_img_dir, sensitive_classes, masked_grads, output_cols_each_task ,col_used, para_batch, impt_type, stop_batch, delta_p = prune_cfgs
        print("Sensitive classes:",sensitive_classes)
        mask = fairness_grad(self.model, prune_ratio, test_csv, new_img_dir, sensitive_classes, masked_grads, output_cols_each_task ,col_used, para_batch, impt_type, stop_batch, delta_p)
        return mask

def fairness_grad(model, prune_ratio, test_csv, new_img_dir=None, sensitive_classes = 2, masked_grads=True, output_cols_each_task=[(0,7),(7,9),(9,18)],col_names=['race','gender'], para_batch=1, impt_type = 0, stop_batch=10000, delta_p=False,n_jobs=1):

    if impt_type == 0:
        _,grad_mag_by_race = importance_by_class0(model, test_csv, new_img_dir=new_img_dir, masked_grads=masked_grads, output_cols_each_task=output_cols_each_task,col_names=col_names,stop_batch=stop_batch)
    elif impt_type == 1:
        _,grad_mag_by_race = importance_by_class1(model, test_csv, new_img_dir=new_img_dir, masked_grads=masked_grads, output_cols_each_task=output_cols_each_task,col_names=col_names, n_classes=sensitive_classes)	
    elif impt_type == 2:
        _,grad_mag_by_race = importance_by_class2(model, test_csv, new_img_dir, output_cols_each_task,col_names)	

    # calculate the target distribution of gradient on pre-pruning model at each layer
    # Note that this input model might have been previously pruned as well.
    grad_mag_each_race = defaultdict(list)
    for race in grad_mag_by_race.keys():
        race_grad_mag = grad_mag_by_race[race]
        for layer_name in race_grad_mag:
            grad_mag_each_race[layer_name].append(torch.sum(race_grad_mag[layer_name].abs()))

        # CAUTION, grads still have negatives.
    n_classes = sensitive_classes
    grads_by_race_merged = make_mask_by_grad(grad_mag_by_race,n_classes)
    unpruned_grad = grad_mag_each_race
    grad_target, grad_target_total = {}, np.array([0.0] * n_classes)
    for layer_name in unpruned_grad:
        grad_this_layer = unpruned_grad[layer_name]
        grad_target[layer_name] = np.array([grad/sum(grad_this_layer) for grad in grad_this_layer])
        #print(np.array(grad_this_layer))
        grad_target_total += np.array(grad_this_layer)

    # [REF-HOOK] 비교용으로 레이어별 target 점유율을 남긴다 (동작 불변)
    global LAST_GRAD_TARGET
    LAST_GRAD_TARGET = {k: np.array([float(v) for v in vals]) for k, vals in grad_target.items()}

    # For each weight, record its group-wise importance and idx within layer.
    # Notice! The selection below can be done using different metrics
    grad_by_layer_sorted = {}
    for name, layer in model.named_modules():
        grad_by_layer_sorted[name] = {}
        if type(layer).__name__ not in supported_layers:
            continue
        selected = layer.mask
        idxs,idxs_tp = selected.nonzero(), selected.nonzero(as_tuple=True)
        # [REF-HOOK] torch 2.x 호환: CPU 텐서(grads_by_race_merged)를 CUDA 인덱스로 색인할 수 없어 인덱스만 CPU 로 옮긴다 (값·순서 불변)
        idxs, idxs_tp = idxs.cpu(), tuple(i.cpu() for i in idxs_tp)
        grad_this_layer = grads_by_race_merged[name][idxs_tp]
        sum_per_node = torch.sum(grad_this_layer, 1)
        for race in range(n_classes):
            race_col = grad_this_layer[:, race]
            # While a node might have high importance for one race, it might also have even 
            # larger importance for another, which ultimately decreases share for this race.
            if delta_p == 1:
                race_col /= sum_per_node
            elif delta_p == 2:
                race_col *= (race_col / sum_per_node)
            _, sorted_idx = torch.topk(race_col, k = len(race_col), sorted=True)
            grad_by_layer_sorted[name][race] = [grad_this_layer[sorted_idx], idxs[sorted_idx]]

    ####################
    # greedy method
    ####################
    mask_list = []
    # record how many weights to select at each layer, use for layer wise connection
    nodes_each_layer = {}
    for i in range(n_classes):
        nodes_each_layer[i] = []

    layer_parameters = []
    for name,layer in model.named_modules():
        layer_parameters.append([name, layer,grad_by_layer_sorted[name],grad_target,n_classes])

    # This function is designed to facilitate parallel processing, but only n_jobs = 1 available for now.
    def greed_one_layer(layer_parameter):
        name, layer,grad_by_layer_sorted_layer,grad_target,n_classes = layer_parameter
        if type(layer).__name__ not in supported_layers:
            return {name:None}
        print("Performing greedy selection on {}".format(name))
        mask_this_layer = torch.zeros(layer.weight.shape)
        layer_total = int(torch.prod(torch.tensor(layer.weight.shape)))
        num_to_select_this_layer = int(layer_total * (1-prune_ratio))

        n_selected_this_layer = 0
        last_printed_freq = 0

        grad_target_this_layer = grad_target[name]
        grads_by_race_selected = np.array([0] * n_classes, dtype=float)
        grads_prop_by_race = np.array([1/n_classes] * n_classes, dtype=float)

        grads_by_race_idx = np.array([0] * n_classes)
        last_race_updated = 0
        while n_selected_this_layer < num_to_select_this_layer:
            # find the race that currently has the larget deficient
            race_diff = grads_prop_by_race - grad_target_this_layer
            if last_race_updated == 0:
                race_to_add = race_diff.argmin()
            last_race_updated = last_race_updated + 1 if last_race_updated < para_batch else 0
            idx_in_seq = grads_by_race_idx[race_to_add]
            # grads here are already abs
            grads = grad_by_layer_sorted_layer[race_to_add][0][idx_in_seq]
            idx = tuple(grad_by_layer_sorted_layer[race_to_add][1][idx_in_seq])
            selected_condition = mask_this_layer[idx]
            # only add weights that have neer been selected
            if selected_condition == 0:
                n_selected_this_layer += 1
                grads_by_race_selected += grads.cpu().numpy()
                grads_prop_by_race = grads_by_race_selected / sum(grads_by_race_selected)
                mask_this_layer[idx] = 1
            grads_by_race_idx[race_to_add] += 1
        
        return {name:mask_this_layer}

    names_and_masks = Parallel(n_jobs=n_jobs)(delayed(greed_one_layer)(lp) for lp in layer_parameters)

    mask_by_layernames = dict([pair for d in names_and_masks for pair in d.items()])
    mask_list = [mask_by_layernames[name].to(device) for name,layer in model.named_modules() if type(layer).__name__ in supported_layers]

    return mask_list

# The last col in the label matrix is for sensitive attr, others for non-protected ones
# This order of label is given by col_names, the last one is the sensitive group.
def importance_by_class0(model_path, test_csv, new_img_dir=None, masked_grads=True, output_cols_each_task=[(0,7),(7,9),(9,18)], col_names=['race','gender'],network=None,optimizer=None, lr=1e-4, stop_batch=10000):
    supported_layers = ['Linear', 'Conv2d', 'Conv1d']

    # Load pruned and retrained model
    model = model_path 

    model.train()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    test_frame = pd.read_csv(test_csv) if isinstance(test_csv, str) else test_csv
    criterion = nn.CrossEntropyLoss()
    criterion_sensitive = nn.BCELoss()
    activation = nn.Sigmoid()

    # Make sure all images in test frame exist
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
    sensitive_cols_in_target = len(output_cols_each_task)
    sensitive_groups = sorted(set(test_frame[col_names[-1]]))

    # do mini-batches to get results
    grad_each_group = {}
    H_each_group = {}
    mask_at_each_layer = {}
    batches = 0
    for batch_idx, sample_batched in enumerate(test_loader):
        if batch_idx >= stop_batch:
            break
        batches += 1
        if batch_idx % 200 == 0:
            print("{}th mini-batch of importance!".format(batch_idx))
        image_batched, label_batched = sample_batched
        image_batched = image_batched.to(device, dtype=torch.float)
        # transfer it all to gpu
        label_batched = label_batched.to(device)
        for group_idx, group in enumerate(sensitive_groups):
           gradients = {}
           hessians = {}
           # calculate non-protected loss for this group only 
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

           # get and save all gradient for this group
           for name, layer in model.named_modules():
                if type(layer).__name__ in supported_layers:
                    grads = layer.weight.grad.clone().detach().cpu()
                    weights = layer.weight.data.clone().detach().cpu()
                    # Confirm the model is actually pruned
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

# Keys of grad_each_group are sensitive groups.
# Keys of grad_at_each_layer are model layers.
def make_mask_by_grad(grad_each_group, n_classes=7):
    groups = [i for i in range(n_classes)]
    layer_names = list(grad_each_group[0].keys())
    grad_at_each_layer = {}
    for layer in layer_names:
        layer_shape = tuple(list(grad_each_group[groups[0]][layer].shape)+[1])
        grad_merged = torch.cat([grad_each_group[group][layer].view(layer_shape) for group in groups], dim=len(layer_shape)-1)
        grad_at_each_layer[layer] = grad_merged
    return grad_at_each_layer
