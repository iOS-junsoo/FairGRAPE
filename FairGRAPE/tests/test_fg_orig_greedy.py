"""impt_type=4 greedy 이식 검증: 같은 그룹별 중요도 H 를 넣었을 때 원본(prune_orig_ref.fairness_grad 의 greedy)과
이식본(prune.fg_orig_greedy_masks, numpy flat-index)이 레이어별로 같은 마스크를 내는지 본다.
점수 pass(모델 forward/backward)는 관여하지 않으므로 수치 비결정성 없이 알고리즘 동일성만 검사한다.

실행: PYTHONPATH=/workspace/FairGRAPE/np_compat /workspace/miniconda/envs/fairgrape_cu128/bin/python tests/test_fg_orig_greedy.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

import prune
import prune_orig_ref


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(3, 8, 3)
        self.c1 = nn.Conv2d(8, 16, 1)
        self.fc = nn.Linear(16, 2)


def make_model():
    m = TinyNet().to(prune.device)
    for layer in m.modules():
        if type(layer).__name__ in prune.supported_layers:
            layer.mask = nn.Parameter(torch.ones_like(layer.weight))
    return m


def run_case(model, names, prune_ratio, heavy, para_batch, delta_p, seed):
    torch.manual_seed(seed)
    mods = dict(model.named_modules())
    H = {g: {n: (torch.rand_like(mods[n].weight).cpu() ** (6 if heavy else 1)) * (1.0 if g == 0 else 0.3)
             for n in names} for g in (0, 1)}
    # 원본 fairness_grad 의 점수 pass 를 같은 H 로 대체 (greedy 부분만 비교)
    prune_orig_ref.importance_by_class0 = lambda *a, **k: (None, {g: {n: t.clone() for n, t in H[g].items()} for g in H})
    masks_A = prune_orig_ref.fairness_grad(model, prune_ratio, None, None, sensitive_classes=2, impt_type=0,
                                           para_batch=para_batch, delta_p=delta_p)
    masks_B, stats = prune.fg_orig_greedy_masks(model, H, prune_ratio, names, para_batch, delta_p)
    for n, a, b in zip(names, masks_A, masks_B):
        a, b = a.cpu(), b.cpu()
        assert a.shape == b.shape
        assert int(a.sum()) == int(b.sum()) == int(a.numel() * (1 - prune_ratio)), (n, int(a.sum()), int(b.sum()))
        agree = (a == b).float().mean().item()
        assert agree == 1.0, f"{n}: mask agreement {agree:.4f} != 1.0 (prune_ratio={prune_ratio}, heavy={heavy}, para_batch={para_batch}, delta_p={delta_p})"
        tgt = prune_orig_ref.LAST_GRAD_TARGET[n]
        st = [s for s in stats if s['name'] == n][0]
        assert np.allclose(tgt, st['target_prop'], atol=1e-6), (n, tgt, st['target_prop'])


def test_greedy_matches_original():
    model = make_model()
    names = [n for n, l in model.named_modules() if type(l).__name__ in prune.supported_layers]
    cases = [(0.025, False, 1, 0), (0.025, True, 1, 0), (0.5, True, 1, 0), (0.9, False, 1, 0),
             (0.3, True, 3, 0), (0.3, True, 1, 1), (0.3, True, 1, 2)]
    for i, (pr, heavy, pb, dp) in enumerate(cases):
        run_case(model, names, pr, heavy, pb, dp, seed=i)
        print(f"case {i}: prune_ratio={pr}, heavy={heavy}, para_batch={pb}, delta_p={dp} → 원본과 마스크 100% 일치")
    # 두 번째 iter(이미 프루닝된 마스크에서 후보 = 활성 가중치만) 도 동일한지
    masks_B, _ = prune.fg_orig_greedy_masks(model, {g: {n: torch.rand_like(dict(model.named_modules())[n].weight).cpu() for n in names} for g in (0, 1)}, 0.3, names, 1, 0)
    for (n, l), mk in zip([(n, l) for n, l in model.named_modules() if type(l).__name__ in prune.supported_layers], masks_B):
        l.mask = nn.Parameter(mk.clone(), requires_grad=False)
    run_case(model, names, 0.5, True, 1, 0, seed=99)
    print("2nd iter (활성 가중치만 후보) → 원본과 마스크 100% 일치")
    print('test_greedy_matches_original ok')


if __name__ == '__main__':
    test_greedy_matches_original()
