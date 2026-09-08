"""§6-5 보조 진단: 같은 H(이식본 faithful 점수 pass, 결정론 설정)를 원본 greedy 와 이식본 greedy 에 넣어
레이어별 마스크 차이가 '정확한 동률(H==0 등) 안에서의 교환'인지 판정한다.
- 원본 torch.topk(sorted=True) 는 동률 순서를 정의하지 않고, 이식본은 np.argsort(kind='stable') 이라
  죽은 채널(ReLU6)처럼 H 가 정확히 0 인 가중치 블록 안에서는 서로 다른 가중치를 고를 수 있다 (설계 문서가 허용하는 차이).
- swapped-values-identical=True 이면 그 층의 차이는 전부 동률 교환이다.

실행: PYTHONPATH=/workspace/FairGRAPE/np_compat /workspace/miniconda/envs/fairgrape_cu128/bin/python tests/check_fg_orig_ties.py
"""
import os, sys, copy
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT); sys.path.insert(0, os.path.join(_ROOT, 'tests'))
import torch, numpy as np
import prune, prune_orig_ref
from util import setseed
import compare_fg_orig as C
torch.backends.cudnn.enabled = False; torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
frames = C.build_frames(42); tf = frames['train']
base = C.build_masked_model(prune.device)
mB = copy.deepcopy(base); setseed(42)
H = prune.fg_orig_importance_by_group(mB, tf, C.FACE_DIR, C.OUTPUT_COLS, C.COL_USED, 5, 'faithful', masked_grads=True)
names = [n for n, _ in C.supported_named_layers(mB)]
prune_orig_ref.importance_by_class0 = lambda *a, **k: (None, {g: {n: t.clone() for n, t in H[g].items()} for g in H})
masks_A = prune_orig_ref.fairness_grad(mB, 0.025, None, None, sensitive_classes=2, impt_type=0, para_batch=1, delta_p=0)
masks_B, _ = prune.fg_orig_greedy_masks(mB, H, 0.025, names, 1, 0)
print(f"{'layer':<24}{'numel':>8}{'removed':>8}{'H==0(both g)':>14}{'ndiff':>7}{'agree':>8}  swapped-values-identical")
tot_diff = 0; all_tie = True
for n, a, b in zip(names, masks_A, masks_B):
    a = a.cpu().reshape(-1); b = b.cpu().reshape(-1)
    h0 = H[0][n].reshape(-1); h1 = H[1][n].reshape(-1)
    nz = int(((h0 == 0) & (h1 == 0)).sum())
    diff = (a != b); nd = int(diff.sum()); tot_diff += nd
    onlyA = (a == 0) & (b == 1); onlyB = (b == 0) & (a == 1)   # A removed / B removed
    va = sorted(zip(h0[onlyA].tolist(), h1[onlyA].tolist())); vb = sorted(zip(h0[onlyB].tolist(), h1[onlyB].tolist()))
    same = (va == vb)
    all_tie &= same
    if nd > 0 or nz > 0:
        print(f"{n:<24}{a.numel():>8}{int((a==0).sum()):>8}{nz:>14}{nd:>7}{(1-nd/a.numel()):>8.4f}  {same}  (H==0 removed by A: {int(((a==0)&(h0==0)&(h1==0)).sum())}, by B: {int(((b==0)&(h0==0)&(h1==0)).sum())})")
print("total differing entries:", tot_diff, "| all differences are exact-tie swaps:", all_tie)
