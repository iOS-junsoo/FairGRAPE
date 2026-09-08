"""§6-5 원본 대조: Activate_diff impt_type=4 (FG_ORIG_MODE='faithful', fg_scope='all') vs 원본 prune.py 사본(prune_orig_ref).

같은 dense 체크포인트·같은 frames['train']·같은 seed 로
  (A) prune_orig_ref.fairness_grad(impt_type=0)  — 원본 코드 그대로
  (B) prune.fairness_grad(impt_type=4)           — 이식본, faithful 모드
를 각각 deepcopy 한 모델에 대해 실행하고 레이어별 kept·grad_target·마스크 일치율을 표로 출력한다.

실행 (저장소 루트에서, GPU 필요):
  PYTHONPATH=/workspace/FairGRAPE/np_compat /workspace/miniconda/envs/fairgrape_cu128/bin/python tests/compare_fg_orig.py \
      [--stop_batch 5] [--prune_ratio 0.025] [--out <결과 md 경로>]
판정 기준 (03_fairgrape_orig_blocks.md §6-5):
  - 53개 레이어 모두 kept 일치
  - 레이어별 grad_target 소수 4자리 일치
  - 마스크 일치율 (mA==mB).mean() ≥ 0.99  (동률·cuDNN 비결정성 때문에 100% 는 요구하지 않음)
"""
import argparse
import copy
import os
import sys
import time
import types

# --deterministic: cuBLAS 결정론 설정은 CUDA 초기화 전(=torch import 전)에 환경변수로 넣어야 한다.
if '--deterministic' in sys.argv:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

import config
import prune
import prune_orig_ref
from dataset import make_frame, make_datasets, apply_fscl_skew, make_balanced_eval
from util import make_model, setseed, custom_forward_conv2d

DENSE_CKPT = 'trained_model/unpruned/DENSE_utk_age_byrace_beta4_seed42.pt'
CSV = 'csv/UTKFace_labels_full.csv'
FACE_DIR = 'Images/UTKFace'
COL_USED = ['age_bin', 'race']
OUTPUT_COLS = [(0, 2)]


def build_frames(seed):
    """main_test.experiment() 와 같은 방식 (--race_binary --skew_beta 4 --loss_type age --sensitive_group race, 가독성 검사 생략)."""
    setseed(seed)
    frames = make_frame(CSV, FACE_DIR, seven_races=False, drop_race=0, race_binary=True, stratify=True,
                        split_csv_out=None)  # 분할 csv 는 다시 쓰지 않는다 (seed 42 고정이라 내용 동일)
    frames['train'] = apply_fscl_skew(frames['train'], 4, skew_col='age_bin')
    frames['val'] = make_balanced_eval(frames['val'], cols=('race', 'age_bin'))
    frames['test'] = make_balanced_eval(frames['test'], cols=('race', 'age_bin'))
    return frames


def build_masked_model(device):
    """dense 체크포인트 로드 → 커스텀 forward(weight*mask) 교체 → mask=1 등록. (Prunner.__init__ + init_mask 와 동일)"""
    model = make_model(network='mobilenetv2', n_classes=2).to(device)
    ck = torch.load(DENSE_CKPT, map_location=device)
    model.load_state_dict(ck['model_state'] if isinstance(ck, dict) and 'model_state' in ck else ck)
    for layer in model.modules():
        if type(layer).__name__ in prune.forward_mapping_dict:
            layer.forward = types.MethodType(prune.forward_mapping_dict[type(layer).__name__], layer)
            layer.mask = nn.Parameter(torch.ones_like(layer.weight).to(device))
    return model


def supported_named_layers(model):
    return [(n, l) for n, l in model.named_modules() if type(l).__name__ in prune.supported_layers and hasattr(l, 'weight')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stop_batch', type=int, default=5)
    ap.add_argument('--prune_ratio', type=float, default=0.025)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default=None, help='결과 표(markdown) 저장 경로')
    ap.add_argument('--deterministic', action='store_true',
                    help='A/B 모두 cuDNN 비활성 + torch.use_deterministic_algorithms 로 실행해 수치 비결정성(cuDNN 알고리즘 차이를 '
                         'faithful 모드의 Adam step 이 증폭)을 제거하고 알고리즘 동일성만 본다')
    args = ap.parse_args()
    device = prune.device
    if args.deterministic:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[deterministic] cudnn=off, use_deterministic_algorithms=True, CUBLAS_WORKSPACE_CONFIG=", os.environ.get('CUBLAS_WORKSPACE_CONFIG'))

    frames = build_frames(args.seed)
    train_frame = frames['train']
    print(f"train {len(train_frame)}행, race 분포 {dict(train_frame['race'].value_counts().sort_index())}, "
          f"age_bin 분포 {dict(train_frame['age_bin'].value_counts().sort_index())}")

    base = build_masked_model(device)
    model_A = copy.deepcopy(base)
    model_B = copy.deepcopy(base)
    # deepcopy 후에도 인스턴스 forward(커스텀, mask 적용)가 유지되는지 확인. 유지되지 않으면 다시 건다.
    for m in (model_A, model_B):
        for layer in m.modules():
            if type(layer).__name__ in prune.forward_mapping_dict:
                fwd = layer.__dict__.get('forward')
                if fwd is None or getattr(fwd, '__func__', None) is not prune.forward_mapping_dict[type(layer).__name__]:
                    layer.forward = types.MethodType(prune.forward_mapping_dict[type(layer).__name__], layer)
    same = all(torch.equal(pa, pb) for pa, pb in zip(model_A.parameters(), model_B.parameters()))
    print(f"모델 A/B 파라미터 동일: {same}, 커스텀 forward 유지: "
          f"{model_A.features[0][0].forward.__func__ is custom_forward_conv2d}")

    # ── (A) 원본 코드 ──
    setseed(args.seed)
    t0 = time.perf_counter()
    masks_A = prune_orig_ref.fairness_grad(model_A, args.prune_ratio, train_frame, FACE_DIR, sensitive_classes=2,
                                           masked_grads=True, output_cols_each_task=OUTPUT_COLS, col_names=COL_USED,
                                           para_batch=1, impt_type=0, stop_batch=args.stop_batch, delta_p=0)
    tA = time.perf_counter() - t0
    target_A = dict(prune_orig_ref.LAST_GRAD_TARGET)

    # ── (B) 이식본 impt_type=4, faithful 모드, scope=all ──
    prune.FG_ORIG_MODE = 'faithful'
    config.glo_fg_scope = 'all'
    config.glo_prune_iter = 0
    config.glo_impt_type = 4
    config.glo_dataset = 'UTKFace'
    config.glo_seed = args.seed
    config.glo_fg_log_run_dir = None
    setseed(args.seed)
    t0 = time.perf_counter()
    masks_B = prune.fairness_grad(model_B, args.prune_ratio, train_frame, FACE_DIR, sensitive_classes=2,
                                  masked_grads=True, output_cols_each_task=OUTPUT_COLS, col_names=COL_USED,
                                  para_batch=1, impt_type=4, stop_batch=args.stop_batch, delta_p=0)
    tB = time.perf_counter() - t0
    stats_B = {s['name']: s for s in prune.FG_ORIG_LAST_STATS}

    layers = supported_named_layers(base)
    assert len(masks_A) == len(masks_B) == len(layers), (len(masks_A), len(masks_B), len(layers))

    rows = []
    n_kept_mismatch = n_target_mismatch = n_agree_low = 0
    for (name, layer), mA, mB in zip(layers, masks_A, masks_B):
        mA = mA.detach().float().cpu()
        mB = mB.detach().float().cpu()
        assert mA.shape == mB.shape == layer.weight.shape
        keptA, keptB = int(mA.sum().item()), int(mB.sum().item())
        tA_ = np.asarray(target_A[name], dtype=float)
        tB_ = np.asarray(stats_B[name]['target_prop'], dtype=float)
        agree = float((mA == mB).float().mean().item())
        kept_ok = keptA == keptB
        target_ok = bool(np.all(np.round(tA_, 4) == np.round(tB_, 4)))
        agree_ok = agree >= 0.99
        n_kept_mismatch += (not kept_ok)
        n_target_mismatch += (not target_ok)
        n_agree_low += (not agree_ok)
        rows.append((name, layer.weight.numel(), keptA, keptB, tA_, tB_, agree, kept_ok and target_ok and agree_ok))

    def fmt_t(t):
        return '(' + ','.join(f'{v:.4f}' for v in t) + ')'

    lines = []
    lines.append(f"# compare_fg_orig  (stop_batch={args.stop_batch}, prune_ratio={args.prune_ratio}, seed={args.seed}, "
                 f"deterministic={args.deterministic}, A={tA:.0f}s, B={tB:.0f}s)")
    lines.append("")
    lines.append("| layer | numel | kept_A | kept_B | grad_target_A (g0,g1) | grad_target_B (g0,g1) | mask_agree | ok |")
    lines.append("|---|---:|---:|---:|---|---|---:|:-:|")
    for name, numel, kA, kB, tA_, tB_, agree, ok in rows:
        lines.append(f"| {name} | {numel} | {kA} | {kB} | {fmt_t(tA_)} | {fmt_t(tB_)} | {agree:.4f} | {'✓' if ok else '✗'} |")
    lines.append("")
    all_agree = np.mean([r[6] for r in rows])
    min_agree = min(r[6] for r in rows)
    lines.append(f"- layers: {len(rows)}, kept mismatch: {n_kept_mismatch}, grad_target(4자리) mismatch: {n_target_mismatch}, "
                 f"agree<0.99: {n_agree_low}")
    lines.append(f"- mask agreement: mean {all_agree:.4f}, min {min_agree:.4f}")
    lines.append(f"- 총 kept: A {sum(r[2] for r in rows)}, B {sum(r[3] for r in rows)} / {sum(r[1] for r in rows)}")
    verdict = 'PASS' if (n_kept_mismatch == 0 and n_target_mismatch == 0 and n_agree_low == 0) else 'FAIL'
    lines.append(f"- 판정: **{verdict}**")
    text = "\n".join(lines)
    print("\n" + text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + "\n")
        print(f"결과 저장: {args.out}")
    sys.exit(0 if verdict == 'PASS' else 1)


if __name__ == '__main__':
    main()
