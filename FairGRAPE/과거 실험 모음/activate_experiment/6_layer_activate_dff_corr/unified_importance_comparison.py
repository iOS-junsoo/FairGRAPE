#!/usr/bin/env python
"""
성능 기여도 통합 실험: conv0+conv1+conv2 통합 importance vs 기존 conv1-only importance

기존 compute_importance()는 각 레이어별 (weight × gradient)² 를 반환.
기존에는 conv1 레이어만 사용했으나, 실제 pruning은 conv0+conv1+conv2를 묶어서 제거하므로
세 레이어의 importance를 채널 단위로 합산하여 통합 성능 기여도를 만든다.

  통합 perf[k] = mean(conv0.importance[k]) + mean(conv1.importance[k]) + mean(conv2.importance[:, k])

이미 측정된 ΔEO, ΔAcc 결과(results_unified/)를 재활용하여,
통합 성능 기여도 vs |ΔAcc|, |ΔEO| 상관관계를 기존(conv1-only)과 비교한다.

사용법:
  cd /workspace/FairGRAPE/FairGRAPE
  /workspace/fairgrape_env_gpu/bin/python \
    activate_experiment/6_layer_activate_dff_corr/unified_importance_comparison.py
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torchvision import models

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "6_layer_activate_dff_corr" else SCRIPT_DIR
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import make_datasets, make_frame
from prune import (
    compute_importance,
    _is_impt_type2_target_layer,
    _get_impt_type2_block_layer_names,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CHECKPOINT = (
    ROOT_DIR / "trained_model" / "unpruned"
    / "CelebA_unpruned_classes_bygender_mobilenetv2_0.pt"
)
DEFAULT_ZEROING_DIR = SCRIPT_DIR / "results_unified"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_unified_importance"

# 5개 레이어 매핑 (features.1은 conv0/conv1 채널 수 불일치로 제외)
LAYER_FILES = [
    ("features.2.conv.1.0",  "unified_channels_features_2_conv_1_0.csv"),
    ("features.3.conv.1.0",  "unified_channels_features_3_conv_1_0.csv"),
    ("features.6.conv.1.0",  "unified_channels_features_6_conv_1_0.csv"),
    ("features.10.conv.1.0", "unified_channels_features_10_conv_1_0.csv"),
    ("features.16.conv.1.0", "unified_channels_features_16_conv_1_0.csv"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Unified importance (conv0+conv1+conv2) comparison")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--zeroing-dir", type=Path, default=DEFAULT_ZEROING_DIR,
                   help="기존 통합 zeroing 결과 CSV 위치")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--stop-batch", type=int, default=10000)
    p.add_argument("--importance-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_model(n_classes):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model


def load_checkpoint(model, path):
    data = torch.load(path, map_location=device)
    if isinstance(data, dict) and "model_state" in data:
        model.load_state_dict(data["model_state"])
    else:
        model.load_state_dict(data)
    return model


def build_celeba_context(split, batch_size, num_workers):
    csv_path = ROOT_DIR / "csv" / "CelebA.csv"
    face_dir = ROOT_DIR / "Images" / "CelebA" / "img_align_celeba"
    frames = make_frame(str(csv_path), str(face_dir), seven_races=False)
    split_frame = frames[split]
    attribute_names = list(frames["train"].columns[2:41])
    output_cols_each_task = [(i * 2, i * 2 + 2) for i in range(len(attribute_names))]
    col_used = attribute_names + ["gender"]
    return {
        "frame": split_frame,
        "face_dir": str(face_dir),
        "attribute_names": attribute_names,
        "output_cols_each_task": output_cols_each_task,
        "col_used": col_used,
    }


def conv1_to_block_name(conv1_name):
    if conv1_name == "features.1.conv.1":
        return "features.1"
    if conv1_name == "features.18.0":
        return "features.18"
    parts = conv1_name.split(".")
    return f"{parts[0]}.{parts[1]}"


def aggregate_per_channel(tensor):
    """importance 텐서를 채널별 스칼라로 집계 (mean). dim 0 기준."""
    t = tensor.detach().cpu()
    if t.ndim == 4:
        return t.mean(dim=(1, 2, 3)).numpy().astype(np.float64)
    elif t.ndim == 3:
        return t.mean(dim=(1, 2)).numpy().astype(np.float64)
    elif t.ndim == 2:
        return t.mean(dim=1).numpy().astype(np.float64)
    elif t.ndim == 1:
        return t.numpy().astype(np.float64)
    else:
        return t.reshape(t.shape[0], -1).mean(dim=1).numpy().astype(np.float64)


def aggregate_per_input_channel(tensor):
    """importance 텐서를 입력 채널별 스칼라로 집계 (mean). dim 1 기준.
    conv2: weight shape [out_ch, in_ch, 1, 1] → in_ch별 평균."""
    t = tensor.detach().cpu()
    if t.ndim == 4:
        return t.mean(dim=(0, 2, 3)).numpy().astype(np.float64)
    elif t.ndim == 3:
        return t.mean(dim=(0, 2)).numpy().astype(np.float64)
    elif t.ndim == 2:
        return t.mean(dim=0).numpy().astype(np.float64)
    else:
        return t.reshape(t.shape[0], -1).mean(dim=1).numpy().astype(np.float64)


def compute_unified_importance(importance_score_all, conv1_name):
    """conv0+conv1+conv2 통합 성능 기여도를 채널별로 계산.

    Returns:
        unified_perf: np.ndarray [num_channels]
        components: dict with conv0/conv1/conv2 개별 값 (디버깅용)
    """
    block_name = conv1_to_block_name(conv1_name)
    conv0_name, conv1_layer_name, conv2_name = _get_impt_type2_block_layer_names(block_name)

    components = {}

    # 기준 채널 수 = conv1의 출력 채널 (pruning 기준)
    ref_channels = None
    if conv1_layer_name and conv1_layer_name in importance_score_all:
        ref_channels = importance_score_all[conv1_layer_name].shape[0]
    elif conv0_name and conv0_name in importance_score_all:
        ref_channels = importance_score_all[conv0_name].shape[0]

    if ref_channels is None:
        return None, components

    unified = np.zeros(ref_channels, dtype=np.float64)

    # conv0: 출력 채널 (dim 0)
    if conv0_name and conv0_name in importance_score_all:
        conv0_perf = aggregate_per_channel(importance_score_all[conv0_name])
        components["conv0"] = conv0_perf
        n = min(ref_channels, len(conv0_perf))
        unified[:n] += conv0_perf[:n]

    # conv1: 출력 채널 (dim 0)
    if conv1_layer_name and conv1_layer_name in importance_score_all:
        conv1_perf = aggregate_per_channel(importance_score_all[conv1_layer_name])
        components["conv1"] = conv1_perf
        n = min(ref_channels, len(conv1_perf))
        unified[:n] += conv1_perf[:n]

    # conv2: 입력 채널 (dim 1)
    if conv2_name and conv2_name in importance_score_all:
        conv2_perf = aggregate_per_input_channel(importance_score_all[conv2_name])
        components["conv2"] = conv2_perf
        n = min(ref_channels, len(conv2_perf))
        unified[:n] += conv2_perf[:n]

    return unified, components


def safe_correlations(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    result = {"n": int(x.shape[0]),
              "pearson_r": np.nan, "pearson_p": np.nan,
              "spearman_rho": np.nan, "spearman_p": np.nan}
    if x.shape[0] < 3:
        return result
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return result
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    result.update(pearson_r=float(pr), pearson_p=float(pp),
                  spearman_rho=float(sr), spearman_p=float(sp))
    return result


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  성능 기여도 통합 비교: conv1-only vs conv0+conv1+conv2")
    print(f"  device: {device}")
    print(f"  zeroing 결과: {args.zeroing_dir}")
    print(f"  output: {output_dir}")
    print("=" * 70)

    # ── [1/3] 모델 + 데이터 로드 ──
    print("\n[1/3] 모델 + 데이터 로드 중...")
    ctx = build_celeba_context(args.split, args.batch_size, args.num_workers)
    n_classes = len(ctx["attribute_names"]) * 2
    model = build_model(n_classes)
    model = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()
    print(f"  모델 로드 완료 ({n_classes} classes)")

    # ── [2/3] 성능 기여도 계산 (전체 레이어) ──
    print("\n[2/3] 성능 기여도 계산 중 (compute_importance)...")
    t0 = time.time()
    importance_score_all, _ = compute_importance(
        model, None,
        ctx["frame"],
        new_img_dir=ctx["face_dir"],
        masked_grads=True,
        output_cols_each_task=ctx["output_cols_each_task"],
        col_names=ctx["col_used"],
        network="mobilenetv2",
        stop_batch=args.stop_batch,
        sensitive_group="gender",
        sensitive_classes=2,
        imp_batch_size=args.importance_batch_size,
    )
    print(f"  계산 완료 ({time.time()-t0:.1f}s)")

    # ── [3/3] 6개 레이어 비교 ──
    print("\n[3/3] conv1-only vs conv0+conv1+conv2 성능 기여도 비교")

    summary_rows = []

    for conv1_name, csv_filename in LAYER_FILES:
        print(f"\n{'='*60}")
        print(f"  {conv1_name}")
        print(f"{'='*60}")

        # 기존 zeroing 결과 로드
        zeroing_csv = args.zeroing_dir / csv_filename
        if not zeroing_csv.exists():
            print(f"  ⚠ zeroing 결과 없음, 스킵: {zeroing_csv}")
            continue
        df = pd.read_csv(zeroing_csv)
        n_ch = len(df)

        delta_eo = df["delta_eo_abs"].to_numpy(np.float64)
        delta_acc = df["delta_acc_abs"].to_numpy(np.float64)
        phi = df["phi"].to_numpy(np.float64)
        old_perf = df["perf"].to_numpy(np.float64)

        # 통합 성능 기여도 계산
        unified_perf, components = compute_unified_importance(importance_score_all, conv1_name)

        if unified_perf is None:
            print(f"  ⚠ importance 데이터 없음, 스킵")
            continue

        unified_perf = unified_perf[:n_ch]

        # 어떤 레이어가 합산되었는지 출력
        block_name = conv1_to_block_name(conv1_name)
        c0, c1, c2 = _get_impt_type2_block_layer_names(block_name)
        parts_used = []
        for name, label in [(c0, "conv0"), (c1, "conv1"), (c2, "conv2")]:
            if name and label in components:
                parts_used.append(f"{name} ({label})")
        print(f"  통합 대상: {' + '.join(parts_used)}")

        # 상관관계: conv1-only perf
        old_perf_dacc = safe_correlations(old_perf, delta_acc)
        old_perf_deo = safe_correlations(old_perf, delta_eo)

        # 상관관계: unified perf
        new_perf_dacc = safe_correlations(unified_perf, delta_acc)
        new_perf_deo = safe_correlations(unified_perf, delta_eo)

        # fairness proxy (변경 없음, 참고용)
        phi_deo = safe_correlations(phi, delta_eo)
        phi_dacc = safe_correlations(phi, delta_acc)

        print(f"\n  성능 기여도 vs |ΔAcc|:")
        print(f"    conv1-only:       Spearman ρ={old_perf_dacc['spearman_rho']:+.4f}  Pearson r={old_perf_dacc['pearson_r']:+.4f}")
        print(f"    conv0+conv1+conv2: Spearman ρ={new_perf_dacc['spearman_rho']:+.4f}  Pearson r={new_perf_dacc['pearson_r']:+.4f}")

        print(f"\n  성능 기여도 vs |ΔEO|:")
        print(f"    conv1-only:       Spearman ρ={old_perf_deo['spearman_rho']:+.4f}  Pearson r={old_perf_deo['pearson_r']:+.4f}")
        print(f"    conv0+conv1+conv2: Spearman ρ={new_perf_deo['spearman_rho']:+.4f}  Pearson r={new_perf_deo['pearson_r']:+.4f}")

        print(f"\n  fairness proxy (변경 없음, 참고):")
        print(f"    fp vs |ΔEO|:  ρ={phi_deo['spearman_rho']:+.4f}  r={phi_deo['pearson_r']:+.4f}")
        print(f"    fp vs |ΔAcc|: ρ={phi_dacc['spearman_rho']:+.4f}  r={phi_dacc['pearson_r']:+.4f}")

        # 결과 CSV 저장 (채널별 raw)
        df_out = df.copy()
        df_out["perf_conv1_only"] = old_perf
        df_out["perf_unified"] = unified_perf
        safe_name = conv1_name.replace(".", "_")
        df_out.to_csv(output_dir / f"comparison_{safe_name}.csv", index=False)

        summary_rows.append({
            "layer": conv1_name,
            "block": block_name,
            "channels": n_ch,
            # 기준 1: 성능 기여도 vs |ΔAcc|
            "old_perf_dacc_spearman": old_perf_dacc["spearman_rho"],
            "old_perf_dacc_pearson": old_perf_dacc["pearson_r"],
            "new_perf_dacc_spearman": new_perf_dacc["spearman_rho"],
            "new_perf_dacc_pearson": new_perf_dacc["pearson_r"],
            # 기준 4b: 성능 기여도 vs |ΔEO|
            "old_perf_deo_spearman": old_perf_deo["spearman_rho"],
            "old_perf_deo_pearson": old_perf_deo["pearson_r"],
            "new_perf_deo_spearman": new_perf_deo["spearman_rho"],
            "new_perf_deo_pearson": new_perf_deo["pearson_r"],
            # fairness proxy (변경 없음)
            "phi_deo_spearman": phi_deo["spearman_rho"],
            "phi_deo_pearson": phi_deo["pearson_r"],
            "phi_dacc_spearman": phi_dacc["spearman_rho"],
            "phi_dacc_pearson": phi_dacc["pearson_r"],
        })

    # ── 최종 요약 ──
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "summary_importance_comparison.csv", index=False)

        print(f"\n{'='*100}")
        print(f"  비교 완료")
        print(f"{'='*100}")

        print(f"\n  ── 기준 1: 성능 기여도 vs |ΔAcc| (개선 여부) ──")
        print(f"  {'layer':<28} {'ch':>4}  "
              f"{'conv1 ρ':>9} {'unified ρ':>10} {'Δρ':>8}  "
              f"{'conv1 r':>9} {'unified r':>10} {'Δr':>8}")
        print(f"  {'-'*90}")
        for _, row in summary_df.iterrows():
            d_s = row["new_perf_dacc_spearman"] - row["old_perf_dacc_spearman"]
            d_p = row["new_perf_dacc_pearson"] - row["old_perf_dacc_pearson"]
            print(
                f"  {row['layer']:<28} {row['channels']:>4}  "
                f"{row['old_perf_dacc_spearman']:>+9.4f} {row['new_perf_dacc_spearman']:>+10.4f} {d_s:>+8.4f}  "
                f"{row['old_perf_dacc_pearson']:>+9.4f} {row['new_perf_dacc_pearson']:>+10.4f} {d_p:>+8.4f}"
            )

        print(f"\n  ── 기준 4b: 성능 기여도 vs |ΔEO| (낮을수록 좋음) ──")
        print(f"  {'layer':<28} {'ch':>4}  "
              f"{'conv1 ρ':>9} {'unified ρ':>10} {'Δρ':>8}  "
              f"{'conv1 r':>9} {'unified r':>10} {'Δr':>8}")
        print(f"  {'-'*90}")
        for _, row in summary_df.iterrows():
            d_s = row["new_perf_deo_spearman"] - row["old_perf_deo_spearman"]
            d_p = row["new_perf_deo_pearson"] - row["old_perf_deo_pearson"]
            print(
                f"  {row['layer']:<28} {row['channels']:>4}  "
                f"{row['old_perf_deo_spearman']:>+9.4f} {row['new_perf_deo_spearman']:>+10.4f} {d_s:>+8.4f}  "
                f"{row['old_perf_deo_pearson']:>+9.4f} {row['new_perf_deo_pearson']:>+10.4f} {d_p:>+8.4f}"
            )

        print(f"\n  결과 저장: {output_dir}")

    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment": "Unified importance comparison: conv1-only vs conv0+conv1+conv2",
        "checkpoint": str(args.checkpoint),
        "zeroing_dir": str(args.zeroing_dir),
        "method_old": "conv1 only: mean((weight * gradient)^2) per output channel",
        "method_new": "conv0+conv1+conv2: sum of per-channel mean((weight * gradient)^2)",
        "seed": args.seed,
        "device": str(device),
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()