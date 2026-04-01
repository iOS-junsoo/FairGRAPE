#!/usr/bin/env python
"""
6개 레이어 상관관계 실험: φ_k vs ΔEO, perf vs ΔEO 비교

실행 흐름:
  [1/5] CelebA 데이터 로드
  [2/5] compute_phi_k → 전체 conv1 레이어 φ_k (39개 속성 합산)
  [3/5] compute_importance → 전체 conv1 레이어 성능 기여도
  [4/5] 17개 레이어 φ_k vs perf 상관관계 계산 → 높은/중간/낮은 각 2개 자동 선정
  [5/5] 선정된 6개 레이어에서 채널 zeroing → ΔEO 측정

출력물:
  표 1) all_layers_phi_vs_perf.csv — 17개 레이어 φ_k vs perf 상관관계 (선정 근거)
  표 2+3) summary_6layer_correlations.csv — 6개 레이어 perf→ΔEO, φ_k→ΔEO 상관관계
  레이어별 채널 raw 데이터, scatter plot, 요약 bar chart

사용법:
  cd /workspace/FairGRAPE/FairGRAPE
  /workspace/fairgrape_env_gpu/bin/python \\
    activate_experiment/6_layer_activate_dff_corr/celeba_6layer_correlation_experiment.py

  # 특정 레이어 직접 지정 (자동 선정 건너뜀)
  ... --layers features.1.conv.1,features.5.conv.1.0

  # 이어서 실행 (완료된 레이어 스킵)
  ... --resume
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torchvision import models

# ── 경로 설정 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent  # 6_layer_activate_dff_corr/
ROOT_DIR = SCRIPT_DIR.parents[1]  # activate_experiment/ → FairGRAPE/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import make_datasets, make_frame
from prune import compute_phi_k, compute_importance, _is_impt_type2_target_layer
from util import safe_forward_with_cudnn_fallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CHECKPOINT = (
    ROOT_DIR / "trained_model" / "unpruned"
    / "CelebA_unpruned_classes_bygender_mobilenetv2_0.pt"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"


# ═══════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="6-layer phi_k vs delta_EO correlation experiment")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target-attr", type=str, default="Attractive",
                   help="delta_EO 측정 대상 속성 (기본: Attractive)")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--stop-batch", type=int, default=10000,
                   help="compute_phi_k / compute_importance 배치 제한")
    p.add_argument("--importance-batch-size", type=int, default=64,
                   help="compute_importance 내부 배치 크기")
    p.add_argument("--max-eo-batches", type=int, default=None,
                   help="EO 평가 시 배치 제한 (None=전체)")
    p.add_argument("--layers", type=str, default=None,
                   help="쉼표로 구분된 conv1 레이어 목록 (지정 시 자동 선정 건너뜀)")
    p.add_argument("--resume", action="store_true",
                   help="이미 완료된 레이어 결과 CSV가 있으면 스킵")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_model(n_classes: int) -> nn.Module:
    model = models.mobilenet_v2(pretrained=False)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, n_classes)
    return model


def load_checkpoint(model: nn.Module, path: Path) -> nn.Module:
    data = torch.load(path, map_location=device)
    if isinstance(data, dict) and "model_state" in data:
        model.load_state_dict(data["model_state"])
    else:
        model.load_state_dict(data)
    return model


def build_celeba_context(split: str, batch_size: int, num_workers: int) -> dict:
    csv_path = ROOT_DIR / "csv" / "CelebA.csv"
    face_dir = ROOT_DIR / "Images" / "CelebA" / "img_align_celeba"

    if not csv_path.exists():
        raise FileNotFoundError(f"CelebA csv not found: {csv_path}")
    if not face_dir.exists():
        raise FileNotFoundError(f"CelebA image dir not found: {face_dir}")

    print(f"[context] CelebA {split} split 구성 중...")
    frames = make_frame(str(csv_path), str(face_dir), seven_races=False)
    split_frame = frames[split]

    attribute_names = list(frames["train"].columns[2:41])  # 39개 속성
    output_cols_each_task = [(i * 2, i * 2 + 2) for i in range(len(attribute_names))]
    col_used = attribute_names + ["gender"]

    _, eval_dataset = make_datasets(split_frame, split_frame, False, batch_size, col_used)
    from torch.utils.data import DataLoader
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    return {
        "frame": split_frame,
        "eval_loader": eval_loader,
        "face_dir": str(face_dir),
        "attribute_names": attribute_names,
        "output_cols_each_task": output_cols_each_task,
        "col_used": col_used,
    }


def resolve_target_attribute(attribute_names: list[str], target_attr: str) -> tuple[int, str]:
    lowered = {name.lower(): (idx, name) for idx, name in enumerate(attribute_names)}
    if target_attr.lower() not in lowered:
        raise ValueError(f"알 수 없는 속성: {target_attr}")
    return lowered[target_attr.lower()]


def safe_correlations(x: np.ndarray, y: np.ndarray) -> dict:
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


def aggregate_channel_importance(score_tensor: torch.Tensor) -> np.ndarray:
    """importance score 텐서를 채널별 스칼라로 집계 (mean)"""
    t = score_tensor.detach().cpu()
    if t.ndim == 4:
        agg = t.mean(dim=(1, 2, 3))
    elif t.ndim == 3:
        agg = t.mean(dim=(1, 2))
    elif t.ndim == 2:
        agg = t.mean(dim=1)
    elif t.ndim == 1:
        agg = t
    else:
        agg = t.reshape(t.shape[0], -1).mean(dim=1)
    return agg.numpy().astype(np.float64)


# ═══════════════════════════════════════════════════════════
# [표 1] 전체 레이어 phi_k vs perf 상관관계 + 자동 선정
# ═══════════════════════════════════════════════════════════

def compute_all_layers_phi_vs_perf(
    phi_by_layer: dict[str, torch.Tensor],
    perf_by_layer: dict[str, np.ndarray],
) -> pd.DataFrame:
    """전체 conv1 레이어에서 phi_k vs perf Spearman 상관관계 계산"""
    rows = []
    common = sorted(set(phi_by_layer.keys()) & set(perf_by_layer.keys()))

    for layer_name in common:
        phi_vec = phi_by_layer[layer_name].detach().cpu().numpy().astype(np.float64)
        perf_vec = perf_by_layer[layer_name]

        if phi_vec.shape != perf_vec.shape:
            continue

        corr = safe_correlations(phi_vec, perf_vec)
        rows.append({
            "layer": layer_name,
            "channels": int(phi_vec.shape[0]),
            "spearman_rho": corr["spearman_rho"],
            "spearman_p": corr["spearman_p"],
            "pearson_r": corr["pearson_r"],
            "pearson_p": corr["pearson_p"],
        })

    return pd.DataFrame(rows)


def auto_select_6_layers(phi_perf_df: pd.DataFrame) -> list[tuple[str, int, str, float]]:
    """
    |Spearman rho| 기준으로 높은/중간/낮은 각 2개 선정.
    Returns: [(conv1_name, channels, corr_group, rho), ...]
    """
    df = phi_perf_df.dropna(subset=["spearman_rho"]).copy()
    df["abs_rho"] = df["spearman_rho"].abs()
    df = df.sort_values("abs_rho", ascending=False).reset_index(drop=True)

    n = len(df)
    if n < 6:
        print(f"  ⚠ 레이어 {n}개로 6개 미만 → 전부 선정")
        return [
            (row["layer"], row["channels"], "all", row["spearman_rho"])
            for _, row in df.iterrows()
        ]

    # 상위 2개 = high, 하위 2개 = low, 중간에서 2개 = medium
    high = df.iloc[:2]
    low = df.iloc[-2:]

    mid_start = n // 2 - 1
    mid = df.iloc[mid_start:mid_start + 2]

    selected = []
    for _, row in high.iterrows():
        selected.append((row["layer"], int(row["channels"]), "high", row["spearman_rho"]))
    for _, row in mid.iterrows():
        selected.append((row["layer"], int(row["channels"]), "medium", row["spearman_rho"]))
    for _, row in low.iterrows():
        selected.append((row["layer"], int(row["channels"]), "low", row["spearman_rho"]))

    # 채널 수 적은 순 정렬 (빠른 것부터 실행)
    selected.sort(key=lambda x: x[1])
    return selected


# ═══════════════════════════════════════════════════════════
# EO 계산
# ═══════════════════════════════════════════════════════════

def compute_attribute_eo(
    model: nn.Module,
    eval_loader,
    attribute_index: int,
    max_batches: int | None = None,
) -> float:
    """단일 속성에 대한 Equalized Odds 계산"""
    model.eval()
    stats = {g: {"tp": 0, "fp": 0, "pos": 0, "neg": 0} for g in (0, 1)}
    out_start = attribute_index * 2
    out_end = out_start + 2

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(eval_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device).long()

            outputs = safe_forward_with_cudnn_fallback(model, images)
            preds = torch.argmax(outputs[:, out_start:out_end], dim=1)
            target_y = labels[:, attribute_index]
            sensitive_a = labels[:, -1]

            for g in (0, 1):
                mask = sensitive_a == g
                if not torch.any(mask):
                    continue
                yt = target_y[mask]
                yp = preds[mask]
                stats[g]["tp"] += ((yp == 1) & (yt == 1)).sum().item()
                stats[g]["fp"] += ((yp == 1) & (yt == 0)).sum().item()
                stats[g]["pos"] += (yt == 1).sum().item()
                stats[g]["neg"] += (yt == 0).sum().item()

    eps = 1e-12
    tpr0 = stats[0]["tp"] / (stats[0]["pos"] + eps)
    tpr1 = stats[1]["tp"] / (stats[1]["pos"] + eps)
    fpr0 = stats[0]["fp"] / (stats[0]["neg"] + eps)
    fpr1 = stats[1]["fp"] / (stats[1]["neg"] + eps)
    return float((abs(tpr0 - tpr1) + abs(fpr0 - fpr1)) / 2.0)


# ═══════════════════════════════════════════════════════════
# [5/5] 채널 zeroing 실험
# ═══════════════════════════════════════════════════════════

def run_layer_experiment(
    model: nn.Module,
    eval_loader,
    layer_name: str,
    attribute_index: int,
    phi_vec: np.ndarray,
    perf_vec: np.ndarray,
    max_eo_batches: int | None,
) -> pd.DataFrame:
    """하나의 레이어에서 채널을 하나씩 zeroing → delta_EO 측정"""
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise KeyError(f"레이어를 찾을 수 없음: {layer_name}")
    target_layer = modules[layer_name]
    num_channels = int(target_layer.weight.shape[0])

    assert phi_vec.shape[0] == num_channels
    assert perf_vec.shape[0] == num_channels

    print(f"  baseline EO 계산 중...")
    baseline_eo = compute_attribute_eo(model, eval_loader, attribute_index, max_eo_batches)
    print(f"  baseline EO = {baseline_eo:.6f}")

    rows = []
    t_start = time.time()

    with torch.no_grad():
        for ch in range(num_channels):
            saved_w = target_layer.weight.data[ch].clone()
            target_layer.weight.data[ch].zero_()

            saved_b = None
            if target_layer.bias is not None:
                saved_b = target_layer.bias.data[ch].clone()
                target_layer.bias.data[ch].zero_()

            pruned_eo = compute_attribute_eo(model, eval_loader, attribute_index, max_eo_batches)

            target_layer.weight.data[ch].copy_(saved_w)
            if saved_b is not None:
                target_layer.bias.data[ch].copy_(saved_b)

            delta_eo = pruned_eo - baseline_eo
            delta_eo_abs = abs(delta_eo)

            rows.append({
                "layer": layer_name,
                "channel": ch,
                "baseline_eo": baseline_eo,
                "pruned_eo": pruned_eo,
                "delta_eo": delta_eo,
                "delta_eo_abs": delta_eo_abs,
                "phi": float(phi_vec[ch]),
                "perf": float(perf_vec[ch]),
            })

            if ch == 0 or (ch + 1) % 10 == 0 or ch + 1 == num_channels:
                elapsed = time.time() - t_start
                rate = (ch + 1) / elapsed if elapsed > 0 else 0
                remain = (num_channels - ch - 1) / rate if rate > 0 else 0
                print(
                    f"  [{ch+1}/{num_channels}] "
                    f"delta_EO={delta_eo:+.6f} |delta_EO|={delta_eo_abs:.6f}  "
                    f"({elapsed:.0f}s elapsed, ~{remain:.0f}s remaining)"
                )

    return pd.DataFrame(rows)


def compute_layer_correlations(channel_df: pd.DataFrame, layer_name: str) -> dict:
    """phi_k vs delta_EO, perf vs delta_EO, phi_k vs perf 상관관계"""
    delta = channel_df["delta_eo_abs"].to_numpy(np.float64)
    phi = channel_df["phi"].to_numpy(np.float64)
    perf = channel_df["perf"].to_numpy(np.float64)

    phi_vs_deo = safe_correlations(phi, delta)
    perf_vs_deo = safe_correlations(perf, delta)
    phi_vs_perf = safe_correlations(phi, perf)

    return {
        "layer": layer_name,
        "channels": len(channel_df),
        "phi_deo_spearman": phi_vs_deo["spearman_rho"],
        "phi_deo_spearman_p": phi_vs_deo["spearman_p"],
        "phi_deo_pearson": phi_vs_deo["pearson_r"],
        "phi_deo_pearson_p": phi_vs_deo["pearson_p"],
        "perf_deo_spearman": perf_vs_deo["spearman_rho"],
        "perf_deo_spearman_p": perf_vs_deo["spearman_p"],
        "perf_deo_pearson": perf_vs_deo["pearson_r"],
        "perf_deo_pearson_p": perf_vs_deo["pearson_p"],
        "phi_perf_spearman": phi_vs_perf["spearman_rho"],
        "phi_perf_spearman_p": phi_vs_perf["spearman_p"],
    }


# ═══════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════

def save_layer_scatter(channel_df: pd.DataFrame, layer_name: str,
                       corr_dict: dict, plot_dir: Path):
    """레이어별 scatter plot 2개: phi vs delta_EO, perf vs delta_EO"""
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_name = layer_name.replace(".", "_")

    for score_col, label, corr_key in [
        ("phi", "phi_k", "phi_deo"),
        ("perf", "perf (importance)", "perf_deo"),
    ]:
        x = channel_df[score_col].to_numpy(np.float64)
        y = channel_df["delta_eo_abs"].to_numpy(np.float64)
        rho = corr_dict[f"{corr_key}_spearman"]
        p_val = corr_dict[f"{corr_key}_spearman_p"]

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(x, y, s=16, alpha=0.5, color="#1f77b4", edgecolors="none")

        if not (np.allclose(x, x[0]) or np.allclose(y, y[0])):
            slope, intercept = np.polyfit(x, y, 1)
            lx = np.linspace(x.min(), x.max(), 200)
            ax.plot(lx, slope * lx + intercept, color="#d62728", lw=2)

        ax.set_title(f"{layer_name}\n{label} vs |delta_EO|  (Spearman rho={rho:.4f}, p={p_val:.2e})")
        ax.set_xlabel(label)
        ax.set_ylabel("|delta_EO|")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_name}__{score_col}_vs_delta_eo.png", dpi=200)
        plt.close(fig)


def save_summary_plot(summary_df: pd.DataFrame, plot_dir: Path):
    """전체 레이어 요약 bar chart: phi vs perf Spearman 비교"""
    plot_dir.mkdir(parents=True, exist_ok=True)

    layers = summary_df["layer"].tolist()
    phi_rho = summary_df["phi_deo_spearman"].tolist()
    perf_rho = summary_df["perf_deo_spearman"].tolist()

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, phi_rho, width, label="phi_k vs |delta_EO|", color="#2196F3")
    ax.bar(x + width/2, perf_rho, width, label="perf vs |delta_EO|", color="#FF9800")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman rho")
    ax.set_title("6-Layer Correlation: phi_k vs perf -> |delta_EO|")
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(".conv.1.0", "").replace(".conv.1", "")
                        for l in layers], rotation=30, ha="right")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "summary_6layer_comparison.png", dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"체크포인트 없음: {args.checkpoint}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  6개 레이어 상관관계 실험: phi_k vs delta_EO, perf vs delta_EO")
    print(f"  device: {device}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  target attribute (delta_EO): {args.target_attr}")
    print(f"  phi_k 방식: compute_phi_k (전체 39개 속성 합산)")
    print(f"  output: {output_dir}")
    print("=" * 70)

    # ── [1/5] CelebA 데이터 로드 ──
    print("\n[1/5] CelebA 데이터 로드 중...")
    ctx = build_celeba_context(args.split, args.batch_size, args.num_workers)
    attr_idx, attr_name = resolve_target_attribute(ctx["attribute_names"], args.target_attr)
    print(f"  대상 속성: {attr_name} (index={attr_idx})")

    # ── 모델 로드 ──
    n_classes = len(ctx["attribute_names"]) * 2
    model = build_model(n_classes)
    model = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()
    print(f"  모델 로드 완료 ({n_classes} classes)")

    # ── [2/5] phi_k 계산 ──
    print("\n[2/5] phi_k 계산 중 (compute_phi_k, 전체 39개 속성 합산)...")
    t0 = time.time()
    phi_model = copy.deepcopy(model)
    phi_by_layer = compute_phi_k(
        phi_model,
        ctx["frame"],
        new_img_dir=ctx["face_dir"],
        output_cols_each_task=ctx["output_cols_each_task"],
        col_names=ctx["col_used"],
        stop_batch=args.stop_batch,
        masked_grads=True,
        sensitive_group="gender",
    )
    del phi_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  phi_k 계산 완료 ({time.time()-t0:.1f}s, {len(phi_by_layer)}개 레이어)")

    # ── [3/5] importance 계산 ──
    print("\n[3/5] 성능 기여도 계산 중 (compute_importance)...")
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
    print(f"  importance 계산 완료 ({time.time()-t0:.1f}s)")

    # conv1 레이어만 채널별 집계
    perf_by_layer = {}
    for name, tensor in importance_score_all.items():
        if _is_impt_type2_target_layer(name):
            perf_by_layer[name] = aggregate_channel_importance(tensor)

    # ── [4/5] 표 1: 전체 레이어 phi_k vs perf 상관관계 + 자동 선정 ──
    print("\n[4/5] 전체 레이어 phi_k vs perf 상관관계 계산 중...")
    phi_perf_df = compute_all_layers_phi_vs_perf(phi_by_layer, perf_by_layer)
    phi_perf_df.to_csv(output_dir / "all_layers_phi_vs_perf.csv", index=False)

    print(f"\n  ── 표 1: 전체 레이어 phi_k vs perf 상관관계 ──")
    print(f"  {'layer':<28} {'ch':>4}  {'Spearman rho':>12} {'p-value':>12}")
    print(f"  {'-'*60}")
    for _, row in phi_perf_df.iterrows():
        print(f"  {row['layer']:<28} {row['channels']:>4}  "
              f"{row['spearman_rho']:>+12.4f} {row['spearman_p']:>12.2e}")
    print(f"  -> 저장: {output_dir / 'all_layers_phi_vs_perf.csv'}")

    # 레이어 선정
    if args.layers:
        # 직접 지정 모드
        specified = set(args.layers.split(","))
        target_layers = []
        for _, row in phi_perf_df.iterrows():
            if row["layer"] in specified:
                target_layers.append(
                    (row["layer"], int(row["channels"]), "manual", row["spearman_rho"])
                )
        if not target_layers:
            raise ValueError(f"지정된 레이어를 찾을 수 없음: {args.layers}")
        print(f"\n  수동 선정: {len(target_layers)}개 레이어")
    else:
        # 자동 선정
        target_layers = auto_select_6_layers(phi_perf_df)
        print(f"\n  ── 자동 선정된 6개 레이어 ──")
        for conv1_name, ch, group, rho in target_layers:
            print(f"  {conv1_name:<28} {ch:>4}ch  {group:<7} phi_k-perf rho={rho:+.4f}")

    # ── [5/5] 채널 zeroing 실험 ──
    total_channels = sum(t[1] for t in target_layers)
    print(f"\n[5/5] 채널 zeroing 실험 시작 ({len(target_layers)}개 레이어, 총 {total_channels}채널)")
    summary_rows = []
    experiment_start = time.time()

    for layer_idx, (conv1_name, n_ch, corr_group, phi_perf_rho) in enumerate(target_layers):
        print(f"\n{'='*60}")
        print(f"[{layer_idx+1}/{len(target_layers)}] {conv1_name} ({n_ch}ch, {corr_group} corr)")
        print(f"{'='*60}")

        layer_csv = output_dir / f"channels_{conv1_name.replace('.', '_')}.csv"

        # resume 모드
        if args.resume and layer_csv.exists():
            print(f"  기존 결과 발견 -> 스킵: {layer_csv}")
            existing_df = pd.read_csv(layer_csv)
            corr = compute_layer_correlations(existing_df, conv1_name)
            corr["corr_group"] = corr_group
            corr["phi_perf_rho"] = phi_perf_rho
            summary_rows.append(corr)
            continue

        # phi, perf 벡터 확인
        if conv1_name not in phi_by_layer:
            print(f"  phi_k 없음, 스킵: {conv1_name}")
            continue
        phi_vec = phi_by_layer[conv1_name].detach().cpu().numpy().astype(np.float64)

        if conv1_name not in perf_by_layer:
            print(f"  importance 없음, 스킵: {conv1_name}")
            continue
        perf_vec = perf_by_layer[conv1_name]

        # 채널 zeroing 실행
        channel_df = run_layer_experiment(
            model=model,
            eval_loader=ctx["eval_loader"],
            layer_name=conv1_name,
            attribute_index=attr_idx,
            phi_vec=phi_vec,
            perf_vec=perf_vec,
            max_eo_batches=args.max_eo_batches,
        )

        # 즉시 저장
        channel_df.to_csv(layer_csv, index=False)
        print(f"  -> 저장: {layer_csv}")

        # 상관관계 계산
        corr = compute_layer_correlations(channel_df, conv1_name)
        corr["corr_group"] = corr_group
        corr["phi_perf_rho"] = phi_perf_rho
        summary_rows.append(corr)

        # scatter plot
        save_layer_scatter(channel_df, conv1_name, corr, output_dir / "plots")

        # 레이어 결과 출력
        print(f"\n  -- {conv1_name} 결과 --")
        print(f"  phi_k  vs |delta_EO|: Spearman rho={corr['phi_deo_spearman']:.4f} "
              f"(p={corr['phi_deo_spearman_p']:.2e})")
        print(f"  perf vs |delta_EO|: Spearman rho={corr['perf_deo_spearman']:.4f} "
              f"(p={corr['perf_deo_spearman_p']:.2e})")

    total_time = time.time() - experiment_start

    # ── 최종 요약 ──
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "summary_6layer_correlations.csv", index=False)
        save_summary_plot(summary_df, output_dir / "plots")

        print(f"\n{'='*80}")
        print(f"  실험 완료 (총 {total_time:.0f}s)")
        print(f"{'='*80}")

        print(f"\n  -- 표 2+3: 선정된 레이어 phi_k->delta_EO, perf->delta_EO 상관관계 --")
        print(f"  {'layer':<28} {'ch':>4} {'group':<7} "
              f"{'phi_k->dEO':>10} {'p':>10}  {'perf->dEO':>10} {'p':>10}")
        print(f"  {'-'*95}")
        for _, row in summary_df.iterrows():
            phi_sig = "***" if row["phi_deo_spearman_p"] < 0.001 else \
                      "** " if row["phi_deo_spearman_p"] < 0.01 else \
                      "*  " if row["phi_deo_spearman_p"] < 0.05 else "   "
            perf_sig = "***" if row["perf_deo_spearman_p"] < 0.001 else \
                       "** " if row["perf_deo_spearman_p"] < 0.01 else \
                       "*  " if row["perf_deo_spearman_p"] < 0.05 else "   "
            print(
                f"  {row['layer']:<28} {row['channels']:>4} "
                f"{row.get('corr_group',''):>7} "
                f"{row['phi_deo_spearman']:>+10.4f}{phi_sig} "
                f"{row['phi_deo_spearman_p']:>9.2e}  "
                f"{row['perf_deo_spearman']:>+10.4f}{perf_sig} "
                f"{row['perf_deo_spearman_p']:>9.2e}"
            )
        print(f"\n  결과 저장: {output_dir}")

    # 메타데이터
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint": str(args.checkpoint),
        "target_attr": attr_name,
        "target_attr_index": attr_idx,
        "phi_method": "compute_phi_k (39-attribute aggregated)",
        "split": args.split,
        "stop_batch": args.stop_batch,
        "importance_batch_size": args.importance_batch_size,
        "max_eo_batches": args.max_eo_batches,
        "seed": args.seed,
        "device": str(device),
        "total_time_sec": total_time,
        "layers_completed": len(summary_rows),
        "selected_layers": [t[0] for t in target_layers],
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()