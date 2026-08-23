#!/usr/bin/env python
"""
상관관계 실험 v2: 통합 성능 기여도 + 통합 zeroing + ΔEO·ΔAcc 동시 측정

v1(celeba_6layer_correlation_experiment.py) 대비 변경점:
  1. 성능 기여도: conv1-only → conv0+conv1+conv2 통합 합산
     - ΔAcc 측정 시 conv0+conv1+conv2를 묶어서 제거하므로, 프록시도 동일 범위로 계산
  2. zeroing: conv1-only → conv0+conv1+conv2 통합 (BlockChannelZeroing)
  3. 측정: ΔEO만 → ΔEO + ΔAcc 동시 (한 번의 순전파)
  4. features.1 제외 → 16개 레이어 (features.2~17)
     - features.1은 conv0(depthwise, 32ch)과 conv1(pointwise, 16ch) 채널 수 불일치로 합산 불가

실행 흐름:
  [1/5] CelebA 데이터 로드
  [2/5] compute_phi_k → 16개 레이어 fairness proxy (39개 속성 합산)
  [3/5] compute_importance → conv0+conv1+conv2 통합 성능 기여도
  [4/5] 16개 레이어 fairness proxy vs 통합 성능 기여도 상관관계 → 6개 레이어 자동 선정
  [5/5] 선정된 6개 레이어에서 conv0+conv1+conv2 통합 zeroing → ΔEO + ΔAcc 동시 측정

출력물:
  all_layers_phi_vs_perf.csv — 16개 레이어 상관관계 (선정 근거)
  summary_correlations.csv — 6개 레이어 4가지 상관관계
  레이어별 채널 raw CSV, scatter plot, 요약 bar chart

사용법:
  cd /workspace/FairGRAPE/FairGRAPE
  /workspace/fairgrape_env_gpu/bin/python \
    activate_experiment/6_layer_activate_dff_corr/celeba_correlation_experiment_v2.py

  # 특정 레이어 직접 지정
  ... --layers features.2.conv.1.0,features.5.conv.1.0

  # 이어서 실행
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
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import make_datasets, make_frame
from prune import (
    compute_phi_k,
    compute_importance,
    _is_impt_type2_target_layer,
    _get_impt_type2_block_layer_names,
)
from util import safe_forward_with_cudnn_fallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CHECKPOINT = (
    ROOT_DIR / "trained_model" / "unpruned"
    / "CelebA_unpruned_classes_bygender_mobilenetv2_0.pt"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_v2"


# ═══════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Correlation experiment v2: unified importance + unified zeroing"
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target-attr", type=str, default="Attractive",
                   help="ΔEO, ΔAcc 측정 대상 속성")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--stop-batch", type=int, default=10000,
                   help="compute_phi_k / compute_importance 배치 제한")
    p.add_argument("--importance-batch-size", type=int, default=64)
    p.add_argument("--max-batches", type=int, default=None,
                   help="EO/Acc 평가 시 배치 제한 (None=전체)")
    p.add_argument("--layers", type=str, default=None,
                   help="쉼표로 구분된 conv1 레이어 목록 (지정 시 자동 선정 건너뜀)")
    p.add_argument("--resume", action="store_true")
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

    attribute_names = list(frames["train"].columns[2:41])
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


def conv1_to_block_name(conv1_name: str) -> str:
    """conv1 레이어명 → 블록명. features.N.conv.1.0 → features.N"""
    if conv1_name == "features.18.0":
        return "features.18"
    parts = conv1_name.split(".")
    return f"{parts[0]}.{parts[1]}"


# ═══════════════════════════════════════════════════════════
# 통합 성능 기여도 (conv0 + conv1 + conv2)
# ═══════════════════════════════════════════════════════════

def aggregate_per_channel(tensor: torch.Tensor) -> np.ndarray:
    """출력 채널(dim 0) 기준 집계."""
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


def aggregate_per_input_channel(tensor: torch.Tensor) -> np.ndarray:
    """입력 채널(dim 1) 기준 집계. conv2용."""
    t = tensor.detach().cpu()
    if t.ndim == 4:
        return t.mean(dim=(0, 2, 3)).numpy().astype(np.float64)
    elif t.ndim == 3:
        return t.mean(dim=(0, 2)).numpy().astype(np.float64)
    elif t.ndim == 2:
        return t.mean(dim=0).numpy().astype(np.float64)
    else:
        return t.reshape(t.shape[0], -1).mean(dim=1).numpy().astype(np.float64)


def compute_unified_perf_by_layer(
    importance_score_all: dict[str, torch.Tensor],
    phi_by_layer: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """conv0+conv1+conv2 통합 성능 기여도를 conv1 레이어 기준으로 계산.

    features.1은 conv0(32ch)/conv1(16ch) 채널 수 불일치로 제외.
    features.2~17은 conv0[k] + conv1[k] + conv2[:, k] 합산.

    Returns:
        {conv1_layer_name: unified_perf_array} — phi_by_layer와 동일한 키 사용
    """
    perf_by_layer = {}

    for conv1_name in sorted(phi_by_layer.keys()):
        # features.1 제외
        if conv1_name == "features.1.conv.1":
            continue

        block_name = conv1_to_block_name(conv1_name)
        conv0_name, conv1_layer_name, conv2_name = _get_impt_type2_block_layer_names(block_name)

        # conv1의 채널 수를 기준으로 사용
        if conv1_layer_name and conv1_layer_name in importance_score_all:
            ref_channels = importance_score_all[conv1_layer_name].shape[0]
        elif conv0_name and conv0_name in importance_score_all:
            ref_channels = importance_score_all[conv0_name].shape[0]
        else:
            continue

        unified = np.zeros(ref_channels, dtype=np.float64)

        if conv0_name and conv0_name in importance_score_all:
            conv0_perf = aggregate_per_channel(importance_score_all[conv0_name])
            n = min(ref_channels, len(conv0_perf))
            unified[:n] += conv0_perf[:n]

        if conv1_layer_name and conv1_layer_name in importance_score_all:
            conv1_perf = aggregate_per_channel(importance_score_all[conv1_layer_name])
            n = min(ref_channels, len(conv1_perf))
            unified[:n] += conv1_perf[:n]

        if conv2_name and conv2_name in importance_score_all:
            conv2_perf = aggregate_per_input_channel(importance_score_all[conv2_name])
            n = min(ref_channels, len(conv2_perf))
            unified[:n] += conv2_perf[:n]

        perf_by_layer[conv1_name] = unified

    return perf_by_layer


# ═══════════════════════════════════════════════════════════
# [표 1] 전체 레이어 fairness proxy vs 통합 성능 기여도
# ═══════════════════════════════════════════════════════════

def compute_all_layers_phi_vs_perf(
    phi_by_layer: dict[str, torch.Tensor],
    perf_by_layer: dict[str, np.ndarray],
) -> pd.DataFrame:
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
    """|Spearman rho| 기준 높은/중간/낮은 각 2개 선정."""
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

    selected.sort(key=lambda x: x[1])
    return selected


# ═══════════════════════════════════════════════════════════
# EO + Acc 동시 계산
# ═══════════════════════════════════════════════════════════

def compute_eo_and_acc(
    model: nn.Module,
    eval_loader,
    attribute_index: int,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """단일 속성에 대한 EO와 Accuracy를 한 번의 순전파로 동시 계산."""
    model.eval()
    stats = {g: {"tp": 0, "fp": 0, "pos": 0, "neg": 0} for g in (0, 1)}
    correct = 0
    total = 0
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

            correct += (preds == target_y).sum().item()
            total += target_y.size(0)

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
    eo = float((abs(tpr0 - tpr1) + abs(fpr0 - fpr1)) / 2.0)
    acc = correct / max(total, 1)

    return eo, acc


# ═══════════════════════════════════════════════════════════
# 통합 zeroing (conv0 + conv1 + conv2)
# ═══════════════════════════════════════════════════════════

class BlockChannelZeroing:
    """블록 내 conv0+conv1+conv2의 channel k를 동시에 zeroing/복원.

    conv0, conv1: 출력 채널 (dim 0) → weight[k].zero_()
    conv2:        입력 채널 (dim 1) → weight[:, k].zero_()
    """

    def __init__(self, model: nn.Module, block_name: str, channel_k: int):
        self.block_name = block_name
        self.channel_k = channel_k
        self.modules = dict(model.named_modules())
        self.saved = []

        conv0_name, conv1_name, conv2_name = _get_impt_type2_block_layer_names(block_name)
        self._register(conv0_name, dim=0)
        self._register(conv1_name, dim=0)
        self._register(conv2_name, dim=1)

    def _register(self, layer_name: str | None, dim: int):
        if layer_name is None or layer_name not in self.modules:
            return
        layer = self.modules[layer_name]
        if not hasattr(layer, 'weight'):
            return
        self.saved.append((layer_name, layer, dim))

    def __enter__(self):
        self._backups = []
        k = self.channel_k

        for layer_name, layer, dim in self.saved:
            if dim == 0:
                if k >= layer.weight.shape[0]:
                    self._backups.append(None)
                    continue
                backup_w = layer.weight.data[k].clone()
                layer.weight.data[k].zero_()
                backup_b = None
                if layer.bias is not None and k < layer.bias.shape[0]:
                    backup_b = layer.bias.data[k].clone()
                    layer.bias.data[k].zero_()
                self._backups.append((backup_w, backup_b, dim))
            else:
                if k >= layer.weight.shape[1]:
                    self._backups.append(None)
                    continue
                backup_w = layer.weight.data[:, k].clone()
                layer.weight.data[:, k].zero_()
                self._backups.append((backup_w, None, dim))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        k = self.channel_k
        for (layer_name, layer, dim), backup in zip(self.saved, self._backups):
            if backup is None:
                continue
            backup_w, backup_b, dim = backup
            if dim == 0:
                layer.weight.data[k].copy_(backup_w)
                if backup_b is not None:
                    layer.bias.data[k].copy_(backup_b)
            else:
                layer.weight.data[:, k].copy_(backup_w)
        return False


# ═══════════════════════════════════════════════════════════
# [5/5] 채널 zeroing 실험
# ═══════════════════════════════════════════════════════════

def run_layer_experiment(
    model: nn.Module,
    eval_loader,
    conv1_name: str,
    attribute_index: int,
    phi_vec: np.ndarray,
    perf_vec: np.ndarray,
    max_batches: int | None,
) -> pd.DataFrame:
    """하나의 블록에서 채널별 conv0+conv1+conv2 통합 zeroing → ΔEO + ΔAcc 동시 측정."""
    block_name = conv1_to_block_name(conv1_name)
    conv0_name, conv1_layer_name, conv2_name = _get_impt_type2_block_layer_names(block_name)
    num_channels = phi_vec.shape[0]

    modules = dict(model.named_modules())
    target_info = []
    for lname, dim_label in [(conv0_name, "dim0"), (conv1_layer_name, "dim0"), (conv2_name, "dim1")]:
        if lname and lname in modules and hasattr(modules[lname], 'weight'):
            shape = tuple(modules[lname].weight.shape)
            target_info.append(f"{lname} {shape} ({dim_label})")
    print(f"  블록: {block_name}")
    print(f"  zeroing 대상: {', '.join(target_info)}")

    print(f"  baseline EO + Acc 계산 중...")
    baseline_eo, baseline_acc = compute_eo_and_acc(model, eval_loader, attribute_index, max_batches)
    print(f"  baseline EO = {baseline_eo:.6f}, baseline Acc = {baseline_acc:.6f}")

    rows = []
    t_start = time.time()

    with torch.no_grad():
        for ch in range(num_channels):
            with BlockChannelZeroing(model, block_name, ch):
                pruned_eo, pruned_acc = compute_eo_and_acc(
                    model, eval_loader, attribute_index, max_batches
                )

            delta_eo = pruned_eo - baseline_eo
            delta_acc = pruned_acc - baseline_acc

            rows.append({
                "layer": conv1_name,
                "block": block_name,
                "channel": ch,
                "baseline_eo": baseline_eo,
                "pruned_eo": pruned_eo,
                "delta_eo": delta_eo,
                "delta_eo_abs": abs(delta_eo),
                "baseline_acc": baseline_acc,
                "pruned_acc": pruned_acc,
                "delta_acc": delta_acc,
                "delta_acc_abs": abs(delta_acc),
                "phi": float(phi_vec[ch]),
                "perf": float(perf_vec[ch]),
            })

            if ch == 0 or (ch + 1) % 10 == 0 or ch + 1 == num_channels:
                elapsed = time.time() - t_start
                rate = (ch + 1) / elapsed if elapsed > 0 else 0
                remain = (num_channels - ch - 1) / rate if rate > 0 else 0
                print(
                    f"  [{ch+1}/{num_channels}] "
                    f"ΔEO={delta_eo:+.6f}  ΔAcc={delta_acc:+.6f}  "
                    f"({elapsed:.0f}s, ~{remain:.0f}s left)"
                )

    return pd.DataFrame(rows)


def compute_layer_correlations(channel_df: pd.DataFrame, conv1_name: str) -> dict:
    """4가지 상관관계: fp vs ΔEO, perf vs ΔEO, fp vs ΔAcc, perf vs ΔAcc."""
    delta_eo = channel_df["delta_eo_abs"].to_numpy(np.float64)
    delta_acc = channel_df["delta_acc_abs"].to_numpy(np.float64)
    phi = channel_df["phi"].to_numpy(np.float64)
    perf = channel_df["perf"].to_numpy(np.float64)

    phi_deo = safe_correlations(phi, delta_eo)
    perf_deo = safe_correlations(perf, delta_eo)
    phi_dacc = safe_correlations(phi, delta_acc)
    perf_dacc = safe_correlations(perf, delta_acc)
    phi_perf = safe_correlations(phi, perf)

    return {
        "layer": conv1_name,
        "block": conv1_to_block_name(conv1_name),
        "channels": len(channel_df),
        "phi_deo_spearman": phi_deo["spearman_rho"],
        "phi_deo_spearman_p": phi_deo["spearman_p"],
        "phi_deo_pearson": phi_deo["pearson_r"],
        "phi_deo_pearson_p": phi_deo["pearson_p"],
        "perf_deo_spearman": perf_deo["spearman_rho"],
        "perf_deo_spearman_p": perf_deo["spearman_p"],
        "perf_deo_pearson": perf_deo["pearson_r"],
        "perf_deo_pearson_p": perf_deo["pearson_p"],
        "phi_dacc_spearman": phi_dacc["spearman_rho"],
        "phi_dacc_spearman_p": phi_dacc["spearman_p"],
        "phi_dacc_pearson": phi_dacc["pearson_r"],
        "phi_dacc_pearson_p": phi_dacc["pearson_p"],
        "perf_dacc_spearman": perf_dacc["spearman_rho"],
        "perf_dacc_spearman_p": perf_dacc["spearman_p"],
        "perf_dacc_pearson": perf_dacc["pearson_r"],
        "perf_dacc_pearson_p": perf_dacc["pearson_p"],
        "phi_perf_spearman": phi_perf["spearman_rho"],
        "phi_perf_pearson": phi_perf["pearson_r"],
    }


# ═══════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════

def save_layer_scatter(channel_df: pd.DataFrame, conv1_name: str,
                       corr_dict: dict, plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_name = conv1_name.replace(".", "_")

    combos = [
        ("phi",  "delta_eo_abs",  "fairness proxy",   "|ΔEO|",  "phi_deo"),
        ("perf", "delta_eo_abs",  "성능 기여도",        "|ΔEO|",  "perf_deo"),
        ("phi",  "delta_acc_abs", "fairness proxy",   "|ΔAcc|", "phi_dacc"),
        ("perf", "delta_acc_abs", "성능 기여도",        "|ΔAcc|", "perf_dacc"),
    ]

    for x_col, y_col, x_label, y_label, corr_key in combos:
        x = channel_df[x_col].to_numpy(np.float64)
        y = channel_df[y_col].to_numpy(np.float64)
        rho = corr_dict[f"{corr_key}_spearman"]
        p_val = corr_dict[f"{corr_key}_spearman_p"]

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(x, y, s=16, alpha=0.5, color="#1f77b4", edgecolors="none")

        if not (np.allclose(x, x[0]) or np.allclose(y, y[0])):
            slope, intercept = np.polyfit(x, y, 1)
            lx = np.linspace(x.min(), x.max(), 200)
            ax.plot(lx, slope * lx + intercept, color="#d62728", lw=2)

        ax.set_title(
            f"{conv1_name}\n{x_label} vs {y_label}  "
            f"(Spearman ρ={rho:.4f}, p={p_val:.2e})"
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_name}__{corr_key}.png", dpi=200)
        plt.close(fig)


def save_summary_plot(summary_df: pd.DataFrame, plot_dir: Path):
    plot_dir.mkdir(parents=True, exist_ok=True)

    layers = summary_df["layer"].tolist()
    short_labels = [l.replace(".conv.1.0", "").replace(".conv.1", "") for l in layers]
    x = np.arange(len(layers))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    ax = axes[0]
    ax.bar(x - width/2, summary_df["phi_deo_spearman"].tolist(), width,
           label="fairness proxy vs |ΔEO|", color="#2196F3")
    ax.bar(x + width/2, summary_df["perf_deo_spearman"].tolist(), width,
           label="통합 성능 기여도 vs |ΔEO|", color="#FF9800")
    ax.set_title("Spearman ρ vs |ΔEO|")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    ax.bar(x - width/2, summary_df["phi_dacc_spearman"].tolist(), width,
           label="fairness proxy vs |ΔAcc|", color="#2196F3")
    ax.bar(x + width/2, summary_df["perf_dacc_spearman"].tolist(), width,
           label="통합 성능 기여도 vs |ΔAcc|", color="#FF9800")
    ax.set_title("Spearman ρ vs |ΔAcc|")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(plot_dir / "summary_comparison.png", dpi=200)
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
    print("  상관관계 실험 v2")
    print("  통합 성능 기여도 (conv0+conv1+conv2)")
    print("  통합 zeroing (conv0+conv1+conv2)")
    print("  ΔEO + ΔAcc 동시 측정")
    print("  features.1 제외 → 16개 레이어 (features.2~17)")
    print(f"  device: {device}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  target attribute: {args.target_attr}")
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

    # ── [2/5] fairness proxy 계산 ──
    print("\n[2/5] fairness proxy 계산 중 (compute_phi_k, 전체 39개 속성 합산)...")
    t0 = time.time()
    phi_model = copy.deepcopy(model)
    phi_result = compute_phi_k(
        phi_model,
        ctx["frame"],
        new_img_dir=ctx["face_dir"],
        output_cols_each_task=ctx["output_cols_each_task"],
        col_names=ctx["col_used"],
        stop_batch=args.stop_batch,
        masked_grads=True,
        sensitive_group="gender",
    )
    if isinstance(phi_result, tuple):
        phi_by_layer = phi_result[0]
    else:
        phi_by_layer = phi_result
    del phi_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # features.1 제외
    phi_by_layer.pop("features.1.conv.1", None)
    print(f"  fairness proxy 계산 완료 ({time.time()-t0:.1f}s, {len(phi_by_layer)}개 레이어)")
    print(f"  features.1 제외 (conv0/conv1 채널 수 불일치)")

    # ── [3/5] 통합 성능 기여도 계산 ──
    print("\n[3/5] 통합 성능 기여도 계산 중 (conv0+conv1+conv2 합산)...")
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

    perf_by_layer = compute_unified_perf_by_layer(importance_score_all, phi_by_layer)
    print(f"  통합 성능 기여도: {len(perf_by_layer)}개 레이어")

    # ── [4/5] 표 1: 16개 레이어 상관관계 + 자동 선정 ──
    print("\n[4/5] 16개 레이어 fairness proxy vs 통합 성능 기여도 상관관계 계산 중...")
    phi_perf_df = compute_all_layers_phi_vs_perf(phi_by_layer, perf_by_layer)
    phi_perf_df.to_csv(output_dir / "all_layers_phi_vs_perf.csv", index=False)

    print(f"\n  ── 표 1: 16개 레이어 fairness proxy vs 통합 성능 기여도 ──")
    print(f"  {'layer':<28} {'ch':>4}  {'Spearman ρ':>12} {'p-value':>12}  {'Pearson r':>12} {'p-value':>12}")
    print(f"  {'-'*85}")
    for _, row in phi_perf_df.iterrows():
        print(f"  {row['layer']:<28} {row['channels']:>4}  "
              f"{row['spearman_rho']:>+12.4f} {row['spearman_p']:>12.2e}  "
              f"{row['pearson_r']:>+12.4f} {row['pearson_p']:>12.2e}")
    print(f"  → 저장: {output_dir / 'all_layers_phi_vs_perf.csv'}")

    # 레이어 선정
    if args.layers:
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
        target_layers = auto_select_6_layers(phi_perf_df)
        print(f"\n  ── 자동 선정된 6개 레이어 ──")
        for conv1_name, ch, group, rho in target_layers:
            print(f"  {conv1_name:<28} {ch:>4}ch  {group:<7} ρ={rho:+.4f}")

    # ── [5/5] 채널 zeroing 실험 ──
    total_channels = sum(t[1] for t in target_layers)
    print(f"\n[5/5] 통합 채널 zeroing 실험 시작 ({len(target_layers)}개 레이어, 총 {total_channels}채널)")
    summary_rows = []
    experiment_start = time.time()

    for layer_idx, (conv1_name, n_ch, corr_group, phi_perf_rho) in enumerate(target_layers):
        print(f"\n{'='*60}")
        print(f"[{layer_idx+1}/{len(target_layers)}] {conv1_name} ({n_ch}ch, {corr_group} corr)")
        print(f"{'='*60}")

        safe_name = conv1_name.replace(".", "_")
        layer_csv = output_dir / f"channels_{safe_name}.csv"

        # resume
        if args.resume and layer_csv.exists():
            print(f"  기존 결과 발견 → 스킵: {layer_csv}")
            existing_df = pd.read_csv(layer_csv)
            corr = compute_layer_correlations(existing_df, conv1_name)
            corr["corr_group"] = corr_group
            corr["phi_perf_rho_selection"] = phi_perf_rho
            summary_rows.append(corr)
            continue

        # phi, perf 벡터 확인
        if conv1_name not in phi_by_layer:
            print(f"  fairness proxy 없음, 스킵: {conv1_name}")
            continue
        phi_vec = phi_by_layer[conv1_name].detach().cpu().numpy().astype(np.float64)

        if conv1_name not in perf_by_layer:
            print(f"  통합 성능 기여도 없음, 스킵: {conv1_name}")
            continue
        perf_vec = perf_by_layer[conv1_name]

        # 채널 zeroing 실행
        channel_df = run_layer_experiment(
            model=model,
            eval_loader=ctx["eval_loader"],
            conv1_name=conv1_name,
            attribute_index=attr_idx,
            phi_vec=phi_vec,
            perf_vec=perf_vec,
            max_batches=args.max_batches,
        )

        channel_df.to_csv(layer_csv, index=False)
        print(f"  → 저장: {layer_csv}")

        corr = compute_layer_correlations(channel_df, conv1_name)
        corr["corr_group"] = corr_group
        corr["phi_perf_rho_selection"] = phi_perf_rho
        summary_rows.append(corr)

        save_layer_scatter(channel_df, conv1_name, corr, output_dir / "plots")

        print(f"\n  -- {conv1_name} 결과 --")
        print(f"  fairness proxy vs |ΔEO|:  ρ={corr['phi_deo_spearman']:+.4f}  r={corr['phi_deo_pearson']:+.4f}")
        print(f"  통합 perf     vs |ΔEO|:  ρ={corr['perf_deo_spearman']:+.4f}  r={corr['perf_deo_pearson']:+.4f}")
        print(f"  fairness proxy vs |ΔAcc|: ρ={corr['phi_dacc_spearman']:+.4f}  r={corr['phi_dacc_pearson']:+.4f}")
        print(f"  통합 perf     vs |ΔAcc|: ρ={corr['perf_dacc_spearman']:+.4f}  r={corr['perf_dacc_pearson']:+.4f}")

    total_time = time.time() - experiment_start

    # ── 최종 요약 ──
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "summary_correlations.csv", index=False)
        save_summary_plot(summary_df, output_dir / "plots")

        print(f"\n{'='*90}")
        print(f"  실험 완료 (총 {total_time:.0f}s)")
        print(f"{'='*90}")

        print(f"\n  ── fairness proxy / 통합 성능 기여도 vs |ΔEO| ──")
        print(f"  {'layer':<28} {'ch':>4} {'group':<7} "
              f"{'fp→ΔEO ρ':>10} {'p':>10}  {'perf→ΔEO ρ':>11} {'p':>10}")
        print(f"  {'-'*90}")
        for _, row in summary_df.iterrows():
            print(
                f"  {row['layer']:<28} {row['channels']:>4} "
                f"{row.get('corr_group',''):>7} "
                f"{row['phi_deo_spearman']:>+10.4f} {row['phi_deo_spearman_p']:>9.2e}  "
                f"{row['perf_deo_spearman']:>+11.4f} {row['perf_deo_spearman_p']:>9.2e}"
            )

        print(f"\n  ── fairness proxy / 통합 성능 기여도 vs |ΔAcc| ──")
        print(f"  {'layer':<28} {'ch':>4} {'group':<7} "
              f"{'fp→ΔAcc ρ':>10} {'p':>10}  {'perf→ΔAcc ρ':>12} {'p':>10}")
        print(f"  {'-'*90}")
        for _, row in summary_df.iterrows():
            print(
                f"  {row['layer']:<28} {row['channels']:>4} "
                f"{row.get('corr_group',''):>7} "
                f"{row['phi_dacc_spearman']:>+10.4f} {row['phi_dacc_spearman_p']:>9.2e}  "
                f"{row['perf_dacc_spearman']:>+12.4f} {row['perf_dacc_spearman_p']:>9.2e}"
            )

        print(f"\n  결과 저장: {output_dir}")

    # 메타데이터
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment": "Correlation experiment v2: unified importance + unified zeroing",
        "version_changes": [
            "성능 기여도: conv1-only → conv0+conv1+conv2 통합 합산",
            "zeroing: conv1-only → conv0+conv1+conv2 통합 (BlockChannelZeroing)",
            "측정: ΔEO만 → ΔEO + ΔAcc 동시 (한 번의 순전파)",
            "features.1 제외 (conv0/conv1 채널 수 불일치) → 16개 레이어",
        ],
        "checkpoint": str(args.checkpoint),
        "target_attr": attr_name,
        "target_attr_index": attr_idx,
        "split": args.split,
        "stop_batch": args.stop_batch,
        "importance_batch_size": args.importance_batch_size,
        "max_batches": args.max_batches,
        "seed": args.seed,
        "device": str(device),
        "total_time_sec": total_time,
        "layers_completed": len(summary_rows),
        "selected_layers": [t[0] for t in target_layers],
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
