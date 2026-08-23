#!/usr/bin/env python
"""
실험 B: conv0+conv1+conv2 통합 zeroing → ΔEO + ΔAcc 동시 측정

기존 6개 레이어 실험(conv1만 zeroing)의 측정 조건 불일치를 해결.
실제 pruning(conv0+conv1+conv2 묶음 제거)과 동일한 조건에서 ΔEO, ΔAcc를 동시에 측정.

블록별 매핑 (_get_impt_type2_block_layer_names 기준):
  features.1  → conv0=features.1.conv.0.0,  conv1=features.1.conv.1,   conv2=None
  features.2~17 → conv0=features.N.conv.0.0, conv1=features.N.conv.1.0, conv2=features.N.conv.2
  features.18 → conv0=features.18.0,         conv1=None,                conv2=classifier.1

Zeroing 방향:
  conv0, conv1: channel k = 출력 채널 (dim 0) → weight[k].zero_()
  conv2:        channel k = 입력 채널 (dim 1) → weight[:, k].zero_()

출력물:
  - 레이어별 채널 raw 데이터 CSV (phi, perf, delta_eo, delta_acc 등)
  - summary_unified_correlations.csv (perf→ΔAcc, φ_k→ΔAcc, perf→ΔEO, φ_k→ΔEO)
  - scatter plots
  - run_metadata.json

사용법:
  cd /workspace/FairGRAPE/FairGRAPE
  /workspace/fairgrape_env_gpu/bin/python \
    activate_experiment/6_layer_activate_dff_corr/unified_channel_zeroing_experiment.py

  # 이어서 실행 (완료된 레이어 스킵)
  ... --resume

  # 특정 레이어만 실행
  ... --layers features.1.conv.1,features.3.conv.1.0
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
ROOT_DIR = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "6_layer_activate_dff_corr" else SCRIPT_DIR
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
DEFAULT_INPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_unified"


# ═══════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment B: unified conv0+conv1+conv2 zeroing → ΔEO + ΔAcc"
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                   help="기존 6개 레이어 결과 CSV 위치 (phi, perf 값 재활용)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target-attr", type=str, default="Attractive",
                   help="ΔEO, ΔAcc 측정 대상 속성 (기본: Attractive)")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=None,
                   help="EO/Acc 평가 시 배치 제한 (None=전체)")
    p.add_argument("--layers", type=str, default=None,
                   help="쉼표로 구분된 conv1 레이어 목록 (지정 시 해당 레이어만 실행)")
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


def conv1_to_block_name(conv1_name: str) -> str:
    """conv1 레이어명 → 블록명 변환.
    features.1.conv.1       → features.1
    features.N.conv.1.0     → features.N
    features.18.0           → features.18
    """
    if conv1_name == "features.1.conv.1":
        return "features.1"
    if conv1_name == "features.18.0":
        return "features.18"
    # features.N.conv.1.0 → features.N
    parts = conv1_name.split(".")
    return f"{parts[0]}.{parts[1]}"


# 6개 레이어 CSV 파일명 (기존 실험에서 생성된 것)
LAYER_FILES = [
    ("features.1.conv.1",    "channels_features_1_conv_1.csv"),
    ("features.2.conv.1.0",  "channels_features_2_conv_1_0.csv"),
    ("features.3.conv.1.0",  "channels_features_3_conv_1_0.csv"),
    ("features.6.conv.1.0",  "channels_features_6_conv_1_0.csv"),
    ("features.10.conv.1.0", "channels_features_10_conv_1_0.csv"),
    ("features.16.conv.1.0", "channels_features_16_conv_1_0.csv"),
]


# ═══════════════════════════════════════════════════════════
# EO + Acc 동시 계산
# ═══════════════════════════════════════════════════════════

def compute_eo_and_acc(
    model: nn.Module,
    eval_loader,
    attribute_index: int,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """단일 속성에 대한 EO와 Accuracy를 한 번의 순전파로 동시 계산.

    Returns:
        (eo, accuracy)
    """
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

            # Accuracy
            correct += (preds == target_y).sum().item()
            total += target_y.size(0)

            # EO
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
    """블록 내 conv0+conv1+conv2의 channel k를 동시에 zeroing/복원하는 컨텍스트 매니저.

    conv0, conv1: 출력 채널 (dim 0) → weight[k].zero_()
    conv2:        입력 채널 (dim 1) → weight[:, k].zero_()
    """

    def __init__(self, model: nn.Module, block_name: str, channel_k: int):
        self.model = model
        self.block_name = block_name
        self.channel_k = channel_k
        self.modules = dict(model.named_modules())
        self.saved = []  # (layer_name, dim, saved_weight, saved_bias_or_None)

        conv0_name, conv1_name, conv2_name = _get_impt_type2_block_layer_names(block_name)
        self._register(conv0_name, dim=0)
        self._register(conv1_name, dim=0)
        self._register(conv2_name, dim=1)

    def _register(self, layer_name: str | None, dim: int):
        if layer_name is None:
            return
        if layer_name not in self.modules:
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
                # 출력 채널 zeroing
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
                # 입력 채널 zeroing (conv2)
                if k >= layer.weight.shape[1]:
                    self._backups.append(None)
                    continue
                backup_w = layer.weight.data[:, k].clone()
                layer.weight.data[:, k].zero_()
                # conv2는 bias가 출력 채널 단위이므로 건드리지 않음
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
# 레이어 실험 실행
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

    # zeroing 대상 레이어 출력
    target_info = []
    modules = dict(model.named_modules())
    for lname, dim_label in [(conv0_name, "dim0"), (conv1_layer_name, "dim0"), (conv2_name, "dim1")]:
        if lname and lname in modules and hasattr(modules[lname], 'weight'):
            shape = tuple(modules[lname].weight.shape)
            target_info.append(f"{lname} {shape} ({dim_label})")
    print(f"  블록: {block_name}")
    print(f"  zeroing 대상: {', '.join(target_info)}")

    # baseline 계산
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


# ═══════════════════════════════════════════════════════════
# 상관관계 계산
# ═══════════════════════════════════════════════════════════

def compute_layer_correlations(channel_df: pd.DataFrame, conv1_name: str) -> dict:
    """4가지 상관관계: φ_k→ΔEO, perf→ΔEO, φ_k→ΔAcc, perf→ΔAcc."""
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
        # φ_k → ΔEO
        "phi_deo_spearman": phi_deo["spearman_rho"],
        "phi_deo_spearman_p": phi_deo["spearman_p"],
        "phi_deo_pearson": phi_deo["pearson_r"],
        "phi_deo_pearson_p": phi_deo["pearson_p"],
        # perf → ΔEO
        "perf_deo_spearman": perf_deo["spearman_rho"],
        "perf_deo_spearman_p": perf_deo["spearman_p"],
        "perf_deo_pearson": perf_deo["pearson_r"],
        "perf_deo_pearson_p": perf_deo["pearson_p"],
        # φ_k → ΔAcc
        "phi_dacc_spearman": phi_dacc["spearman_rho"],
        "phi_dacc_spearman_p": phi_dacc["spearman_p"],
        "phi_dacc_pearson": phi_dacc["pearson_r"],
        "phi_dacc_pearson_p": phi_dacc["pearson_p"],
        # perf → ΔAcc
        "perf_dacc_spearman": perf_dacc["spearman_rho"],
        "perf_dacc_spearman_p": perf_dacc["spearman_p"],
        "perf_dacc_pearson": perf_dacc["pearson_r"],
        "perf_dacc_pearson_p": perf_dacc["pearson_p"],
        # φ_k vs perf (참고)
        "phi_perf_spearman": phi_perf["spearman_rho"],
    }


# ═══════════════════════════════════════════════════════════
# 시각화
# ═══════════════════════════════════════════════════════════

def save_layer_scatter(channel_df: pd.DataFrame, conv1_name: str,
                       corr_dict: dict, plot_dir: Path):
    """레이어별 scatter plot 4개: phi/perf × ΔEO/ΔAcc."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    safe_name = conv1_name.replace(".", "_")

    combos = [
        ("phi",  "delta_eo_abs",  "φ_k",              "|ΔEO|",  "phi_deo"),
        ("perf", "delta_eo_abs",  "perf (importance)", "|ΔEO|",  "perf_deo"),
        ("phi",  "delta_acc_abs", "φ_k",              "|ΔAcc|", "phi_dacc"),
        ("perf", "delta_acc_abs", "perf (importance)", "|ΔAcc|", "perf_dacc"),
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
    """요약 bar chart: φ_k vs perf → ΔEO, ΔAcc 각각 비교."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    layers = summary_df["layer"].tolist()
    short_labels = [l.replace(".conv.1.0", "").replace(".conv.1", "") for l in layers]
    x = np.arange(len(layers))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # (a) → ΔEO
    ax = axes[0]
    ax.bar(x - width/2, summary_df["phi_deo_spearman"].tolist(), width,
           label="φ_k → |ΔEO|", color="#2196F3")
    ax.bar(x + width/2, summary_df["perf_deo_spearman"].tolist(), width,
           label="perf → |ΔEO|", color="#FF9800")
    ax.set_title("Spearman ρ → |ΔEO|  (conv0+conv1+conv2 zeroing)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)

    # (b) → ΔAcc
    ax = axes[1]
    ax.bar(x - width/2, summary_df["phi_dacc_spearman"].tolist(), width,
           label="φ_k → |ΔAcc|", color="#2196F3")
    ax.bar(x + width/2, summary_df["perf_dacc_spearman"].tolist(), width,
           label="perf → |ΔAcc|", color="#FF9800")
    ax.set_title("Spearman ρ → |ΔAcc|  (conv0+conv1+conv2 zeroing)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(plot_dir / "summary_unified_comparison.png", dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════
# 기존 conv1-only 결과와 비교 테이블 출력
# ═══════════════════════════════════════════════════════════

def print_comparison_with_conv1_only(summary_df: pd.DataFrame, input_dir: Path):
    """기존 conv1-only 결과가 있으면 비교 테이블 출력."""
    old_summary = input_dir / "summary_6layer_correlations.csv"
    if not old_summary.exists():
        return

    old_df = pd.read_csv(old_summary)
    print(f"\n  ── conv1-only vs conv0+conv1+conv2 비교 (φ_k → |ΔEO| Spearman ρ) ──")
    print(f"  {'layer':<28} {'conv1-only':>12} {'unified':>12} {'diff':>10}")
    print(f"  {'-'*65}")

    for _, new_row in summary_df.iterrows():
        layer = new_row["layer"]
        old_row = old_df[old_df["layer"] == layer]
        if old_row.empty:
            continue
        old_rho = float(old_row.iloc[0]["phi_deo_spearman"])
        new_rho = new_row["phi_deo_spearman"]
        diff = new_rho - old_rho
        print(f"  {layer:<28} {old_rho:>+12.4f} {new_rho:>+12.4f} {diff:>+10.4f}")


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
    print("  실험 B: conv0+conv1+conv2 통합 zeroing → ΔEO + ΔAcc 동시 측정")
    print(f"  device: {device}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  target attribute: {args.target_attr}")
    print(f"  input (기존 CSV): {args.input_dir}")
    print(f"  output: {output_dir}")
    print("=" * 70)

    # ── [1/3] CelebA 데이터 + 모델 로드 ──
    print("\n[1/3] CelebA 데이터 + 모델 로드 중...")
    ctx = build_celeba_context(args.split, args.batch_size, args.num_workers)
    attr_idx, attr_name = resolve_target_attribute(ctx["attribute_names"], args.target_attr)
    print(f"  대상 속성: {attr_name} (index={attr_idx})")

    n_classes = len(ctx["attribute_names"]) * 2
    model = build_model(n_classes)
    model = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()
    print(f"  모델 로드 완료 ({n_classes} classes)")

    # ── [2/3] 대상 레이어 결정 ──
    print("\n[2/3] 대상 레이어 결정...")

    if args.layers:
        specified = set(args.layers.split(","))
        target_layers = [
            (conv1, csv_fn) for conv1, csv_fn in LAYER_FILES
            if conv1 in specified
        ]
        if not target_layers:
            raise ValueError(f"지정된 레이어를 찾을 수 없음: {args.layers}")
    else:
        target_layers = LAYER_FILES

    for conv1_name, csv_fn in target_layers:
        block = conv1_to_block_name(conv1_name)
        c0, c1, c2 = _get_impt_type2_block_layer_names(block)
        parts = [c0, c1, c2]
        parts_str = " + ".join(p for p in parts if p is not None)
        print(f"  {conv1_name} → {parts_str}")

    # ── [3/3] 채널 zeroing 실험 ──
    print(f"\n[3/3] 통합 채널 zeroing 실험 시작 ({len(target_layers)}개 레이어)")
    summary_rows = []
    experiment_start = time.time()

    for layer_idx, (conv1_name, csv_filename) in enumerate(target_layers):
        print(f"\n{'='*60}")
        print(f"[{layer_idx+1}/{len(target_layers)}] {conv1_name}")
        print(f"{'='*60}")

        # 기존 CSV에서 phi, perf 로드
        input_csv = args.input_dir / csv_filename
        if not input_csv.exists():
            print(f"  ⚠ 기존 CSV 없음, 스킵: {input_csv}")
            print(f"    → 먼저 celeba_6layer_correlation_experiment.py를 실행하세요")
            continue
        existing_df = pd.read_csv(input_csv)
        num_channels = len(existing_df)
        phi_vec = existing_df["phi"].to_numpy(np.float64)
        perf_vec = existing_df["perf"].to_numpy(np.float64)
        print(f"  {num_channels}채널 로드 (phi, perf from {csv_filename})")

        # 출력 CSV 경로
        output_csv = output_dir / f"unified_{csv_filename}"

        # resume 모드
        if args.resume and output_csv.exists():
            print(f"  기존 결과 발견 → 스킵: {output_csv}")
            result_df = pd.read_csv(output_csv)
            corr = compute_layer_correlations(result_df, conv1_name)
            summary_rows.append(corr)
            continue

        # 실험 실행
        channel_df = run_layer_experiment(
            model=model,
            eval_loader=ctx["eval_loader"],
            conv1_name=conv1_name,
            attribute_index=attr_idx,
            phi_vec=phi_vec,
            perf_vec=perf_vec,
            max_batches=args.max_batches,
        )

        # 즉시 저장
        channel_df.to_csv(output_csv, index=False)
        print(f"  → 저장: {output_csv}")

        # 상관관계 계산
        corr = compute_layer_correlations(channel_df, conv1_name)
        summary_rows.append(corr)

        # scatter plot
        save_layer_scatter(channel_df, conv1_name, corr, output_dir / "plots")

        # 레이어 결과 출력
        print(f"\n  -- {conv1_name} 결과 (conv0+conv1+conv2 zeroing) --")
        print(f"  φ_k  → |ΔEO|:  Spearman ρ={corr['phi_deo_spearman']:+.4f} "
              f"(p={corr['phi_deo_spearman_p']:.2e})")
        print(f"  perf → |ΔEO|:  Spearman ρ={corr['perf_deo_spearman']:+.4f} "
              f"(p={corr['perf_deo_spearman_p']:.2e})")
        print(f"  φ_k  → |ΔAcc|: Spearman ρ={corr['phi_dacc_spearman']:+.4f} "
              f"(p={corr['phi_dacc_spearman_p']:.2e})")
        print(f"  perf → |ΔAcc|: Spearman ρ={corr['perf_dacc_spearman']:+.4f} "
              f"(p={corr['perf_dacc_spearman_p']:.2e})")

    total_time = time.time() - experiment_start

    # ── 최종 요약 ──
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "summary_unified_correlations.csv", index=False)
        save_summary_plot(summary_df, output_dir / "plots")

        print(f"\n{'='*90}")
        print(f"  실험 완료 (총 {total_time:.0f}s)")
        print(f"{'='*90}")

        # 표 출력: φ_k vs perf → ΔEO
        print(f"\n  ── φ_k vs perf → |ΔEO| (Spearman ρ) ──")
        print(f"  {'layer':<28} {'ch':>4}  {'φ_k→ΔEO':>10} {'p':>10}  {'perf→ΔEO':>10} {'p':>10}")
        print(f"  {'-'*80}")
        for _, row in summary_df.iterrows():
            phi_sig = "***" if row["phi_deo_spearman_p"] < 0.001 else \
                      "** " if row["phi_deo_spearman_p"] < 0.01 else \
                      "*  " if row["phi_deo_spearman_p"] < 0.05 else "   "
            perf_sig = "***" if row["perf_deo_spearman_p"] < 0.001 else \
                       "** " if row["perf_deo_spearman_p"] < 0.01 else \
                       "*  " if row["perf_deo_spearman_p"] < 0.05 else "   "
            print(
                f"  {row['layer']:<28} {row['channels']:>4}  "
                f"{row['phi_deo_spearman']:>+10.4f}{phi_sig} {row['phi_deo_spearman_p']:>9.2e}  "
                f"{row['perf_deo_spearman']:>+10.4f}{perf_sig} {row['perf_deo_spearman_p']:>9.2e}"
            )

        # 표 출력: φ_k vs perf → ΔAcc
        print(f"\n  ── φ_k vs perf → |ΔAcc| (Spearman ρ) ──")
        print(f"  {'layer':<28} {'ch':>4}  {'φ_k→ΔAcc':>10} {'p':>10}  {'perf→ΔAcc':>10} {'p':>10}")
        print(f"  {'-'*80}")
        for _, row in summary_df.iterrows():
            phi_sig = "***" if row["phi_dacc_spearman_p"] < 0.001 else \
                      "** " if row["phi_dacc_spearman_p"] < 0.01 else \
                      "*  " if row["phi_dacc_spearman_p"] < 0.05 else "   "
            perf_sig = "***" if row["perf_dacc_spearman_p"] < 0.001 else \
                       "** " if row["perf_dacc_spearman_p"] < 0.01 else \
                       "*  " if row["perf_dacc_spearman_p"] < 0.05 else "   "
            print(
                f"  {row['layer']:<28} {row['channels']:>4}  "
                f"{row['phi_dacc_spearman']:>+10.4f}{phi_sig} {row['phi_dacc_spearman_p']:>9.2e}  "
                f"{row['perf_dacc_spearman']:>+10.4f}{perf_sig} {row['perf_dacc_spearman_p']:>9.2e}"
            )

        # 기존 conv1-only 결과와 비교
        print_comparison_with_conv1_only(summary_df, args.input_dir)

        print(f"\n  결과 저장: {output_dir}")

    # 메타데이터
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment": "B: unified conv0+conv1+conv2 zeroing",
        "checkpoint": str(args.checkpoint),
        "target_attr": attr_name,
        "target_attr_index": attr_idx,
        "zeroing_scope": "conv0+conv1+conv2 (matches actual pruning)",
        "phi_source": "existing CSV from 6-layer experiment (compute_phi_k, 39-attr aggregated)",
        "perf_source": "existing CSV from 6-layer experiment (compute_importance)",
        "split": args.split,
        "max_batches": args.max_batches,
        "seed": args.seed,
        "device": str(device),
        "total_time_sec": total_time,
        "layers_completed": len(summary_rows),
        "block_mapping": {
            conv1: {
                "block": conv1_to_block_name(conv1),
                "layers": _get_impt_type2_block_layer_names(conv1_to_block_name(conv1))
            }
            for conv1, _ in target_layers
        },
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
