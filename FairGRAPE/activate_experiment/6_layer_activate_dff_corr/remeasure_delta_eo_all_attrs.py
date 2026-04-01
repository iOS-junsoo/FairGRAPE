#!/usr/bin/env python
"""
ΔEO 재측정: Attractive 단일 → 39개 속성 전체 평균 EO

기존 zeroing 실험(results_v2/)에서 ΔEO가 Attractive 1개 속성만으로 측정되었음.
실제 pruning 코드(train_and_val.py)에서는 39개 속성 각각 EO를 구해 평균을 사용.
fairness proxy(compute_phi_k)도 39개 속성 전체 gap의 누적 평균.
→ 측정 조건을 맞추기 위해 ΔEO를 39개 속성 전체 평균으로 재측정.

phi, perf는 results_v2/ CSV에서 그대로 재활용.
ΔAcc도 속성과 무관하므로 재활용 (전체 모델 정확도).

사용법:
  cd /workspace/FairGRAPE/FairGRAPE
  /workspace/fairgrape_env_gpu/bin/python \
    activate_experiment/6_layer_activate_dff_corr/remeasure_delta_eo_all_attrs.py

  # 이어서 실행
  ... --resume
"""

from __future__ import annotations

import argparse
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
from prune import _get_impt_type2_block_layer_names
from util import safe_forward_with_cudnn_fallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CHECKPOINT = (
    ROOT_DIR / "trained_model" / "unpruned"
    / "CelebA_unpruned_classes_bygender_mobilenetv2_0.pt"
)
DEFAULT_V2_DIR = SCRIPT_DIR / "results_v2"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_v2_eo_all_attrs"


# ═══════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Remeasure ΔEO with all 39 attributes mean EO")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR,
                   help="기존 results_v2/ 디렉토리 (phi, perf 재활용)")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--resume", action="store_true")
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
    _, eval_dataset = make_datasets(split_frame, split_frame, False, batch_size, col_used)
    from torch.utils.data import DataLoader
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return {
        "eval_loader": eval_loader,
        "attribute_names": attribute_names,
        "output_cols_each_task": output_cols_each_task,
        "n_attrs": len(attribute_names),
    }


def conv1_to_block_name(conv1_name):
    if conv1_name == "features.18.0":
        return "features.18"
    parts = conv1_name.split(".")
    return f"{parts[0]}.{parts[1]}"


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


# ═══════════════════════════════════════════════════════════
# 39개 속성 전체 평균 EO 계산
# ═══════════════════════════════════════════════════════════

def compute_mean_eo_all_attrs(
    model: nn.Module,
    eval_loader,
    n_attrs: int,
    max_batches: int | None = None,
) -> float:
    """39개 속성 각각의 EO를 계산한 뒤 평균. 실제 pruning 코드와 동일한 방식."""
    model.eval()

    # 속성별 성별별 통계
    stats = {}
    for attr_idx in range(n_attrs):
        stats[attr_idx] = {
            g: {"tp": 0, "fp": 0, "pos": 0, "neg": 0}
            for g in (0, 1)
        }

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(eval_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device).long()

            outputs = safe_forward_with_cudnn_fallback(model, images)
            sensitive_a = labels[:, -1]

            for attr_idx in range(n_attrs):
                out_start = attr_idx * 2
                out_end = out_start + 2
                preds = torch.argmax(outputs[:, out_start:out_end], dim=1)
                target_y = labels[:, attr_idx]

                for g in (0, 1):
                    mask = sensitive_a == g
                    if not torch.any(mask):
                        continue
                    yt = target_y[mask]
                    yp = preds[mask]
                    stats[attr_idx][g]["tp"] += ((yp == 1) & (yt == 1)).sum().item()
                    stats[attr_idx][g]["fp"] += ((yp == 1) & (yt == 0)).sum().item()
                    stats[attr_idx][g]["pos"] += (yt == 1).sum().item()
                    stats[attr_idx][g]["neg"] += (yt == 0).sum().item()

    # 속성별 EO 계산 후 평균
    eps = 1e-12
    eos = []
    for attr_idx in range(n_attrs):
        tpr0 = stats[attr_idx][0]["tp"] / (stats[attr_idx][0]["pos"] + eps)
        tpr1 = stats[attr_idx][1]["tp"] / (stats[attr_idx][1]["pos"] + eps)
        fpr0 = stats[attr_idx][0]["fp"] / (stats[attr_idx][0]["neg"] + eps)
        fpr1 = stats[attr_idx][1]["fp"] / (stats[attr_idx][1]["neg"] + eps)
        eo = (abs(tpr0 - tpr1) + abs(fpr0 - fpr1)) / 2.0
        eos.append(eo)

    return float(np.mean(eos))


# ═══════════════════════════════════════════════════════════
# 통합 zeroing
# ═══════════════════════════════════════════════════════════

class BlockChannelZeroing:
    def __init__(self, model, block_name, channel_k):
        self.block_name = block_name
        self.channel_k = channel_k
        self.modules = dict(model.named_modules())
        self.saved = []

        conv0_name, conv1_name, conv2_name = _get_impt_type2_block_layer_names(block_name)
        self._register(conv0_name, dim=0)
        self._register(conv1_name, dim=0)
        self._register(conv2_name, dim=1)

    def _register(self, layer_name, dim):
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
# 메인
# ═══════════════════════════════════════════════════════════

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  ΔEO 재측정: 39개 속성 전체 평균 EO")
    print("  phi, perf는 results_v2/에서 재활용")
    print(f"  device: {device}")
    print(f"  v2 결과: {args.v2_dir}")
    print(f"  output: {output_dir}")
    print("=" * 70)

    # 데이터 + 모델 로드
    print("\n모델/데이터 로드 중...")
    ctx = build_celeba_context(args.split, args.batch_size, args.num_workers)
    n_attrs = ctx["n_attrs"]
    print(f"  속성 수: {n_attrs}")

    n_classes = n_attrs * 2
    model = build_model(n_classes)
    model = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()
    print(f"  모델 로드 완료 ({n_classes} classes)")

    # baseline EO (39개 속성 평균)
    print("\nbaseline EO (39개 속성 평균) 계산 중...")
    baseline_eo = compute_mean_eo_all_attrs(model, ctx["eval_loader"], n_attrs, args.max_batches)
    print(f"  baseline mean EO = {baseline_eo:.6f}")

    # results_v2/ 에서 채널별 CSV 찾기
    v2_csvs = sorted(args.v2_dir.glob("channels_*.csv"))
    if not v2_csvs:
        raise FileNotFoundError(f"results_v2/ 에서 channels_*.csv를 찾을 수 없음: {args.v2_dir}")

    print(f"\n{len(v2_csvs)}개 레이어 발견")
    summary_rows = []
    experiment_start = time.time()

    for csv_idx, v2_csv in enumerate(v2_csvs):
        df = pd.read_csv(v2_csv)
        conv1_name = df["layer"].iloc[0]
        block_name = df["block"].iloc[0] if "block" in df.columns else conv1_to_block_name(conv1_name)
        n_ch = len(df)

        print(f"\n{'='*60}")
        print(f"[{csv_idx+1}/{len(v2_csvs)}] {conv1_name} ({n_ch}ch)")
        print(f"{'='*60}")

        output_csv = output_dir / v2_csv.name

        # resume
        if args.resume and output_csv.exists():
            print(f"  기존 결과 발견 → 스킵")
            result_df = pd.read_csv(output_csv)
            phi = result_df["phi"].to_numpy(np.float64)
            perf = result_df["perf"].to_numpy(np.float64)
            delta_eo_abs = result_df["delta_eo_all_abs"].to_numpy(np.float64)

            phi_deo = safe_correlations(phi, delta_eo_abs)
            perf_deo = safe_correlations(perf, delta_eo_abs)

            summary_rows.append({
                "layer": conv1_name,
                "channels": n_ch,
                "phi_deo_spearman": phi_deo["spearman_rho"],
                "phi_deo_spearman_p": phi_deo["spearman_p"],
                "phi_deo_pearson": phi_deo["pearson_r"],
                "phi_deo_pearson_p": phi_deo["pearson_p"],
                "perf_deo_spearman": perf_deo["spearman_rho"],
                "perf_deo_spearman_p": perf_deo["spearman_p"],
                "perf_deo_pearson": perf_deo["pearson_r"],
                "perf_deo_pearson_p": perf_deo["pearson_p"],
            })
            continue

        phi_vec = df["phi"].to_numpy(np.float64)
        perf_vec = df["perf"].to_numpy(np.float64)

        # 채널별 zeroing → 39개 속성 평균 ΔEO
        rows = []
        t_start = time.time()

        with torch.no_grad():
            for ch in range(n_ch):
                with BlockChannelZeroing(model, block_name, ch):
                    pruned_eo = compute_mean_eo_all_attrs(
                        model, ctx["eval_loader"], n_attrs, args.max_batches
                    )

                delta_eo = pruned_eo - baseline_eo

                rows.append({
                    "layer": conv1_name,
                    "block": block_name,
                    "channel": ch,
                    "baseline_eo_all": baseline_eo,
                    "pruned_eo_all": pruned_eo,
                    "delta_eo_all": delta_eo,
                    "delta_eo_all_abs": abs(delta_eo),
                    "phi": float(phi_vec[ch]),
                    "perf": float(perf_vec[ch]),
                })

                if ch == 0 or (ch + 1) % 10 == 0 or ch + 1 == n_ch:
                    elapsed = time.time() - t_start
                    rate = (ch + 1) / elapsed if elapsed > 0 else 0
                    remain = (n_ch - ch - 1) / rate if rate > 0 else 0
                    print(
                        f"  [{ch+1}/{n_ch}] "
                        f"ΔEO(39attr)={delta_eo:+.6f}  "
                        f"({elapsed:.0f}s, ~{remain:.0f}s left)"
                    )

        result_df = pd.DataFrame(rows)
        result_df.to_csv(output_csv, index=False)
        print(f"  → 저장: {output_csv}")

        # 상관관계 계산
        delta_eo_abs = result_df["delta_eo_all_abs"].to_numpy(np.float64)
        phi_deo = safe_correlations(phi_vec, delta_eo_abs)
        perf_deo = safe_correlations(perf_vec, delta_eo_abs)

        summary_rows.append({
            "layer": conv1_name,
            "channels": n_ch,
            "phi_deo_spearman": phi_deo["spearman_rho"],
            "phi_deo_spearman_p": phi_deo["spearman_p"],
            "phi_deo_pearson": phi_deo["pearson_r"],
            "phi_deo_pearson_p": phi_deo["pearson_p"],
            "perf_deo_spearman": perf_deo["spearman_rho"],
            "perf_deo_spearman_p": perf_deo["spearman_p"],
            "perf_deo_pearson": perf_deo["pearson_r"],
            "perf_deo_pearson_p": perf_deo["pearson_p"],
        })

        print(f"\n  -- {conv1_name} 결과 (39개 속성 평균 EO) --")
        print(f"  fairness proxy vs |ΔEO|:  ρ={phi_deo['spearman_rho']:+.4f}  r={phi_deo['pearson_r']:+.4f}")
        print(f"  통합 성능 기여도 vs |ΔEO|:  ρ={perf_deo['spearman_rho']:+.4f}  r={perf_deo['pearson_r']:+.4f}")

    total_time = time.time() - experiment_start

    # 최종 요약
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "summary_deo_all_attrs.csv", index=False)

        print(f"\n{'='*80}")
        print(f"  실험 완료 (총 {total_time:.0f}s)")
        print(f"{'='*80}")

        print(f"\n  ── fairness proxy / 통합 성능 기여도 vs |ΔEO| (39개 속성 평균) ──")
        print(f"  {'layer':<28} {'ch':>4}  "
              f"{'fp→ΔEO ρ':>10} {'fp→ΔEO r':>10}  "
              f"{'perf→ΔEO ρ':>11} {'perf→ΔEO r':>11}")
        print(f"  {'-'*80}")
        for _, row in summary_df.iterrows():
            print(
                f"  {row['layer']:<28} {row['channels']:>4}  "
                f"{row['phi_deo_spearman']:>+10.4f} {row['phi_deo_pearson']:>+10.4f}  "
                f"{row['perf_deo_spearman']:>+11.4f} {row['perf_deo_pearson']:>+11.4f}"
            )

        print(f"\n  결과 저장: {output_dir}")

    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment": "ΔEO remeasurement: single attr (Attractive) → 39-attr mean EO",
        "reason": "phi(compute_phi_k)는 39개 속성 전체 gap 누적, 실제 pruning도 39개 평균 EO 사용. 기존 zeroing은 Attractive 1개만 측정하여 조건 불일치.",
        "checkpoint": str(args.checkpoint),
        "eo_method": "39개 속성 각각 EO 계산 후 평균 (train_and_val.py와 동일)",
        "phi_perf_source": str(args.v2_dir),
        "split": args.split,
        "max_batches": args.max_batches,
        "baseline_eo_all_attrs": baseline_eo,
        "seed": args.seed,
        "device": str(device),
        "total_time_sec": total_time,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
