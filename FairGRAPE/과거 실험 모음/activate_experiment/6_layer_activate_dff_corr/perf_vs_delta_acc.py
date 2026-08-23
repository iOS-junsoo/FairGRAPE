#!/usr/bin/env python
"""
성능 기여도 vs ΔAcc 상관관계 실험

기존 6개 레이어 CSV에서 perf 값을 읽고,
채널별 zeroing → ΔAcc를 측정하여 perf vs ΔAcc 상관관계를 계산.

사용법:
  cd /workspace/FairGRAPE/FairGRAPE
  /workspace/fairgrape_env_gpu/bin/python \
    activate_experiment/6_layer_activate_dff_corr/perf_vs_delta_acc.py

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
ROOT_DIR = SCRIPT_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import make_datasets, make_frame
from util import safe_forward_with_cudnn_fallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CHECKPOINT = (
    ROOT_DIR / "trained_model" / "unpruned"
    / "CelebA_unpruned_classes_bygender_mobilenetv2_0.pt"
)
DEFAULT_INPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results_delta_acc"

# 6개 레이어 CSV 파일명
LAYER_FILES = [
    ("features.1.conv.1", "channels_features_1_conv_1.csv"),
    ("features.2.conv.1.0", "channels_features_2_conv_1_0.csv"),
    ("features.3.conv.1.0", "channels_features_3_conv_1_0.csv"),
    ("features.6.conv.1.0", "channels_features_6_conv_1_0.csv"),
    ("features.10.conv.1.0", "channels_features_10_conv_1_0.csv"),
    ("features.16.conv.1.0", "channels_features_16_conv_1_0.csv"),
]


def parse_args():
    p = argparse.ArgumentParser(description="perf vs delta_Acc correlation")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
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
    col_used = attribute_names + ["gender"]
    _, eval_dataset = make_datasets(split_frame, split_frame, False, batch_size, col_used)
    from torch.utils.data import DataLoader
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    return eval_loader, attribute_names


def compute_accuracy(model, eval_loader, attribute_index, max_batches=None):
    """단일 속성에 대한 Accuracy 계산"""
    model.eval()
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
            correct += (preds == target_y).sum().item()
            total += target_y.size(0)

    return correct / max(total, 1)


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

    print("=" * 60)
    print("  성능 기여도 vs ΔAcc 상관관계 실험")
    print(f"  device: {device}")
    print(f"  input: {args.input_dir}")
    print(f"  output: {output_dir}")
    print("=" * 60)

    # 데이터 + 모델 로드
    print("\n모델/데이터 로드 중...")
    eval_loader, attribute_names = build_celeba_context(
        args.split, args.batch_size, args.num_workers)

    # Attractive 속성 인덱스
    attr_idx = next(i for i, name in enumerate(attribute_names)
                    if name.lower() == "attractive")
    print(f"  대상 속성: Attractive (index={attr_idx})")

    n_classes = len(attribute_names) * 2
    model = build_model(n_classes)
    model = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()
    print(f"  모델 로드 완료")

    # baseline Acc
    print("\nbaseline Acc 계산 중...")
    baseline_acc = compute_accuracy(model, eval_loader, attr_idx, args.max_batches)
    print(f"  baseline Acc = {baseline_acc:.6f}")

    summary_rows = []
    experiment_start = time.time()

    for layer_name, csv_filename in LAYER_FILES:
        print(f"\n{'='*50}")
        print(f"{layer_name}")
        print(f"{'='*50}")

        # 기존 CSV 로드
        input_csv = args.input_dir / csv_filename
        if not input_csv.exists():
            print(f"  CSV 없음, 스킵: {input_csv}")
            continue
        existing_df = pd.read_csv(input_csv)
        num_channels = len(existing_df)
        print(f"  {num_channels}채널 로드")

        output_csv = output_dir / csv_filename
        if args.resume and output_csv.exists():
            print(f"  기존 결과 발견 -> 스킵")
            result_df = pd.read_csv(output_csv)
            perf = result_df["perf"].to_numpy(np.float64)
            delta_acc_abs = result_df["delta_acc_abs"].to_numpy(np.float64)
            corr = safe_correlations(perf, delta_acc_abs)
            corr["layer"] = layer_name
            corr["channels"] = num_channels
            summary_rows.append(corr)
            continue

        # 레이어 모듈 찾기
        modules = dict(model.named_modules())
        target_layer = modules[layer_name]

        # 채널별 zeroing → ΔAcc
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

                pruned_acc = compute_accuracy(model, eval_loader, attr_idx, args.max_batches)

                target_layer.weight.data[ch].copy_(saved_w)
                if saved_b is not None:
                    target_layer.bias.data[ch].copy_(saved_b)

                delta_acc = pruned_acc - baseline_acc
                delta_acc_abs = abs(delta_acc)
                perf_val = float(existing_df.iloc[ch]["perf"])

                rows.append({
                    "layer": layer_name,
                    "channel": ch,
                    "baseline_acc": baseline_acc,
                    "pruned_acc": pruned_acc,
                    "delta_acc": delta_acc,
                    "delta_acc_abs": delta_acc_abs,
                    "perf": perf_val,
                    "phi": float(existing_df.iloc[ch]["phi"]),
                })

                if ch == 0 or (ch + 1) % 10 == 0 or ch + 1 == num_channels:
                    elapsed = time.time() - t_start
                    rate = (ch + 1) / elapsed if elapsed > 0 else 0
                    remain = (num_channels - ch - 1) / rate if rate > 0 else 0
                    print(f"  [{ch+1}/{num_channels}] "
                          f"delta_Acc={delta_acc:+.6f}  "
                          f"({elapsed:.0f}s, ~{remain:.0f}s left)")

        result_df = pd.DataFrame(rows)
        result_df.to_csv(output_csv, index=False)
        print(f"  -> 저장: {output_csv}")

        # 상관관계 계산
        perf = result_df["perf"].to_numpy(np.float64)
        delta_acc_abs_arr = result_df["delta_acc_abs"].to_numpy(np.float64)

        perf_vs_dacc = safe_correlations(perf, delta_acc_abs_arr)
        phi = result_df["phi"].to_numpy(np.float64)
        phi_vs_dacc = safe_correlations(phi, delta_acc_abs_arr)

        summary_rows.append({
            "layer": layer_name,
            "channels": num_channels,
            "perf_dacc_spearman": perf_vs_dacc["spearman_rho"],
            "perf_dacc_spearman_p": perf_vs_dacc["spearman_p"],
            "perf_dacc_pearson": perf_vs_dacc["pearson_r"],
            "perf_dacc_pearson_p": perf_vs_dacc["pearson_p"],
            "phi_dacc_spearman": phi_vs_dacc["spearman_rho"],
            "phi_dacc_spearman_p": phi_vs_dacc["spearman_p"],
            "phi_dacc_pearson": phi_vs_dacc["pearson_r"],
            "phi_dacc_pearson_p": phi_vs_dacc["pearson_p"],
        })

        print(f"  perf vs |ΔAcc|: Spearman={perf_vs_dacc['spearman_rho']:.4f}, "
              f"Pearson={perf_vs_dacc['pearson_r']:.4f}")
        print(f"  phi  vs |ΔAcc|: Spearman={phi_vs_dacc['spearman_rho']:.4f}, "
              f"Pearson={phi_vs_dacc['pearson_r']:.4f}")

    total_time = time.time() - experiment_start

    # 요약 저장
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "summary_perf_vs_delta_acc.csv", index=False)

        print(f"\n{'='*70}")
        print(f"  실험 완료 ({total_time:.0f}s)")
        print(f"{'='*70}")
        print(f"\n  {'layer':<28} {'ch':>4}  "
              f"{'perf→ΔAcc(ρ)':>12} {'perf→ΔAcc(r)':>12}  "
              f"{'φ_k→ΔAcc(ρ)':>12} {'φ_k→ΔAcc(r)':>12}")
        print(f"  {'-'*85}")
        for _, row in summary_df.iterrows():
            print(f"  {row['layer']:<28} {row['channels']:>4}  "
                  f"{row.get('perf_dacc_spearman', row.get('spearman_rho', float('nan'))):>+12.4f} "
                  f"{row.get('perf_dacc_pearson', row.get('pearson_r', float('nan'))):>+12.4f}  "
                  f"{row.get('phi_dacc_spearman', float('nan')):>+12.4f} "
                  f"{row.get('phi_dacc_pearson', float('nan')):>+12.4f}")

        print(f"\n  결과: {output_dir}")

    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "checkpoint": str(args.checkpoint),
        "target_attr": "Attractive",
        "target_attr_index": attr_idx,
        "baseline_acc": baseline_acc,
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
        "total_time_sec": total_time,
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()