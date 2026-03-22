from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import make_datasets, make_frame
from util import filter_readable_images_in_frames, safe_forward_with_cudnn_fallback

DEFAULT_CHECKPOINT = ROOT_DIR / "trained_model" / "unpruned" / "CelebA_unpruned_classes_bygender_mobilenetv2_0.pt"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "activate_experiment" / "results" / "celeba_unpruned_activation_gap"
GROUP_KEYS = ((0, 0), (0, 1), (1, 0), (1, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CelebA unpruned MobileNetV2에 대해 레이어별 activation gap 기반 fairness proxy(phi)를 계산합니다."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="분석할 체크포인트 경로")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="결과 저장 디렉토리")
    parser.add_argument("--batch-size", type=int, default=64, help="데이터 로더 배치 크기")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="분석할 데이터 split")
    parser.add_argument(
        "--target-attr",
        type=str,
        default="all",
        help="분석할 CelebA attribute 이름 또는 인덱스. all이면 39개 속성을 모두 분석합니다.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="디버깅용 배치 제한. 지정하지 않으면 전체 split을 사용합니다.",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker 수")
    return parser.parse_args()


def build_model(n_classes: int) -> nn.Module:
    model = models.mobilenet_v2(pretrained=False)
    classifier_in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(classifier_in_features, n_classes)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state"])
    else:
        model.load_state_dict(checkpoint_data)


def get_target_layers(model: nn.Module) -> OrderedDict[str, nn.Module]:
    target_layers: OrderedDict[str, nn.Module] = OrderedDict()
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            target_layers[name] = module
    return target_layers


def reduce_feature_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 4:
        return F.adaptive_avg_pool2d(tensor, 1).flatten(1)
    if tensor.dim() == 2:
        return tensor
    return tensor.reshape(tensor.size(0), -1)


class LayerCollector:
    def __init__(self, target_layers: OrderedDict[str, nn.Module], capture_gradients: bool) -> None:
        self.target_layers = target_layers
        self.capture_gradients = capture_gradients
        self.activations: dict[str, torch.Tensor] = {}
        self.gradients: dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, layer in self.target_layers.items():
            self.hooks.append(layer.register_forward_hook(self._make_forward_hook(name)))

    def _make_forward_hook(self, layer_name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            reduced_output = reduce_feature_tensor(output)
            self.activations[layer_name] = reduced_output.detach().cpu()

            if self.capture_gradients and isinstance(output, torch.Tensor) and output.requires_grad:
                output.register_hook(self._make_gradient_hook(layer_name))

        return hook

    def _make_gradient_hook(self, layer_name: str):
        def hook(grad: torch.Tensor) -> None:
            reduced_grad = reduce_feature_tensor(grad.abs())
            self.gradients[layer_name] = reduced_grad.detach().cpu()

        return hook

    def clear(self) -> None:
        self.activations = {}
        self.gradients = {}

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def build_celeba_loader(batch_size: int, split: str, num_workers: int):
    csv_path = ROOT_DIR / "csv" / "CelebA.csv"
    face_dir = ROOT_DIR / "Images" / "CelebA" / "img_align_celeba"
    report_dir = ROOT_DIR / "data_quality_logs" / "CelebA"

    frames = make_frame(str(csv_path), str(face_dir), seven_races=False)
    frames = filter_readable_images_in_frames(
        frames,
        loader="dlib",
        report_dir=str(report_dir),
        report_prefix="activation_gap_excluded_images",
    )

    attribute_names = list(frames["train"].columns[2:41])
    col_used = attribute_names + ["gender"]

    _, eval_dataset = make_datasets(frames["train"], frames[split], False, batch_size, col_used)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return frames, eval_loader, attribute_names


def resolve_target_attributes(attribute_names: list[str], target_attr: str) -> list[tuple[int, str]]:
    if target_attr.lower() == "all":
        return list(enumerate(attribute_names))

    if target_attr.isdigit():
        attr_index = int(target_attr)
        if attr_index < 0 or attr_index >= len(attribute_names):
            raise ValueError(f"attribute index out of range: {attr_index}")
        return [(attr_index, attribute_names[attr_index])]

    lowered = {name.lower(): (idx, name) for idx, name in enumerate(attribute_names)}
    if target_attr.lower() not in lowered:
        raise ValueError(f"unknown attribute: {target_attr}")
    return [lowered[target_attr.lower()]]


def init_group_accumulator() -> dict[tuple[int, int], dict[str, object]]:
    return {group_key: {"sum": None, "count": 0} for group_key in GROUP_KEYS}


def update_group_stats(
    grouped_stats: dict[str, dict[tuple[int, int], dict[str, object]]],
    activations: dict[str, torch.Tensor],
    labels_y: torch.Tensor,
    labels_a: torch.Tensor,
) -> None:
    labels_y = labels_y.cpu()
    labels_a = labels_a.cpu()

    for layer_name, layer_activation in activations.items():
        layer_activation = layer_activation.to(torch.float64)
        layer_stats = grouped_stats.setdefault(layer_name, init_group_accumulator())
        for y_value, a_value in GROUP_KEYS:
            mask = (labels_y == y_value) & (labels_a == a_value)
            if not torch.any(mask):
                continue
            masked_values = layer_activation[mask]
            current_sum = masked_values.sum(dim=0)
            if layer_stats[(y_value, a_value)]["sum"] is None:
                layer_stats[(y_value, a_value)]["sum"] = current_sum
            else:
                layer_stats[(y_value, a_value)]["sum"] += current_sum
            layer_stats[(y_value, a_value)]["count"] += int(mask.sum().item())


def update_gradient_stats(
    gradient_sums: dict[str, torch.Tensor],
    gradient_counts: dict[str, int],
    gradients: dict[str, torch.Tensor],
) -> None:
    for layer_name, layer_gradient in gradients.items():
        layer_gradient = layer_gradient.to(torch.float64)
        if layer_name not in gradient_sums:
            gradient_sums[layer_name] = layer_gradient.sum(dim=0)
            gradient_counts[layer_name] = layer_gradient.shape[0]
        else:
            gradient_sums[layer_name] += layer_gradient.sum(dim=0)
            gradient_counts[layer_name] += layer_gradient.shape[0]


def compute_activation_gap_for_attribute(
    model: nn.Module,
    eval_loader,
    target_layers: OrderedDict[str, nn.Module],
    attribute_index: int,
    max_batches: int | None,
    device: torch.device,
) -> dict[str, dict[tuple[int, int], dict[str, object]]]:
    grouped_stats: dict[str, dict[tuple[int, int], dict[str, object]]] = {}
    collector = LayerCollector(target_layers, capture_gradients=False)

    model.eval()
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(eval_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(device)
            labels = labels.long()
            target_y = labels[:, attribute_index]
            sensitive_a = labels[:, -1]

            collector.clear()
            _ = safe_forward_with_cudnn_fallback(model, images)
            update_group_stats(grouped_stats, collector.activations, target_y, sensitive_a)

    collector.remove()
    return grouped_stats


def compute_mean_gradients_for_attribute(
    model: nn.Module,
    eval_loader,
    target_layers: OrderedDict[str, nn.Module],
    attribute_index: int,
    max_batches: int | None,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    criterion = nn.CrossEntropyLoss()
    gradient_sums: dict[str, torch.Tensor] = {}
    gradient_counts: dict[str, int] = {}
    collector = LayerCollector(target_layers, capture_gradients=True)

    model.eval()
    for batch_index, (images, labels) in enumerate(eval_loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device).long()

        collector.clear()
        model.zero_grad(set_to_none=True)
        start = attribute_index * 2
        end = start + 2

        try:
            outputs = safe_forward_with_cudnn_fallback(model, images)
            loss = criterion(outputs[:, start:end], labels[:, attribute_index])
            loss.backward()
        except RuntimeError as error:
            if "cuDNN" not in str(error) and "CUDNN" not in str(error):
                raise
            collector.clear()
            model.zero_grad(set_to_none=True)
            with torch.backends.cudnn.flags(enabled=False):
                outputs = model(images)
                loss = criterion(outputs[:, start:end], labels[:, attribute_index])
                loss.backward()

        update_gradient_stats(gradient_sums, gradient_counts, collector.gradients)

    collector.remove()
    return gradient_sums, gradient_counts


def nanmean_or_none(values: np.ndarray) -> float | None:
    if not np.isfinite(values).any():
        return None
    return float(np.nanmean(values))


def nanmax_or_none(values: np.ndarray) -> float | None:
    if not np.isfinite(values).any():
        return None
    return float(np.nanmax(values))


def safe_top_index(values: np.ndarray) -> int | None:
    if not np.isfinite(values).any():
        return None
    return int(np.nanargmax(values))


def summarize_attribute(
    target_layers: OrderedDict[str, nn.Module],
    grouped_stats: dict[str, dict[tuple[int, int], dict[str, object]]],
    gradient_sums: dict[str, torch.Tensor],
    gradient_counts: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, int]]:
    summary_rows = []
    raw_arrays: dict[str, np.ndarray] = {}
    group_counts: dict[str, int] = {}

    for layer_name in target_layers:
        layer_groups = grouped_stats[layer_name]
        layer_means: dict[tuple[int, int], np.ndarray] = {}
        layer_width = None

        for group_key in GROUP_KEYS:
            group_sum = layer_groups[group_key]["sum"]
            group_count = int(layer_groups[group_key]["count"])
            group_counts[f"{layer_name}:{group_key[0]}_{group_key[1]}"] = group_count

            if group_sum is not None:
                layer_width = int(group_sum.shape[0])
                layer_means[group_key] = (group_sum / group_count).numpy()
            elif layer_width is not None:
                layer_means[group_key] = np.full(layer_width, np.nan, dtype=np.float64)

        if layer_width is None:
            continue

        for group_key in GROUP_KEYS:
            if group_key not in layer_means:
                layer_means[group_key] = np.full(layer_width, np.nan, dtype=np.float64)

        gap_y0 = np.abs(layer_means[(0, 1)] - layer_means[(0, 0)])
        gap_y1 = np.abs(layer_means[(1, 1)] - layer_means[(1, 0)])
        activation_gap = gap_y0 + gap_y1

        if layer_name in gradient_sums and gradient_counts.get(layer_name, 0) > 0:
            mean_gradient = (gradient_sums[layer_name] / gradient_counts[layer_name]).numpy()
        else:
            mean_gradient = np.full(layer_width, np.nan, dtype=np.float64)

        phi = activation_gap * mean_gradient
        top_unit = safe_top_index(phi)

        raw_arrays[f"{layer_name}__gap_y0"] = gap_y0
        raw_arrays[f"{layer_name}__gap_y1"] = gap_y1
        raw_arrays[f"{layer_name}__activation_gap"] = activation_gap
        raw_arrays[f"{layer_name}__mean_gradient"] = mean_gradient
        raw_arrays[f"{layer_name}__phi"] = phi

        summary_rows.append(
            {
                "layer_name": layer_name,
                "num_units": layer_width,
                "gap_y0_mean": nanmean_or_none(gap_y0),
                "gap_y1_mean": nanmean_or_none(gap_y1),
                "activation_gap_mean": nanmean_or_none(activation_gap),
                "activation_gap_max": nanmax_or_none(activation_gap),
                "mean_gradient_mean": nanmean_or_none(mean_gradient),
                "mean_gradient_max": nanmax_or_none(mean_gradient),
                "phi_mean": nanmean_or_none(phi),
                "phi_max": nanmax_or_none(phi),
                "top_unit_index": top_unit,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, raw_arrays, group_counts


def save_summary_plots(summary_df: pd.DataFrame, output_dir: Path, attribute_name: str) -> None:
    if not HAS_MATPLOTLIB or summary_df.empty:
        return

    metrics = [
        ("gap_y1_mean", "Conditional activation gap (Y=1)", "gap_y1_mean.png"),
        ("gap_y0_mean", "Conditional activation gap (Y=0)", "gap_y0_mean.png"),
        ("phi_mean", "Fairness proxy phi", "phi_mean.png"),
    ]

    for metric_column, title, filename in metrics:
        values = summary_df[metric_column].astype(float).to_numpy()
        if not np.isfinite(values).any():
            continue

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(summary_df["layer_name"], values, color="#1f77b4")
        ax.set_title(f"{attribute_name}: {title}")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_column)
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)


def save_attribute_outputs(
    output_dir: Path,
    attribute_name: str,
    summary_df: pd.DataFrame,
    raw_arrays: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    attribute_dir = output_dir / attribute_name
    attribute_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(attribute_dir / "layer_summary.csv", index=False)
    np.savez_compressed(attribute_dir / "layer_raw_scores.npz", **raw_arrays)
    with open(attribute_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    save_summary_plots(summary_df, attribute_dir, attribute_name)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames, eval_loader, attribute_names = build_celeba_loader(
        batch_size=args.batch_size,
        split=args.split,
        num_workers=args.num_workers,
    )
    selected_attributes = resolve_target_attributes(attribute_names, args.target_attr)

    model = build_model(n_classes=len(attribute_names) * 2)
    load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    target_layers = get_target_layers(model)

    run_metadata = {
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "device": str(device),
        "num_eval_samples": len(frames[args.split]),
        "num_target_layers": len(target_layers),
        "target_layers": list(target_layers.keys()),
        "selected_attributes": [attribute_name for _, attribute_name in selected_attributes],
    }

    with open(args.output_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2, ensure_ascii=False)

    for attribute_index, attribute_name in selected_attributes:
        print(f"[start] attribute={attribute_name} (index={attribute_index})")
        grouped_stats = compute_activation_gap_for_attribute(
            model=model,
            eval_loader=eval_loader,
            target_layers=target_layers,
            attribute_index=attribute_index,
            max_batches=args.max_batches,
            device=device,
        )
        gradient_sums, gradient_counts = compute_mean_gradients_for_attribute(
            model=model,
            eval_loader=eval_loader,
            target_layers=target_layers,
            attribute_index=attribute_index,
            max_batches=args.max_batches,
            device=device,
        )
        summary_df, raw_arrays, group_counts = summarize_attribute(
            target_layers=target_layers,
            grouped_stats=grouped_stats,
            gradient_sums=gradient_sums,
            gradient_counts=gradient_counts,
        )

        attribute_metadata = {
            "attribute_name": attribute_name,
            "attribute_index": attribute_index,
            "group_counts": group_counts,
            "num_layers": int(summary_df.shape[0]),
        }
        save_attribute_outputs(args.output_dir, attribute_name, summary_df, raw_arrays, attribute_metadata)
        print(f"[done] attribute={attribute_name} -> {args.output_dir / attribute_name}")


if __name__ == "__main__":
    main()