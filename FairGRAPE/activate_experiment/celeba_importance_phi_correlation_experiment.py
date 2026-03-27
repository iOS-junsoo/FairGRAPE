from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from torchvision import models


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset import make_datasets, make_frame
from prune import compute_importance
from util import filter_readable_images_in_frames, safe_forward_with_cudnn_fallback


DEFAULT_CHECKPOINT = ROOT_DIR / "trained_model" / "unpruned" / "CelebA_unpruned_classes_bygender_mobilenetv2_0.pt"
DEFAULT_ACTIVATION_RESULTS = (
    ROOT_DIR
    / "activate_experiment"
    / "results"
    / "celeba_unpruned_activation_gap"
    / "Attractive"
    / "layer_raw_scores.npz"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "activate_experiment" / "corr_result"
DEFAULT_TARGET_LAYER = "features.17.conv.2"
ACTIVATION_METRICS = ("phi", "gap_y0", "gap_y1", "activation_gap", "mean_gradient")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare compute_importance channel scores against activation-gap phi scores on CelebA."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--activation-results", type=Path, default=DEFAULT_ACTIVATION_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--target-attr", type=str, default="Attractive")
    parser.add_argument("--target-layer", type=str, default=DEFAULT_TARGET_LAYER)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--check-readable-images",
        action="store_true",
        help="Run dlib-based readability filtering on the selected split before analysis.",
    )
    parser.add_argument(
        "--importance-stop-batch",
        type=int,
        default=10000,
        help="Maximum number of batches used inside compute_importance.",
    )
    parser.add_argument(
        "--importance-batch-size",
        type=int,
        default=64,
        help="Mini-batch size used inside compute_importance.",
    )
    parser.add_argument(
        "--max-eo-batches",
        type=int,
        default=None,
        help="Optional batch cap for EO evaluation during experiment 2.",
    )
    parser.add_argument(
        "--max-zeroed-channels",
        type=int,
        default=None,
        help="Optional debug cap for experiment 2 channel zeroing.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=5000,
        help="Maximum number of points to render per scatter plot.",
    )
    return parser.parse_args()


def build_model(n_classes: int) -> nn.Module:
    model = models.mobilenet_v2(pretrained=False)
    classifier_in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(classifier_in_features, n_classes)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state"])
    else:
        model.load_state_dict(checkpoint_data)
    return model


def build_celeba_context(
    split: str,
    batch_size: int,
    num_workers: int,
    check_readable_images: bool,
) -> dict[str, object]:
    csv_path = ROOT_DIR / "csv" / "CelebA.csv"
    face_dir = ROOT_DIR / "Images" / "CelebA" / "img_align_celeba"
    report_dir = ROOT_DIR / "data_quality_logs" / "CelebA"

    if not csv_path.exists():
        raise FileNotFoundError(f"CelebA csv not found: {csv_path}")
    if not face_dir.exists():
        raise FileNotFoundError(f"CelebA image directory not found: {face_dir}")

    print(f"[context] building CelebA {split} split context")
    frames = make_frame(str(csv_path), str(face_dir), seven_races=False)
    filtered_split_frame = frames[split]
    if check_readable_images:
        print(f"[context] running readability check for split={split}")
        filtered_split_frame = filter_readable_images_in_frames(
            {split: frames[split]},
            loader="dlib",
            report_dir=str(report_dir),
            report_prefix="importance_phi_correlation_excluded_images",
        )[split]
    else:
        print(f"[context] skipping readability check for split={split}")

    attribute_names = list(frames["train"].columns[2:41])
    output_cols_each_task = [(index * 2, index * 2 + 2) for index in range(len(attribute_names))]
    col_used = attribute_names + ["gender"]

    _, eval_dataset = make_datasets(filtered_split_frame, filtered_split_frame, False, batch_size, col_used)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {
        "frames": frames,
        "frame": filtered_split_frame,
        "eval_loader": eval_loader,
        "face_dir": str(face_dir),
        "attribute_names": attribute_names,
        "output_cols_each_task": output_cols_each_task,
        "col_used": col_used,
    }


def resolve_target_attribute(attribute_names: list[str], target_attr: str) -> tuple[int, str]:
    if target_attr.isdigit():
        attr_index = int(target_attr)
        if attr_index < 0 or attr_index >= len(attribute_names):
            raise ValueError(f"attribute index out of range: {attr_index}")
        return attr_index, attribute_names[attr_index]

    lowered = {name.lower(): (idx, name) for idx, name in enumerate(attribute_names)}
    if target_attr.lower() not in lowered:
        raise ValueError(f"unknown attribute: {target_attr}")
    return lowered[target_attr.lower()]


def load_activation_scores(npz_path: Path) -> dict[str, dict[str, np.ndarray]]:
    if not npz_path.exists():
        raise FileNotFoundError(f"activation score file not found: {npz_path}")

    npz_data = np.load(npz_path)
    layer_scores: dict[str, dict[str, np.ndarray]] = {}

    for key in npz_data.files:
        if "__" not in key:
            continue
        layer_name, metric_name = key.split("__", 1)
        if metric_name not in ACTIVATION_METRICS:
            continue
        layer_scores.setdefault(layer_name, {})[metric_name] = np.asarray(npz_data[key], dtype=np.float64)

    missing_metric_layers = [
        layer_name
        for layer_name, metrics in layer_scores.items()
        if any(metric_name not in metrics for metric_name in ACTIVATION_METRICS)
    ]
    if missing_metric_layers:
        raise ValueError(
            "activation npz is missing metrics for layers: " + ", ".join(sorted(missing_metric_layers)[:10])
        )

    return layer_scores


def aggregate_tensor_to_channels(score_tensor: torch.Tensor) -> np.ndarray:
    score_tensor = score_tensor.detach().cpu()

    if score_tensor.ndim == 4:
        aggregated = score_tensor.mean(dim=(1, 2, 3))  # sum → mean
    elif score_tensor.ndim == 3:
        aggregated = score_tensor.mean(dim=(1, 2))     # sum → mean
    elif score_tensor.ndim == 2:
        aggregated = score_tensor.mean(dim=1)          # sum → mean
    elif score_tensor.ndim == 1:
        aggregated = score_tensor
    else:
        aggregated = score_tensor.reshape(score_tensor.shape[0], -1).mean(dim=1)  # sum → mean

    return aggregated.numpy().astype(np.float64)


def aggregate_scores_by_layer(score_by_layer: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    aggregated_scores = {}
    for layer_name, score_tensor in score_by_layer.items():
        aggregated_scores[layer_name] = aggregate_tensor_to_channels(score_tensor)
    return aggregated_scores


def filter_valid_pair(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert x_values.shape == y_values.shape, f"length mismatch: {x_values.shape} vs {y_values.shape}"
    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[valid_mask], y_values[valid_mask]


def safe_correlations(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float | int]:
    filtered_x, filtered_y = filter_valid_pair(x_values, y_values)
    result = {
        "num_points": int(filtered_x.shape[0]),
        "pearson": np.nan,
        "pearson_p": np.nan,
        "spearman": np.nan,
        "spearman_p": np.nan,
    }

    if filtered_x.shape[0] < 2:
        return result
    if np.allclose(filtered_x, filtered_x[0]) or np.allclose(filtered_y, filtered_y[0]):
        return result

    pearson_value, pearson_p = pearsonr(filtered_x, filtered_y)
    spearman_value, spearman_p = spearmanr(filtered_x, filtered_y)

    result["pearson"] = float(pearson_value)
    result["pearson_p"] = float(pearson_p)
    result["spearman"] = float(spearman_value)
    result["spearman_p"] = float(spearman_p)
    return result


def sample_points_for_plot(
    x_values: np.ndarray,
    y_values: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    filtered_x, filtered_y = filter_valid_pair(x_values, y_values)
    if filtered_x.shape[0] <= max_points:
        return filtered_x, filtered_y

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(filtered_x.shape[0], size=max_points, replace=False)
    return filtered_x[sample_indices], filtered_y[sample_indices]


def save_scatter_with_regression(
    x_values: np.ndarray,
    y_values: np.ndarray,
    save_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    correlation_summary: dict[str, float | int],
    max_points: int,
) -> None:
    plot_x, plot_y = sample_points_for_plot(x_values, y_values, max_points=max_points)
    if plot_x.shape[0] < 2:
        return

    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    axis.scatter(plot_x, plot_y, s=14, alpha=0.45, color="#1f77b4", edgecolors="none")

    if not np.allclose(plot_x, plot_x[0]) and not np.allclose(plot_y, plot_y[0]):
        slope, intercept = np.polyfit(plot_x, plot_y, deg=1)
        line_x = np.linspace(plot_x.min(), plot_x.max(), 200)
        line_y = slope * line_x + intercept
        axis.plot(line_x, line_y, color="#d62728", linewidth=2.0)

    subtitle = (
        f"Pearson={correlation_summary['pearson']:.4f}, "
        f"Spearman={correlation_summary['spearman']:.4f}, "
        f"N={correlation_summary['num_points']}"
    )
    axis.set_title(f"{title}\n{subtitle}")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_experiment1_plots(
    activation_scores: dict[str, dict[str, np.ndarray]],
    importance_channel_scores: dict[str, np.ndarray],
    global_df: pd.DataFrame,
    plot_dir: Path,
    max_plot_points: int,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    for _, row in global_df.iterrows():
        score_name = row["score_name"]
        metric_name = row["activation_metric"]
        x_parts = []
        y_parts = []

        common_layers = sorted(set(importance_channel_scores.keys()) & set(activation_scores.keys()))
        for layer_name in common_layers:
            x_vector = importance_channel_scores[layer_name]
            y_vector = activation_scores[layer_name][metric_name]
            if x_vector.shape != y_vector.shape:
                continue
            filtered_x, filtered_y = filter_valid_pair(x_vector, y_vector)
            if filtered_x.size == 0:
                continue
            x_parts.append(filtered_x)
            y_parts.append(filtered_y)

        if not x_parts:
            continue

        x_values = np.concatenate(x_parts, axis=0)
        y_values = np.concatenate(y_parts, axis=0)
        save_scatter_with_regression(
            x_values=x_values,
            y_values=y_values,
            save_path=plot_dir / f"{score_name}_vs_{metric_name}.png",
            title=f"Experiment 1: {score_name} vs {metric_name}",
            xlabel=score_name,
            ylabel=metric_name,
            correlation_summary=row.to_dict(),
            max_points=max_plot_points,
        )


def save_experiment2_plots(
    channel_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    target_layer_name: str,
    plot_dir: Path,
    max_plot_points: int,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    for _, row in correlation_df.iterrows():
        score_name = row["score_name"]
        save_scatter_with_regression(
            x_values=channel_df[score_name].to_numpy(dtype=np.float64),
            y_values=channel_df["delta_eo_abs"].to_numpy(dtype=np.float64),
            save_path=plot_dir / f"{target_layer_name.replace('.', '_')}__{score_name}_vs_delta_eo_abs.png",
            title=f"Experiment 2: {target_layer_name} {score_name} vs |delta EO|",
            xlabel=score_name,
            ylabel="delta_eo_abs",
            correlation_summary=row.to_dict(),
            max_points=max_plot_points,
        )


def compute_experiment1(
    activation_scores: dict[str, dict[str, np.ndarray]],
    importance_channel_scores: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_layer_rows = []
    global_pairs: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    skipped_layers: set[tuple[str, str, str]] = set()

    common_layers = sorted(set(activation_scores.keys()) & set(importance_channel_scores.keys()))
    if not common_layers:
        raise RuntimeError("No common layers found across activation and importance_score.")

    for layer_name in common_layers:
        metrics = activation_scores[layer_name]
        score_vectors = {
            "importance_score": importance_channel_scores[layer_name],
        }

        for score_name, score_vector in score_vectors.items():
            for metric_name in ACTIVATION_METRICS:
                metric_vector = metrics[metric_name]
                if score_vector.shape != metric_vector.shape:
                    skipped_layers.add((layer_name, score_name, metric_name))
                    continue

                corr_values = safe_correlations(score_vector, metric_vector)
                per_layer_rows.append(
                    {
                        "layer_name": layer_name,
                        "num_channels": int(score_vector.shape[0]),
                        "score_name": score_name,
                        "activation_metric": metric_name,
                        **corr_values,
                    }
                )

                filtered_x, filtered_y = filter_valid_pair(score_vector, metric_vector)
                global_pairs.setdefault((score_name, metric_name), []).append((filtered_x, filtered_y))

    global_rows = []
    for (score_name, metric_name), value_pairs in sorted(global_pairs.items()):
        stacked_x = np.concatenate([pair[0] for pair in value_pairs if pair[0].size > 0], axis=0)
        stacked_y = np.concatenate([pair[1] for pair in value_pairs if pair[1].size > 0], axis=0)
        corr_values = safe_correlations(stacked_x, stacked_y)
        global_rows.append(
            {
                "score_name": score_name,
                "activation_metric": metric_name,
                **corr_values,
            }
        )

    per_layer_df = pd.DataFrame(per_layer_rows)
    global_df = pd.DataFrame(global_rows)

    if skipped_layers:
        print(f"[experiment1] skipped {len(skipped_layers)} mismatched layer-score pairs")
        for layer_name, score_name, metric_name in sorted(skipped_layers):
            print(f"  - skip {layer_name}: {score_name} vs {metric_name}")

    return per_layer_df, global_df


def compute_attribute_eo(
    model: nn.Module,
    eval_loader,
    attribute_index: int,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()

    stats = {
        0: {"tp": 0, "fp": 0, "total_p": 0, "total_n": 0},
        1: {"tp": 0, "fp": 0, "total_p": 0, "total_n": 0},
    }

    output_start = attribute_index * 2
    output_end = output_start + 2

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(eval_loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device).long()

            outputs = safe_forward_with_cudnn_fallback(model, images)
            logits = outputs[:, output_start:output_end]
            preds = torch.argmax(logits, dim=1)

            target_y = labels[:, attribute_index]
            sensitive_a = labels[:, -1]

            for group_value in (0, 1):
                group_mask = sensitive_a == group_value
                if not torch.any(group_mask):
                    continue

                y_true = target_y[group_mask]
                y_pred = preds[group_mask]

                tp = ((y_pred == 1) & (y_true == 1)).sum().item()
                fp = ((y_pred == 1) & (y_true == 0)).sum().item()
                total_p = (y_true == 1).sum().item()
                total_n = (y_true == 0).sum().item()

                stats[group_value]["tp"] += tp
                stats[group_value]["fp"] += fp
                stats[group_value]["total_p"] += total_p
                stats[group_value]["total_n"] += total_n

    if min(stats[0]["total_p"], stats[1]["total_p"], stats[0]["total_n"], stats[1]["total_n"]) == 0:
        raise RuntimeError(f"Insufficient samples to compute EO: {stats}")

    eps = 1e-12
    tpr0 = stats[0]["tp"] / (stats[0]["total_p"] + eps)
    tpr1 = stats[1]["tp"] / (stats[1]["total_p"] + eps)
    fpr0 = stats[0]["fp"] / (stats[0]["total_n"] + eps)
    fpr1 = stats[1]["fp"] / (stats[1]["total_n"] + eps)
    return float((abs(tpr0 - tpr1) + abs(fpr0 - fpr1)) / 2.0)


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(f"module not found: {module_name}")
    return modules[module_name]


def compute_experiment2(
    model: nn.Module,
    eval_loader,
    target_layer_name: str,
    attribute_index: int,
    activation_scores: dict[str, dict[str, np.ndarray]],
    importance_channel_scores: dict[str, np.ndarray],
    device: torch.device,
    max_batches: int | None,
    max_zeroed_channels: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if target_layer_name not in activation_scores:
        raise KeyError(f"target layer not found in activation scores: {target_layer_name}")
    if target_layer_name not in importance_channel_scores:
        raise KeyError(f"target layer not found in importance_score: {target_layer_name}")

    target_layer = get_module_by_name(model, target_layer_name)
    if not hasattr(target_layer, "weight"):
        raise TypeError(f"target layer has no weight parameter: {target_layer_name}")

    target_weight = target_layer.weight
    num_channels = int(target_weight.shape[0])

    phi = activation_scores[target_layer_name]["phi"]
    gap_y0 = activation_scores[target_layer_name]["gap_y0"]
    gap_y1 = activation_scores[target_layer_name]["gap_y1"]
    activation_gap = activation_scores[target_layer_name]["activation_gap"]
    mean_gradient = activation_scores[target_layer_name]["mean_gradient"]
    importance_channel = importance_channel_scores[target_layer_name]

    for vector_name, vector in {
        "phi": phi,
        "gap_y0": gap_y0,
        "gap_y1": gap_y1,
        "activation_gap": activation_gap,
        "mean_gradient": mean_gradient,
        "importance_channel": importance_channel,
    }.items():
        assert vector.shape == (num_channels,), (
            f"length mismatch for {target_layer_name}: {vector_name}={vector.shape}, expected={(num_channels,)}"
        )

    baseline_eo = compute_attribute_eo(
        model=model,
        eval_loader=eval_loader,
        attribute_index=attribute_index,
        device=device,
        max_batches=max_batches,
    )

    limit = num_channels if max_zeroed_channels is None else min(num_channels, max_zeroed_channels)
    channel_rows = []

    with torch.no_grad():
        for channel_index in range(limit):
            saved_weight = target_layer.weight.data[channel_index].clone()
            target_layer.weight.data[channel_index].zero_()

            saved_bias = None
            if target_layer.bias is not None:
                saved_bias = target_layer.bias.data[channel_index].clone()
                target_layer.bias.data[channel_index].zero_()

            pruned_eo = compute_attribute_eo(
                model=model,
                eval_loader=eval_loader,
                attribute_index=attribute_index,
                device=device,
                max_batches=max_batches,
            )
            delta_eo_abs = abs(pruned_eo - baseline_eo)

            target_layer.weight.data[channel_index].copy_(saved_weight)
            if saved_bias is not None:
                target_layer.bias.data[channel_index].copy_(saved_bias)

            channel_rows.append(
                {
                    "channel_index": channel_index,
                    "baseline_eo": baseline_eo,
                    "pruned_eo": pruned_eo,
                    "delta_eo_abs": delta_eo_abs,
                    "phi": float(phi[channel_index]),
                    "gap_y0": float(gap_y0[channel_index]),
                    "gap_y1": float(gap_y1[channel_index]),
                    "activation_gap": float(activation_gap[channel_index]),
                    "mean_gradient": float(mean_gradient[channel_index]),
                    "importance_channel": float(importance_channel[channel_index]),
                }
            )

            if channel_index == 0 or (channel_index + 1) % 10 == 0 or channel_index + 1 == limit:
                print(
                    f"[experiment2] {channel_index + 1}/{limit} channels processed "
                    f"for {target_layer_name}; baseline_eo={baseline_eo:.8f}"
                )

    channel_df = pd.DataFrame(channel_rows)
    correlation_rows = []
    for score_name in ("phi", "importance_channel"):
        abs_corr_values = safe_correlations(
            channel_df[score_name].to_numpy(dtype=np.float64),
            channel_df["delta_eo_abs"].to_numpy(dtype=np.float64),
        )
        correlation_rows.append(
            {
                "target_layer": target_layer_name,
                "score_name": score_name,
                "delta_metric": "delta_eo_abs",
                **abs_corr_values,
            }
        )

    correlation_df = pd.DataFrame(correlation_rows)
    return channel_df, correlation_df, baseline_eo


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[start] device={device}, target_attr={args.target_attr}, split={args.split}")
    context = build_celeba_context(
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        check_readable_images=args.check_readable_images,
    )
    attribute_index, attribute_name = resolve_target_attribute(context["attribute_names"], args.target_attr)

    activation_scores = load_activation_scores(args.activation_results)
    print(f"[load] activation scores loaded from {args.activation_results}")

    model = build_model(n_classes=len(context["attribute_names"]) * 2)
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    model.eval()
    print(f"[load] base checkpoint loaded from {args.checkpoint}")

    print("[experiment1] computing importance_score")
    importance_score, _ = compute_importance(
        model,
        None,
        context["frame"],
        new_img_dir=context["face_dir"],
        masked_grads=True,
        output_cols_each_task=context["output_cols_each_task"],
        col_names=context["col_used"],
        network="mobilenetv2",
        stop_batch=args.importance_stop_batch,
        sensitive_group="gender",
        sensitive_classes=2,
        imp_batch_size=args.importance_batch_size,
    )

    importance_channel_scores = aggregate_scores_by_layer(importance_score)
    print(f"[experiment1] aggregated {len(importance_channel_scores)} importance layers")

    run_output_dir = args.output_dir / attribute_name
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] saving results under {run_output_dir}")

    experiment1_per_layer_df, experiment1_global_df = compute_experiment1(
        activation_scores=activation_scores,
        importance_channel_scores=importance_channel_scores,
    )
    experiment1_per_layer_df.to_csv(run_output_dir / "experiment1_per_layer_correlations.csv", index=False)
    experiment1_global_df.to_csv(run_output_dir / "experiment1_global_correlations.csv", index=False)
    save_experiment1_plots(
        activation_scores=activation_scores,
        importance_channel_scores=importance_channel_scores,
        global_df=experiment1_global_df,
        plot_dir=run_output_dir / "plots_experiment1",
        max_plot_points=args.max_plot_points,
    )
    print("[experiment1] correlation tables saved")

    print(f"[experiment2] measuring |delta EO| on layer {args.target_layer}")
    experiment2_channels_df, experiment2_correlations_df, baseline_eo = compute_experiment2(
        model=model,
        eval_loader=context["eval_loader"],
        target_layer_name=args.target_layer,
        attribute_index=attribute_index,
        activation_scores=activation_scores,
        importance_channel_scores=importance_channel_scores,
        device=device,
        max_batches=args.max_eo_batches,
        max_zeroed_channels=args.max_zeroed_channels,
    )
    experiment2_channels_df.to_csv(run_output_dir / "experiment2_target_layer_channels.csv", index=False)
    experiment2_correlations_df.to_csv(run_output_dir / "experiment2_target_layer_correlations.csv", index=False)
    save_experiment2_plots(
        channel_df=experiment2_channels_df,
        correlation_df=experiment2_correlations_df,
        target_layer_name=args.target_layer,
        plot_dir=run_output_dir / "plots_experiment2",
        max_plot_points=args.max_plot_points,
    )

    metadata = {
        "checkpoint": str(args.checkpoint),
        "activation_results": str(args.activation_results),
        "split": args.split,
        "target_attr": attribute_name,
        "target_attr_index": attribute_index,
        "target_layer": args.target_layer,
        "device": str(device),
        "batch_size": args.batch_size,
        "importance_stop_batch": args.importance_stop_batch,
        "importance_batch_size": args.importance_batch_size,
        "max_eo_batches": args.max_eo_batches,
        "max_zeroed_channels": args.max_zeroed_channels,
        "max_plot_points": args.max_plot_points,
        "num_layers_activation": len(activation_scores),
        "num_layers_importance": len(importance_channel_scores),
        "experiment2_baseline_eo": baseline_eo,
        "experiment2_num_channels": int(experiment2_channels_df.shape[0]),
        "experiment2_delta_metric": "delta_eo_abs",
    }
    with open(run_output_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"[saved] {run_output_dir / 'experiment1_per_layer_correlations.csv'}")
    print(f"[saved] {run_output_dir / 'experiment1_global_correlations.csv'}")
    print(f"[saved] {run_output_dir / 'experiment2_target_layer_channels.csv'}")
    print(f"[saved] {run_output_dir / 'experiment2_target_layer_correlations.csv'}")
    print(f"[saved] {run_output_dir / 'run_metadata.json'}")


if __name__ == "__main__":
    main()