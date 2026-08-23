import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import make_frame
from prune import compute_importance
from util import download_dataset, make_model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state"])
        load_mode = "dict['model_state']"
    else:
        model.load_state_dict(checkpoint_data)
        load_mode = "raw state_dict"
    print(f"Loaded checkpoint: {load_mode}")
    return model


def load_gender_model(checkpoint_path, device):
    gender_model = make_model(network="mobilenetv2", n_classes=2).to(device)
    checkpoint_data = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        state_dict = checkpoint_data["model_state_dict"]
        load_mode = "dict['model_state_dict']"
    elif isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data:
        state_dict = checkpoint_data["model_state"]
        load_mode = "dict['model_state']"
    else:
        state_dict = checkpoint_data
        load_mode = "raw state_dict"

    state_dict = {key: value for key, value in state_dict.items() if not key.endswith(".mask")}
    gender_model.load_state_dict(state_dict, strict=False)
    gender_model.eval()
    print(f"Loaded gender checkpoint: {load_mode}")
    return gender_model


def build_celeba_context(split):
    csv_path = "csv/CelebA.csv"
    face_dir = "Images/CelebA/img_align_celeba"
    download_dataset("CelebA", face_dir)

    frames = make_frame(csv_path, face_dir, seven_races=False)
    output_cols_each_task = [(index * 2, index * 2 + 2) for index in range(39)]
    col_used_training = [frames["train"].columns[index] for index in range(2, 41)]
    col_used = col_used_training + ["gender"]

    if split not in frames:
        raise ValueError(f"Unknown split: {split}")

    return {
        "frame": frames[split],
        "face_dir": face_dir,
        "output_cols_each_task": output_cols_each_task,
        "col_used": col_used,
    }


def compute_scores(model, gender_model, frame, face_dir, output_cols_each_task, col_used, stop_batch):
    importance_by_layer, gender_by_layer = compute_importance(
        model,
        gender_model,
        frame,
        new_img_dir=face_dir,
        masked_grads=True,
        output_cols_each_task=output_cols_each_task,
        col_names=col_used,
        network="mobilenetv2",
        stop_batch=stop_batch,
        sensitive_group="gender",
        sensitive_classes=2,
    )
    return importance_by_layer, gender_by_layer


def align_common_layers(importance_by_layer, gender_by_layer):
    aligned = []
    skipped = []

    common_layer_names = sorted(set(importance_by_layer.keys()) & set(gender_by_layer.keys()))
    for layer_name in common_layer_names:
        if layer_name.startswith("classifier."):
            continue

        importance = importance_by_layer[layer_name].detach().cpu().numpy().reshape(-1).astype(np.float64)
        gender = gender_by_layer[layer_name].detach().cpu().numpy().reshape(-1).astype(np.float64)

        if importance.shape != gender.shape:
            skipped.append((layer_name, importance.shape[0], gender.shape[0], "shape_mismatch"))
            continue

        aligned.append((layer_name, importance, gender))

    return aligned, skipped


def safe_spearman(x_values, y_values):
    if x_values.size == 0 or y_values.size == 0:
        return np.nan, np.nan
    if np.allclose(x_values, x_values[0]) or np.allclose(y_values, y_values[0]):
        return np.nan, np.nan
    rho, p_value = spearmanr(x_values, y_values)
    return float(rho), float(p_value)


def save_global_summary(output_dir, rho, p_value, total_weights, layer_count, skipped):
    output_path = output_dir / "spearman_global.txt"
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Global Spearman correlation between importance_score and gender_score\n")
        handle.write("score_definition: compute_importance(model, gender_model, ...) from prune.py impt_type 0 path\n")
        handle.write(f"layer_count: {layer_count}\n")
        handle.write(f"total_weights: {total_weights}\n")
        handle.write(f"spearman_rho: {rho:.10f}\n")
        handle.write(f"p_value: {p_value:.10e}\n")
        handle.write(f"skipped_layers: {len(skipped)}\n")
        for layer_name, importance_len, gender_len, reason in skipped:
            handle.write(
                f"skipped: layer={layer_name}, importance_len={importance_len}, gender_len={gender_len}, reason={reason}\n"
            )
    return output_path


def save_per_layer_csv(output_dir, aligned_layers):
    output_path = output_dir / "spearman_per_layer.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "layer_name",
            "num_weights",
            "spearman_rho",
            "p_value",
            "importance_mean",
            "importance_std",
            "gender_mean",
            "gender_std",
        ])

        for layer_name, importance, gender in aligned_layers:
            rho, p_value = safe_spearman(importance, gender)
            writer.writerow([
                layer_name,
                importance.shape[0],
                rho,
                p_value,
                float(np.mean(importance)),
                float(np.std(importance)),
                float(np.mean(gender)),
                float(np.std(gender)),
            ])
    return output_path


def sample_for_plot(x_values, y_values, max_points, seed):
    if x_values.shape[0] <= max_points:
        return x_values, y_values

    rng = np.random.default_rng(seed)
    selected = rng.choice(x_values.shape[0], size=max_points, replace=False)
    return x_values[selected], y_values[selected]


def save_global_scatter(output_dir, x_values, y_values, rho, p_value, max_points, seed):
    sampled_x, sampled_y = sample_for_plot(x_values, y_values, max_points=max_points, seed=seed)

    output_path = output_dir / "scatter_global.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(sampled_x, sampled_y, s=8, alpha=0.35)
    plt.xlabel("importance_score = (weight x gradient)^2")
    plt.ylabel("gender_score = (weight x gender_gradient)^2")
    plt.title(f"Global Spearman rho={rho:.6f}, p={p_value:.3e}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_top_layer_scatter_grid(output_dir, aligned_layers, max_layers, max_points_per_layer, seed):
    top_layers = sorted(aligned_layers, key=lambda item: item[1].shape[0], reverse=True)[:max_layers]
    if not top_layers:
        return None

    n_cols = 3
    n_rows = int(np.ceil(len(top_layers) / n_cols))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for axis, (layer_name, importance, gender) in zip(axes, top_layers):
        rho, p_value = safe_spearman(importance, gender)
        sampled_x, sampled_y = sample_for_plot(importance, gender, max_points=max_points_per_layer, seed=seed)
        axis.scatter(sampled_x, sampled_y, s=7, alpha=0.35)
        axis.set_title(f"{layer_name}\nrho={rho:.4f}, p={p_value:.2e}")
        axis.set_xlabel("importance_score")
        axis.set_ylabel("gender_score")

    for axis in axes[len(top_layers):]:
        axis.axis("off")

    output_path = output_dir / "scatter_per_layer_top.png"
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare importance_score and gender_score from prune.py impt_type 0 path with Spearman correlation."
    )
    parser.add_argument(
        "--checkpoint",
        default="trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_0.pt",
    )
    parser.add_argument(
        "--gender-checkpoint",
        default="gender_model/best_gender_model.pth",
    )
    parser.add_argument(
        "--output-dir",
        default="Gradient_norm/ex5",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument(
        "--stop-batch",
        type=int,
        default=10000,
        help="Maximum number of batches to use when computing gradients.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--top-layer-count",
        type=int,
        default=9,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Gender checkpoint: {args.gender_checkpoint}")

    context = build_celeba_context(args.split)
    frame = context["frame"]
    face_dir = context["face_dir"]
    output_cols_each_task = context["output_cols_each_task"]
    col_used = context["col_used"]

    model = make_model(network="mobilenetv2", n_classes=78).to(device)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    gender_model = load_gender_model(args.gender_checkpoint, device)

    importance_by_layer, gender_by_layer = compute_scores(
        model,
        gender_model,
        frame,
        face_dir,
        output_cols_each_task,
        col_used,
        stop_batch=args.stop_batch,
    )

    aligned_layers, skipped_layers = align_common_layers(importance_by_layer, gender_by_layer)
    if not aligned_layers:
        raise RuntimeError("No common non-classifier layers were aligned between importance_score and gender_score.")

    global_importance = np.concatenate([importance for _, importance, _ in aligned_layers], axis=0)
    global_gender = np.concatenate([gender for _, _, gender in aligned_layers], axis=0)
    global_rho, global_p_value = safe_spearman(global_importance, global_gender)

    global_txt_path = save_global_summary(
        output_dir,
        global_rho,
        global_p_value,
        total_weights=global_importance.shape[0],
        layer_count=len(aligned_layers),
        skipped=skipped_layers,
    )
    per_layer_csv_path = save_per_layer_csv(output_dir, aligned_layers)
    global_scatter_path = save_global_scatter(
        output_dir,
        global_importance,
        global_gender,
        global_rho,
        global_p_value,
        max_points=args.max_plot_points,
        seed=args.seed,
    )
    per_layer_scatter_path = save_top_layer_scatter_grid(
        output_dir,
        aligned_layers,
        max_layers=args.top_layer_count,
        max_points_per_layer=min(args.max_plot_points, 10000),
        seed=args.seed,
    )

    print("Saved files:")
    print(f"  - {global_txt_path}")
    print(f"  - {per_layer_csv_path}")
    print(f"  - {global_scatter_path}")
    if per_layer_scatter_path is not None:
        print(f"  - {per_layer_scatter_path}")


if __name__ == "__main__":
    main()