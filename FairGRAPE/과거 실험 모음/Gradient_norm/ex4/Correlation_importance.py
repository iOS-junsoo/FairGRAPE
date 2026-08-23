import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import make_frame, make_datasets
from prune import compute_importance
from util import download_dataset, make_model, safe_forward_with_cudnn_fallback


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data:
        model.load_state_dict(checkpoint_data["model_state"])
        print("Loaded checkpoint: dict['model_state']")
    else:
        model.load_state_dict(checkpoint_data)
        print("Loaded checkpoint: raw state_dict")
    return model


def build_celeba_val_loader(batch_size):
    csv = "csv/CelebA.csv"
    face_dir = "Images/CelebA/img_align_celeba"
    download_dataset("CelebA", face_dir)

    frames = make_frame(csv, face_dir, seven_races=False)

    output_cols_each_task = [(index * 2, index * 2 + 2) for index in range(39)]
    col_used_training = [frames["train"].columns[index] for index in range(2, 41)]
    col_used = col_used_training + ["gender"]

    _, val_loader = make_datasets(
        frames["train"],
        frames["val"],
        True,
        batch_size,
        col_used,
    )

    return frames, val_loader, face_dir, col_used_training, col_used, output_cols_each_task


def compute_mean_eo(model, dataloader, output_cols_each_task, num_tasks, device):
    model.eval()

    eqodds_stats = [
        {group: {"TP": 0, "P": 0, "FP": 0, "N": 0} for group in [0, 1]}
        for _ in range(num_tasks)
    ]

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx == 0:
                print("[EO] first validation batch loaded")

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)

            outputs = safe_forward_with_cudnn_fallback(model, images)
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(0)

            task_labels = labels[:, :num_tasks]
            gender_labels = labels[:, -1]

            for task_idx, (start_idx, end_idx) in enumerate(output_cols_each_task):
                task_out = outputs[:, start_idx:end_idx]
                task_truth = task_labels[:, task_idx]
                task_pred = torch.argmax(task_out, dim=1)

                for group in [0, 1]:
                    mask_pos = (gender_labels == group) & (task_truth == 1)
                    mask_neg = (gender_labels == group) & (task_truth == 0)

                    TP = ((task_pred == 1) & mask_pos).sum().item()
                    P = mask_pos.sum().item()
                    FP = ((task_pred == 1) & mask_neg).sum().item()
                    N = mask_neg.sum().item()

                    eqodds_stats[task_idx][group]["TP"] += TP
                    eqodds_stats[task_idx][group]["P"] += P
                    eqodds_stats[task_idx][group]["FP"] += FP
                    eqodds_stats[task_idx][group]["N"] += N

    eo_list = []
    for task_idx in range(num_tasks):
        TP_f = eqodds_stats[task_idx][1]["TP"]
        P_f = eqodds_stats[task_idx][1]["P"]
        FP_f = eqodds_stats[task_idx][1]["FP"]
        N_f = eqodds_stats[task_idx][1]["N"]

        TP_m = eqodds_stats[task_idx][0]["TP"]
        P_m = eqodds_stats[task_idx][0]["P"]
        FP_m = eqodds_stats[task_idx][0]["FP"]
        N_m = eqodds_stats[task_idx][0]["N"]

        if P_f == 0 or P_m == 0 or N_f == 0 or N_m == 0:
            continue

        tpr_f = TP_f / P_f
        tpr_m = TP_m / P_m
        fpr_f = FP_f / N_f
        fpr_m = FP_m / N_m
        eo = (abs(tpr_f - tpr_m) + abs(fpr_f - fpr_m)) / 2.0
        eo_list.append(eo)

    if len(eo_list) == 0:
        raise RuntimeError("EO를 계산할 수 있는 task가 없습니다.")

    return float(np.mean(eo_list))


def format_seconds(seconds):
    seconds = int(max(seconds, 0))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_target_weight_params(model, target_name="features.1.conv.0.0.weight"):
    return [(name, param) for name, param in model.named_parameters() if name == target_name]


def compute_delta_eo_per_weight(
    model,
    val_loader,
    output_cols_each_task,
    num_tasks,
    device,
    target_name="features.1.conv.0.0.weight",
):
    baseline_start = time.time()
    baseline_eo = compute_mean_eo(
        model=model,
        dataloader=val_loader,
        output_cols_each_task=output_cols_each_task,
        num_tasks=num_tasks,
        device=device,
    )
    baseline_elapsed = time.time() - baseline_start

    print(f"Baseline EO: {baseline_eo:.8f}")
    print(f"Baseline EO eval time: {baseline_elapsed:.2f} sec")

    target_params = get_target_weight_params(model, target_name=target_name)
    if len(target_params) == 0:
        raise RuntimeError(f"{target_name} 파라미터를 찾지 못했습니다.")

    delta_eo_all = []
    param_name_order = []

    total_weights = sum(param.numel() for _, param in target_params)
    estimated_total_sec = baseline_elapsed * total_weights

    print(f"Target parameter tensors: {len(target_params)}")
    print(f"Total weights to test: {total_weights}")
    print(f"Estimated total time: {format_seconds(estimated_total_sec)}")

    processed = 0
    overall_start = time.time()

    for name, param in target_params:
        print(f"Processing parameter: {name}, shape={tuple(param.shape)}, numel={param.numel()}")
        param_name_order.append(name)

        with torch.no_grad():
            flat = param.view(-1)

            for idx in range(flat.numel()):
                weight_start = time.time()

                original_value = flat[idx].item()
                flat[idx] = 0.0

                masked_eo = compute_mean_eo(
                    model=model,
                    dataloader=val_loader,
                    output_cols_each_task=output_cols_each_task,
                    num_tasks=num_tasks,
                    device=device,
                )
                delta_eo = abs(masked_eo - baseline_eo)
                delta_eo_all.append(delta_eo)

                flat[idx] = original_value

                processed += 1
                elapsed = time.time() - overall_start
                avg_sec_per_weight = elapsed / processed
                eta_sec = avg_sec_per_weight * (total_weights - processed)
                weight_elapsed = time.time() - weight_start

                if processed == 1 or processed % 10 == 0 or processed == total_weights:
                    print(
                        f"[{processed}/{total_weights}] "
                        f"last={weight_elapsed:.2f}s, "
                        f"avg={avg_sec_per_weight:.2f}s/weight, "
                        f"elapsed={format_seconds(elapsed)}, "
                        f"ETA={format_seconds(eta_sec)}"
                    )

    return baseline_eo, target_params, param_name_order, np.asarray(delta_eo_all, dtype=np.float64)


def load_importance_vector(model, frame, face_dir, output_cols_each_task, col_used, stop_batch, target_name):
    importance_score, _ = compute_importance(
        model,
        None,
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

    target_layer_name = target_name[:-7] if target_name.endswith(".weight") else target_name
    if target_layer_name not in importance_score:
        raise KeyError(f"importance_score에 {target_layer_name} 레이어가 없습니다.")

    return importance_score[target_layer_name].reshape(-1).cpu().numpy().astype(np.float64)


def save_summary(summary_path, baseline_eo, num_weights, rho, p_value, target_name):
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Performance importance vs |delta EO| Spearman correlation\n")
        handle.write(f"target_name: {target_name}\n")
        handle.write(f"baseline_eo: {baseline_eo:.8f}\n")
        handle.write(f"num_weights: {num_weights}\n")
        handle.write(f"spearman_rho: {rho:.8f}\n")
        handle.write(f"p_value: {p_value:.8e}\n")


def plot_scatter(importance_vector, delta_eo_vector, rho, p_value, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(importance_vector, delta_eo_vector, s=10, alpha=0.5)
    plt.xlabel("(weight x gradient)^2")
    plt.ylabel("|ΔEO|")
    plt.title(f"Spearman rho={rho:.6f}, p={p_value:.6e}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    checkpoint_path = "trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_0.pt"
    scatter_save_path = "Gradient_norm/ex4/features1_importance_spearman_scatter.png"
    summary_save_path = "Gradient_norm/ex4/features1_importance_spearman.txt"
    batch_size = 128
    stop_batch = 10000
    target_name = "features.1.conv.0.0.weight"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Target weight tensor:", target_name)

    frames, val_loader, face_dir, col_used_training, col_used, output_cols_each_task = build_celeba_val_loader(
        batch_size=batch_size
    )

    model = make_model(network="mobilenetv2", n_classes=78).to(device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    baseline_eo, target_params, param_name_order, delta_eo_vector = compute_delta_eo_per_weight(
        model=model,
        val_loader=val_loader,
        output_cols_each_task=output_cols_each_task,
        num_tasks=len(col_used_training),
        device=device,
        target_name=target_name,
    )

    importance_vector = load_importance_vector(
        model=model,
        frame=frames["val"],
        face_dir=face_dir,
        output_cols_each_task=output_cols_each_task,
        col_used=col_used,
        stop_batch=stop_batch,
        target_name=target_name,
    )

    if len(importance_vector) != len(delta_eo_vector):
        raise ValueError(
            f"길이 불일치: importance={len(importance_vector)}, ΔEO={len(delta_eo_vector)}"
        )

    rho, p_value = spearmanr(importance_vector, delta_eo_vector)

    print("=" * 80)
    print(f"Baseline EO: {baseline_eo:.8f}")
    print(f"Number of weights compared: {len(delta_eo_vector)}")
    print(f"Spearman rho: {rho:.8f}")
    print(f"p-value: {p_value:.8e}")
    print("=" * 80)

    os.makedirs(os.path.dirname(scatter_save_path), exist_ok=True)
    plot_scatter(
        importance_vector=importance_vector,
        delta_eo_vector=delta_eo_vector,
        rho=rho,
        p_value=p_value,
        save_path=scatter_save_path,
    )
    save_summary(
        summary_path=summary_save_path,
        baseline_eo=baseline_eo,
        num_weights=len(delta_eo_vector),
        rho=rho,
        p_value=p_value,
        target_name=target_name,
    )

    print(f"Scatter saved to: {scatter_save_path}")
    print(f"Summary saved to: {summary_save_path}")


if __name__ == "__main__":
    main()