import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from dataset import make_frame, make_datasets
from util import make_model, download_dataset, safe_forward_with_cudnn_fallback


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

    output_cols_each_task = [(i * 2, i * 2 + 2) for i in range(39)]
    col_used_training = [frames["train"].columns[i] for i in range(2, 41)]
    col_used = col_used_training + ["gender"]

    _, val_loader = make_datasets(
        frames["train"],
        frames["val"],
        True,
        batch_size,
        col_used
    )

    return frames, val_loader, face_dir, col_used_training, col_used, output_cols_each_task


def compute_mean_eo(model, dataloader, output_cols_each_task, num_tasks, device):
    model.eval()

    eqodds_stats = [
        {g: {"TP": 0, "P": 0, "FP": 0, "N": 0} for g in [0, 1]}
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

            for task_idx, (st, ed) in enumerate(output_cols_each_task):
                task_out = outputs[:, st:ed]
                task_truth = task_labels[:, task_idx]
                task_pred = torch.argmax(task_out, dim=1)

                for g in [0, 1]:
                    mask_pos = (gender_labels == g) & (task_truth == 1)
                    mask_neg = (gender_labels == g) & (task_truth == 0)

                    TP = ((task_pred == 1) & mask_pos).sum().item()
                    P = mask_pos.sum().item()
                    FP = ((task_pred == 1) & mask_neg).sum().item()
                    N = mask_neg.sum().item()

                    eqodds_stats[task_idx][g]["TP"] += TP
                    eqodds_stats[task_idx][g]["P"] += P
                    eqodds_stats[task_idx][g]["FP"] += FP
                    eqodds_stats[task_idx][g]["N"] += N

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
    target = []
    for name, param in model.named_parameters():
        if name == target_name:
            target.append((name, param))
    return target


def compute_delta_eo_per_weight(
    model,
    val_loader,
    output_cols_each_task,
    num_tasks,
    device,
    target_name="features.1.conv.0.0.weight"
):
    baseline_start = time.time()
    baseline_eo = compute_mean_eo(
        model=model,
        dataloader=val_loader,
        output_cols_each_task=output_cols_each_task,
        num_tasks=num_tasks,
        device=device
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
                    device=device
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


def load_abs_grad_vector(npz_path, param_name_order):
    npz_data = np.load(npz_path)

    abs_grad_list = []
    for param_name in param_name_order:
        key = f"abs_diff::{param_name}"
        if key not in npz_data:
            raise KeyError(f"npz에 {key} 키가 없습니다.")
        abs_grad_list.append(npz_data[key].reshape(-1))

    abs_grad_vector = np.concatenate(abs_grad_list, axis=0).astype(np.float64)
    return abs_grad_vector


def plot_scatter(abs_grad_vector, delta_eo_vector, rho, p_value, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(abs_grad_vector, delta_eo_vector, s=10, alpha=0.5)
    plt.xlabel("|Δgradient|")
    plt.ylabel("|ΔEO|")
    plt.title(f"Spearman rho={rho:.6f}, p={p_value:.6e}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    checkpoint_path = "trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_0.pt"
    npz_path = "Gradient_norm/ex1/celeba_gender_grad_diff.npz"
    scatter_save_path = "Gradient_norm/ex2/features1_spearman_scatter.png"
    batch_size = 128

    # 288개만 보려면 이 이름을 유지
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
        target_name=target_name
    )

    abs_grad_vector = load_abs_grad_vector(npz_path=npz_path, param_name_order=param_name_order)

    if len(abs_grad_vector) != len(delta_eo_vector):
        raise ValueError(
            f"길이 불일치: |Δgradient|={len(abs_grad_vector)}, ΔEO={len(delta_eo_vector)}"
        )

    rho, p_value = spearmanr(abs_grad_vector, delta_eo_vector)

    print("=" * 80)
    print(f"Baseline EO: {baseline_eo:.8f}")
    print(f"Number of weights compared: {len(delta_eo_vector)}")
    print(f"Spearman rho: {rho:.8f}")
    print(f"p-value: {p_value:.8e}")
    print("=" * 80)

    os.makedirs(os.path.dirname(scatter_save_path), exist_ok=True)
    plot_scatter(
        abs_grad_vector=abs_grad_vector,
        delta_eo_vector=delta_eo_vector,
        rho=rho,
        p_value=p_value,
        save_path=scatter_save_path
    )
    print(f"Scatter saved to: {scatter_save_path}")


if __name__ == "__main__":
    main()