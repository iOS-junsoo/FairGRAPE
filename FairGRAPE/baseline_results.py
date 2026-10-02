"""Held-out DogsCats metrics for the final FairGRAPE model.

Consumes the predictions already returned by util.save_output, so the summary
and its prediction CSV describe the same model and the same inference pass.
"""

import csv
import json
from pathlib import Path


def dogscats_metrics(fair_df):
    """ACC, group x class BA, and equalized-odds gap, all on a 0..1 scale."""
    required = ("species", "color", "species_preds_fair")
    for name in required:
        if list(fair_df.columns).count(name) != 1:
            raise ValueError(f"Expected exactly one {name!r} column")
        if not fair_df[name].isin([0, 1]).all():
            raise ValueError(f"DogsCats {name!r} must contain only binary 0/1 values")
    if len(fair_df) == 0:
        raise ValueError("Cannot evaluate empty DogsCats predictions")

    truth, pred = fair_df["species"], fair_df["species_preds_fair"]
    correct = truth == pred
    cells, groups = [], []
    for color in (0, 1):
        selected = fair_df["color"] == color
        for species in (0, 1):
            mask = selected & (truth == species)
            count = int(mask.sum())
            n_correct = int((correct & mask).sum())
            cells.append({"color": color, "species": species, "count": count,
                          "correct": n_correct,
                          "accuracy": n_correct / count if count else None})
        tp = int((selected & (truth == 1) & (pred == 1)).sum())
        fn = int((selected & (truth == 1) & (pred == 0)).sum())
        fp = int((selected & (truth == 0) & (pred == 1)).sum())
        tn = int((selected & (truth == 0) & (pred == 0)).sum())
        groups.append({"color": color, "count": tp + fn + fp + tn,
                       "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                       "TPR": tp / (tp + fn) if tp + fn else None,
                       "FPR": fp / (fp + tn) if fp + tn else None})

    valid_cells = [cell["accuracy"] for cell in cells if cell["count"]]
    eo = None
    if all(group[rate] is not None for group in groups for rate in ("TPR", "FPR")):
        eo = (abs(groups[0]["TPR"] - groups[1]["TPR"])
              + abs(groups[0]["FPR"] - groups[1]["FPR"])) / 2
    return {"n_samples": len(fair_df), "ACC": int(correct.sum()) / len(fair_df),
            "EO": eo, "BA": sum(valid_cells) / len(valid_cells),
            "ba_valid_cells": len(valid_cells), "cells": cells, "groups": groups}


def _mask_sparsity(model, scope, target_layer_names):
    """Count mask zeros, excluding mask parameters from the model size.

    The repository's supported Conv2d/Linear weights have rank 4/2. Tensor
    operations are used through their ordinary interface; importing torch here
    is unnecessary. Naturally zero weights are not counted as removed weights.
    """
    base = getattr(model, "base_model", model)
    total_params = sum(p.numel() for name, p in base.named_parameters()
                       if name.rsplit(".", 1)[-1] != "mask")
    weights = {}
    for name, layer in base.named_modules():
        weight = getattr(layer, "weight", None)
        if weight is None or len(weight.shape) not in (2, 4):
            continue
        mask = getattr(layer, "mask", None)
        if mask is not None and tuple(mask.shape) != tuple(weight.shape):
            raise ValueError(f"Mask shape differs from weight shape for {name}")
        zeros = int((mask == 0).sum().item()) if mask is not None else 0
        weights[name] = {"total": weight.numel(), "pruned": zeros,
                         "masked": mask is not None, "rank": len(weight.shape)}

    if target_layer_names is not None:
        targets = set(target_layer_names)
        unknown = targets.difference(weights)
        if unknown:
            raise ValueError(f"Unknown pruning target layers: {sorted(unknown)}")
    elif scope == "all":
        targets = set(weights)
    elif scope == "blocks":
        targets = {name for name, info in weights.items()
                   if info["rank"] == 4 and len(name.split(".")) > 2
                   and name.split(".")[0] == "features"
                   and name.split(".")[1] in {str(i) for i in range(1, 18)}}
    else:
        raise ValueError(f"Unknown FairGRAPE scope: {scope!r}")

    pruned = sum(info["pruned"] for info in weights.values())
    weight_total = sum(info["total"] for info in weights.values())
    masked_total = sum(info["total"] for info in weights.values() if info["masked"])
    target_total = sum(weights[name]["total"] for name in targets)
    target_pruned = sum(weights[name]["pruned"] for name in targets)
    return {"total_parameters_excluding_masks": total_params,
            "pruned_parameters": pruned, "remaining_parameters": total_params - pruned,
            "model_sparsity": pruned / total_params if total_params else None,
            "conv_linear_weights": weight_total,
            "conv_linear_weight_sparsity": pruned / weight_total if weight_total else None,
            "masked_weights": masked_total,
            "masked_weight_sparsity": pruned / masked_total if masked_total else None,
            "target_weights": target_total, "pruned_target_weights": target_pruned,
            "target_sparsity": target_pruned / target_total if target_total else None,
            "target_layer_names": sorted(targets)}


def save_dogscats_test_results(fair_df, model, args, output_dir, *,
                             mode=None, scope=None, target_layer_names=None):
    """Write final Test predictions and ACC/EO/BA JSON + CSV; return the summary.

    ``args`` is an argparse Namespace or a dict. ``output_dir`` should be the
    current run's results directory. Optional target names override scope-based
    sparsity counting, without changing any masks or predictions.
    """
    arguments = dict(args) if isinstance(args, dict) else vars(args).copy()
    mode = mode if mode is not None else arguments.get("fg_mode")
    scope = scope if scope is not None else arguments.get("fg_scope", "blocks")
    metrics = dogscats_metrics(fair_df)
    sparsity = _mask_sparsity(model, scope, target_layer_names)
    output_dir = Path(output_dir)
    paths = {"predictions": str(output_dir / "final_test_predictions.csv"),
             "metrics_json": str(output_dir / "final_test_metrics.json"),
             "metrics_csv": str(output_dir / "final_test_metrics.csv")}
    summary = {"dataset": "DogsCats", "split": "test", "task": "species",
               "sensitive_group": "color", "method": arguments.get("prune_type", "FairGRAPE"),
               "fg_mode": mode, "fg_scope": scope, "seed": arguments.get("seed"),
               "skew_beta": arguments.get("skew_beta"),
               "requested_prune_rate": arguments.get("prune_rate"), **metrics,
               "metric_definitions": {
                   "ACC": "Fraction of all samples predicted correctly.",
                   "BA": "Mean accuracy over nonempty color x species cells.",
                   "EO": "(abs(TPR_color0-TPR_color1) + abs(FPR_color0-FPR_color1))/2; null if any denominator is zero.",
                   "scale": "0..1; higher ACC/BA and lower EO are better."},
               "sparsity": sparsity, "arguments": arguments, "files": paths}
    output_dir.mkdir(parents=True, exist_ok=True)
    fair_df.to_csv(paths["predictions"], index=False)
    with open(paths["metrics_json"], "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        stream.write("\n")
    row = {key: value for key, value in summary.items()
           if key not in ("cells", "groups", "metric_definitions", "sparsity", "arguments", "files")}
    row.update({key: value for key, value in sparsity.items() if key != "target_layer_names"})
    with open(paths["metrics_csv"], "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    eo_text = f"{metrics['EO']:.6f}" if metrics["EO"] is not None else "N/A"
    print(f"[Final Test] ACC={metrics['ACC']:.6f} EO={eo_text} BA={metrics['BA']:.6f}")
    print(f"[Final Test] metrics: {paths['metrics_json']}")
    return summary
