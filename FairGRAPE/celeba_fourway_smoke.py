"""Three-round CelebA / MobileNetV2 channel-pruning comparison.

Uses the repository's impt_type=2 weight importance and channel masks. The
two-stage selector and explicit fairness evaluator are isolated from legacy runs.
"""
import argparse
import contextlib
import copy
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shlex
import sys
import time
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset

import dataset
import prune
from train_and_val import loss_multi_tasks

ROOT = Path(__file__).resolve().parent
METHODS = ("A_activation", "B_activation", "A_contribution", "B_contribution")
SLICES = [(2 * t, 2 * t + 2) for t in range(39)]
DEFINITIONS = """TASK / 과제
39 attributes in CSV columns [2:41], each represented by two logits (78 total).
Male is CSV gender, verified against the original annotations for every row.
Male=0 means original Male=-1; Male=1 means original Male=+1.
Loss = arithmetic mean of 39 binary-task CrossEntropyLoss values; no Male loss,
no adversarial/GRL loss. Prediction = argmax of each pair (tie -> class 0).

PERFORMANCE CANDIDATES / 성능 후보
Call unchanged prune.compute_importance with a fixed cached selection loader.
Per weight: batch mean of (weight * gradient of the 39-task mean CE)^2.
The square is applied BEFORE averaging batches, exactly as in the repository.
Original masks are applied to weight gradients. For expansion and depthwise
convolutions average over all axes except output channel; for projection average
over all axes except input channel; ADD these vectors, as impt_type=2 does.
Use raw importance ascending, excluding already removed and protected channels;
retain the existing per-layer floor and performance protection ratios. The
legacy alpha mixture is intentionally replaced by the requested two-stage rule.
Smoke selection inputs use the repository evaluation transform, fixed across
methods and candidates (legacy importance uses random training augmentation).

SIGNALS AND GAPS / 신호 및 격차
Analysis layers: impt_type=2 depthwise Conv outputs in features.1..17, BEFORE BN
and ReLU. features.18 and classifier are not pruned or included in the gap.
a[i,l,c] = mean over H,W of signed Conv activation A[i,l,c,h,w].
p[i,t] = softmax(logits[i,2*t:2*t+2])[1], positive-class probability.
q[i,t,l,c] = mean over H,W of A * d p[i,t] / d A, retaining the sign.
Group means = sum of these per-image values / number of images in that group.
Accumulate sums across the entire dataset BEFORE subtracting group means.
D(signal,condition) = mean over the 17 layers of:
  mean over REMAINING channels of abs(mean_group0 - mean_group1).
Absolute value is applied AFTER the group means; never abs(A) or abs(q).
Removed channels are excluded from the denominator. Layers have equal weight.
A_activation = D(a,all), ONE global score; never a sum/mean of 39 copies.
B_activation = mean_t max(D(a,y_t=1), D(a,y_t=0)), all 39 attributes.
A_contribution = mean_t D(q_t,all), all 39 attributes.
B_contribution = mean_t max(D(q_t,y_t=1), D(q_t,y_t=0)), all 39 attributes.
B scores require all 4 cells for EVERY selection attribute; otherwise fail.
Neither |D_positive-D_negative| nor activation mass nor dense ratios is used.
Candidate masks are actually applied and signals/gradients recomputed on the
remaining model. Greedy removal maximizes before-minus-after score. If every
candidate worsens the score, take the least worsening and report that fact.
Tie key: after-score rounded to 12 decimal places, lower raw performance
importance, numeric block index, original channel index. No test-set selection.

PRUNING AND RETRAINING / 제거 및 재학습
Repository cumulative expansion-output / depthwise-output / projection-input
weight masks; fixed-size tensors (logical channel removal, not tensor compaction).
Gradient masks and post-optimizer projection prevent regrowth. Projection input
masks also block residual BN offsets of removed internal channels.
Every round recomputes performance and selection signals after previous training.
Each method starts from a fresh strict load of the SAME checkpoint.
Retraining: repository ImgAugTransform, shared cached batches per round, Adam
with repository default betas/eps/weight_decay and CLI learning rate, fresh
optimizer per round, no test-based early stopping. RNG reset equally per round.

EVALUATION / 평가
ACC=(TP+TN)/N. BA=(global TPR+global TNR)/2; N/A if either true class absent.
For each Male group: ACC, TPR=TP/P, FPR=FP/N_negative, PPR=(TP+FP)/N_group.
EqualOpportunity=abs(TPR0-TPR1).
EqualizedOdds=max(abs(TPR0-TPR1),abs(FPR0-FPR1)).
DemographicParity=abs(PPR0-PPR1).
Legacy EO in util.py/train_and_val.py is the MEAN of TPR and FPR gaps; also
reported as legacy_EO_mean_gap, not substituted for requested EqualizedOdds.
Legacy training BA averages nonempty (label,group) cell accuracies; requested BA
here instead balances the two true classes across the complete evaluation set.
Activation unconditional D is GLOBAL and listed once outside macro aggregation;
per-attribute records reference that value. Conditional activation and all three
contribution gaps are per attribute. All other macros are arithmetic means over
the SAME defined attributes across dense/all methods/all stages, with n/39 and
attribute names recorded separately for EACH metric. Missing denominators ->
N/A, never zero. Sample variability and rare cells limit this smoke experiment.
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset.ia.seed(seed)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def reference_layers(model):
    return {n: m for n, m in prune._get_impt_type2_target_layers(model).items()
            if n != "features.18.0"}


def block_number(name):
    return int(name.split(".")[1])


def weight_layers(model):
    return {n: m for n, m in model.named_modules()
            if type(m).__name__ in prune.supported_layers and hasattr(m, "weight")}


def install_masks(model):
    """Dynamic gradient mask: one hook per parameter, not one hook per round."""
    for layer in weight_layers(model).values():
        layer.register_buffer("mask", torch.ones_like(layer.weight))
        layer.weight.register_hook(lambda grad, layer=layer: grad * layer.mask)


@torch.no_grad()
def enforce_masks(model):
    for layer in weight_layers(model).values():
        layer.weight.mul_(layer.mask)


def apply_channels(model, channels):
    with contextlib.redirect_stdout(io.StringIO()):
        masks = prune._build_channel_mask_list(model, channels,
                                               next(model.parameters()).device)
    layers = weight_layers(model)
    assert len(layers) == len(masks)
    for layer, mask in zip(layers.values(), masks):
        layer.mask.copy_(mask)
    enforce_masks(model)


def assert_masks(model):
    for name, layer in weight_layers(model).items():
        assert torch.count_nonzero(layer.weight.detach()[layer.mask == 0]) == 0, name


@contextlib.contextmanager
def temporary_channels(model, channels):
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    try:
        apply_channels(model, channels)
        yield
    finally:
        model.load_state_dict(snapshot, strict=True)


def active_masks(model):
    return {name: layer.mask.flatten(1).any(1)
            for name, layer in reference_layers(model).items()}


def aggregate_importance(scores):
    """Same aggregation and float32 conversions as prune.py impt_type == 2."""
    result = {}
    for b in range(1, 18):
        c0, c1, c2 = prune._get_impt_type2_block_layer_names(f"features.{b}")
        combined = prune._aggregate_channel_importance(scores[c0]).float()
        if c1 is not None:
            combined = combined + prune._aggregate_channel_importance(scores[c1]).float()
        combined = combined + scores[c2].float().mean(dim=(0, 2, 3))
        result[c1 if c1 else c0] = combined
    return result


def performance_importance(model, loader, frame, attrs, log_path):
    # Only replace data construction. Execute the existing gradient/weight code
    # unchanged, on the exact cached images used for candidate comparisons.
    with patch.object(prune, "make_datasets", return_value=(loader, loader)):
        with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            scores, _ = prune.compute_importance(
                model, None, frame, new_img_dir=None, masked_grads=True,
                output_cols_each_task=SLICES, col_names=attrs + ["gender"],
                stop_batch=len(loader), network="mobilenetv2", sensitive_group="gender",
                sensitive_classes=78, imp_batch_size=loader.batch_size)
    return aggregate_importance(scores)


def candidates(model, perf, count):
    protected, _, _ = prune._compute_protected_channels(
        model, perf, prune.IMPT2_PROTECTION_RATIO)
    rows = []
    for name, active in active_masks(model).items():
        block = f"features.{block_number(name)}"
        floor = max(1, math.ceil(len(active) * prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER))
        if int(active.sum()) <= floor:
            continue
        for channel in active.nonzero().flatten().tolist():
            if (block, channel) in protected:
                continue
            score = float(perf[name][channel])
            if not math.isfinite(score):
                raise ValueError(f"Nonfinite performance importance: {name}:{channel}")
            rows.append(dict(block=block, layer=name, channel=channel, performance=score))
    rows.sort(key=lambda r: (r["performance"], block_number(r["block"]), r["channel"]))
    if len(rows) < count:
        raise ValueError("Not enough eligible channels for the requested candidate count")
    return rows[:count]


def membership(labels):
    """Rows: group 0/1, then task x truth 0/1 x group 0/1."""
    y, g = labels[:, :-1], labels[:, -1]
    rows = [(g == a) for a in (0, 1)]
    rows.extend((y[:, t] == v) & (g == a)
                for t in range(y.shape[1]) for v in (0, 1) for a in (0, 1))
    return torch.stack(rows).double()


def gap_from_sums(sums, counts, active):
    """Leading axes arbitrary; penultimate axis is the two groups."""
    means = sums / counts[..., None].clamp_min(1)
    gap = (means[..., 0, :] - means[..., 1, :]).abs()[..., active].mean(-1)
    valid = (counts > 0).all(-1)
    return torch.where(valid, gap, torch.full_like(gap, float("nan")))


def collect_signals(model, loader, contribution=True):
    """Full-dataset group sums, with per-task gradients of positive probability."""
    model.eval()
    layers = reference_layers(model)
    active = active_masks(model)
    acts, handles, a_sums, q_sums = {}, [], {}, {}
    device = next(model.parameters()).device
    counts = torch.zeros(158, dtype=torch.float64, device=device)
    predictions, truths, groups = [], [], []
    for name, layer in layers.items():
        handles.append(layer.register_forward_hook(
            lambda _m, _i, output, name=name: acts.__setitem__(name, output)))
        a_sums[name] = torch.zeros(158, layer.out_channels, dtype=torch.float64, device=device)
        if contribution:
            q_sums[name] = torch.zeros(39, 6, layer.out_channels, dtype=torch.float64, device=device)
    try:
        with torch.set_grad_enabled(contribution):
            for x, labels in loader:
                x, labels = x.to(device), labels.to(device)
                logits = model(x).reshape(-1, 39, 2)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite logits")
                predictions.append(logits.detach().argmax(2).cpu().numpy())
                truths.append(labels[:, :39].cpu().numpy())
                groups.append(labels[:, 39].cpu().numpy())
                member = membership(labels)
                counts += member.sum(1)
                values = [acts[name] for name in layers]
                for name, activation in zip(layers, values):
                    pooled = activation.detach().mean(dim=(2, 3)).double()
                    a_sums[name] += member @ pooled
                if contribution:
                    positive = logits.softmax(2)[:, :, 1]
                    for t in range(39):
                        gradients = torch.autograd.grad(positive[:, t].sum(), values,
                                                        retain_graph=t < 38)
                        selected = torch.cat((member[:2], member[2+4*t:6+4*t]))
                        for name, activation, gradient in zip(layers, values, gradients):
                            pooled = (activation.detach() * gradient.detach()).mean((2, 3)).double()
                            q_sums[name][t] += selected @ pooled
                acts.clear()
    finally:
        for handle in handles:
            handle.remove()
    act_all, act_cond, con_all, con_cond = [], [], [], []
    for name in layers:
        act_all.append(gap_from_sums(a_sums[name][:2], counts[:2], active[name]))
        act_cond.append(gap_from_sums(a_sums[name][2:].reshape(39, 2, 2, -1),
                                     counts[2:].reshape(39, 2, 2), active[name]))
        if contribution:
            con_all.append(gap_from_sums(q_sums[name][:, :2], counts[:2].expand(39, 2), active[name]))
            con_cond.append(gap_from_sums(q_sums[name][:, 2:].reshape(39, 2, 2, -1),
                                         counts[2:].reshape(39, 2, 2), active[name]))
    result = dict(activation_all=float(torch.stack(act_all).mean()),
                  activation_cond=torch.stack(act_cond).mean(0).cpu().numpy(),
                  predictions=np.concatenate(predictions), truth=np.concatenate(truths),
                  groups=np.concatenate(groups), cell_counts=counts[2:].reshape(39, 2, 2).cpu().numpy())
    if contribution:
        result.update(contribution_all=torch.stack(con_all).mean(0).cpu().numpy(),
                      contribution_cond=torch.stack(con_cond).mean(0).cpu().numpy())
    return result


def selection_score(signals, method):
    if method == "A_activation":
        value = signals["activation_all"]
    elif method == "B_activation":
        value = signals["activation_cond"].max(axis=1).mean()
    elif method == "A_contribution":
        value = signals["contribution_all"].mean()
    elif method == "B_contribution":
        value = signals["contribution_cond"].max(axis=1).mean()
    else:
        raise ValueError(method)
    if not math.isfinite(value):
        raise ValueError(f"{method}: all 39 selection attributes must be defined")
    return float(value)


def rates(y, p):
    tp, fn = int(((y == 1) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    fp, tn = int(((y == 0) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
    div = lambda n, d: n / d if d else float("nan")
    return dict(ACC=div(tp + tn, len(y)), TPR=div(tp, tp + fn), FPR=div(fp, fp + tn),
                PPR=div(tp + fp, len(y)), TP=tp, FN=fn, FP=fp, TN=tn,
                BA=(div(tp, tp + fn) + div(tn, tn + fp)) / 2)


def classification_metrics(truth, prediction, group, attrs):
    rows = []
    for t, attr in enumerate(attrs):
        y, p = truth[:, t], prediction[:, t]
        global_rates = rates(y, p)
        row = dict(attribute=attr, ACC=global_rates["ACC"], BA=global_rates["BA"])
        r = [rates(y[group == a], p[group == a]) for a in (0, 1)]
        for a in (0, 1):
            row.update({f"Male{a}_{k}": r[a][k] for k in ("ACC", "TPR", "FPR", "PPR", "TP", "FN", "FP", "TN")})
        dt = abs(r[0]["TPR"] - r[1]["TPR"])
        df = abs(r[0]["FPR"] - r[1]["FPR"])
        row.update(EqualOpportunity=dt, EqualizedOdds=float(np.maximum(dt, df)),
                   DemographicParity=abs(r[0]["PPR"] - r[1]["PPR"]), legacy_EO_mean_gap=(dt + df) / 2)
        rows.append(row)
    return rows


def evaluate(model, loader, attrs, method, round_index, stage):
    started = time.monotonic()
    signals = collect_signals(model, loader, contribution=True)
    rows = classification_metrics(signals["truth"], signals["predictions"], signals["groups"], attrs)
    for t, row in enumerate(rows):
        row.update(activation_all_global_ref=signals["activation_all"],
                   activation_positive=signals["activation_cond"][t, 1],
                   activation_negative=signals["activation_cond"][t, 0],
                   contribution_all=signals["contribution_all"][t],
                   contribution_positive=signals["contribution_cond"][t, 1],
                   contribution_negative=signals["contribution_cond"][t, 0])
    sync()
    return dict(method=method, round=round_index, stage=stage, attributes=rows,
                global_activation_gap=signals["activation_all"],
                seconds=time.monotonic()-started)


def macro_metrics(records):
    """One common attribute mask per metric over every compared model state."""
    keys = [k for k in records[0]["attributes"][0] if k != "attribute"
            and k != "activation_all_global_ref" and not k.endswith(("_TP", "_TN", "_FP", "_FN"))]
    masks = {}
    for key in keys:
        masks[key] = [i for i in range(39) if all(
            r["attributes"][i][key] is not None and math.isfinite(r["attributes"][i][key])
            for r in records)]
    for record in records:
        record["macro"] = {
            key: dict(value=float(np.mean([record["attributes"][i][key] for i in ids])) if ids else None,
                      n_attributes=len(ids),
                      attributes=[record["attributes"][i]["attribute"] for i in ids])
            for key, ids in masks.items()}
    return masks


def validate_annotations(csv_path, raw_path):
    frame = pd.read_csv(csv_path)
    attrs = frame.columns[2:41].tolist()
    assert len(attrs) == 39 and "Male" not in attrs and "gender" not in attrs
    assert list(frame.columns[:2]) == ["image_name", "face_name_align"]
    assert frame.columns[-1] == "gender" and frame.image_name.is_unique
    raw = pd.read_csv(raw_path, sep=r"\s+", skiprows=1).loc[frame.image_name]
    raw_attrs = [a.replace("X5_o_Clock_Shadow", "5_o_Clock_Shadow") for a in attrs]
    assert set(raw_attrs) == set(raw.columns) - {"Male"}
    assert np.array_equal(frame.gender.to_numpy(), (raw.Male.to_numpy()+1)//2)
    assert np.array_equal(frame[attrs].to_numpy(), (raw[raw_attrs].to_numpy()+1)//2)
    return attrs, dict(rows_verified=len(frame), csv_sha256=sha(csv_path),
                       original_annotations=str(raw_path), original_sha256=sha(raw_path),
                       gender_mapping="CSV gender = (original Male + 1) / 2; zero mismatches",
                       attributes_in_head_order=attrs)


def subset_frames(frames, attrs, args):
    rng = np.random.RandomState(args.seed)
    train = frames["train"]
    order = rng.permutation(len(train)).tolist()
    chosen = order[:args.selection_samples]
    chosen_set = set(chosen)
    # Cover rare cells using training labels ONLY, not held-out test labels.
    for attr in attrs:
        for y in (0, 1):
            for g in (0, 1):
                mask = (train[attr].to_numpy() == y) & (train.gender.to_numpy() == g)
                if not mask[chosen].any():
                    extra = next((i for i in order if mask[i]), None)
                    if extra is None:
                        raise ValueError(f"Cannot define all-39 B selection: empty train cell {attr},y={y},Male={g}")
                    chosen.append(extra)
                    chosen_set.add(extra)
    remaining = [i for i in order if i not in chosen_set]
    if len(remaining) < args.train_samples:
        raise ValueError("Not enough disjoint retraining samples")
    test_order = np.random.RandomState(args.seed+1).permutation(len(frames["test"]))
    if args.eval_samples > len(test_order):
        raise ValueError("eval-samples exceeds the test split")
    subsets = dict(selection=train.iloc[chosen].reset_index(drop=True),
                   retrain=train.iloc[remaining[:args.train_samples]].reset_index(drop=True),
                   evaluation=frames["test"].iloc[test_order[:args.eval_samples]].reset_index(drop=True))
    names = {k: set(f.image_name) for k, f in subsets.items()}
    assert names["selection"].isdisjoint(names["retrain"])
    assert names["evaluation"].isdisjoint(set(frames["train"].image_name))
    assert names["evaluation"].isdisjoint(set(frames["val"].image_name))
    return subsets


def cached_loader(frame, attrs, batch_size):
    _, images = dataset.make_datasets(frame, frame, False, batch_size, attrs+["gender"])
    x, y = zip(*(images[i] for i in range(len(images))))
    return DataLoader(TensorDataset(torch.stack(x), torch.stack(y)), batch_size=batch_size,
                      shuffle=False, num_workers=0)


def training_batches(frame, attrs, args, round_index):
    seed_all(args.seed + round_index)
    images, _ = dataset.make_datasets(frame, frame, False, args.batch_size, attrs+["gender"])
    batches = []
    for step in range(args.retrain_steps):
        indices = [(step*args.batch_size+i) % len(images) for i in range(args.batch_size)]
        x, y = zip(*(images[i] for i in indices))
        batches.append((torch.stack(x), torch.stack(y)))
    return batches


def retrain(model, batches, args, round_index):
    started = time.monotonic()
    seed_all(args.seed + round_index)
    model.train()
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=args.retrain_lr)
    losses = []
    for x, y in batches:
        optimizer.zero_grad(set_to_none=True)
        output = model(x.to(device))
        loss = loss_multi_tasks(output, y.to(device), torch.nn.CrossEntropyLoss(), SLICES)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite task loss")
        loss.backward()
        optimizer.step()
        enforce_masks(model)
        losses.append(float(loss))
    assert_masks(model)
    model.eval()
    sync()
    return dict(losses=losses, steps=len(losses), seconds=time.monotonic()-started)


def format_value(value):
    return "N/A" if value is None or not math.isfinite(value) else f"{value:.6f}"


def comparison(records):
    final = [r for r in records if r["method"] == "dense" or (r["round"] == 3 and r["stage"] == "post")]
    keys = ["ACC", "BA", "EqualOpportunity", "EqualizedOdds", "DemographicParity",
            "activation_positive", "activation_negative", "contribution_all",
            "contribution_positive", "contribution_negative"]
    rows = []
    for record in final:
        row = dict(method=record["method"], activation_global=format_value(record["global_activation_gap"]))
        row.update({k: f"{format_value(record['macro'][k]['value'])} ({record['macro'][k]['n_attributes']}/39)"
                    for k in keys})
        rows.append(row)
    return pd.DataFrame(rows).to_string(index=False)


def write_report(out, result):
    records = result["evaluations"]
    if records:
        macro_metrics(records)
    dump(out / "results.json", result)
    with open(out / "results.txt", "w", encoding="utf-8") as f:
        f.write("CELEBA FOUR-WAY THREE-ROUND SMOKE / 39속성 채널 프루닝\n\n")
        f.write("STATUS: " + result["status"] + "\n\n")
        f.write(DEFINITIONS + "\n\n")
        f.write("CONFIGURATION / 실제 설정\n" + json.dumps(clean(result["metadata"]), ensure_ascii=False, indent=2) + "\n\n")
        if records:
            f.write("FINAL COMPARISON / 덴스 및 3회 재학습 후 비교\n" + comparison(records) + "\n\n")
            f.write("COMMON VALID ATTRIBUTE SETS / 지표별 동일 유효 속성\n")
            for key, item in records[0]["macro"].items():
                f.write(f"{key}: {item['n_attributes']}/39: {', '.join(item['attributes'])}\n")
        f.write("\nROUND SELECTIONS AND TIMINGS / 회차별 후보·제거 채널·시간\n")
        f.write(json.dumps(clean(result["rounds"]), ensure_ascii=False, indent=2) + "\n")
        for record in records:
            f.write(f"\n{'='*100}\n{record['method']} round={record['round']} {record['stage']} seconds={record['seconds']:.3f}\n")
            f.write(f"GLOBAL activation gap (one value): {record['global_activation_gap']:.9f}\n")
            f.write("MACRO MEANS / 속성 산술평균\n")
            for key, item in record["macro"].items():
                f.write(f"  {key:28s} {format_value(item['value'])}  n={item['n_attributes']}/39\n")
            table = pd.DataFrame(record["attributes"])
            groups = [list(table.columns[:7]), ["attribute"]+[k for k in table if k.startswith("Male")],
                      ["attribute", "EqualOpportunity", "EqualizedOdds", "DemographicParity", "legacy_EO_mean_gap"],
                      ["attribute"]+[k for k in table if k.startswith(("activation_", "contribution_"))]]
            for columns in groups:
                f.write("\n" + table[columns].to_string(index=False, na_rep="N/A", float_format=lambda x: f"{x:.7f}") + "\n")
        if records:
            f.write("\nREGRESSIONS VERSUS DENSE / 덴스 대비 악화 (모든 상태)\n")
            dense = records[0]
            for record in records[1:]:
                f.write(f"\n{record['method']} round={record['round']} {record['stage']}\n")
                da = record["global_activation_gap"] - dense["global_activation_gap"]
                f.write(f"  global activation gap delta={da:+.9f}\n")
                for key in ("ACC", "BA", "EqualOpportunity", "EqualizedOdds", "DemographicParity"):
                    v, base = record["macro"][key]["value"], dense["macro"][key]["value"]
                    if v is not None and base is not None:
                        f.write(f"  macro {key} delta={v-base:+.7f}\n")
                for row, base in zip(record["attributes"], dense["attributes"]):
                    changes = []
                    for key in ("ACC", "BA", "Male0_ACC", "Male1_ACC", "Male0_TPR", "Male1_TPR",
                                "Male0_FPR", "Male1_FPR", "EqualOpportunity", "EqualizedOdds", "DemographicParity"):
                        if not math.isfinite(row[key]) or not math.isfinite(base[key]):
                            continue
                        delta = row[key]-base[key]
                        worse = delta < -1e-12 if key in ("ACC", "BA", "Male0_ACC", "Male1_ACC", "Male0_TPR", "Male1_TPR") else delta > 1e-12
                        if worse:
                            changes.append(f"{key}:{base[key]:.5f}->{row[key]:.5f}")
                    if changes:
                        f.write(f"  {row['attribute']}: {'; '.join(changes)}\n")
        f.write(f"\nTOTAL ELAPSED SECONDS: {result.get('elapsed_seconds', 0):.3f}\n")
        if result.get("error"):
            f.write("ERROR: " + result["error"] + "\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT/"trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_0.pt")
    p.add_argument("--csv", type=Path, default=ROOT/"csv/CelebA.csv")
    p.add_argument("--image-dir", type=Path, default=ROOT/"Images/CelebA/img_align_celeba")
    p.add_argument("--original-attributes", type=Path, default=ROOT.parent/"Balancing_Act/data/celeba/list_attr_celeba.txt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, choices=[3], default=3)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--selection-samples", type=int, default=128)
    p.add_argument("--train-samples", type=int, default=256)
    p.add_argument("--eval-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--remove-per-round", type=int, default=1)
    p.add_argument("--retrain-steps", type=int, default=5)
    p.add_argument("--retrain-lr", type=float, default=1e-5)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--preflight-only", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ("selection_samples", "train_samples", "eval_samples", "batch_size", "candidates", "remove_per_round", "retrain_steps"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    assert args.candidates >= args.remove_per_round
    assert len(set(args.methods)) == len(args.methods)
    for key in ("checkpoint", "csv", "image_dir", "original_attributes"):
        if not getattr(args, key).exists():
            raise FileNotFoundError(f"Missing --{key.replace('_','-')}: {getattr(args,key)}")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(status="preparing", metadata=dict(arguments=vars(args),
        command=shlex.join([sys.executable, str(Path(__file__).resolve())]+sys.argv[1:]),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), evaluations=[], rounds=[])
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        seed_all(args.seed)
        attrs, annotation = validate_annotations(args.csv, args.original_attributes)
        print("Annotation verification:", annotation["gender_mapping"], flush=True)
        frames = dataset.make_frame(str(args.csv), str(args.image_dir), seven_races=False)
        assert frames["train"].columns[2:41].tolist() == attrs
        subsets = subset_frames(frames, attrs, args)
        for name, frame in subsets.items():
            frame.to_csv(out/f"{name}_samples.csv", index=False)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        state = saved
        checkpoint_format = "plain state_dict"
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in saved:
                state, checkpoint_format = saved[key], key
                break
        assert state["classifier.1.weight"].shape == (78, 1280)
        assert not any("mask" in k for k in state), "Initial checkpoint must be dense"

        def fresh_model():
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = torch.nn.Linear(1280, 78)
            model.load_state_dict(state, strict=True)
            model.to(args.device)
            install_masks(model)
            model.eval()
            return model

        model = fresh_model()
        assert len(reference_layers(model)) == 17
        result["metadata"].update(annotation=annotation, checkpoint_format=checkpoint_format,
            checkpoint_sha256=sha(args.checkpoint), checkpoint_head_shape=[78,1280],
            legacy_checkpoint_training_metadata="Not embedded; attribute order uses verified project CSV/head convention",
            environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                             torchvision=torchvision.__version__, numpy=np.__version__, pandas=pd.__version__,
                             cuda=torch.version.cuda, device=args.device,
                             gpu=torch.cuda.get_device_name() if args.device.startswith("cuda") else None),
            split_sizes={s:len(frames[s]) for s in ("train","val","test")}, split_seed=42,
            subset_sizes={k:len(f) for k,f in subsets.items()},
            subset_files={k:dict(path=str(out/f"{k}_samples.csv"),sha256=sha(out/f"{k}_samples.csv")) for k in subsets},
            analysis_layers={n:m.out_channels for n,m in reference_layers(model).items()},
            floor_ratio=prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER, protection_ratio=prune.IMPT2_PROTECTION_RATIO,
            layer_cap="disabled (channel count budget)", performance_normalization="none; raw importance",
            optimizer=dict(name="Adam",lr=args.retrain_lr,betas=[0.9,0.999],eps=1e-8,weight_decay=0),
            source_sha256={n:sha(ROOT/n) for n in ("celeba_fourway_smoke.py","prune.py","dataset.py","train_and_val.py","util.py")})
        selection = cached_loader(subsets["selection"], attrs, args.batch_size)
        evaluation = cached_loader(subsets["evaluation"], attrs, args.batch_size)
        counts = membership(selection.dataset.tensors[1]).sum(1)[2:].reshape(39,2,2)
        assert (counts > 0).all()
        result["metadata"]["selection_cell_counts_y_by_Male"] = counts.tolist()
        result["metadata"]["evaluation_cell_counts_y_by_Male"] = membership(evaluation.dataset.tensors[1]).sum(1)[2:].reshape(39,2,2).tolist()
        for loader in (selection,evaluation):
            assert torch.isfinite(loader.dataset.tensors[0]).all()
        print("Subsets:", result["metadata"]["subset_sizes"], flush=True)
        if args.preflight_only:
            result.update(status="preflight_complete",elapsed_seconds=time.monotonic()-started)
            write_report(out,result)
            print("Preflight complete:",out/"results.txt",flush=True)
            return
        # Cache the exact augmented minibatches once for all four methods.
        batches = {r:training_batches(subsets["retrain"],attrs,args,r) for r in range(1,4)}
        used_indices = [(s*args.batch_size+i) % len(subsets["retrain"])
                        for s in range(args.retrain_steps) for i in range(args.batch_size)]
        result["metadata"]["retraining_samples_consumed_per_round"] = len(used_indices)
        result["metadata"]["retraining_unique_images_per_round"] = len(set(used_indices))
        result["metadata"]["retraining_image_sequence_each_round"] = subsets["retrain"].iloc[used_indices].image_name.tolist()
        result["metadata"]["retraining_batch_sha256"] = {
            r:hashlib.sha256(b"".join(x.numpy().tobytes()+y.numpy().tobytes() for x,y in b)).hexdigest()
            for r,b in batches.items()}
        result["status"] = "running"
        print("Evaluating dense model...",flush=True)
        result["evaluations"].append(evaluate(model,evaluation,attrs,"dense",0,"dense"))
        write_report(out,result)
        del model
        first_perf_hash = None
        for method in args.methods:
            model = fresh_model()
            removed = []
            for r in range(1,4):
                round_started=time.monotonic()
                print(f"{method} round {r}/3: recomputing performance importance",flush=True)
                seed_all(args.seed+r)
                t=time.monotonic()
                perf=performance_importance(model,selection,subsets["selection"],attrs,out/f"{method}_round{r}_importance.txt")
                perf_hash=hashlib.sha256(b"".join(v.detach().cpu().numpy().tobytes() for v in perf.values())).hexdigest()
                if r == 1:
                    if first_perf_hash is None: first_perf_hash=perf_hash
                    assert perf_hash == first_perf_hash, "First-round candidates must have identical performance importance"
                pool=candidates(model,perf,args.candidates)
                entry=dict(method=method,round=r,importance_seconds=time.monotonic()-t,
                           importance_sha256=perf_hash,candidate_pool=pool,choices=[])
                for pick in range(args.remove_per_round):
                    t=time.monotonic()
                    contribution="contribution" in method
                    baseline=collect_signals(model,selection,contribution)
                    before=selection_score(baseline,method)
                    trials=[]
                    for candidate in pool:
                        channel=(candidate["block"],candidate["channel"])
                        if channel in removed: continue
                        layer=reference_layers(model)[candidate["layer"]]
                        active=int(layer.mask.flatten(1).any(1).sum())
                        if active <= max(1,math.ceil(layer.out_channels*prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER)): continue
                        with temporary_channels(model,[channel]):
                            measured=collect_signals(model,selection,contribution)
                            after=selection_score(measured,method)
                        trial=dict(candidate,score_before=before,score_after=after,improvement=before-after)
                        signal_key="contribution_cond" if contribution else "activation_cond"
                        trial["conditional_after_per_attribute_negative_positive"]=measured[signal_key].tolist()
                        trials.append(trial)
                    if not trials: raise RuntimeError("No valid candidate remains")
                    trials.sort(key=lambda z:(round(z["score_after"],12),z["performance"],block_number(z["block"]),z["channel"]))
                    best=trials[0]
                    chosen=(best["block"],best["channel"])
                    apply_channels(model,[chosen])
                    removed.append(chosen)
                    entry["choices"].append(dict(selected=chosen,trials=trials,
                        all_candidates_worsen=all(z["improvement"]<0 for z in trials),selection_seconds=time.monotonic()-t))
                    print(f"  removed {chosen}, score {before:.8g} -> {best['score_after']:.8g}",flush=True)
                entry["removed_cumulative"]=copy.deepcopy(removed)
                entry["masked_weights"]=sum(int((m.mask==0).sum()) for m in weight_layers(model).values())
                print(f"{method} round {r}: evaluating pre-retrain",flush=True)
                result["evaluations"].append(evaluate(model,evaluation,attrs,method,r,"pre"))
                entry["retraining"]=retrain(model,batches[r],args,r)
                print(f"{method} round {r}: evaluating post-retrain",flush=True)
                result["evaluations"].append(evaluate(model,evaluation,attrs,method,r,"post"))
                entry["round_seconds"]=time.monotonic()-round_started
                result["rounds"].append(entry)
                result["elapsed_seconds"]=time.monotonic()-started
                write_report(out,result)
            torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            removed_channels=removed,attributes=attrs,arguments=clean(vars(args))),out/f"{method}_round3.pt")
            del model
        assert sha(args.checkpoint)==result["metadata"]["checkpoint_sha256"]
        result.update(status="complete",elapsed_seconds=time.monotonic()-started)
        write_report(out,result)
        print("\nFINAL THREE-ROUND COMPARISON\n"+comparison(result["evaluations"]),flush=True)
        print("\nTXT:",out/"results.txt",flush=True)
    except Exception as exc:
        result.update(status="failed",error=repr(exc),elapsed_seconds=time.monotonic()-started)
        write_report(out,result)
        raise


if __name__ == "__main__":
    main()
