"""LFWA data contracts, official split preparation and binary fairness metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ATTRIBUTES = (
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "young",
)
TARGETS = list(ATTRIBUTES[:-1])
LABEL_COLUMNS = TARGETS + ["young"]
OUTPUT_SLICES = [(2 * i, 2 * i + 2) for i in range(len(TARGETS))]
SPLIT_SIZES = {"train": 5000, "val": 1263, "test": 6880}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def metadata_for(frame):
    """Path-independent fingerprint of the actual labels and ordered splits."""
    rows = frame[["source_index", "image_name", "split"] + LABEL_COLUMNS].to_dict("records")
    return {
        "dataset": "LFWA", "schema_version": 1, "targets": TARGETS,
        "sensitive_attribute": "young", "sensitive_values": {"0": "non-young", "1": "young"},
        "output_slices": [list(x) for x in OUTPUT_SLICES],
        "split_sizes": SPLIT_SIZES, "manifest_sha256": _digest(rows),
        "split_sha256": {
            split: _digest(frame.loc[frame.split == split, "source_index"].tolist())
            for split in SPLIT_SIZES
        },
    }


def validate_frame(frame, check_paths=False):
    required = ["image_name", "face_name_align", "identity", "source_index", "split"] + LABEL_COLUMNS
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing LFWA columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError("LFWA contains missing labels or metadata")
    for row in frame[['image_name', 'face_name_align', 'identity']].itertuples(index=False):
        suffix = '/'.join(Path(row.face_name_align).parts[-2:])
        if suffix != row.image_name or row.identity != row.image_name.split('/')[0]:
            raise ValueError(f'Image path, annotation name and identity disagree: {row.image_name}')
    if frame.split.value_counts().to_dict() != SPLIT_SIZES:
        raise ValueError(f"LFWA split sizes differ from {SPLIT_SIZES}: {frame.split.value_counts().to_dict()}")
    for column in ("image_name", "face_name_align", "source_index"):
        if frame[column].duplicated().any():
            raise ValueError(f"Duplicate LFWA {column}")
    if set(frame.source_index) != set(range(1, 13144)):
        raise ValueError("LFWA source indices must cover 1..13143 exactly")
    for column in LABEL_COLUMNS:
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"LFWA {column} must contain only 0/1")
    for split in SPLIT_SIZES:
        if set(frame.loc[frame.split == split, "young"]) != {0, 1}:
            raise ValueError(f"LFWA {split} is missing a Young group")
    if check_paths:
        missing_paths = [str(path) for path in frame.face_name_align if not Path(path).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Missing {len(missing_paths)} LFWA images: {missing_paths[:5]}")


def load_frames(csv="csv/LFWA.csv", image_root="Images/LFWA"):
    frame = pd.read_csv(csv)
    validate_frame(frame)
    metadata = metadata_for(frame)
    sidecar = Path(csv).with_suffix(".metadata.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing {sidecar}; run scripts/prepare_lfwa.py")
    recorded = json.loads(sidecar.read_text())
    if recorded["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("LFWA CSV differs from its prepared manifest")
    root = Path(image_root).resolve()
    from concurrent.futures import ThreadPoolExecutor

    def resolve_image(path):
        resolved = (root / path).resolve()
        if root not in resolved.parents:
            raise ValueError('LFWA image path escapes the selected image root')
        if not resolved.is_file():
            raise FileNotFoundError(f'Missing LFWA image: {resolved}')
        return str(resolved)

    with ThreadPoolExecutor(max_workers=8) as pool:
        frame["face_name_align"] = list(pool.map(resolve_image, frame.face_name_align))
    validate_frame(frame)
    frames = {split: frame[frame.split == split].reset_index(drop=True) for split in SPLIT_SIZES}
    frames["all"] = frame
    for part in frames.values():
        part.attrs["lfwa_metadata"] = metadata
    return frames, metadata


def validate_checkpoint(checkpoint, expected):
    actual = checkpoint.get("experiment_metadata") if isinstance(checkpoint, dict) else None
    if actual != expected:
        raise ValueError("Checkpoint LFWA targets, sensitive attribute or split/labels do not match this run")


def _read_strings(handle, key):
    return [
        "".join(chr(int(c)) for c in np.asarray(handle[ref]).ravel())
        for ref in np.asarray(handle[key]).ravel()
    ]


def official_annotations(raw):
    import h5py
    raw = Path(raw)
    with h5py.File(raw / "lfw_att_40.mat", "r") as source:
        attributes = _read_strings(source, "AttrName")
        names = [s.replace("\\", "/") for s in _read_strings(source, "name")]
        labels = np.asarray(source["label"])
    attributes = ["young" if a == "Young" else a for a in attributes]
    if len(set(attributes)) != 40 or set(attributes) != set(ATTRIBUTES):
        raise ValueError(f"Unexpected official LFWA attributes: {attributes}")
    if labels.shape == (40, len(names)):
        labels = labels.T
    if labels.shape != (len(names), 40) or len(names) != 13143:
        raise ValueError(f"Unexpected LFWA annotation shape: {labels.shape}, names={len(names)}")
    values = set(np.unique(labels))
    if values <= {-1, 1}:
        labels = (labels + 1) / 2
    elif not values <= {0, 1}:
        raise ValueError(f"Unexpected LFWA label encoding: {values}")
    with h5py.File(raw / "indices_train_test.mat", "r") as source:
        indices = {}
        for split, size in (("train", 6263), ("test", 6880)):
            original = np.asarray(source[f"indices_img_{split}"]).ravel()
            if len(original) != size or not np.equal(original, original.astype(int)).all():
                raise ValueError(f"Invalid official {split} indices")
            indices[split] = original.astype(int) - 1
    all_indices = np.concatenate(list(indices.values()))
    if set(all_indices) != set(range(len(names))) or len(set(all_indices)) != len(all_indices):
        raise ValueError("Official LFWA train/test indices overlap or do not cover the labels")
    return names, pd.DataFrame(labels.astype(np.int64), columns=attributes)[LABEL_COLUMNS], indices


def prepare_csv(raw, image_root, output):
    from PIL import Image
    import importlib.util
    spec = importlib.util.spec_from_file_location("lfwa_preparation", Path(__file__).parent / "scripts/prepare_lfwa.py")
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    FILES, sha256_file = preparation.FILES, preparation.sha256_file
    names, labels, indices = official_annotations(raw)
    image_root, output = Path(image_root).resolve(), Path(output)
    # Match the annotation's person/file.jpg suffix, preserving the archive layout.
    import zipfile
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import PurePosixPath
    image_index = {}
    with zipfile.ZipFile(Path(raw) / 'lfw.zip') as archive:
        for member in archive.infolist():
            name_path = PurePosixPath(member.filename)
            if name_path.is_absolute() or '..' in name_path.parts:
                raise ValueError(f'Invalid archive path: {member.filename}')
            if name_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            name = '/'.join(name_path.parts[-2:])
            if name in image_index:
                raise ValueError(f'Duplicate image in LFWA archive: {name}')
            image_index[name] = image_root / member.filename

    def verify_image(name):
        if name not in image_index:
            raise FileNotFoundError(f'Official LFWA annotation has no image: {name}')
        path = image_index[name]
        with Image.open(path) as image:
            image.convert('RGB').load()
        # Match the existing FaceDataset decoder as well as PIL.
        import dlib
        decoded = dlib.load_rgb_image(str(path))
        if decoded.ndim != 3 or decoded.shape[2] != 3:
            raise ValueError(f'Invalid RGB image: {path}')
        return path.relative_to(image_root).as_posix()

    paths = []
    # Parallel reads avoid serial metadata latency on the shared filesystem.
    # map preserves annotation order and propagates every decoding error.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for number, path in enumerate(pool.map(verify_image, names), 1):
            paths.append(path)
            if number % 1000 == 0:
                print(f'Verified {number:,}/{len(names):,} annotated images', flush=True)
    frame = pd.DataFrame({
        "image_name": names, "face_name_align": paths,
        "identity": [name.split("/")[0] for name in names],
        "source_index": np.arange(1, len(names) + 1),
    })
    frame = pd.concat([frame, labels], axis=1)
    parts = []
    for split, positions in (("train", indices["train"][:5000]), ("val", indices["train"][5000:]), ("test", indices["test"])):
        part = frame.iloc[positions].copy()
        part["split"] = split
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    validate_frame(frame)
    metadata = metadata_for(frame)
    metadata["source_files"] = {
        name: {"google_drive_id": FILES[name][0], "bytes": (Path(raw) / name).stat().st_size,
               "sha256": sha256_file(Path(raw) / name)} for name in FILES
    }
    identities = {split: set(frame.loc[frame.split == split, "identity"]) for split in SPLIT_SIZES}
    metadata["identity_overlap"] = {
        f"{a}_{b}": len(identities[a] & identities[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    metadata["young_counts"] = {
        split: {str(k): int(v) for k, v in frame.loc[frame.split == split, "young"].value_counts().items()}
        for split in SPLIT_SIZES
    }
    metadata["archive_images"] = len(image_index)
    metadata["annotated_images"] = len(frame)
    metadata["unannotated_images"] = sorted(set(image_index) - set(names))
    metadata["identity_counts"] = {split: len(values) for split, values in identities.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    counts = []
    for split in SPLIT_SIZES:
        part = frame[frame.split == split]
        for attr in TARGETS:
            for group in (0, 1):
                for label in (0, 1):
                    counts.append({"split": split, "attribute": attr, "young": group, "label": label,
                                   "count": int(((part.young == group) & (part[attr] == label)).sum())})
    pd.DataFrame(counts).to_csv(output.with_suffix(".cell_counts.csv"), index=False)
    print(json.dumps({k: metadata[k] for k in ("split_sizes", "young_counts", "identity_overlap", "manifest_sha256")}, indent=2))
    print(f"Prepared and verified {len(frame):,} images: {output}", flush=True)


def binary_metrics(truth, predictions, sensitive, attributes=TARGETS):
    """Group/class BA and equalized odds; undefined rates remain missing."""
    truth, predictions, sensitive = np.asarray(truth), np.asarray(predictions), np.asarray(sensitive)
    if truth.shape != predictions.shape or truth.shape != (len(sensitive), len(attributes)):
        raise ValueError("Metric shapes do not match the named targets")
    if not len(sensitive) or not np.isin(sensitive, [0, 1]).all():
        raise ValueError("Metrics require nonempty binary sensitive labels")
    if not np.isin(truth, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Metrics require binary targets and predictions")
    rows = []
    for j, attr in enumerate(attributes):
        row = {"attribute": attr, "accuracy": float((truth[:, j] == predictions[:, j]).mean())}
        rates = []
        for group in (0, 1):
            mask = sensitive == group
            y, pred = truth[mask, j], predictions[mask, j]
            tp, fn = int(((y == 1) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
            fp, tn = int(((y == 0) & (pred == 1)).sum()), int(((y == 0) & (pred == 0)).sum())
            tpr = tp / (tp + fn) if tp + fn else np.nan
            fpr = fp / (fp + tn) if fp + tn else np.nan
            row.update({f"g{group}_{key}": value for key, value in {
                "n": len(y), "tp": tp, "fn": fn, "fp": fp, "tn": tn, "tpr": tpr, "fpr": fpr,
                "accuracy": float((y == pred).mean()) if len(y) else np.nan,
            }.items()})
            rates.extend([tpr, 1 - fpr])
        row["ba"] = float(np.nanmean(rates)) if np.isfinite(rates).any() else np.nan
        row["eo"] = (abs(row["g0_tpr"] - row["g1_tpr"]) + abs(row["g0_fpr"] - row["g1_fpr"])) / 2
        rows.append(row)
    per_attribute = pd.DataFrame(rows)
    summary = {"n_images": len(sensitive), "n_attributes": len(attributes),
               "accuracy": float(per_attribute.accuracy.mean()), "ba": float(per_attribute.ba.mean()),
               "eo": float(per_attribute.eo.mean()) if per_attribute.eo.notna().any() else None,
               "eo_valid_attributes": int(per_attribute.eo.notna().sum())}
    for group in (0, 1):
        summary[f"young_{group}_accuracy"] = float(per_attribute[f"g{group}_accuracy"].mean()) if (sensitive == group).any() else None
    return per_attribute, summary


def sparsity(model):
    """Count removed entries from binary masks, excluding masks from denominators."""
    from prune import _get_impt_type2_block_layer_names
    target_names = {n for b in range(1, 18) for n in _get_impt_type2_block_layer_names(f'features.{b}') if n}
    target_total = target_removed = removed = 0
    total = sum(p.numel() for n, p in model.named_parameters() if not n.endswith('.mask'))
    for name, layer in model.named_modules():
        weight, mask = getattr(layer, 'weight', None), getattr(layer, 'mask', None)
        if weight is None:
            continue
        zeros = int((mask == 0).sum()) if mask is not None else 0
        removed += zeros
        if name in target_names:
            target_total += weight.numel()
            target_removed += zeros
    return {'target_weights': target_total, 'target_removed': target_removed,
            'target_sparsity': target_removed / target_total if target_total else 0.,
            'model_parameters': total, 'model_removed': removed,
            'model_sparsity': removed / total if total else 0.}


def assert_masks(model, before=None):
    import torch
    current = {}
    for name, layer in model.named_modules():
        mask = getattr(layer, 'mask', None)
        if mask is None:
            continue
        if mask.requires_grad or not torch.all((mask == 0) | (mask == 1)):
            raise AssertionError(f'Invalid or trainable mask: {name}')
        if torch.count_nonzero(layer.weight.detach()[mask == 0]):
            raise AssertionError(f'Pruned weights regrew: {name}')
        current[name] = mask.detach().cpu().clone()
    if before is not None:
        if current.keys() != before.keys() or any(not torch.equal(current[n], before[n]) for n in current):
            raise AssertionError('Pruning masks changed during retraining')
    return current


def save_checkpoint(model, path, metadata, **details):
    import os
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert_masks(model)
    checkpoint = {'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  'experiment_metadata': metadata, 'sparsity': sparsity(model),
                  'validation': getattr(model, 'lfwa_validation_metrics', None), **details}
    partial = path.with_suffix(path.suffix + '.part')
    torch.save(checkpoint, partial)
    os.replace(partial, path)
    print(f'LFWA checkpoint: {path}; sparsity={checkpoint["sparsity"]}', flush=True)
    return checkpoint


def load_checkpoint_model(path, metadata, device='cpu'):
    import torch
    import types
    from util import make_model
    from prune import forward_mapping_dict
    checkpoint = torch.load(path, map_location='cpu')
    validate_checkpoint(checkpoint, metadata)
    model = make_model(network='mobilenetv2', n_classes=78, pretrained=False)
    state = checkpoint.get('model_state', checkpoint.get('model_state_dict'))
    if state is None:
        raise ValueError('LFWA checkpoint has no model state')
    modules = dict(model.named_modules())
    for key, value in state.items():
        if key.endswith('.mask'):
            layer = modules[key[:-5]]
            layer.mask = torch.nn.Parameter(torch.ones_like(value), requires_grad=False)
            layer.forward = types.MethodType(forward_mapping_dict[type(layer).__name__], layer)
    model.load_state_dict(state, strict=True)
    model.to(device)
    assert_masks(model)
    return model, checkpoint


def record_epoch(row):
    import config
    root = getattr(config, 'glo_lfwa_run_dir', None)
    if root is None:
        return
    row = {'stage': config.glo_train_stage, 'epoch': config.glo_epoch_count,
           'prune_iteration': config.glo_prune_iter + 1 if config.glo_train_stage != 'dense' else 0, **row}
    output = Path(root) / 'epochs.csv'
    pd.DataFrame([row]).to_csv(output, mode='a', header=not output.exists(), index=False, na_rep='N/A')
    print('LFWA epoch:', row, flush=True)


def evaluate_model(model, frame, directory, batch_size=256):
    import torch
    from dataset import make_datasets
    from util import safe_forward_with_cudnn_fallback
    if len(set(frame.split)) != 1:
        raise ValueError('Evaluation requires exactly one split')
    _, loader = make_datasets(frame, frame, True, batch_size, LABEL_COLUMNS)
    device = next(model.parameters()).device
    model.eval()
    truth, predictions, sensitive, probabilities = [], [], [], []
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        for images, labels in loader:
            logits = safe_forward_with_cudnn_fallback(model, images.to(device)).reshape(-1, 39, 2)
            truth.append(labels[:, :39].numpy())
            sensitive.append(labels[:, -1].numpy())
            predictions.append(logits.argmax(-1).cpu().numpy())
            probabilities.append(logits.softmax(-1)[:, :, 1].cpu().numpy())
    y, pred, group, prob = [np.concatenate(parts) for parts in (truth, predictions, sensitive, probabilities)]
    if not np.array_equal(y, frame[TARGETS].to_numpy()) or not np.array_equal(group, frame.young.to_numpy()):
        raise AssertionError('Evaluation loader changed manifest row order')
    per_attribute, summary = binary_metrics(y, pred, group)
    summary.update(sparsity(model))
    summary['split'] = frame.split.iloc[0]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = frame[['source_index', 'image_name', 'identity', 'split', 'young']].copy()
    for j, name in enumerate(TARGETS):
        output[name + '_true'] = y[:, j]
        output[name + '_pred'] = pred[:, j]
        output[name + '_probability'] = prob[:, j]
    output.to_csv(directory / 'predictions.csv', index=False)
    per_attribute.to_csv(directory / 'attributes.csv', index=False, na_rep='N/A')
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print('LFWA evaluation:', json.dumps(summary), flush=True)
    return summary
