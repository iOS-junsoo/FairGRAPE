"""Contracts for ImageNet -> official CelebA single-attribute experiments."""
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PARTITION_MD5 = 'd32c9cbf5e040fd4025c592c306e6668'
PARTITION_FILE_ID = '0B7EVK8r0v71pY0NSMzRuSXJEVkk'
SPLIT_SIZES = {'train': 162770, 'val': 19867, 'test': 19962}
IMAGENET_WEIGHTS = 'MobileNet_V2_Weights.IMAGENET1K_V1'
OUTPUT_SLICES = [(0, 2)]


def file_hash(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_partition(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}; run scripts/prepare_celeba_partition.py')
    if file_hash(path, 'md5') != PARTITION_MD5:
        raise ValueError('CelebA partition checksum differs from the official file')
    part = pd.read_csv(path, sep=r'\s+', header=None, names=['image_name', 'partition'])
    if part.isna().any().any() or part.image_name.duplicated().any():
        raise ValueError('Missing or duplicate names in partition')
    if not part.partition.isin([0, 1, 2]).all():
        raise ValueError('Partition must be 0 (train), 1 (val), or 2 (test)')
    part['split'] = part.partition.map({0: 'train', 1: 'val', 2: 'test'})
    if part.split.value_counts().to_dict() != SPLIT_SIZES:
        raise ValueError('Official partition sizes do not match')
    return part[['image_name', 'split']]


def load_frames(csv, image_root, partition_file, target, sensitive='gender',
                *, check_readable=True, smoke_samples=0):
    """Join by filename, preserving official membership even when CSV order changes."""
    part = read_partition(partition_file)
    frame = pd.read_csv(csv)
    targets = list(frame.columns[2:41])
    if len(targets) != 39 or target not in targets or target == sensitive:
        raise ValueError(f'--target_attr must be one of the 39 prediction attributes: {targets}')
    # The supported CelebA schema excludes Male from the 39 tasks and calls it gender.
    if sensitive != 'gender':
        raise ValueError('CelebA impt=5/6 requires --sensitive_group gender')
    required = ['image_name', 'face_name_align', target, sensitive]
    if not set(required).issubset(frame.columns) or frame[required].isna().any().any():
        raise ValueError('Missing image names or selected labels')
    for column in ('image_name', 'face_name_align'):
        frame[column] = frame[column].map(lambda name: str(name).replace('\\', '/').rsplit('/', 1)[-1])
        if frame[column].duplicated().any():
            raise ValueError(f'Duplicate {column}')
    if not frame.image_name.equals(frame.face_name_align):
        raise ValueError('Image names and aligned filenames disagree')
    if set(frame.image_name) != set(part.image_name):
        raise ValueError('CSV and official partition must cover exactly the same images')
    for column in (target, sensitive):
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f'{column} must contain only 0/1')
        frame[column] = frame[column].astype('int64')
    frame = frame.drop(columns=['split'], errors='ignore').merge(part, on='image_name', validate='one_to_one')
    frame = frame.sort_values('image_name').reset_index(drop=True)
    official_hashes = {s: digest(frame.loc[frame.split == s, 'image_name'].tolist()) for s in SPLIT_SIZES}
    if smoke_samples:
        if smoke_samples < 4:
            raise ValueError('Smoke samples must be >= 4 per split (one per label/group cell)')
        selected = []
        for split in SPLIT_SIZES:
            rows = frame.loc[frame.split == split]
            if len(rows) < smoke_samples:
                raise ValueError('Smoke sample count exceeds split size')
            cells = rows.groupby([target, sensitive], sort=True).head(1)
            if len(cells) != 4:
                raise ValueError(f'{split} lacks a target/group cell for smoke validation')
            rest = rows.drop(cells.index).head(smoke_samples - 4)
            selected.append(pd.concat([cells, rest]))
        frame = pd.concat(selected).sort_values('image_name').reset_index(drop=True)
    root = Path(image_root).resolve()

    def check(name):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'Missing official split image: {path}')
        if check_readable:
            from PIL import Image
            with Image.open(path) as im:
                im.load()
        return str(path)

    # Smoke checks only its declared subset; production never silently drops images.
    with ThreadPoolExecutor(max_workers=8) as pool:
        frame['face_name_align'] = list(pool.map(check, frame.image_name))
    frames = {s: frame.loc[frame.split == s].reset_index(drop=True) for s in SPLIT_SIZES}
    metadata = {
        'schema_version': 1, 'dataset': 'CelebA', 'network': 'mobilenetv2',
        'targets': [target], 'sensitive_attribute': sensitive, 'num_classes': 2,
        'output_slices': [[0, 2]], 'initialization': IMAGENET_WEIGHTS,
        'split_protocol': 'official', 'partition_sha256': file_hash(partition_file),
        'official_split_sha256': official_hashes,
        'split_sizes': {s: len(f) for s, f in frames.items()},
        'split_sha256': {s: digest(f.image_name.tolist()) for s, f in frames.items()},
        'labels_sha256': digest(frame[['image_name', target, sensitive]].to_dict('records')),
        'smoke_samples': smoke_samples,
    }
    return frames, metadata


def selection_key(metrics):
    eo = metrics['eo']
    eo = float(eo) if eo is not None and math.isfinite(float(eo)) else math.inf
    return (float(metrics['accuracy']), -eo, -float(metrics['loss']))


def validate_checkpoint(checkpoint, expected, *, allow_pruned=False, impt_type=None):
    if not isinstance(checkpoint, dict) or checkpoint.get('experiment_metadata') != expected:
        raise ValueError('Checkpoint attribute, ImageNet initialization, official splits or labels differ')
    if checkpoint.get('finetuning_complete') is not True:
        raise ValueError('Single-attribute fine-tuning must be complete')
    state = checkpoint.get('model_state')
    network = expected.get('network', 'mobilenetv2')
    heads = {'mobilenetv2': ('classifier.1.weight', (2, 1280)), 'resnet18': ('fc.weight', (2, 512))}
    if network not in heads:
        raise ValueError(f'Unsupported checkpoint network: {network}')
    key, shape = heads[network]
    if not isinstance(state, dict) or tuple(state.get(key, np.empty(0)).shape) != shape:
        raise ValueError(f'Checkpoint must contain a two-output {network} classifier')
    pruned = any(k.endswith('.mask') for k in state)
    if checkpoint.get('stage') not in ('dense', 'pruned'):
        raise ValueError('Unknown checkpoint stage')
    if pruned != (checkpoint['stage'] == 'pruned'):
        raise ValueError('Checkpoint stage and masks disagree')
    if not allow_pruned and (pruned or checkpoint.get('prune_iteration') != 0):
        raise ValueError('Start pruning from a dense single-attribute checkpoint')
    if pruned:
        origin = checkpoint.get('source_checkpoint', {})
        if not isinstance(origin, dict) or len(origin.get('sha256', '')) != 64:
            raise ValueError('Pruned checkpoint is missing its dense checkpoint provenance')
        method = checkpoint.get('pruning_method')
        if impt_type == 7:
            from taylor_pruning import METHOD
            if method != METHOD:
                raise ValueError('impt=7 requires an all-backbone Taylor neuron checkpoint')
        if impt_type == 6 and method != dict(impt_type=6, unit='weight', mode='original', scope='all'):
            raise ValueError('impt=6 requires an original/all single-attribute pruning checkpoint')
        if impt_type == 5 and method is not None and (not isinstance(method, dict) or method.get('impt_type') != 5):
            raise ValueError('impt=5 requires a channel-pruning checkpoint')
        for key, mask in state.items():
            if key.endswith('.mask') and not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError(f'Non-binary pruning mask: {key}')
    return state


def save_checkpoint(model, path, metadata, *, iteration=0, source=None, settings=None):
    import torch
    pruned = any(n.endswith('.mask') for n in model.state_dict())
    checkpoint = {
        'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'experiment_metadata': metadata, 'finetuning_complete': True,
        'stage': 'pruned' if pruned else 'dense', 'prune_iteration': iteration,
        'source_checkpoint': source, 'settings': settings,
        'validation_metrics': getattr(model, 'single_validation_metrics', None),
    }
    if pruned:
        impt = (settings or {}).get('impt_type', (settings or {}).get('arguments', {}).get('impt', 5))
        if impt == 7:
            from taylor_pruning import METHOD, statistics, validate_masks
            validate_masks(model)
            checkpoint['pruning_method'] = dict(METHOD)
            checkpoint['pruning_statistics'] = statistics(model)
            checkpoint['online_state'] = dict(getattr(model, 'taylor_online_state', {}))
        else:
            checkpoint['pruning_method'] = (dict(impt_type=6, unit='weight', mode='original', scope='all')
                                            if impt == 6 else dict(impt_type=5, unit='channel'))
    validate_checkpoint(checkpoint, metadata, allow_pruned=pruned)
    torch.save(checkpoint, path)


def metrics_from_counts(counts, loss):
    """Compute binary group gaps; missing denominators invalidate only affected metrics."""
    cells, groups = [], {}
    correct = samples = 0
    for g in (0, 1):
        c = {k: int(v) for k, v in counts[g].items()}
        tp, fp, p, n = (c[k] for k in ('TP', 'FP', 'P', 'N'))
        tpr, fpr = tp / p if p else None, fp / n if n else None
        if p: cells.append(tpr)
        if n: cells.append((n - fp) / n)
        groups[str(g)] = dict(c, tpr=tpr, fpr=fpr,
                              positive_prediction_rate=(tp + fp) / (p + n) if p + n else None)
        correct += tp + n - fp
        samples += p + n
    if not samples:
        raise ValueError('Cannot evaluate an empty split')

    def gap(key):
        a, b = (groups[str(g)][key] for g in (0, 1))
        return abs(a - b) if a is not None and b is not None else None

    eopp, fpr_gap, dp = gap('tpr'), gap('fpr'), gap('positive_prediction_rate')
    eo = (eopp + fpr_gap) / 2 if eopp is not None and fpr_gap is not None else None
    return dict(accuracy=correct / samples, loss=float(loss), ba=float(np.mean(cells)),
                eo=eo, eopp=eopp, dp=dp, samples=samples, groups=groups)


def binary_metrics(truth, pred, group, loss):
    truth, pred, group = map(np.asarray, (truth, pred, group))
    counts = {}
    for g in (0, 1):
        pos, neg = (group == g) & (truth == 1), (group == g) & (truth == 0)
        counts[g] = dict(TP=int(((pred == 1) & pos).sum()), P=int(pos.sum()),
                         FP=int(((pred == 1) & neg).sum()), N=int(neg.sum()))
    return metrics_from_counts(counts, loss)


def evaluate(model, loader, directory, target, sensitive, split):
    import torch
    import config
    from util import safe_forward_with_cudnn_fallback
    model.eval()
    device = next(model.parameters()).device
    truth, group, prediction, probability = [], [], [], []
    loss, count = 0., 0
    with torch.no_grad():
        for images, labels in loader:
            out = safe_forward_with_cudnn_fallback(model, images.to(device, dtype=torch.float32))
            y = labels[:, 0].to(device).long()
            loss += torch.nn.functional.cross_entropy(out, y, reduction='sum').item()
            count += len(y)
            truth.extend(y.cpu().tolist()); group.extend(labels[:, -1].tolist())
            prediction.extend(out.argmax(1).cpu().tolist())
            probability.extend(out.softmax(1)[:, 1].cpu().tolist())
    if count == 0:
        raise ValueError('Cannot evaluate an empty split')
    result = binary_metrics(truth, prediction, group, loss / count)
    result.update(split=split, attribute=target, split_protocol='official',
                  impt_type=getattr(config, 'glo_impt_type', None) or 5)
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    (directory / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    frame = pd.DataFrame({target: truth, f'{target}_pred': prediction,
                          f'{target}_prob_1': probability, sensitive: group})
    frame.insert(0, 'image_name', loader.dataset.data_frame.image_name.tolist())
    frame.to_csv(directory / 'predictions.csv', index=False)
    return result
