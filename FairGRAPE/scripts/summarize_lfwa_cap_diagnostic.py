#!/usr/bin/env python3
"""Summarize paired LFWA cap diagnostic; bootstrap fixed predictions by identity."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args()
    root, out = args.run_dir.resolve(), args.out_dir.resolve()
    if not (root / 'complete.json').is_file():
        raise ValueError('Both arms must be complete before summarizing')
    out.mkdir(parents=True, exist_ok=True)
    arms = ['baseline', 'strict_cap10']
    rows, summaries, attrs, preds = [], {}, {}, {}
    before = json.loads((root / 'before_prune/summary.json').read_text())
    for arm in arms:
        for phase in ['post_prune', 'post_retrain']:
            path = root / arm / phase
            summaries[arm, phase] = json.loads((path / 'summary.json').read_text())
            attrs[arm, phase] = pd.read_csv(path / 'attributes.csv').set_index('attribute')
            preds[arm, phase] = pd.read_csv(path / 'predictions.csv')
            row = {'arm': arm, 'phase': phase, **summaries[arm, phase]}
            row['ba_at_most_055_attributes'] = int((attrs[arm, phase].ba <= .55).sum())
            rows.append({k: v for k, v in row.items() if not isinstance(v, (list, dict))})
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    a, b = [summaries[arm, 'post_retrain'] for arm in arms]
    delta = {metric: b[metric] - a[metric] for metric in ['accuracy', 'ba', 'eo', 'model_sparsity']}
    gate = delta['ba'] >= .005 and delta['accuracy'] >= -.002 and b['near_constant_99pct_attributes'] <= a['near_constant_99pct_attributes'] and abs(delta['model_sparsity']) <= .0005
    merged = attrs['baseline', 'post_retrain'].add_suffix('_baseline').join(attrs['strict_cap10', 'post_retrain'].add_suffix('_cap10'))
    for key in ['accuracy', 'ba', 'eo']:
        merged[key + '_delta'] = merged[key + '_cap10'] - merged[key + '_baseline']
    merged.to_csv(out / 'attribute_comparison.csv')
    names = list(merged.index)
    attr_counts = {key: int((merged[key + '_delta'] > 0).sum()) for key in ['accuracy', 'ba', 'eo']}
    near = sorted(set(a['near_constant_99pct_attribute_names']) | set(b['near_constant_99pct_attribute_names']))
    decomposed = {}
    for label, subset in [('near_constant_union', near), ('other_attributes', [n for n in names if n not in near])]:
        part = merged.loc[subset]
        decomposed[label] = {'n_attributes': len(subset), 'attributes': subset, **{k: float(part[k].mean()) for k in ['ba_baseline', 'ba_cap10', 'eo_baseline', 'eo_cap10']}}
    frames = [preds[arm, 'post_retrain'] for arm in arms]
    for key in ['source_index', 'identity', 'young'] + [n + '_true' for n in names]:
        assert np.array_equal(frames[0][key], frames[1][key]), key
    truth = frames[0][[n + '_true' for n in names]].to_numpy()
    groups = frames[0].young.to_numpy()
    identity_codes, identity_names = pd.factorize(frames[0].identity, sort=True)
    count_ids = len(identity_names)
    # Per-identity sufficient statistics, retaining correlations between images,
    # attributes, sensitive groups, and the two fixed fitted models.
    identity_counts = np.bincount(identity_codes, minlength=count_ids)
    sufficient = np.zeros((count_ids, 2, 2, 4, len(names)), dtype=np.float64)
    for arm_i, frame in enumerate(frames):
        pred = frame[[n + '_pred' for n in names]].to_numpy()
        indicators = [(truth == 1) & (pred == 1), (truth == 1) & (pred == 0), (truth == 0) & (pred == 1), (truth == 0) & (pred == 0)]
        for group in [0, 1]:
            for cell_i, indicator in enumerate(indicators):
                np.add.at(sufficient[:, arm_i, group, cell_i, :], identity_codes, indicator * (groups == group)[:, None])

    def compute(weights):
        counts = (weights @ sufficient.reshape(count_ids, -1)).reshape(len(weights), 2, 2, 4, len(names))
        tp, fn, fp, tn = [counts[:, :, :, i, :] for i in range(4)]
        with np.errstate(divide='ignore', invalid='ignore'):
            tpr, fpr = tp / (tp + fn), fp / (fp + tn)
            ba = np.nanmean(np.nanmean(np.stack([tpr, 1 - fpr], axis=3), axis=(2, 3)), axis=2)
            eo_attr = (np.abs(tpr[:, :, 0, :] - tpr[:, :, 1, :]) + np.abs(fpr[:, :, 0, :] - fpr[:, :, 1, :])) / 2
            eo = np.nanmean(eo_attr, axis=2)
            accuracy = (tp + tn).sum(axis=(2, 3)) / ((weights @ identity_counts)[:, None] * len(names))
        return np.stack([accuracy, ba, eo], axis=2), np.isfinite(eo_attr).sum(axis=2)

    check, _ = compute(np.ones((1, count_ids)))
    for i, arm in enumerate(arms):
        for j, metric in enumerate(['accuracy', 'ba', 'eo']):
            assert abs(check[0, i, j] - summaries[arm, 'post_retrain'][metric]) < 1e-12
    rng = np.random.default_rng(20260917)
    weights = rng.multinomial(count_ids, np.full(count_ids, 1 / count_ids), size=args.bootstrap)
    boot, valid = compute(weights)
    differences = boot[:, 1, :] - boot[:, 0, :]
    intervals = {name: {'delta': delta[name], 'percentile_95': np.percentile(differences[:, i], [2.5, 97.5]).tolist()} for i, name in enumerate(['accuracy', 'ba', 'eo'])}
    result = {'source_run': str(root), 'delta_cap_minus_baseline': delta, 'practical_extension_gate_passed': bool(gate), 'bootstrap': {'replicates': args.bootstrap, 'seed': 20260917, 'sampling_unit': 'identity', 'n_identities': count_ids, 'n_images': len(frames[0]), 'eo_valid_attributes_min': int(valid.min()), 'eo_valid_attributes_max': int(valid.max()), 'intervals': intervals, 'scope': 'Exploratory validation-sample uncertainty conditional on the two fitted checkpoints; not training-seed variation or evidence of whole-trajectory replication.'}, 'attribute_improvement_counts': {'accuracy_increased': attr_counts['accuracy'], 'ba_increased': attr_counts['ba'], 'eo_decreased': int((merged.eo_delta < 0).sum())}, 'attribute_decomposition': decomposed}
    (out / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, metric, title in zip(axes, ['accuracy', 'ba', 'eo'], ['Accuracy (higher is better)', 'Group/class BA (higher is better)', 'EO (lower is better)']):
        for arm, label, color in [('baseline', 'Original selector', '#c04b3e'), ('strict_cap10', '10% per-layer cap', '#177f92')]:
            epochs = pd.read_csv(root / arm / 'epochs.csv')
            vals = epochs.loc[epochs.phase == 'val', metric].tolist()
            values = [before[metric], summaries[arm, 'post_prune'][metric]] + vals
            ax.plot(range(len(values)), values, 'o-', label=label, color=color, markersize=4)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(7), ['Source', 'Pruned', '1', '2', '3', '4', '5'], rotation=30)
        ax.set_xlabel('Same source → pruning → retraining epoch')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('LFWA α=0.7, step 13: paired intervention, validation only', fontsize=12)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(out / ('paired_step13.' + ext), dpi=180, bbox_inches='tight')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
