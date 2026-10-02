#!/usr/bin/env python3
"""Read completed paired iterations and report progress without touching training."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--initial-run', type=Path, required=True)
    args = parser.parse_args()
    root, initial = args.run_dir.resolve(), args.initial_run.resolve()
    records = []
    for arm, initial_arm in [('baseline', 'baseline'), ('cap10', 'strict_cap10')]:
        for phase in ['post_prune', 'post_retrain']:
            s = json.loads((initial / initial_arm / phase / 'summary.json').read_text())
            records.append({'iteration': 13, 'arm': arm, 'phase': phase, **s})
        for marker in sorted((root / arm).glob('iteration_*/complete.json')):
            data = json.loads(marker.read_text())
            if data['arm'] != arm:
                raise ValueError('Unexpected arm metadata')
            for phase in ['post_prune', 'post_retrain']:
                records.append({'iteration': data['iteration'], 'arm': arm, 'phase': phase, **data[phase]})
    frame = pd.DataFrame(records)
    for column in list(frame.columns):
        if any(isinstance(x, (list, dict)) for x in frame[column]):
            frame = frame.drop(columns=[column])
    out = root / 'analysis'
    out.mkdir(exist_ok=True)
    frame.to_csv(out / 'all_completed_metrics.csv', index=False)
    final = frame[frame.phase == 'post_retrain']
    paired = final[final.arm == 'baseline'].set_index('iteration').add_suffix('_baseline').join(final[final.arm == 'cap10'].set_index('iteration').add_suffix('_cap10'), how='inner')
    for key in ['accuracy', 'ba', 'eo', 'model_sparsity', 'constant_prediction_attributes', 'near_constant_99pct_attributes']:
        paired[key + '_delta'] = paired[key + '_cap10'] - paired[key + '_baseline']
    paired.to_csv(out / 'paired_metrics.csv')
    last = paired.iloc[-1]
    last_iter = int(paired.index[-1])
    near_delta = int(last.near_constant_99pct_attributes_delta)
    budget_ok = abs(last.model_sparsity_delta) <= .0005
    recent = paired.tail(5)
    gate = None
    if last_iter >= 25:
        gate = bool(budget_ok and last.ba_delta >= .01 and last.accuracy_delta >= -.005 and near_delta <= 0 and (recent.ba_delta > 0).sum() >= 3)
    report = {'last_paired_iteration': last_iter, 'latest_delta_cap_minus_baseline': {k: float(last[k + '_delta']) for k in ['accuracy','ba','eo','model_sparsity']}, 'latest_near_constant_count_delta': near_delta, 'latest_whole_model_sparsity_difference_within_005_percentage_points': bool(budget_ok), 'practical_step25_extension_gate': gate, 'complete_marker_exists': (root/'complete.json').exists(), 'gate_note': 'Practical resource-allocation rule; not statistical significance. No stage25 conclusion before both models finish.'}
    (out / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for arm, label, color in [('baseline', 'Original selector', '#c04b3e'), ('cap10', '10% per-layer cap', '#177f92')]:
        part = final[final.arm == arm].sort_values('iteration')
        for ax, metric, title in zip(axes, ['accuracy','ba','eo'], ['Accuracy ↑', 'Group/class BA ↑', 'EO ↓']):
            ax.plot(part.model_sparsity * 100, part[metric], 'o-', color=color, label=label, markersize=4)
            ax.set_title(title)
            ax.set_xlabel('Whole-model parameter sparsity (%)')
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('LFWA α=0.7: continuation after paired step13, validation only', fontsize=11)
    fig.tight_layout()
    for extension in ['png','pdf']:
        fig.savefig(out / ('trajectory.' + extension), dpi=180, bbox_inches='tight')
    table = ['| Step | Acc original / cap | BA original / cap | EO original / cap | Sparsity difference (%p) |', '|---:|---:|---:|---:|---:|']
    for iteration, row in paired.iterrows():
        values = [f'{row[k + "_baseline"]:.4f} / {row[k + "_cap10"]:.4f}' for k in ['accuracy','ba','eo']]
        table.append(f'| {iteration} | ' + ' | '.join(values) + f' | {row.model_sparsity_delta * 100:+.4f} |')
    text = 'Paired validation progress, completed iterations only. Step13 is the prior paired intervention; subsequent scores are recomputed separately for each model.\n\n' + '\n'.join(table) + '\n\nStep25 extension gate: ' + str(gate) + ' (None means pending).\n'
    (out / 'progress.md').write_text(text)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
