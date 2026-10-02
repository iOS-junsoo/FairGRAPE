"""Standalone LFWA comparison tables and exportable training figures."""
import json
from pathlib import Path

import pandas as pd


def build_report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    output = root / 'evaluation'
    summaries, tables = {}, {}
    for name in ('dense', 'pruned'):
        summaries[name] = json.loads((output / name / 'summary.json').read_text())
        tables[name] = pd.read_csv(output / name / 'attributes.csv')
    pd.DataFrame(summaries).T.to_csv(output / 'comparison.csv', na_rep='N/A')
    comparison = tables['dense'].merge(tables['pruned'], on='attribute', suffixes=('_dense', '_pruned'), validate='one_to_one')
    for metric in ('accuracy', 'ba', 'eo'):
        comparison[metric + '_delta'] = comparison[metric + '_pruned'] - comparison[metric + '_dense']
    comparison.to_csv(output / 'attribute_changes.csv', index=False, na_rep='N/A')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for name in ('dense', 'pruned'):
        history = pd.read_csv(root / name / 'epochs.csv')
        history[history.phase == 'val'].to_csv(output / f'{name}_validation_history.csv', index=False, na_rep='N/A')
        for phase, style in [('train', '--'), ('val', '-')]:
            selected = history[history.phase == phase]
            axes[0, 0].plot(selected.epoch, selected.accuracy, style, label=f'{name}/{phase}')
            axes[0, 1].plot(selected.epoch, selected.loss, style, label=f'{name}/{phase}')
        selected = history[history.phase == 'val']
        axes[1, 0].plot(selected.epoch, selected.eo, label=name)
    # Read recorded mask counts, not 1 - keep**iteration.
    import torch
    records = []
    for checkpoint in sorted((root / 'pruned').glob('prune_*.pt')):
        data = torch.load(checkpoint, map_location='cpu')
        validation = data.get('validation') or {}
        records.append({'iteration': data['iteration'], **data['sparsity'],
                        **{'validation_' + key: value for key, value in validation.items()}})
    sparsity = pd.DataFrame(records)
    sparsity.to_csv(output / 'sparsity_history.csv', index=False, na_rep='N/A')
    sparsity.to_csv(output / 'pruning_history.csv', index=False, na_rep='N/A')
    if len(sparsity):
        axes[1, 1].plot(sparsity.iteration, sparsity.target_sparsity, label='Target weights')
        axes[1, 1].plot(sparsity.iteration, sparsity.model_sparsity, label='Whole model')
    for ax, title, xlabel in zip(axes.flat, ['Attribute accuracy', 'Main loss', 'Validation EO', 'Actual mask sparsity'], ['Epoch', 'Epoch', 'Epoch', 'Pruning iteration']):
        ax.set(title=title, xlabel=xlabel)
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png', dpi=180)
    fig.savefig(output / 'training_curves.pdf')
    plt.close(fig)
    lines = ['# LFWA seed 42 results', '', 'Test was evaluated after Dense and pruning model selection.', '',
             '| Model | Accuracy | BA | EO | Valid EO attributes | Target sparsity | Model sparsity |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, values in summaries.items():
        eo = 'N/A' if values['eo'] is None else f"{values['eo']:.6f}"
        lines.append(f"| {name} | {values['accuracy']:.6f} | {values['ba']:.6f} | {eo} | {values['eo_valid_attributes']}/39 | {values['target_sparsity']:.6f} | {values['model_sparsity']:.6f} |")
    lines += ['', 'Young group accuracies: see comparison.csv.',
              'Attribute deltas (pruned minus Dense): attribute_changes.csv.',
              'Prediction CSVs include source index, truth, prediction and positive-class probability.',
              'EO with a missing group/class cell is N/A and excluded from the mean.',
              'BA averages available group/class recalls, preserving the existing convention.',
              'The official train list is split by position; identity overlap is recorded in LFWA.metadata.json.',
              '', '![Training curves](training_curves.png)', '']
    (output / 'README.md').write_text('\n'.join(lines))
