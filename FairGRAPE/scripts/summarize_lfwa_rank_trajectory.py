#!/usr/bin/env python3
"""Read completed rank/max continuations without changing or evaluating models."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank-dir', type=Path, required=True)
    parser.add_argument('--max-dir', type=Path, required=True)
    parser.add_argument('--rank-step13', type=Path, required=True)
    parser.add_argument('--max-step13', type=Path, required=True)
    args = parser.parse_args()
    roots = [(args.max_dir, args.max_step13, 'baseline', 'baseline'),
             (args.max_dir, args.max_step13, 'cap10', 'strict_cap10'),
             (args.rank_dir, args.rank_step13, 'rank_no_cap', 'rank_shared')]
    records = []
    for root, initial, arm, initial_arm in roots:
        assert read(initial / initial_arm / 'complete.json')['status'] == 'complete'
        for phase in ['post_prune', 'post_retrain']:
            metric = read(initial / initial_arm / phase / 'summary.json')
            records.append({'iteration': 13, 'arm': arm, 'phase': phase, **metric})
        last = 13
        for path in sorted((root / arm).glob('iteration_*/complete.json')):
            data = read(path)
            assert data['arm'] == arm and data['iteration'] == last + 1
            assert data['score_seed'] == 42 + data['iteration']
            assert data['training_seed'] == 10042 + data['iteration']
            for phase in ['post_prune', 'post_retrain']:
                records.append({'iteration': data['iteration'], 'arm': arm,
                                'phase': phase, **data[phase]})
            last = data['iteration']
    frame = pd.DataFrame(records)
    for column in list(frame.columns):
        if any(isinstance(value, (dict, list)) for value in frame[column]):
            frame = frame.drop(columns=[column])
    output = args.rank_dir / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'all_completed_metrics.csv', index=False)
    final = frame[frame.phase == 'post_retrain']
    base = final[final.arm == 'baseline'].set_index('iteration')
    rank = final[final.arm == 'rank_no_cap'].set_index('iteration')
    pairs = base.add_suffix('_baseline').join(rank.add_suffix('_rank'), how='inner')
    for field in ['accuracy', 'ba', 'eo', 'model_sparsity',
                  'constant_prediction_attributes', 'near_constant_99pct_attributes']:
        pairs[field + '_delta'] = pairs[field + '_rank'] - pairs[field + '_baseline']
    pairs.to_csv(output / 'paired_metrics.csv')
    latest = pairs.iloc[-1]
    gate = None
    criteria = None
    if 25 in pairs.index:
        point = pairs.loc[25]
        recent = pairs.loc[21:25]
        assert list(recent.index) == list(range(21, 26))
        criteria = {'ba': bool(point.ba_delta >= .01),
                    'accuracy': bool(point.accuracy_delta >= -.005),
                    'near_constant': bool(point.near_constant_99pct_attributes_delta <= 0),
                    'recent_ba_positive': bool((recent.ba_delta > 0).sum() >= 3),
                    'sparsity_difference': bool(abs(point.model_sparsity_delta) <= .0005)}
        gate = all(criteria.values())
    report = {'last_paired_iteration': int(pairs.index[-1]),
              'latest_rank_minus_baseline': {key: float(latest[key + '_delta']) for key in
                  ['accuracy', 'ba', 'eo', 'model_sparsity', 'near_constant_99pct_attributes']},
              'step25_utility_resource_gate': gate, 'step25_criteria': criteria,
              'note': 'Same practical utility gate as the max/cap trajectory. EO is reported independently. No whole60 or fairness-success claim.'}
    (output / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for arm, label, color in [('baseline', 'Max baseline', '#bc4b38'),
                               ('cap10', 'Max + cap10', '#357a54'),
                               ('rank_no_cap', 'Rank, no cap', '#286bb0')]:
        values = final[final.arm == arm].sort_values('iteration')
        for ax, field in zip(axes, ['accuracy', 'ba', 'eo']):
            ax.plot(values.model_sparsity * 100, values[field], '-o', label=label,
                    color=color, markersize=3)
    for ax, label in zip(axes, ['Accuracy', 'Group/class balanced accuracy', 'Hard EO gap']):
        ax.set_ylabel(label)
        ax.set_xlabel('Whole-model parameter sparsity (%)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('LFWA alpha=0.7: same source12; five epochs per step; validation only')
    for extension in ['png', 'pdf']:
        fig.savefig(output / ('trajectory.' + extension), dpi=180)
    plt.close(fig)
    lines = ['완료 단계만 집계. 각 모델의 현재 상태에서 P/φ를 다시 계산한다.', '',
             '| 단계 | Acc 기존 / rank | BA 기존 / rank | EO 기존 / rank | 희소도 차이 (%p) |',
             '|---:|---:|---:|---:|---:|']
    for step, row in pairs.iterrows():
        values = [f'{row[key + "_baseline"]:.6f} / {row[key + "_rank"]:.6f}'
                  for key in ['accuracy', 'ba', 'eo']]
        lines.append(f'| {step} | ' + ' | '.join(values) + f' | {row.model_sparsity_delta * 100:+.5f} |')
    lines += ['', '25단계 성능 진행 기준: ' + str(gate) + ' (None=미완료).', '',
              'Rank는 P/φ의 상대 간격과 유효 α도 바꾼다. 개선되더라도 max 정규화만이 유일한 원인이라는 뜻은 아니다. '
              'EO 감소와 예측 능력 보존을 함께 판단하며, 이 표는 전체60%나 test 검증을 포함하지 않는다.']
    (output / 'progress.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
