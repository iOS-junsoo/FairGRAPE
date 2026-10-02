#!/usr/bin/env python3
"""Compare the completed rank step13 with the original paired max controls."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank-dir', type=Path, required=True)
    parser.add_argument('--controls-dir', type=Path, required=True)
    args = parser.parse_args()
    rank_root, controls = args.rank_dir.resolve(), args.controls_dir.resolve()
    rank_run = rank_root / 'run'
    assert read(rank_run / 'complete.json')['status'] == 'complete'
    assert read(controls / 'complete.json')['status'] == 'complete'
    assert read(rank_run / 'gpu_rank_preflight.json')['status'] == 'passed'
    new_meta, old_meta = read(rank_run / 'diagnostic_run.json'), read(controls / 'diagnostic_run.json')
    for key in ['source_checkpoint_sha256', 'source_channel_csv_sha256', 'iteration', 'alpha',
                'seed', 'batch', 'retrain_epochs', 'retrain_lr', 'layer_floor', 'keep_per_iter', 'protection_ratio']:
        assert new_meta['plan'][key] == old_meta['plan'][key], key
    assert new_meta['plan']['normalization'] == 'rank'
    assert old_meta['plan']['normalization'] == 'max'
    assert new_meta['source_sha256'] == old_meta['source_sha256'], 'Original implementation changed'
    sources = [('max_baseline', controls / 'baseline'), ('max_cap10', controls / 'strict_cap10'),
               ('rank_shared', rank_run / 'rank_shared')]
    before = read(rank_run / 'before_prune/summary.json')
    old_before = read(controls / 'before_prune/summary.json')
    for key in ['accuracy', 'ba', 'eo']:
        assert abs(before[key] - old_before[key]) < 1e-10
    rows, trained, directories = [], {}, {}
    for arm, directory in sources:
        assert read(directory / 'complete.json')['status'] == 'complete'
        directories[arm] = directory
        for phase in ['post_prune', 'post_retrain']:
            metric = read(directory / phase / 'summary.json')
            rows.append({'arm': arm, 'phase': phase, **metric})
            if phase == 'post_retrain': trained[arm] = metric
    out = rank_root / 'analysis'; out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'all_metrics.csv', index=False)
    rank = trained['rank_shared']
    delta_keys = ['accuracy', 'ba', 'eo', 'model_sparsity', 'constant_prediction_attributes', 'near_constant_99pct_attributes']
    comparisons = {arm: {key: rank[key] - metric[key] for key in delta_keys}
                   for arm, metric in trained.items() if arm != 'rank_shared'}
    delta = comparisons['max_baseline']
    criteria = {'ba': delta['ba'] >= .005, 'accuracy': delta['accuracy'] >= -.002,
                'near_constant': delta['near_constant_99pct_attributes'] <= 0}
    source = directories['rank_shared'] / 'retrained.pt'
    baseline_source = directories['max_baseline'] / 'retrained.pt'
    gate = {'schema_version': 1, 'passes': all(criteria.values()), 'criteria': criteria,
            'source_checkpoint': str(source), 'source_checkpoint_sha256': sha(source),
            'reference_baseline_checkpoint': str(baseline_source),
            'reference_baseline_checkpoint_sha256': sha(baseline_source),
            'reference_metrics': trained['max_baseline'], 'rank_metrics': rank,
            'deltas_rank_minus_baseline': delta,
            'thresholds': {'delta_ba_min': .005, 'delta_accuracy_min': -.002, 'near_constant_delta_max': 0},
            'evaluated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'scope': 'utility-only resource-allocation gate; EO reported separately, not fairness success',
            'plan_sha256': sha(rank_root / 'plan.json'),
            'gpu_rank_preflight_sha256': sha(rank_run / 'gpu_rank_preflight.json'),
            'comparator': 'Same source12 and paired training seed42, same original five-epoch implementation; rank normalization changes selection.'}
    write(out / 'continuation_gate.json', gate)
    write(out / 'comparison.json', {'rank_minus_controls': comparisons, 'trained': trained,
                                    'source_validation_identical': True,
                                    'original_implementation_hashes_identical': True,
                                    'same_nominal_not_exact_actual_budget': True})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    colors = {'max_baseline': '#bc4b38', 'max_cap10': '#357a54', 'rank_shared': '#286bb0'}
    labels = {'max_baseline': 'Max baseline', 'max_cap10': 'Max + cap10', 'rank_shared': 'Rank (cap inactive)'}
    for metric, ax in zip(['accuracy', 'ba', 'eo'], axes):
        for arm, directory in sources:
            epochs = pd.read_csv(directory / 'epochs.csv')
            epochs = epochs[(epochs.phase == 'val') & (epochs.prune_iteration == 13)]
            assert len(epochs) == 5
            post = read(directory / 'post_prune/summary.json')
            values = [before[metric], post[metric]] + epochs[metric].tolist()
            ax.plot([-1, 0, 1, 2, 3, 4, 5], values, '-o', color=colors[arm], label=labels[arm], markersize=4)
            matches = epochs[(epochs.accuracy - trained[arm]['accuracy']).abs() < 1e-10]
            matches = matches[(matches.ba - trained[arm]['ba']).abs() < 1e-10]
            matches = matches[(matches.eo - trained[arm]['eo']).abs() < 1e-10]
            assert len(matches) >= 1
            selected = list(epochs.index).index(matches.index[-1]) + 1
            ax.scatter([selected], [trained[arm][metric]], marker='*', s=95, color=colors[arm], zorder=5)
        ax.set_xticks([-1, 0, 1, 2, 3, 4, 5], ['Source', 'Pruned', '1', '2', '3', '4', '5'])
        ax.set_xlabel('State / retraining epoch; star = selected checkpoint')
        ax.set_ylabel({'accuracy': 'Accuracy', 'ba': 'Group/class balanced accuracy', 'eo': 'Hard equalized-odds gap'}[metric])
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle('LFWA step13: same source and five-epoch training; validation only')
    fig.savefig(out / 'paired_rank_step13.png', dpi=170)
    fig.savefig(out / 'paired_rank_step13.pdf')
    plt.close(fig)
    lines = ['**Rank 정규화: 동일 source12의 단일 단계 비교**', '',
             '| 5 epoch 후 선택 모델 | Accuracy | BA | EO↓ | 전체 모델 희소도 | 근상수 속성 |',
             '|---|---:|---:|---:|---:|---:|']
    for arm, metric in trained.items():
        lines.append(f"| {arm} | {metric['accuracy']:.6f} | {metric['ba']:.6f} | {metric['eo']:.6f} | {metric['model_sparsity']:.6%} | {metric['near_constant_99pct_attributes']} |")
    lines += ['', '사전 성능 진행 기준: ' + ('통과' if gate['passes'] else '미통과') + '. 이는 자원 배분 기준이며 공정성 성공이나 통계적 유의성 판단이 아니다.',
              '', 'rank의 cap 유무는 이 단계에서 같은167채널 선택이라 한 번만 학습했다. 다음 단계에서도 같다는 뜻은 아니다. 기존 max 두 대조와 source·학습 설정·원본 구현 해시가 같고 GPU rank 선택도 CPU plan과 같음을 확인했다. Raw score는 기존 반올림 CSV를 사용했다. 전체60%나 새 seed의 결과는 이 표에 포함되지 않는다.']
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'passes': gate['passes'], 'criteria': criteria, 'rank_minus_controls': comparisons,
                      'gate': str(out / 'continuation_gate.json')}, indent=2))


if __name__ == '__main__':
    main()
