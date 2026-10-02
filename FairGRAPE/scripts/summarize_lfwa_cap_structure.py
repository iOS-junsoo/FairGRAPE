#!/usr/bin/env python3
"""Summarize completed paired cap iterations from saved selection records only."""
import argparse
import csv
import datetime
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--initial-dir', type=Path, required=True)
    args = parser.parse_args()
    root, initial = args.run_dir, args.initial_dir
    complete = {arm: {int(p.parent.name.split('_')[-1]): p.parent
                     for p in (root / arm).glob('iteration_*/complete.json')}
                for arm in ('baseline', 'cap10')}
    paired = sorted(set(complete['baseline']) & set(complete['cap10']))
    rows, trajectories = [], []
    for arm, oldarm in [('baseline', 'baseline'), ('cap10', 'strict_cap10')]:
        audit = read(initial / 'run' / oldarm / 'mask_audit.json')
        start = {'layers': [dict(item, removed_channels=item['selected_channels'])
                            for item in audit['layers']]}
        plan_arm = next(item for item in read(initial / 'plan.json')['arms'] if item['name'] == oldarm)
        selected = {tuple(item) for item in plan_arm['selected_channels']}
        survived = ('features.1', 4) not in selected
        survived2 = ('features.2', 20) not in selected
        steps = [(13, start, None)] + [(i, read(complete[arm][i] / 'selection_audit.json'), complete[arm][i]) for i in paired]
        for iteration, audit, directory in steps:
            if directory:
                table = list(csv.DictReader((directory / 'channel_pruning' / f'alpha0.7_iter{iteration:02}_channels.csv').open()))
                marker = next(item for item in table if item['block'] == 'features.1' and item['ch'] == '4')
                survived = marker['status'] in ('active', 'protected')
                marker2 = next(item for item in table if item['block'] == 'features.2' and item['ch'] == '20')
                survived2 = marker2['status'] in ('active', 'protected')
            for item in audit['layers']:
                rows.append({'iteration': iteration, 'arm': arm, 'block': item['block'],
                             'active_before': item['active_before'], 'removed': item['removed_channels'],
                             'active_after': item['active_after'],
                             'fraction_removed': item['removed_channels'] / item['active_before'],
                             'floor': item['minimum_keep']})
            early = {item['block']: item for item in audit['layers'][:2]}
            trajectories.append({'iteration': iteration, 'arm': arm,
                                 'features1_channels': early['features.1']['active_after'],
                                 'features2_channels': early['features.2']['active_after'],
                                 'features1_channel4_survives': survived,
                                 'features2_channel20_survives': survived2,
                                 'max_layer_removal_fraction': max(item['removed_channels'] / item['active_before'] for item in audit['layers'])})
    out = root / 'analysis'; out.mkdir(exist_ok=True)
    with (out / 'structure_snapshot_20260917.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'last_completed_pair': max([13] + paired), 'early_layer_trajectories': trajectories,
              'note': 'Completed paired iterations only; observational structure differences are not causal mediation estimates.'}
    (out / 'structure_snapshot_20260917.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['**10% 상한의 채널 보존 경로 — 완료된 짝 단계만 집계**', '',
             '| 단계 | 군 | f1 잔존 | f2 잔존 | f1 채널4 생존 | f2 채널20 생존 | 최대 층별 이번 제거율 |',
             '|---|---|---:|---:|---|---|---:|']
    for row in sorted(trajectories, key=lambda row: (row['iteration'], row['arm'])):
        lines.append('| {iteration} | {arm} | {features1_channels} | {features2_channels} | {features1_channel4_survives} | {features2_channel20_survives} | {ratio:.2%} |'.format(**row, ratio=row['max_layer_removal_fraction']))
    lines += ['', '상한은 매 단계의 제거 속도를 제한한다. 같은 채널을 영구적으로 보호하는 규칙은 아니므로 초기에 보존한 채널이 후속 단계에서 제거될 수 있다. 10% 내림으로 활성 채널이 9개 이하인 층에서는 제거 상한이 0이 되는 별도 효과도 있다. 이후 층별 차이는 각 군의 재학습·점수 재추정을 함께 포함하므로 특정 채널의 효과로 인과 분해하지 않는다.']
    (out / 'structure_snapshot_20260917.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
