"""Independently check a completed smoke report and final masks (stdlib + torch)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_dir', type=Path)
    args = p.parse_args()
    result = json.loads((args.run_dir / 'results.json').read_text())
    assert result['status'] == 'complete'
    records, rounds = result['evaluations'], result['rounds']
    methods = result['metadata']['arguments']['methods']
    assert len(records) == 1 + 6 * len(methods)
    assert len(rounds) == 3 * len(methods)
    assert len({r['importance_sha256'] for r in rounds if r['round'] == 1}) == 1
    subsets = {}
    for split in ('selection', 'retrain', 'evaluation'):
        with (args.run_dir / f'{split}_samples.csv').open() as f:
            subsets[split] = list(csv.DictReader(f))
    names = {k: {r['image_name'] for r in rows} for k, rows in subsets.items()}
    assert names['selection'].isdisjoint(names['retrain'])
    assert names['selection'].isdisjoint(names['evaluation'])
    assert names['retrain'].isdisjoint(names['evaluation'])
    truth = subsets['evaluation']
    attrs = result['metadata']['annotation']['attributes_in_head_order']
    assert len(attrs) == len(set(attrs)) == 39
    assert 'Male' not in attrs and 'gender' not in attrs

    def same(got, expected):
        if expected is None:
            assert got is None, (got, expected)
        else:
            assert got is not None and math.isclose(got, expected, rel_tol=1e-10, abs_tol=1e-12), (got, expected)

    def div(x, y):
        return x / y if y else None

    for record in records:
        assert [r['attribute'] for r in record['attributes']] == attrs
        for t, row in enumerate(record['attributes']):
            group = []
            for g in (0, 1):
                r = {k: row[f'Male{g}_{k}'] for k in ('TP', 'FN', 'FP', 'TN', 'ACC', 'TPR', 'FPR', 'PPR')}
                total = r['TP'] + r['FN'] + r['FP'] + r['TN']
                positive = r['TP'] + r['FN']
                negative = r['FP'] + r['TN']
                assert positive == sum(int(x[attrs[t]]) == 1 and int(x['gender']) == g for x in truth)
                assert negative == sum(int(x[attrs[t]]) == 0 and int(x['gender']) == g for x in truth)
                same(r['ACC'], div(r['TP'] + r['TN'], total))
                same(r['TPR'], div(r['TP'], positive))
                same(r['FPR'], div(r['FP'], negative))
                same(r['PPR'], div(r['TP'] + r['FP'], total))
                group.append(r)
            dt = None if any(r['TPR'] is None for r in group) else abs(group[0]['TPR'] - group[1]['TPR'])
            df = None if any(r['FPR'] is None for r in group) else abs(group[0]['FPR'] - group[1]['FPR'])
            same(row['EqualOpportunity'], dt)
            same(row['EqualizedOdds'], None if dt is None or df is None else max(dt, df))
            same(row['legacy_EO_mean_gap'], None if dt is None or df is None else (dt+df)/2)
            dp = None if any(r['PPR'] is None for r in group) else abs(group[0]['PPR']-group[1]['PPR'])
            same(row['DemographicParity'], dp)
            tp, fn, fp, tn = [sum(r[k] for r in group) for k in ('TP', 'FN', 'FP', 'TN')]
            same(row['ACC'], (tp+tn)/(tp+tn+fp+fn))
            same(row['BA'], None if not (tp+fn and tn+fp) else (tp/(tp+fn)+tn/(tn+fp))/2)
            same(row['activation_all_global_ref'], record['global_activation_gap'])
            for label, col in ((1, 'positive'), (0, 'negative')):
                valid = all(any(int(x[attrs[t]]) == label and int(x['gender']) == g for x in truth) for g in (0, 1))
                assert (row[f'activation_{col}'] is not None) == valid
                assert (row[f'contribution_{col}'] is not None) == valid
        for key, item in record['macro'].items():
            ids = [i for i in range(39) if all(r['attributes'][i][key] is not None for r in records)]
            assert item['n_attributes'] == len(ids)
            assert item['attributes'] == [attrs[i] for i in ids]
            same(item['value'], sum(record['attributes'][i][key] for i in ids)/len(ids) if ids else None)
        assert 'activation_all_global_ref' not in record['macro']

    import torch
    checkpoints = {}
    for method in methods:
        trajectory = [r for r in rounds if r['method'] == method]
        assert [r['round'] for r in trajectory] == [1, 2, 3]
        removed = []
        for r in trajectory:
            for choice in r['choices']:
                assert choice['selected'] not in removed
                trials = choice['trials']
                winner = min(trials, key=lambda x: (round(x['score_after'], 12), x['performance'], int(x['block'].split('.')[1]), x['channel']))
                assert choice['selected'] == [winner['block'], winner['channel']]
                removed.append(choice['selected'])
            assert removed == r['removed_cumulative']
            assert r['retraining']['steps'] == result['metadata']['arguments']['retrain_steps']
        saved = torch.load(args.run_dir / f'{method}_round3.pt', weights_only=True, map_location='cpu')
        assert [list(c) for c in saved['removed_channels']] == removed
        state = saved['model_state']
        masked = 0
        for k, value in state.items():
            if not k.endswith('.mask'):
                continue
            weight = state[k[:-4] + 'weight']
            assert torch.count_nonzero(weight[value == 0]) == 0
            masked += int((value == 0).sum())
        assert masked == trajectory[-1]['masked_weights']
        checkpoints[method] = dict(removed_channels=removed, masked_weights=masked)
    checkpoint = Path(result['metadata']['arguments']['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == result['metadata']['checkpoint_sha256']
    summary = dict(status='PASS', evaluations=len(records), per_attribute_rows=len(records)*39,
                   rounds=len(rounds), selection_samples=len(subsets['selection']),
                   evaluation_samples=len(truth), independent_checks='confusion counts, missing values, all macro denominators, shared initial importance, trial tie rule, cumulative masks, no regrowth, unchanged initial checkpoint',
                   final_checkpoints=checkpoints)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    (args.run_dir / 'verification.txt').write_text(text + '\n')
    print(text)
    final = [r for r in records if r['method'] == 'dense' or (r['round'] == 3 and r['stage'] == 'post')]
    lines = [f"{'Metric':28s} {'n':>5s} " + ' '.join(f"{r['method']:>17s}" for r in final)]
    lines.append(f"{'activation_global':28s} {'1':>5s} " + ' '.join(f"{r['global_activation_gap']:17.8g}" for r in final))
    for key, item in final[0]['macro'].items():
        values = [('N/A' if r['macro'][key]['value'] is None else format(r['macro'][key]['value'], '.8g')) for r in final]
        lines.append(f"{key:28s} {str(item['n_attributes'])+'/39':>5s} " + ' '.join(f"{v:>17s}" for v in values))
    table = '\n'.join(lines) + '\n'
    (args.run_dir / 'comparison.txt').write_text(table)
    print('\nFINAL COMPARISON (all macro metrics)\n' + table)


if __name__ == '__main__':
    main()
