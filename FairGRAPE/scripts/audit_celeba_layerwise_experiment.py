"""Audit completed results, channel eligibility, budgets, greedy traces and masks."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output_dir', type=Path)
    args = p.parse_args()
    out = args.output_dir
    r = json.loads((out/'results.json').read_text())
    assert r['status'] == 'complete'
    methods = r['metadata']['arguments']['methods']
    assert methods and len(set(methods)) == len(methods)
    rounds=r['metadata']['arguments']['rounds']
    assert len(r['rounds']) == rounds*len(methods) and len(r['evaluations']) == 1+2*rounds*len(methods)
    assert {(e['method'],e['round']) for e in r['rounds']} == {(m,n) for m in methods for n in range(1,rounds+1)}
    assert {(e['method'],e['round'],e['stage']) for e in r['evaluations']} == {('dense',0,'dense')} | {(m,n,s) for m in methods for n in range(1,rounds+1) for s in ('pre','post')}
    assert all(len(v['attributes']) == 39 for v in r['evaluations'])
    methods = r['metadata']['arguments']['methods']
    deferred = r['metadata']['arguments'].get('defer_contribution', False)
    if deferred:
        assert all(m in ('A_activation','B_activation') for m in methods)
    train_n = r['metadata']['sample_sizes']['retrain']
    batch = r['metadata']['arguments']['batch_size']
    masks, budgets, first_hash, input_hash = {}, {}, set(), {}
    for entry in r['rounds']:
        method, rnd = entry['method'], entry['round']
        cumulative = masks.setdefault(method, set())
        selected = torch.load(out/f'{method}_round{rnd}_selection.pt', map_location='cpu', weights_only=False)
        if rnd == 1:
            first_hash.add(entry['importance_sha256'])
        actual_removed = []
        for name, d in entry['layers'].items():
            block = '.'.join(name.split('.')[:2])
            n = d['live_before']
            assert d['remove_count'] == n//10
            assert d['candidate_count'] == int(n*.3)
            assert d['protected_count'] == math.ceil(n*.1)
            assert len(d['removed']) == n//10
            assert len(d['kept']) == n-n//10
            assert set(d['protected']) <= set(d['fixed_kept']) <= set(d['kept'])
            assert set(d['removed']) <= set(d['candidates'])
            assert not set(d['removed']) & set(d['kept'])
            assert set(d['fixed_kept']).isdisjoint(d['candidates'])
            assert len(d['fixed_kept'])+len(d['candidates']) == n
            imp = selected['performance'][name]
            ranking = sorted(d['kept']+d['removed'], key=lambda c: (-float(imp[c]), c))
            assert set(ranking[-int(.3*n):]) == set(d['candidates'])
            assert ranking[:math.ceil(.1*n)] == d['protected']
            bk = (rnd, name)
            if bk in budgets:
                assert budgets[bk] == (n, len(d['removed']))
            budgets[bk] = (n, len(d['removed']))
            key = ('activation' if 'activation' in method else 'contribution') + ('_all' if method[0] == 'A' else '_cond')
            diff = selected['signed_differences'][name][key]
            fixed = list(d['fixed_kept'])
            pool = set(d['candidates'])
            def score(keep):
                value = np.abs(diff[..., keep].sum(-1))
                if method[0] == 'B':
                    return float(value.max(axis=1).mean())
                return float(value.mean())
            for choice in d['choices']:
                # Independent full re-sum (runner uses incremental sums).
                trials = [(score(fixed+[c]), -float(imp[c]), c) for c in pool]
                best_score = min(v[0] for v in trials)
                c = choice['kept_channel']
                assert c in pool
                actual = score(fixed+[c])
                assert abs(actual-choice['score']) < 1e-9
                assert actual <= best_score+2e-12, (method, rnd, name, actual, best_score)
                fixed.append(c)
                pool.remove(c)
            assert pool == set(d['removed'])
            assert math.isclose(score(fixed), d['kept_proxy_score'], abs_tol=1e-9)
            for c in d['removed']:
                assert (block,c) not in cumulative
                cumulative.add((block,c))
                actual_removed.append((block,c))
        assert actual_removed == [tuple(x) for x in entry['removed_this_round']]
        assert cumulative == set(map(tuple, entry['removed_cumulative']))
        ckpt = torch.load(out/f'{method}_round{rnd}.pt', map_location='cpu', weights_only=True)
        state = ckpt['model_state']
        for name, mask in state.items():
            if name.endswith('.mask'):
                weight = state[name[:-5]+'.weight']
                assert torch.count_nonzero(weight[mask == 0]) == 0
        for block, channel in cumulative:
            n = int(block.split('.')[1])
            dw = f'{block}.conv.0.0' if n == 1 else f'{block}.conv.1.0'
            project = f'{block}.conv.1' if n == 1 else f'{block}.conv.2'
            assert torch.count_nonzero(state[dw+'.mask'][channel]) == 0
            assert torch.count_nonzero(state[project+'.mask'][:,channel]) == 0
            if n > 1:
                assert torch.count_nonzero(state[f'{block}.conv.0.0.mask'][channel]) == 0
        hist = entry['retraining']['epochs']
        assert len(hist) == 5
        assert entry['retraining']['total_optimizer_steps'] == 5*math.ceil(train_n/batch)
        for h in hist:
            assert h['samples'] == train_n and h['optimizer_steps'] == math.ceil(train_n/batch)
            hk = (rnd,h['epoch'])
            if hk in input_hash:
                assert input_hash[hk] == h['first_batch_sha256']
            input_hash[hk] = h['first_batch_sha256']
        best = min(hist, key=lambda h: (-h['validation']['ACC'], h['validation']['legacy_EO'], h['validation']['loss']))
        assert best['epoch'] == entry['retraining']['best_epoch']
    assert len(first_hash) == 1
    counts = np.asarray(r['metadata']['evaluation_cell_counts_y_by_Male'])
    finite = lambda x: x is not None and math.isfinite(x)
    for ev in r['evaluations']:
        if deferred:
            assert ev['contribution_metrics_status'] == 'deferred_by_user'
            for row in ev['attributes']:
                assert all(row[k] is None for k in ('contribution_all','contribution_positive','contribution_negative'))
            assert all(not any(k.startswith('contribution_') for k in layer) for layer in ev['per_layer_signal_gaps'].values())
        for t, row in enumerate(ev['attributes']):
            assert finite(row['EqualOpportunity']) == bool((counts[t,1] > 0).all())
            assert finite(row['EqualizedOdds']) == bool((counts[t] > 0).all())
            if finite(row['EqualizedOdds']):
                expected = max(abs(row['Male0_TPR']-row['Male1_TPR']), abs(row['Male0_FPR']-row['Male1_FPR']))
                assert abs(expected-row['EqualizedOdds']) < 1e-12
                mean_gap=(abs(row['Male0_TPR']-row['Male1_TPR'])+abs(row['Male0_FPR']-row['Male1_FPR']))/2
                assert abs(mean_gap-row['legacy_EO_mean_gap']) < 1e-12
        for metric, macro in ev['macro'].items():
            ids = [i for i in range(39) if all(finite(x['attributes'][i][metric]) for x in r['evaluations'])]
            assert len(ids) == macro['n_attributes']
            assert macro['attributes'] == [ev['attributes'][i]['attribute'] for i in ids]
            if ids:
                expected = np.mean([ev['attributes'][i][metric] for i in ids])
                assert abs(expected-macro['value']) < 1e-12
            else:
                assert macro['value'] is None
    lines = [f'PASS: {len(methods)} methods x {rounds} rounds x 5 complete epochs',
             f'PASS: {len(r["evaluations"])} evaluations / {39*len(r["evaluations"])} attribute rows, shared per-metric valid sets',
             'PASS: per-layer bottom-30% eligibility, top-10%+middle fixed, live-10% budgets',
             'PASS: independent greedy-score replay, no repeated removal, all masks stay zero',
             'PASS: identical initial importance, augmentation hashes and validation-best rule',
             'PASS: missing cells are N/A, original mean-gap EO and macro arithmetic means verified',
             f'Total elapsed seconds: {r["elapsed_seconds"]:.3f}']
    (out/'verification.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
