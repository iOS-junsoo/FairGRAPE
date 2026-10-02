"""Layer-independent keep-greedy CelebA experiment; three rounds, five epochs."""
import argparse
import contextlib
import copy
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import shlex
import sys
import time

import celeba_fourway_smoke as base
from celeba_fourway_smoke import (torch, np, pd, models, dataset, DataLoader,
    TensorDataset, ROOT, METHODS, SLICES)

DEFINITIONS = """LAYERWISE KEEP-GREEDY / 레이어별 남길 채널 선택
TASK: CSV columns [2:41], 39 attributes excluding Male; two logits per task.
Male is only CSV gender, checked against original annotations for ALL rows.
Loss: existing loss_multi_tasks, mean of 39 task cross-entropies, no Male/GRL.
Predictions: original pairwise softmax probabilities, argmax (tie -> class 0).
Checkpoint has no embedded attribute metadata; head order follows the verified
project CSV and training convention, not newly invented/reordered attributes.

PERFORMANCE: unchanged prune.compute_importance, batch average of (w*dL/dw)^2
(square BEFORE averaging batches), using cached selection evaluation transforms.
Original channel aggregation: expansion+depthwise output-axis weight means plus
projection input-axis weight mean. NO activation substitution or normalization.
Analysis: depthwise Conv outputs BEFORE BN in features.1 through features.17.
Every round uses CURRENT surviving channels, separately in EVERY layer.
N live channels: protect ceil(.10*N); candidate pool=floor(.30*N) lowest
importance; remove=floor(.10*N); fixed keep=all channels outside candidate pool.
Performance ranking: descending importance, then ascending original channel ID.
Thus top 10% AND middle 60% are fixed. Starting with fixed channels, greedily
ADD candidate channels until N-floor(.10*N) channels are kept. Unchosen pool
channels are removed. All candidate additions are checked at every greedy step.
Tie key: objective rounded to 12 decimals, HIGHER importance, LOWER channel ID.
All 17 decisions use the same round-start model, then masks are applied together.
Three rounds ONLY. Every method has identical layer budgets, starting checkpoint,
data, optimizer, round/epoch seeds, transforms and validation selection rules.

SIGNALS: a[i,l,c]=mean_HW A[i,l,c,h,w], retaining SIGNED pre-BN activations.
q[i,t,l,c]=mean_HW(A * d z[i,t,positive]/d A), retaining sign; z is the RAW
positive logit, NOT probability and NOT positive-minus-negative logit margin.
The SAME positive logit is used for both positive and negative labelled samples.
Group means are per IMAGE: sum over images / count of images in that group/cell.
d_l,c(condition)=mean_group0(signal)-mean_group1(signal).
For kept set K: D_l(K,condition)=abs(sum_{c in K} d_l,c(condition)).
Sum channels BEFORE abs so opposing channel contributions can compensate within
a layer. No cross-layer cancellation; no abs(A), activation-mass objective,
dense ratio target or channel-count normalization. Removal can lower a summed
gap simply by reducing signal, so this proxy alone does not establish fairness.
A_activation: D_l(K,all), ONE global objective, never 39 duplicated scores.
B_activation: mean over ALL 39 t of max(D_l(K,y_t=1), D_l(K,y_t=0)).
A_contribution: mean over ALL 39 t of D_l(K,q_t,all).
B_contribution: mean over ALL 39 t of max(D_l(K,q_t,y_t=1),D_l(K,q_t,y_t=0)).
B minimizes the larger of the two gaps, not their difference. One conditional
gap can rise while max falls; both values are reported. Selection requires all
four cells for each attribute. Each greedy choice uses FROZEN round-start
channel statistics: it is a proxy, not an exact forward of every kept submodel.
After actual pruning and after retraining, signals are recomputed on that model.

RETRAIN: five COMPLETE epochs over the recorded retraining subset per round,
fresh Adam each round (lr=1e-5, betas=.9,.999, eps=1e-8, weight_decay=0), batch64,
repository ImgAugTransform, shuffle=False. Independent worker seeds and same
round/epoch seeds across methods; no test-set tuning. Validation best of five:
higher ACC, then lower LEGACY mean-gap EO, then lower task loss. Restore best
before the next round. No additional training after the third round.
Masks: original expansion-output/depthwise-output/projection-input masks,
gradient masking AND post-step masking prevent regrowth; no tensor compaction.
The runner preserves original math but enables deterministic cuDNN kernels;
legacy train_model0 disables cuDNN. FP32, no AMP or TF32. This backend difference
is shared across all methods. Checkpoints after each epoch and each round.

EVALUATION: dense + pre/post retraining of each round, identical held-out test
images, separate from selection and validation. ACC=(TP+TN)/N;
BA=(global TPR+global TNR)/2. Group ACC,TPR=TP/P,FPR=FP/Nneg,PPR=(TP+FP)/Ngroup.
EqualOpportunity=abs(TPR0-TPR1); EqualizedOdds=max(abs(TPR0-TPR1),abs(FPR0-FPR1));
DemographicParity=abs(PPR0-PPR1). Legacy EO uses MEAN of TPR/FPR gaps, also listed.
Legacy BA averages nonempty label/group cell accuracies; requested BA balances
the TWO labels globally instead. Missing denominator -> N/A, never zero.
Signal gaps reported for evaluation are arithmetic means of D_l over 17 layers.
Unconditional activation is one GLOBAL value (per-attribute rows only reference
it); conditional activation and all three contribution gaps are per attribute.
All macro metrics use arithmetic means over common valid attributes across ALL
compared states, with metric-specific n/39 and exact names. No zero imputation.
Small smoke samples are not a full-data accuracy/fairness experiment.
"""


def signed_difference(sums, counts):
    means = sums / counts[..., None].clamp_min(1)
    delta = means[..., 0, :] - means[..., 1, :]
    return torch.where((counts > 0).all(-1)[..., None], delta,
                       torch.full_like(delta, float('nan')))


def collect(model, loader, contribution=True, progress=None):
    """Per-image group means; retain channel-wise signs for keep-greedy."""
    model.eval()
    layers = base.reference_layers(model)
    device = next(model.parameters()).device
    acts, handles, sa, sq = {}, [], {}, {}
    counts = torch.zeros(158, dtype=torch.float64, device=device)
    preds, truth, groups = [], [], []
    start = last = time.monotonic()
    for name, layer in layers.items():
        handles.append(layer.register_forward_hook(
            lambda m, inp, out, name=name: acts.__setitem__(name, out)))
        sa[name] = torch.zeros(158, layer.out_channels, dtype=torch.float64, device=device)
        if contribution:
            sq[name] = torch.zeros(39, 6, layer.out_channels, dtype=torch.float64, device=device)
    try:
        with torch.set_grad_enabled(contribution):
            for bi, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                logits = model(x).reshape(-1, 39, 2)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite logits')
                preds.append(logits.detach().softmax(2).argmax(2).cpu().numpy())
                truth.append(y[:, :39].cpu().numpy())
                groups.append(y[:, 39].cpu().numpy())
                member = base.membership(y)
                counts += member.sum(1)
                activations = list(acts[n] for n in layers)
                for n, a in zip(layers, activations):
                    sa[n] += member @ a.detach().mean((2, 3)).double()
                if contribution:
                    for t in range(39):
                        grads = torch.autograd.grad(logits[:, t, 1].sum(), activations,
                                                    retain_graph=t < 38)
                        cells = torch.cat((member[:2], member[2+4*t:6+4*t]))
                        for n, a, g in zip(layers, activations, grads):
                            sq[n][t] += cells @ (a.detach()*g.detach()).mean((2, 3)).double()
                acts.clear()
                if progress and time.monotonic()-last > 45:
                    print(f'  {progress}: signals batch {bi+1}/{len(loader)}', flush=True)
                    last = time.monotonic()
    finally:
        for h in handles:
            h.remove()
    deltas = {}
    for n in layers:
        d = dict(activation_all=signed_difference(sa[n][:2], counts[:2]),
                 activation_cond=signed_difference(sa[n][2:].reshape(39, 2, 2, -1),
                                                  counts[2:].reshape(39, 2, 2)))
        if contribution:
            d.update(contribution_all=signed_difference(sq[n][:, :2], counts[:2].expand(39, 2)),
                     contribution_cond=signed_difference(sq[n][:, 2:].reshape(39, 2, 2, -1),
                                                        counts[2:].reshape(39, 2, 2)))
        deltas[n] = {k: v.cpu().numpy() for k, v in d.items()}
    return dict(deltas=deltas, truth=np.concatenate(truth), predictions=np.concatenate(preds),
                groups=np.concatenate(groups), cell_counts=counts[2:].reshape(39, 2, 2).cpu().numpy(),
                seconds=time.monotonic()-start)


def objective(values, method):
    """Last axis is candidate trial axis; preceding axes are tasks/conditions."""
    if method == 'A_activation':
        return np.abs(values)
    if method == 'A_contribution':
        return np.abs(values).mean(axis=0)
    if method in ('B_activation', 'B_contribution'):
        return np.abs(values).max(axis=1).mean(axis=0)
    raise ValueError(method)


def greedy_keep(importance, live, deltas, method):
    ids = np.flatnonzero(live).tolist()
    ranking = sorted(ids, key=lambda c: (-float(importance[c]), c))
    n = len(ids)
    protected_n, pool_n, remove_n = math.ceil(.10*n), math.floor(.30*n), math.floor(.10*n)
    if n < 1 or pool_n < remove_n or protected_n + pool_n > n:
        raise ValueError(f'Unsupported live channel budget {n}')
    fixed, pool = ranking[:n-pool_n], ranking[n-pool_n:]
    key = ('activation' if 'activation' in method else 'contribution') + ('_all' if method.startswith('A_') else '_cond')
    d = np.asarray(deltas[key], dtype=np.float64)
    if not np.isfinite(d[..., ids]).all():
        raise ValueError('Selection must define every required group/attribute cell')
    current = d[..., fixed].sum(axis=-1)
    initial = float(objective(current[..., None], method)[0])
    available, chosen, trajectory = list(pool), [], []
    for _ in range(pool_n-remove_n):
        scores = objective(current[..., None] + d[..., available], method)
        idx = min(range(len(available)), key=lambda j: (round(float(scores[j]), 12),
                  -float(importance[available[j]]), available[j]))
        channel = available.pop(idx)
        current = current + d[..., channel]
        chosen.append(channel)
        trajectory.append(dict(kept_channel=channel, score=float(scores[idx])))
    kept = sorted(fixed+chosen)
    assert len(kept) == n-remove_n and set(ranking[:protected_n]) <= set(kept)
    assert set(available) <= set(pool) and len(available) == remove_n
    return dict(live_before=n, protected_count=protected_n, candidate_count=pool_n,
                remove_count=remove_n, protected=ranking[:protected_n], fixed_kept=sorted(fixed),
                candidates=sorted(pool), kept=kept, removed=sorted(available),
                importance_by_channel={str(c): float(importance[c]) for c in ids},
                fixed_only_score=initial, all_live_score=float(objective(d[..., ids].sum(-1)[..., None], method)[0]),
                kept_proxy_score=float(objective(current[..., None], method)[0]),
                kept_signed_totals=current.tolist(), choices=trajectory)


def evaluate(model, loader, attrs, method, round_index, stage, contribution=True):
    s = collect(model, loader, contribution, f'{method} r{round_index} {stage}')
    active = base.active_masks(model)
    keys = ['activation_all', 'activation_cond']
    if contribution:
        keys += ['contribution_all', 'contribution_cond']
    gaps = {k: [] for k in keys}
    layer_gaps = {}
    for n, ds in s['deltas'].items():
        ids = active[n].cpu().numpy()
        layer_gaps[n] = {k: np.abs(d[..., ids].sum(-1)) for k, d in ds.items()}
        for k in gaps:
            gaps[k].append(layer_gaps[n][k])
    gaps = {k: np.stack(v).mean(0) for k, v in gaps.items()}
    rows = base.classification_metrics(s['truth'], s['predictions'], s['groups'], attrs)
    for t, row in enumerate(rows):
        row.update(activation_all_global_ref=float(gaps['activation_all']),
                   activation_positive=gaps['activation_cond'][t, 1],
                   activation_negative=gaps['activation_cond'][t, 0],
                   contribution_all=gaps['contribution_all'][t] if contribution else float('nan'),
                   contribution_positive=gaps['contribution_cond'][t, 1] if contribution else float('nan'),
                   contribution_negative=gaps['contribution_cond'][t, 0] if contribution else float('nan'))
    return dict(method=method, round=round_index, stage=stage, attributes=rows,
                global_activation_gap=float(gaps['activation_all']),
                per_layer_signal_gaps=layer_gaps, seconds=s['seconds'],
                contribution_metrics_status='computed' if contribution else 'deferred_by_user')


def subsets(frames, attrs, args):
    train = frames['train']
    order = np.random.RandomState(args.seed).permutation(len(train)).tolist()
    chosen = order[:args.selection_samples]
    for a in attrs:
        for y in (0, 1):
            for g in (0, 1):
                mask = (train[a].to_numpy() == y) & (train.gender.to_numpy() == g)
                if not mask[chosen].any():
                    extra = next((i for i in order if mask[i]), None)
                    if extra is None:
                        raise ValueError(f'Empty TRAIN cell {a},y={y},Male={g}')
                    chosen.append(extra)
    def take(frame, count, seed):
        if count == 0:
            return frame.reset_index(drop=True)
        if count > len(frame):
            raise ValueError('Requested samples exceed split')
        ids = np.random.RandomState(seed).permutation(len(frame))[:count]
        return frame.iloc[sorted(ids)].reset_index(drop=True)
    result = dict(selection=train.iloc[chosen].reset_index(drop=True),
                  retrain=take(train, args.train_samples, args.seed+2),
                  validation=take(frames['val'], args.val_samples, args.seed+3),
                  evaluation=take(frames['test'], args.eval_samples, args.seed+1))
    assert set(result['evaluation'].image_name).isdisjoint(set(train.image_name))
    assert set(result['evaluation'].image_name).isdisjoint(set(frames['val'].image_name))
    return result


def cached(frame, attrs, batch, workers):
    _, ds = dataset.make_datasets(frame, frame, False, batch, attrs+['gender'])
    loader = DataLoader(ds, batch_size=batch, num_workers=workers, shuffle=False,
                        worker_init_fn=dataset.seed_worker)
    xx, yy = [], []
    for x, y in loader:
        xx.append(x)
        yy.append(y)
    return DataLoader(TensorDataset(torch.cat(xx), torch.cat(yy)), batch_size=batch, shuffle=False)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def atomic_save(obj, path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def validation(model, loader, attrs):
    device = next(model.parameters()).device
    model.eval()
    pp, yy, gg, total, n = [], [], [], 0., 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z = model(x)
            loss = base.loss_multi_tasks(z, y, torch.nn.CrossEntropyLoss(), SLICES)
            total += float(loss)*len(x)
            n += len(x)
            pp.append(z.reshape(-1, 39, 2).softmax(2).argmax(2).cpu().numpy())
            yy.append(y[:, :39].cpu().numpy())
            gg.append(y[:, 39].cpu().numpy())
    rows = base.classification_metrics(np.concatenate(yy), np.concatenate(pp), np.concatenate(gg), attrs)
    eo = [r['legacy_EO_mean_gap'] for r in rows if math.isfinite(r['legacy_EO_mean_gap'])]
    return dict(ACC=float(np.mean([r['ACC'] for r in rows])),
                legacy_EO=float(np.mean(eo)) if eo else float('nan'), loss=total/n,
                legacy_EO_n=len(eo), samples=n)


def retrain(model, frame, val_loader, attrs, args, method, rnd, out):
    start = time.monotonic()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.retrain_lr)
    device = next(model.parameters()).device
    history, best_key, best_state, best_epoch = [], None, None, None
    for epoch in range(1, args.retrain_epochs+1):
        epoch_start = time.monotonic()
        seed = args.seed + 1000*rnd + epoch
        base.seed_all(seed)
        # Build a fresh augmentation pipeline after reseeding, identically in all methods.
        ds, _ = dataset.make_datasets(frame, frame, False, args.batch_size, attrs+['gender'])
        gen = torch.Generator().manual_seed(seed)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, worker_init_fn=dataset.seed_worker, generator=gen,
            pin_memory=device.type == 'cuda')
        model.train()
        total, samples, steps, last = 0., 0, 0, time.monotonic()
        first_hash = None
        for x, y in loader:
            if first_hash is None:
                first_hash = hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            z = model(x)
            loss = base.loss_multi_tasks(z, y, torch.nn.CrossEntropyLoss(), SLICES)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            base.enforce_masks(model)
            total += float(loss)*len(x)
            samples += len(x)
            steps += 1
            if time.monotonic()-last > 45:
                print(f'  {method} r{rnd} epoch{epoch}: {samples}/{len(frame)} images', flush=True)
                last = time.monotonic()
        assert samples == len(frame) and steps == math.ceil(len(frame)/args.batch_size)
        base.assert_masks(model)
        val = validation(model, val_loader, attrs)
        key = (-val['ACC'], val['legacy_EO'] if math.isfinite(val['legacy_EO']) else float('inf'), val['loss'])
        if best_key is None or key < best_key:
            best_key, best_state, best_epoch = key, cpu_state(model), epoch
        row = dict(epoch=epoch, seed=seed, samples=samples, optimizer_steps=steps,
                   train_loss=total/samples, validation=val, first_batch_sha256=first_hash,
                   seconds=time.monotonic()-epoch_start)
        history.append(row)
        atomic_save(dict(model_state=cpu_state(model), optimizer_state=optimizer.state_dict(),
            best_model_state=best_state, best_epoch=best_epoch, method=method, round=rnd,
            history=history, attributes=attrs), out/f'{method}_round{rnd}_latest_epoch.pt')
        print(f'  {method} r{rnd} epoch {epoch}/5 loss={total/samples:.6f} '
              f'val_ACC={val["ACC"]:.6f}, {row["seconds"]:.1f}s', flush=True)
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return dict(epochs=history, best_epoch=best_epoch, seconds=time.monotonic()-start,
                samples_per_epoch=len(frame), total_optimizer_steps=sum(h['optimizer_steps'] for h in history))


def report(out, result, start):
    result['elapsed_seconds'] = time.monotonic()-start
    base.DEFINITIONS = DEFINITIONS
    base.write_report(out, result)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_0.pt')
    p.add_argument('--csv', type=Path, default=ROOT/'csv/CelebA.csv')
    p.add_argument('--image-dir', type=Path, default=ROOT/'Images/CelebA/img_align_celeba')
    p.add_argument('--original-attributes', type=Path, default=ROOT.parent/'Balancing_Act/data/celeba/list_attr_celeba.txt')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--rounds', type=int, choices=[3], default=3)
    p.add_argument('--retrain-epochs', type=int, choices=[5], default=5)
    p.add_argument('--selection-samples', type=int, default=512)
    p.add_argument('--train-samples', type=int, default=8192, help='0 = full original train split')
    p.add_argument('--val-samples', type=int, default=1024, help='0 = full original validation split')
    p.add_argument('--eval-samples', type=int, default=1024, help='0 = full original test split')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--signal-batch-size', type=int, default=16)
    p.add_argument('--importance-batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--retrain-lr', type=float, default=1e-5)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--preflight-only', action='store_true')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    assert len(set(args.methods)) == len(args.methods)
    assert all(getattr(args, k) > 0 for k in ('selection_samples', 'batch_size', 'signal_batch_size', 'importance_batch_size'))
    assert min(args.train_samples, args.val_samples, args.eval_samples, args.workers) >= 0
    for k in ('checkpoint', 'csv', 'image_dir', 'original_attributes'):
        if not getattr(args, k).exists():
            raise FileNotFoundError(f'Missing --{k.replace("_", "-")}: {getattr(args, k)}')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = dict(status='preparing', metadata=dict(arguments=vars(args),
        command=shlex.join([sys.executable, str(Path(__file__).resolve())]+(sys.argv[1:] if argv is None else argv)),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), evaluations=[], rounds=[])
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        base.seed_all(args.seed)
        attrs, annotation = base.validate_annotations(args.csv, args.original_attributes)
        print('Annotation order/Male: all rows verified', flush=True)
        frames = dataset.make_frame(str(args.csv), str(args.image_dir), seven_races=False)
        ss = subsets(frames, attrs, args)
        for name, frame in ss.items():
            frame.to_csv(out/f'{name}_samples.csv', index=False)
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        assert state['classifier.1.weight'].shape == (78, 1280)
        assert not any('mask' in k for k in state)
        def fresh():
            m = models.mobilenet_v2(weights=None)
            m.classifier[1] = torch.nn.Linear(1280, 78)
            m.load_state_dict(state, strict=True)
            m.to(args.device)
            base.install_masks(m)
            return m.eval()
        model = fresh()
        result['metadata'].update(annotation=annotation, checkpoint_sha256=base.sha(args.checkpoint),
            checkpoint_format='plain dense state_dict, classifier [78,1280]', split_seed=42,
            split_sizes={k: len(frames[k]) for k in ('train', 'val', 'test')},
            sample_sizes={k: len(f) for k, f in ss.items()},
            sample_files={k: dict(path=str(out/f'{k}_samples.csv'), sha256=base.sha(out/f'{k}_samples.csv')) for k in ss},
            analysis_layers={n: m.out_channels for n, m in base.reference_layers(model).items()},
            source_sha256={n: base.sha(ROOT/n) for n in ('celeba_layerwise_experiment.py', 'celeba_fourway_smoke.py', 'prune.py', 'dataset.py', 'train_and_val.py', 'util.py')},
            environment=dict(python=sys.version, torch=torch.__version__, torchvision=base.torchvision.__version__,
                numpy=np.__version__, pandas=pd.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None),
            source_checkpoint_metadata='Attribute order not embedded; verified project convention',
            backend='deterministic cuDNN enabled; FP32, AMP/TF32 disabled',
            retraining_scope='5 complete epochs over recorded subset; 0 sample limit selects complete original split')
        print('Sample sizes:', result['metadata']['sample_sizes'], flush=True)
        loaders = {k: cached(ss[k], attrs, args.signal_batch_size if k != 'validation' else args.batch_size, args.workers)
                   for k in ('selection', 'validation', 'evaluation')}
        for k in ('selection', 'evaluation'):
            counts = base.membership(loaders[k].dataset.tensors[1]).sum(1)[2:].reshape(39, 2, 2)
            result['metadata'][f'{k}_cell_counts_y_by_Male'] = counts.tolist()
            if k == 'selection':
                assert (counts > 0).all()
        if args.preflight_only:
            result['status'] = 'preflight_complete'
            report(out, result, start)
            return
        result['status'] = 'running'
        print('Evaluating dense model...', flush=True)
        result['evaluations'].append(evaluate(model, loaders['evaluation'], attrs, 'dense', 0, 'dense'))
        report(out, result, start)
        del model
        initial_perf_hash = None
        expected_budgets, epoch_hashes = {}, {}
        for method in args.methods:
            model = fresh()
            removed = []
            for rnd in range(1, 4):
                round_start = time.monotonic()
                base.seed_all(args.seed+rnd)
                print(f'{method} round {rnd}/3: performance importance', flush=True)
                perf_loader = DataLoader(loaders['selection'].dataset, batch_size=args.importance_batch_size, shuffle=False)
                perf_start = time.monotonic()
                perf = base.performance_importance(model, perf_loader, ss['selection'], attrs,
                                                   out/f'{method}_round{rnd}_importance.txt')
                perf = {n: p.detach().cpu().numpy() for n, p in perf.items()}
                perf_hash = hashlib.sha256(b''.join(p.tobytes() for p in perf.values())).hexdigest()
                if rnd == 1:
                    if initial_perf_hash is None:
                        initial_perf_hash = perf_hash
                    assert initial_perf_hash == perf_hash, 'Initial performance scores differ'
                entry = dict(method=method, round=rnd, importance_seconds=time.monotonic()-perf_start,
                             importance_sha256=perf_hash, layers={})
                signals = collect(model, loaders['selection'], 'contribution' in method,
                                  f'{method} r{rnd} selection')
                entry['signals_seconds'] = signals['seconds']
                live = base.active_masks(model)
                picks = []
                select_start = time.monotonic()
                for n in base.reference_layers(model):
                    decision = greedy_keep(perf[n], live[n].cpu().numpy(), signals['deltas'][n], method)
                    entry['layers'][n] = decision
                    picks.extend((f'features.{base.block_number(n)}', c) for c in decision['removed'])
                    budget_key = (rnd, n)
                    budget = (decision['live_before'], decision['remove_count'])
                    if budget_key in expected_budgets:
                        assert budget == expected_budgets[budget_key]
                    expected_budgets[budget_key] = budget
                entry['greedy_seconds'] = time.monotonic()-select_start
                atomic_save(dict(performance=perf, signed_differences=signals['deltas']), out/f'{method}_round{rnd}_selection.pt')
                base.apply_channels(model, picks)
                removed.extend(picks)
                base.assert_masks(model)
                entry.update(removed_this_round=picks, removed_cumulative=copy.deepcopy(removed),
                    live_channels_after=sum(int(v.sum()) for v in base.active_masks(model).values()),
                    masked_weights=sum(int((m.mask == 0).sum()) for m in base.weight_layers(model).values()))
                print(f'{method} r{rnd}: removed {len(picks)} channels; remaining {entry["live_channels_after"]}', flush=True)
                result['rounds'].append(entry)
                result['evaluations'].append(evaluate(model, loaders['evaluation'], attrs, method, rnd, 'pre'))
                report(out, result, start)
                entry['retraining'] = retrain(model, ss['retrain'], loaders['validation'], attrs, args, method, rnd, out)
                for h in entry['retraining']['epochs']:
                    hk = (rnd, h['epoch'])
                    if hk in epoch_hashes:
                        assert h['first_batch_sha256'] == epoch_hashes[hk], 'Methods saw different augmented inputs'
                    epoch_hashes[hk] = h['first_batch_sha256']
                result['evaluations'].append(evaluate(model, loaders['evaluation'], attrs, method, rnd, 'post'))
                entry['round_seconds'] = time.monotonic()-round_start
                atomic_save(dict(model_state=cpu_state(model), removed_channels=removed,
                    attributes=attrs, method=method, round=rnd, arguments=base.clean(vars(args))), out/f'{method}_round{rnd}.pt')
                report(out, result, start)
            del model
        assert base.sha(args.checkpoint) == result['metadata']['checkpoint_sha256']
        result['status'] = 'complete'
        report(out, result, start)
        table = base.comparison(result['evaluations'])
        (out/'comparison.txt').write_text(table+'\n', encoding='utf-8')
        print('\nFINAL THREE-ROUND COMPARISON\n'+table, flush=True)
        print('\nTXT:', out/'results.txt', flush=True)
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        report(out, result, start)
        raise


if __name__ == '__main__':
    main()
