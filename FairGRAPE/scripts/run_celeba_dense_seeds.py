"""Train independent CelebA dense seeds with the repository training routine."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / 'trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_0.pt'
EVALUATOR = REPO / 'docs/celeba_unpruned_test_20260921/evaluate.py'
CORE = ['dataset.py', 'util.py', 'train_and_val.py', 'adversarial_debiasing.py',
        'config.py', 'prune.py', 'channel_cap.py', 'fairgrape_original.py', 'prune_orig_ref.py']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.part')
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    partial.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(seed):
    return REPO / f'trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_seed{seed}.pt'


def prepare(root, args):
    if (root / 'manifest.json').exists():
        return json.loads((root / 'manifest.json').read_text())
    assert len(set(args.seeds)) == len(args.seeds)
    for seed in args.seeds:
        if checkpoint(seed).exists():
            raise FileExistsError(checkpoint(seed))
    hashes = {}
    for name in CORE:
        dst = root / 'source' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, dst)
        hashes[name] = sha(dst)
    shutil.copy2(__file__, root / 'source' / Path(__file__).name)
    shutil.copy2(EVALUATOR, root / 'source' / 'reference_evaluate.py')
    manifest = dict(created=now(), seeds=args.seeds, batch=args.batch, grl=args.grl,
                    model='mobilenetv2', pretrained='ImageNet IMAGENET1K_V1',
                    target_attributes=39, sensitive_attribute='gender',
                    learning_rates=[1e-4, 1e-5, 1e-6], epochs=[8, 1, 1], optimizer='Adam',
                    split_seed=42, split='repository 80/10/10, fixed across training seeds',
                    train_shuffle=False, augmentation='dataset.ImgAugTransform',
                    selection='Unchanged train_and_val.train / train_model0; validation only',
                    torch_threads=2, reference_checkpoint=str(REFERENCE), reference_sha256=sha(REFERENCE),
                    reference_training_metadata='Unavailable in the legacy state_dict; current repository dense protocol is used.',
                    source_sha256=hashes, evaluator_sha256=sha(EVALUATOR), csv_sha256=sha(REPO/'csv/CelebA.csv'),
                    runner_sha256=sha(__file__), exports={str(s): str(checkpoint(s)) for s in args.seeds})
    write(root / 'manifest.json', manifest)
    for seed in args.seeds:
        out = root / f'seed{seed}' / 'test'
        out.mkdir(parents=True)
        source = EVALUATOR.read_text()
        replacements = {
            'ROOT = Path(__file__).resolve().parents[2]': f'ROOT = Path({str(REPO)!r})',
            'OUT = Path(__file__).resolve().parent': f'OUT = Path({str(out)!r})',
            'sys.path.insert(0, str(ROOT))': f'sys.path.insert(0, {str(root / "source")!r})',
            'checkpoint = ROOT / "trained_model/unpruned/CelebA_unpruned_classes_bygender_mobilenetv2_0.pt"': f'checkpoint = Path({str(checkpoint(seed))!r})',
        }
        for old, new in replacements.items():
            assert source.count(old) == 1, old
            source = source.replace(old, new)
        (out / 'evaluate.py').write_text(source)
    write(root / 'status.json', dict(status='prepared', time=now(), seeds=args.seeds))
    return manifest


def activate(root):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(root / 'source'))
    import numpy as np
    # Same local compatibility aliases used by the repository's existing runners.
    for key, value in dict(bool=bool, int=int, float=float, complex=complex, object=object, unicode=str).items():
        if key not in np.__dict__:
            setattr(np, key, value)
    import torch
    import config
    from dataset import make_frame, make_datasets
    from util import setseed, make_model
    from train_and_val import train
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        assert sha(root / 'source' / name) == digest, name
    assert sha(REFERENCE) == manifest['reference_sha256']
    assert sha(REPO / 'csv/CelebA.csv') == manifest['csv_sha256']
    return torch, config, make_frame, make_datasets, setseed, make_model, train, manifest


def worker(root, seed, preflight=False):
    torch, config, make_frame, make_datasets, setseed, make_model, train, manifest = activate(root)
    out = root / ('preflight' if preflight else f'seed{seed}')
    out.mkdir(exist_ok=True)
    if (out / 'started.json').exists():
        raise RuntimeError(f'Run already started; explicit recovery is required: {out}')
    write(out / 'started.json', dict(seed=seed, time=now(), pid=os.getpid()))
    setseed(seed)
    config.glo_dataset, config.glo_seed = 'CelebA', seed
    config.glo_use_grl, config.glo_pretrained = manifest['grl'], True
    config.glo_impt_type, config.glo_n_groups, config.glo_prune_iter = 0, 2, -1
    for key, folder in [('glo_results_run_dir', 'history'), ('glo_model_run_dir', 'selected_models')]:
        path = out / folder
        path.mkdir(exist_ok=True)
        setattr(config, key, str(path))
    frames = make_frame('csv/CelebA.csv', 'Images/CelebA/img_align_celeba', seven_races=False)
    attrs = frames['train'].columns[2:41].tolist()
    columns, slices = attrs + ['gender'], [(i*2, i*2+2) for i in range(39)]
    assert [len(frames[s]) for s in ('train', 'val', 'test')] == [162079, 20260, 20260]
    split_hashes = {s: hashlib.sha256(('\n'.join(frames[s].image_name)+'\n').encode()).hexdigest()
                    for s in ('train', 'val', 'test')}
    expected = json.loads((EVALUATOR.parent / 'independent_split_check.json').read_text())
    assert split_hashes['test'] == expected['ordered_test_names_sha256']
    train_frame, val_frame = frames['train'], frames['val']
    if preflight:
        train_frame = train_frame.iloc[:manifest['batch'] * 2].reset_index(drop=True)
        val_frame = val_frame.iloc[:manifest['batch'] * 2].reset_index(drop=True)
    train_loader, val_loader = make_datasets(train_frame, val_frame, True, manifest['batch'], columns)
    model = make_model(network='mobilenetv2', n_classes=78, pretrained=True).to('cuda:0')
    initial_hash = hashlib.sha256(model.classifier[1].weight.detach().cpu().numpy().tobytes()).hexdigest()
    write(out / 'run.json', dict(seed=seed, split_sha256=split_hashes, classifier_initial_sha256=initial_hash,
                                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                                python=sys.executable, started=now(), batch=manifest['batch'], grl=manifest['grl']))
    state = dict(seed=seed, epoch=0, phase='initializing', pid=os.getpid(), batch=0)
    epochs = []
    started = time.monotonic()

    class ObservedLoader:
        def __init__(self, loader, phase):
            self.loader, self.phase, self.dataset = loader, phase, loader.dataset
        def __len__(self):
            return len(self.loader)
        def __iter__(self):
            if self.phase == 'train':
                state['epoch'] += 1
            phase_started = time.monotonic()
            last = 0
            for number, batch in enumerate(self.loader, 1):
                state.update(phase=self.phase, batch=number, total_batches=len(self.loader))
                if time.monotonic() - last >= 30 or number == len(self.loader):
                    record = dict(state, time=now(), elapsed_seconds=time.monotonic()-started,
                                  phase_elapsed_seconds=time.monotonic()-phase_started)
                    write(out / 'status.json', record)
                    print('DENSE_PROGRESS ' + json.dumps(record), file=sys.__stdout__, flush=True)
                    last = time.monotonic()
                yield batch

    def observe(model, epoch, metrics):
        record = dict(epoch=state['epoch'], time=now(),
                      **{k: float(metrics[k]) for k in ('accuracy', 'ba', 'eo', 'loss')})
        epochs.append(record)
        write(out / 'epochs.json', epochs)
        print('DENSE_VALIDATION ' + json.dumps(record), file=sys.__stdout__, flush=True)

    config.glo_epoch_observer = observe
    loaders = {'train': ObservedLoader(train_loader, 'train'), 'test': ObservedLoader(val_loader, 'val')}
    cfgs = ['CelebA', 'Full', 'classes', 0., frames['test'], 'Images/CelebA/img_align_celeba',
            78, 'mobilenetv2', columns, slices, 'gender', seed, seed]
    model, optimizer = train(model, torch.nn.CrossEntropyLoss(), loaders,
                             [1e-4] if preflight else manifest['learning_rates'],
                             [1] if preflight else manifest['epochs'], attrs, slices,
                             cfgs=cfgs, use_debiasing=manifest['grl'])
    assert len(epochs) == (1 if preflight else sum(manifest['epochs']))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not any(n.endswith('.mask') for n in model.state_dict())
    if preflight:
        assert initial_hash != hashlib.sha256(model.classifier[1].weight.detach().cpu().numpy().tobytes()).hexdigest()
        write(out / 'complete.json', dict(status='passed', seconds=time.monotonic()-started, epochs=epochs))
        return
    target = checkpoint(seed)
    if target.exists():
        raise FileExistsError(target)
    model.cpu()
    torch.save(dict(model_state=model.state_dict(), optimizer_state=optimizer, seed=seed,
                    manifest=manifest, split_sha256=split_hashes), out / 'training_checkpoint.pt')
    partial = target.with_suffix('.pt.part')
    torch.save(model.state_dict(), partial)
    partial.replace(target)
    restored = torch.load(target, map_location='cpu', weights_only=True)
    assert all(torch.equal(value, restored[name]) for name, value in model.state_dict().items())
    write(out / 'training_complete.json', dict(checkpoint=str(target), sha256=sha(target),
                                              time=now(), epochs=len(epochs), seconds=time.monotonic()-started))
    write(out / 'status.json', dict(status='trained', checkpoint=str(target), time=now()))


def queue(root, manifest):
    if not (root / 'preflight/complete.json').exists():
        raise RuntimeError('Run --preflight before starting the full queue')
    for seed in manifest['seeds']:
        out = root / f'seed{seed}'
        if (out / 'complete.json').exists():
            continue
        write(root / 'status.json', dict(status='training', seed=seed, time=now(), pid=os.getpid()))
        with (out / 'train.log').open('a', buffering=1) as log:
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--run-dir', str(root),
                            '--worker', str(seed)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        write(root / 'status.json', dict(status='evaluating_test', seed=seed, time=now(), pid=os.getpid()))
        with (out / 'test/evaluation.log').open('w') as log:
            subprocess.run([sys.executable, '-u', str(out / 'test/evaluate.py')], cwd=REPO,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads((out / 'test/metrics.json').read_text())
        assert metrics['n_test'] == metrics['n_evaluated'] == 20260
        write(out / 'complete.json', dict(status='complete', seed=seed, time=now(),
                                         checkpoint=str(checkpoint(seed)), **{k:metrics[k] for k in ('ACC','BA','EO')}))
    results = [json.loads((root / f'seed{s}/complete.json').read_text()) for s in manifest['seeds']]
    write(root / 'complete.json', dict(status='complete', time=now(), results=results))
    write(root / 'status.json', dict(status='complete', time=now(), results=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[100,120])
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--grl', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    try:
        manifest = prepare(root, args)
        if args.prepare_only:
            print(json.dumps(manifest, indent=2), flush=True)
        elif args.preflight:
            worker(root, manifest['seeds'][0], preflight=True)
        elif args.worker is not None:
            assert args.worker in manifest['seeds']
            worker(root, args.worker)
        else:
            queue(root, manifest)
    except Exception as error:
        failure = dict(status='failed', time=now(), error=repr(error), worker=args.worker)
        write(root / ('failure_preflight.json' if args.preflight else 'failure.json'), failure)
        if not args.preflight:
            write(root / 'status.json', failure)
        raise
