"""LFWA original FairGRAPE baseline over every supported Conv/Linear layer.

Separate from the frozen impt2 runner; reference scoring/selection is unchanged.
"""
import datetime
import json
import math
from pathlib import Path
import time

import torch

import config
import lfwa
import prune
from dataset import make_datasets
from train_and_val import train
from util import make_model, setseed


def planned_iterations(keep, rate, initial=0.):
    if not 0 < keep < 1 or not 0 <= initial < rate < 1:
        raise ValueError('Require 0 < keep < 1 and 0 <= initial < rate < 1')
    return math.ceil(math.log((1 - rate) / (1 - initial), keep))


def _configure(args, root, metadata):
    prune.FG_ORIG_MODE = 'original'
    config.glo_fg_scope = args.fg_scope
    lfwa.sparsity = all_scope_sparsity  # Process-local accounting for the full scope.
    config.glo_dataset, config.glo_seed = 'LFWA', args.seed
    config.glo_use_grl, config.glo_pretrained = False, not args.no_pretrained
    config.glo_impt_type, config.glo_n_groups = 4, 2
    config.glo_keep_per_iter = args.keep_per_iter
    config.glo_phi_gap_only = config.glo_phi_analysis = config.glo_perf_only = False
    config.glo_experiment_metadata = metadata
    config.glo_lfwa_run_dir = str(root)
    config.glo_train_stage, config.glo_epoch_count, config.glo_prune_iter = 'dense', 0, -1
    config.glo_imp_rate = None
    for attr, subdir in [('glo_results_run_dir', 'history'), ('glo_model_run_dir', 'selected_models'),
                         ('glo_fg_log_run_dir', 'original_pruning')]:
        folder = root / subdir
        folder.mkdir(exist_ok=True)
        setattr(config, attr, str(folder))


def run(args):
    if (args.network, args.loss_type, args.sensitive_group, args.impt) != ('mobilenetv2', 'classes', 'young', 4):
        raise ValueError('LFWA requires --network mobilenetv2 --loss_type classes --sensitive_group young --impt 4')
    if not args.no_grl or args.drop_race or args.race_binary or args.skew_beta:
        raise ValueError('LFWA requires --no_grl and the unmodified official splits')
    if args.prune_type not in ('Full', 'FairGRAPE'):
        raise ValueError('LFWA supports Full and FairGRAPE')
    if args.score_only or args.save_impt or args.print_acc or args.perf_only or args.phi_gap_only or getattr(args, 'phi_analysis', False):
        raise ValueError('Use the LFWA runner for this fixed experiment and final Test evaluation')
    if args.init_pruned != 0:
        raise ValueError('LFWA starts from a Dense model; --init_pruned must be 0')
    if args.no_init_train and not args.checkpoint:
        raise ValueError('--no_init_train requires an LFWA Dense checkpoint')
    if args.checkpoint and not args.no_init_train:
        raise ValueError('Loading an LFWA checkpoint requires --no_init_train')
    if args.retrain_iter < 1:
        raise ValueError('LFWA requires at least one retraining epoch per iteration')
    if args.fg_mode != 'original' or args.fg_scope != 'all':
        raise ValueError('This baseline requires original mode and ALL supported layers')
    if args.impt2_alpha is not None or args.delta_p != 0 or args.para_batch != 1:
        raise ValueError('Original baseline has no alpha; delta_p=0 and para_batch=1 are required')
    if not args.no_retrain or not args.no_init_train or args.prune_type != 'FairGRAPE':
        raise ValueError('This baseline starts from the shared dense and excludes final extra training')
    setseed(args.seed)
    frames, metadata = lfwa.load_frames(args.lfwa_csv, args.lfwa_image_root)
    root = Path(args.lfwa_run_dir or ('lfwa_runs/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'run.json').exists():
        raise FileExistsError(f'Refusing to overwrite an existing LFWA run: {root}')
    _configure(args, root, metadata)
    iterations = 0 if args.prune_type == 'Full' else (
        args.prune_iter if args.prune_iter is not None else planned_iterations(args.keep_per_iter, args.prune_rate, args.init_pruned))
    if iterations < 0:
        raise ValueError('Pruning iterations cannot be negative')
    info = {'arguments': vars(args), 'experiment_metadata': metadata,
            'prune_iterations': iterations, 'dense_epochs': [8, 1, 1], 'dense_lrs': [1e-4, 1e-5, 1e-6],
            'importance_split': 'train', 'selection_split': 'val', 'alpha': None,
            'importance_batch': 64, 'phi_batch': None,
            'normalization': None, 'protection_ratio': None, 'layer_floor': None, 'layer_cap': None,
            'impt_type': 4, 'fg_mode': 'original', 'fg_scope': args.fg_scope,
            'scoring_implementation': 'prune_orig_ref.fairness_grad(impt_type=0)',
            'reference_commit': 'b677eb9', 'pruning_unit': 'weight', 'sensitive_classes': 2,
            'scoring_adam_lr': 1e-4, 'final_extra_epochs': 0,
            'scope_adaptation': None, 'target_scope': 'all_supported_conv_linear_weights',
            'mask_trainability': 'Initial disposable scoring masks trainable; returned masks frozen',
            'torch': torch.__version__, 'cuda': torch.version.cuda}
    (root / 'run.json').write_text(json.dumps(info, indent=2))
    print('LFWA configuration:', json.dumps(info, indent=2), flush=True)
    train_loader, val_loader = make_datasets(frames['train'], frames['val'], True, args.batch, lfwa.LABEL_COLUMNS)
    loaders = {'train': train_loader, 'test': val_loader}  # legacy phase name "test" means Validation.
    device = torch.device('cuda:0')
    if args.checkpoint:
        model, ck = lfwa.load_checkpoint_model(args.checkpoint, metadata, device)
        if ck['sparsity']['model_removed']:
            raise ValueError('Start LFWA pruning from the Dense checkpoint, not a smoke/pruned checkpoint')
    else:
        model = make_model(network='mobilenetv2', n_classes=78, pretrained=not args.no_pretrained).to(device)
    info['device'] = str(device)
    if hasattr(torch, 'cuda'):
        info['gpu_name'] = torch.cuda.get_device_name(device)
    (root / 'run.json').write_text(json.dumps(info, indent=2))
    criterion = torch.nn.CrossEntropyLoss()

    def fit(model, rates, epochs, rate):
        masks = lfwa.assert_masks(model)
        cfgs = ['LFWA', args.prune_type, 'classes', rate, frames['val'], None, 78, 'mobilenetv2',
                lfwa.LABEL_COLUMNS, lfwa.OUTPUT_SLICES, 'young', args.exp_idx, args.seed]
        result, _ = train(model, criterion, loaders, rates, epochs, lfwa.TARGETS,
                          lfwa.OUTPUT_SLICES, cfgs=cfgs, use_debiasing=False)
        lfwa.assert_masks(result, masks)
        return result

    if not args.no_init_train:
        model = fit(model, [1e-4, 1e-5, 1e-6], [8, 1, 1], 0.)
        lfwa.save_checkpoint(model, root / 'dense.pt', metadata, stage='dense', iteration=0)
    if iterations:
        pruner = prune.FairGRAPE(model, criterion, train_loader, lfwa.OUTPUT_SLICES, False)
        pruner.init_mask()
        # Mirror main_test.py: retain masked forwards for scoring and training.
        # Initial disposable masks stay trainable, as in the reference pipeline.
        pruner.update_model(pruner.get_model())
        for i in range(iterations):
            config.glo_prune_iter, config.glo_train_stage = i, 'prune'
            started = time.monotonic()
            cumulative_rate = min(1 - (1 - args.init_pruned) * args.keep_per_iter ** (i + 1), args.prune_rate)
            cfgs = prune_configuration(args, cumulative_rate, frames['train'])
            prior_masks = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                           if n.endswith('.mask')}
            model = pruner.prune(cfgs, True)
            validate_pruning(model, cumulative_rate, prior_masks)
            before = lfwa.sparsity(model)
            model = fit(model, [args.retrain_lr], [args.retrain_iter], cumulative_rate)
            if lfwa.sparsity(model) != before:
                raise AssertionError('Sparsity changed during retraining')
            for name, param in model.named_parameters():
                param.requires_grad_(not name.endswith('.mask'))
            lfwa.save_checkpoint(model, root / f'prune_{i+1:03d}.pt', metadata,
                                 stage='prune', iteration=i+1, seconds=time.monotonic()-started)
            pruner.update_model(model)
            print(f'LFWA prune stage {i+1}/{iterations} completed', flush=True)
        del pruner
        if not args.no_retrain:
            config.glo_train_stage = 'final_retrain'
            model = fit(model, [1e-4, 1e-5, 1e-6], [8, 1, 1], args.prune_rate)
    lfwa.save_checkpoint(model, root / 'final.pt', metadata, stage=config.glo_train_stage, iteration=iterations)
    lfwa.evaluate_model(model, frames['val'], root / 'validation', args.batch)
    if not args.defer_test:
        lfwa.evaluate_model(model, frames['test'], root / 'test', args.batch)
    (root / 'complete.json').write_text(json.dumps({'status': 'complete', 'checkpoint': str(root / 'final.pt')}))
    return model


def validate_pruning(model, cumulative_rate, prior_masks):
    """Check the baseline scope, exact weight budget, finite model and no regrowth."""
    masks = lfwa.assert_masks(model)
    scope = set(prune._get_fg_scope_layers(model, 'all'))
    assert len(scope) == len(masks) == 53
    for name, layer in model.named_modules():
        if name not in masks:
            continue
        mask = masks[name]
        assert torch.isfinite(layer.weight).all(), name
        if name in scope:
            assert int(mask.sum()) == int(layer.weight.numel() * (1 - cumulative_rate)), name
        else:
            assert torch.all(mask == 1), name
        if prior_masks:
            assert name + '.mask' in prior_masks
            assert torch.all(mask <= prior_masks[name + '.mask']), name
    stats = prune.FG_ORIG_LAST_STATS
    assert len(stats) == 53 and sum(s['in_scope'] for s in stats) == 53
    for stat in stats:
        assert stat['kept'] == stat['keep_target'], stat['name']
        if stat['in_scope']:
            assert torch.isfinite(torch.as_tensor(stat['target_prop'])).all(), stat['name']
    for name, value in model.state_dict().items():
        assert torch.isfinite(value).all(), name


def prune_configuration(args, cumulative_rate, train_frame):
    """The 78 logits represent 39 targets, while sensitive_classes must be two."""
    return [cumulative_rate, train_frame, None, 2, True, lfwa.OUTPUT_SLICES,
            lfwa.LABEL_COLUMNS, args.para_batch, 4, args.stop_batch, args.delta_p,
            'mobilenetv2', 'young']


def all_scope_sparsity(model):
    """Count removed entries from binary masks, excluding masks from denominators."""
    target_names = set(prune._get_fg_scope_layers(model, 'all'))
    target_total = target_removed = removed = 0
    total = sum(p.numel() for n, p in model.named_parameters() if not n.endswith('.mask'))
    for name, layer in model.named_modules():
        weight, mask = getattr(layer, 'weight', None), getattr(layer, 'mask', None)
        if weight is None:
            continue
        zeros = int((mask == 0).sum()) if mask is not None else 0
        removed += zeros
        if name in target_names:
            target_total += weight.numel()
            target_removed += zeros
    return {'target_scope': 'all', 'target_weights': target_total, 'target_removed': target_removed,
            'target_sparsity': target_removed / target_total if target_total else 0.,
            'model_parameters': total, 'model_removed': removed,
            'model_sparsity': removed / total if total else 0.}

