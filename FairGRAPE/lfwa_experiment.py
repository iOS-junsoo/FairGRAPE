"""LFWA entry point using the existing training, transforms and FairGRAPE pruning."""
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
    prune.IMPT2_NORM = 'max'
    prune.IMPT2_PROTECTION_RATIO = .005
    prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER = .03
    prune.IMPT2_LAYER_CAP_MULTIPLIER = None
    prune.IMPT_TYPE2_IMPORTANCE_BATCH_SIZE = 128
    config.glo_dataset, config.glo_seed = 'LFWA', args.seed
    config.glo_use_grl, config.glo_pretrained = False, not args.no_pretrained
    config.glo_impt_type, config.glo_n_groups = 2, 2
    config.glo_keep_per_iter = args.keep_per_iter
    config.glo_phi_gap_only = config.glo_phi_analysis = config.glo_perf_only = False
    config.glo_experiment_metadata = metadata
    config.glo_lfwa_run_dir = str(root)
    config.glo_train_stage, config.glo_epoch_count, config.glo_prune_iter = 'dense', 0, -1
    config.glo_imp_rate = prune.IMPT_TYPE2_ALPHA
    for attr, subdir in [('glo_results_run_dir', 'history'), ('glo_model_run_dir', 'selected_models'),
                         ('glo_channel_log_run_dir', 'channel_pruning')]:
        folder = root / subdir
        folder.mkdir(exist_ok=True)
        setattr(config, attr, str(folder))


def run(args):
    if (args.network, args.loss_type, args.sensitive_group, args.impt) != ('mobilenetv2', 'classes', 'young', 2):
        raise ValueError('LFWA requires --network mobilenetv2 --loss_type classes --sensitive_group young --impt 2')
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
    setseed(args.seed)
    frames, metadata = lfwa.load_frames(args.lfwa_csv, args.lfwa_image_root)
    root = Path(args.lfwa_run_dir or ('lfwa_runs/' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'run.json').exists():
        raise FileExistsError(f'Refusing to overwrite an existing LFWA run: {root}')
    _configure(args, root, metadata)
    if args.impt2_alpha is not None:
        prune.IMPT_TYPE2_ALPHA = args.impt2_alpha
        config.glo_imp_rate = args.impt2_alpha
    iterations = 0 if args.prune_type == 'Full' else (
        args.prune_iter if args.prune_iter is not None else planned_iterations(args.keep_per_iter, args.prune_rate, args.init_pruned))
    if iterations < 0:
        raise ValueError('Pruning iterations cannot be negative')
    info = {'arguments': vars(args), 'experiment_metadata': metadata,
            'prune_iterations': iterations, 'dense_epochs': [8, 1, 1], 'dense_lrs': [1e-4, 1e-5, 1e-6],
            'importance_split': 'train', 'selection_split': 'val', 'alpha': prune.IMPT_TYPE2_ALPHA,
            'importance_batch': prune.IMPT_TYPE2_IMPORTANCE_BATCH_SIZE, 'phi_batch': 64,
            'normalization': prune.IMPT2_NORM, 'protection_ratio': prune.IMPT2_PROTECTION_RATIO,
            'layer_floor': prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER, 'layer_cap': prune.IMPT2_LAYER_CAP_MULTIPLIER,
            'torch': torch.__version__, 'cuda': torch.version.cuda}
    (root / 'run.json').write_text(json.dumps(info, indent=2))
    print('LFWA configuration:', json.dumps(info, indent=2), flush=True)
    train_loader, val_loader = make_datasets(frames['train'], frames['val'], True, args.batch, lfwa.LABEL_COLUMNS)
    loaders = {'train': train_loader, 'test': val_loader}  # legacy phase name "test" means Validation.
    device = torch.device('cuda:0')
    if args.checkpoint:
        model, ck = lfwa.load_checkpoint_model(args.checkpoint, metadata, device)
        if ck['sparsity']['target_removed']:
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
        for name, param in pruner.model.named_parameters():
            param.requires_grad_(not name.endswith('.mask'))
        for i in range(iterations):
            config.glo_prune_iter, config.glo_train_stage = i, 'prune'
            started = time.monotonic()
            cumulative_rate = min(1 - (1 - args.init_pruned) * args.keep_per_iter ** (i + 1), args.prune_rate)
            cfgs = [cumulative_rate, frames['train'], None, 78, True, lfwa.OUTPUT_SLICES,
                    lfwa.LABEL_COLUMNS, args.para_batch, 2, args.stop_batch, args.delta_p, 'mobilenetv2', 'young']
            model = pruner.prune(cfgs, True)
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
