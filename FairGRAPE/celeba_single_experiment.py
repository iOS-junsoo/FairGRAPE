"""impt=5/6 orchestration: Train signals, Validation selection, final-only Test."""
import datetime
import json
import math
import shlex
import sys
from pathlib import Path

import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import celeba_single as single
import celeba_single_reporting as reporting
import config
import prune
from channel_cap import validate_layer_remove_ratio
from dataset import make_datasets
from train_and_val import train
from util import make_model, setseed

DEFAULT_ALPHA = prune.IMPT_TYPE2_ALPHA


def validate_args(args):
    if args.impt not in (5, 6) or (args.dataset, args.network, args.prune_type) != ('CelebA', 'mobilenetv2', 'FairGRAPE'):
        raise ValueError('impt=5/6 supports CelebA / mobilenetv2 / FairGRAPE only')
    if not args.target_attr:
        raise ValueError('--impt 5/6 requires --target_attr')
    if args.no_pretrained:
        raise ValueError('impt=5/6 requires ImageNet initialization; --no_pretrained is forbidden')
    if args.no_init_train and not args.checkpoint:
        raise ValueError('--no_init_train requires a validated single-attribute checkpoint')
    if getattr(args, 'single_attr_eval_only', False):
        if not args.checkpoint or args.score_only or args.defer_test:
            raise ValueError('--single_attr_eval_only requires a checkpoint and conflicts with score_only/defer_test')
    if args.score_only and not args.checkpoint:
        raise ValueError('--score_only requires a validated checkpoint')
    if args.drop_race or args.race_binary or args.skew_beta or args.init_pruned != 0:
        raise ValueError('impt=5/6 requires unmodified official splits and a dense starting checkpoint')
    if args.save_impt:
        raise ValueError('Use --score_only instead of legacy --save_impt')
    if args.impt == 5 and getattr(args, 'fg_mode', None) is not None:
        raise ValueError('impt=5 does not support --fg_mode or legacy --save_impt; use --score_only')
    if args.perf_only and (args.phi_analysis or args.phi_gap_only):
        raise ValueError('--perf_only conflicts with phi options')
    if not 0 < args.keep_per_iter < 1 or not 0 <= args.prune_rate < 1:
        raise ValueError('Require 0 < keep_per_iter < 1 and 0 <= prune_rate < 1')
    if args.prune_iter is not None and args.prune_iter < 0:
        raise ValueError('--prune_iter must be nonnegative')
    if args.batch < 1 or args.stop_batch < 1 or args.retrain_iter < 0:
        raise ValueError('Batch sizes, stop_batch and enabled retraining epochs must be positive')
    if args.single_attr_smoke_samples < 0:
        raise ValueError('Smoke sample count must be nonnegative')
    if args.impt == 6:
        if getattr(args, 'fg_mode', None) not in (None, 'original'):
            raise ValueError('impt=6 requires --fg_mode original')
        if getattr(args, 'fg_scope', None) not in (None, 'all'):
            raise ValueError('impt=6 requires --fg_scope all')
        if (args.impt2_alpha is not None or args.impt2_layer_max_remove_ratio is not None
                or args.perf_only or args.phi_analysis or args.phi_gap_only):
            raise ValueError('impt=6 does not support alpha, phi, perf_only or channel caps')
        args.fg_mode, args.fg_scope = 'original', 'all'
        if getattr(args, 'delta_p', None) is None:
            args.delta_p = 0
        if args.delta_p not in (0, 1, 2) or args.para_batch < 1:
            raise ValueError('impt=6 requires delta_p in {0,1,2} and positive para_batch')
        return None, None
    alpha = DEFAULT_ALPHA if args.impt2_alpha is None else args.impt2_alpha
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('--impt2_alpha must be in [0, 1]')
    cap = validate_layer_remove_ratio(args.impt2_layer_max_remove_ratio)
    if cap is not None and (args.perf_only or prune.IMPT2_LAYER_CAP_MULTIPLIER is not None):
        raise ValueError('Strict cap conflicts with perf_only or legacy cap_multiplier')
    return alpha, cap


def configure(args, root, metadata, alpha, cap, model_root=None, log_root=None):
    config.glo_dataset, config.glo_impt_type, config.glo_seed = 'CelebA', args.impt, args.seed
    config.glo_n_groups, config.glo_pretrained, config.glo_use_grl = 2, True, not args.no_grl
    config.glo_keep_per_iter, config.glo_impt2_layer_max_remove_ratio = args.keep_per_iter, cap
    config.glo_perf_only, config.glo_phi_analysis, config.glo_phi_gap_only = args.perf_only, args.phi_analysis, args.phi_gap_only
    config.glo_experiment_metadata = metadata
    config.glo_train_stage, config.glo_prune_iter = 'dense', -1
    config.glo_lfwa_run_dir = None
    config.glo_epoch_observer = None
    config.glo_collapse_notes = []
    config.glo_skew_beta = 0
    if args.impt == 5:
        prune.IMPT_TYPE2_ALPHA = alpha
    config.glo_imp_rate = 1.0 if args.perf_only else alpha
    config.glo_single_results_root = str(root)
    config.glo_single_norm = None if args.impt in (6, 7) else ('raw' if args.perf_only else prune.IMPT2_NORM)
    paths = {'glo_results_run_dir': root / 'dense',
             'glo_model_run_dir': model_root or root / 'models',
             'glo_channel_log_run_dir': log_root or root / 'channel_pruning',
             'glo_phi_analysis_run_dir': root / 'phi_analysis'}
    if args.impt == 7:
        paths.pop('glo_phi_analysis_run_dir')
    if args.impt == 6:
        config.glo_fg_scope = 'all'
        config.glo_fg_para_batch, config.glo_fg_delta_p = args.para_batch, args.delta_p
        paths.pop('glo_channel_log_run_dir')
        paths.pop('glo_phi_analysis_run_dir')
        paths['glo_fg_log_run_dir'] = log_root or root / 'fg_original'
    for attr, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        setattr(config, attr, str(path))


def load_model(path, metadata, device, *, allow_pruned=False, impt_type=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = single.validate_checkpoint(checkpoint, metadata, allow_pruned=allow_pruned, impt_type=impt_type)
    model = make_model(network='mobilenetv2', n_classes=2, pretrained=False)
    modules = dict(model.named_modules())
    if impt_type == 6 and checkpoint['stage'] == 'pruned':
        expected_masks = {name + '.mask' for name in prune._get_fg_scope_layers(model, 'all')}
        if {name for name in state if name.endswith('.mask')} != expected_masks:
            raise ValueError('impt=6 checkpoint must contain masks for every Conv2d/Linear')
    for name, mask in state.items():
        if name.endswith('.mask'):
            layer = modules[name[:-5]]
            layer.register_parameter('mask', torch.nn.Parameter(torch.ones_like(layer.weight), requires_grad=False))
    model.load_state_dict(state, strict=True)
    if impt_type == 6 and checkpoint['stage'] == 'pruned':
        for name in prune._get_fg_scope_layers(model, 'all'):
            layer = modules[name]
            if bool((layer.weight.detach()[layer.mask == 0] != 0).any()):
                raise ValueError(f'impt=6 checkpoint has nonzero masked weights: {name}')
    model.single_validation_metrics = checkpoint.get('validation_metrics')
    return model.to(device), checkpoint


def freeze_masks(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.endswith('.mask'))


def validate_original_masks(model, rate, previous):
    """Check the reference's per-layer budget and cumulative binary masks."""
    for name, layer in model.named_modules():
        if type(layer).__name__ not in prune.supported_layers:
            continue
        mask = layer.mask.detach()
        if mask.shape != layer.weight.shape or not bool(((mask == 0) | (mask == 1)).all()):
            raise AssertionError(f'Invalid original FairGRAPE mask: {name}')
        if int(mask.sum()) != int(layer.weight.numel() * (1 - rate)):
            raise AssertionError(f'Original FairGRAPE layer budget differs: {name}')
        if name in previous and bool((mask > previous[name]).any()):
            raise AssertionError(f'Previously pruned weights were restored: {name}')
        if bool((layer.weight.detach()[mask == 0] != 0).any()):
            raise AssertionError(f'Masked weights are nonzero: {name}')


def fit(model, args, frames, loaders, rates, epochs, rate):
    """The legacy phase called 'test' receives only the official Validation loader."""
    directory = reporting.stage_directory(config.glo_single_results_root,
                                          config.glo_train_stage, config.glo_prune_iter + 1)
    directory.mkdir(parents=True, exist_ok=True)
    config.glo_results_run_dir = str(directory)
    config.glo_single_epoch, config.glo_single_total_epochs = 0, sum(epochs)
    labels = [args.target_attr, args.sensitive_group]
    cfgs = ['CelebA', 'FairGRAPE', args.target_attr, rate, frames['val'], None,
            2, args.network, labels, single.OUTPUT_SLICES, args.sensitive_group, args.exp_idx, args.seed]
    masks = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith('.mask')}
    result, _ = train(model, torch.nn.CrossEntropyLoss(), loaders, rates, epochs,
                      [args.target_attr], single.OUTPUT_SLICES, cfgs=cfgs, use_debiasing=not args.no_grl)
    for name, mask in masks.items():
        if not torch.equal(result.state_dict()[name], mask):
            raise AssertionError(f'Mask changed during training: {name}')
    freeze_masks(result)
    return result


def run(args):
    alpha, cap = validate_args(args)
    setseed(args.seed)
    frames, metadata = single.load_frames(args.celeba_csv, args.celeba_image_root,
        args.celeba_partition_file, args.target_attr, args.sensitive_group,
        check_readable=not args.skip_readable_check, smoke_samples=args.single_attr_smoke_samples)
    metadata['grl_enabled'] = not args.no_grl
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = None
    # Validate provenance before making any output directories or starting training.
    if args.checkpoint:
        model, checkpoint = load_model(args.checkpoint, metadata, device,
            allow_pruned=args.score_only or getattr(args, 'single_attr_eval_only', False), impt_type=args.impt)
    else:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
        model = model.to(device)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    tag = '_smoke' if args.single_attr_smoke_samples else ''
    tag += '_perfonly' if args.perf_only else '_gaponly' if args.phi_gap_only else ''
    if args.impt == 6:
        tag += '_original_all'
    norm = None if args.impt == 6 else ('raw' if args.perf_only else prune.IMPT2_NORM)
    effective_alpha = 1.0 if args.perf_only else alpha
    model_tag = 'fgoriginal_all' if args.impt == 6 else f'alpha{effective_alpha}_{norm}'
    root, model_root, log_root = reporting.run_paths(args, stamp, tag, model_tag, original=args.impt == 6)
    configure(args, root, metadata, alpha, cap, model_root, log_root)
    for split, frame in frames.items():
        frame[['image_name', 'split']].to_csv(root / f'{split}_manifest.csv', index=False)
    epochs = [1] if args.single_attr_smoke_samples else [8, 1, 1]
    rates = [1e-4] if args.single_attr_smoke_samples else [1e-4, 1e-5, 1e-6]
    iterations = args.prune_iter if args.prune_iter is not None else (
        math.ceil(math.log(1 - args.prune_rate, args.keep_per_iter)) if args.prune_rate else 0)
    info = {'arguments': vars(args), 'experiment_metadata': metadata,
            'dense_lrs': rates, 'dense_epochs': epochs, 'prune_iterations': iterations,
            'importance_split': 'train', 'selection_split': 'val', 'final_evaluation_split': 'test',
            'impt_type': args.impt, 'pruning_unit': 'weight' if args.impt == 6 else 'channel',
            'alpha': effective_alpha, 'norm': norm, 'gamma': None if args.impt == 6 else prune.IMPT2_PROTECTION_RATIO,
            'floor': None if args.impt == 6 else prune.IMPT2_MIN_KEEP_RATIO_PER_LAYER, 'strict_cap': cap,
            'importance_batch': 64 if args.impt == 6 else prune.IMPT_TYPE2_IMPORTANCE_BATCH_SIZE,
            'phi_batch': None if args.impt == 6 else 64,
            'torch': str(torch.__version__), 'device': str(device)}
    if args.impt == 6:
        info.update(fg_mode='original', fg_scope='all', reference_commit='b677eb9',
                    scoring_implementation='prune_orig_ref.fairness_grad(impt_type=0)')
    info['paths'] = dict(results=str(root), models=str(model_root))
    info['paths']['fg_original_logs' if args.impt == 6 else 'channel_logs'] = str(log_root)
    (root / 'run.json').write_text(json.dumps(info, indent=2))
    run_text = '실행 명령어: ' + shlex.join([sys.executable] + sys.argv) + '\n' + json.dumps(info, indent=2, ensure_ascii=False) + '\n'
    for directory in (root, model_root, log_root):
        (directory / '0_run_info.txt').write_text(run_text, encoding='utf-8')

    def evaluate_stage(model, loader, stage, iteration=0, split='val'):
        directory = reporting.stage_directory(root, stage, iteration)
        result = single.evaluate(model, loader, directory / split, args.target_attr, args.sensitive_group, split)
        selected = getattr(model, 'single_validation_metrics', None) or {}
        selected_epoch = selected.get('epoch', selected.get('selected_epoch'))
        if split == 'val' and selected_epoch is not None:
            result['selected_epoch'] = selected_epoch
        reporting.write_summary(directory, model, result, args, stage=stage, iteration=iteration,
                                alpha=effective_alpha, norm=norm, split=split)
        reporting.append_metrics(root, result, stage, iteration, split, args.target_attr)
        return result

    def checkpoint_path(stage, metrics, iteration=0):
        return reporting.checkpoint_path(model_root, stage, metrics, iteration)
    labels = [args.target_attr, args.sensitive_group]
    if getattr(args, 'single_attr_eval_only', False):
        _, test_loader = make_datasets(frames['test'], frames['test'], True, args.batch, labels)
        stage = 'dense' if checkpoint['stage'] == 'dense' else 'final'
        evaluate_stage(model, test_loader, stage, checkpoint.get('prune_iteration', 0), 'test')
        (root / 'complete.json').write_text(json.dumps({'status': 'evaluation_only', 'checkpoint': args.checkpoint}))
        return model
    train_loader, val_loader = make_datasets(frames['train'], frames['val'], True, args.batch, labels)
    loaders = {'train': train_loader, 'test': val_loader}
    source = {'path': str(Path(args.checkpoint).resolve()), 'sha256': single.file_hash(args.checkpoint)} if args.checkpoint else None

    def evaluate_val(model, stage, iteration=0):
        return evaluate_stage(model, val_loader, stage, iteration)

    if not args.checkpoint:
        model = fit(model, args, frames, loaders, rates, epochs, 0.)
        model.single_validation_metrics = evaluate_val(model, 'dense')
        dense_path = checkpoint_path('dense', model.single_validation_metrics)
        single.save_checkpoint(model, dense_path, metadata, settings=info)
        # Exercise the same validation and loading contract used by a separate invocation.
        model, checkpoint = load_model(dense_path, metadata, device, impt_type=args.impt)
        source = {'path': str(dense_path), 'sha256': single.file_hash(dense_path)}
    if args.score_only:
        source = checkpoint.get('source_checkpoint') or source
    elif args.checkpoint:
        model.single_validation_metrics = evaluate_val(model, 'dense')
        single.save_checkpoint(model, checkpoint_path('dense', model.single_validation_metrics), metadata, settings=info)
    if iterations or args.score_only:
        existing_masks = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith('.mask')}
        pruner = prune.FairGRAPE(model, torch.nn.CrossEntropyLoss(), train_loader, single.OUTPUT_SLICES, False)
        if not any(n.endswith('.mask') for n, _ in pruner.model.named_parameters()):
            pruner.init_mask()
        if args.impt == 6:
            # FairGRAPE's constructor initializes masks. Restore a pruned scoring
            # checkpoint rather than replacing its cumulative masks with ones.
            for name, parameter in pruner.model.named_parameters():
                if name in existing_masks:
                    with torch.no_grad():
                        parameter.copy_(existing_masks[name])
                    parameter.requires_grad_(False)
            # Fresh dense scoring masks remain trainable, as in impt=4/original.
            # Keep masked forwards on the retraining copy too: train() deepcopies
            # models, which does not preserve parameter gradient hooks.
            pruner.update_model(pruner.get_model())
        else:
            freeze_masks(pruner.model)
        if args.score_only:
            config.glo_prune_iter = checkpoint.get('prune_iteration', 0)
            # Compute and log scores using Train. Discard proposed masks; do not mutate/save the input model.
            scoring_rate = args.prune_rate
            if args.impt == 6 and existing_masks:
                # A score-only proposal must fit the currently live candidates.
                removed = [1 - float((p != 0).sum()) / p.numel() for p in existing_masks.values()]
                scoring_rate = math.nextafter(max(scoring_rate, max(removed)), 1.0)
            pruner.get_mask([scoring_rate, frames['train'], None, 2, True, single.OUTPUT_SLICES,
                             labels, args.para_batch, args.impt, args.stop_batch, args.delta_p,
                             'mobilenetv2', args.sensitive_group])
            (root / 'complete.json').write_text(json.dumps({'status': 'scores_only', 'input': source}))
            return model
        for i in range(iterations):
            config.glo_prune_iter, config.glo_train_stage = i, 'prune'
            rate = min(1 - args.keep_per_iter ** (i + 1), args.prune_rate)
            previous = {n[:-5]: p.detach().clone() for n, p in model.named_parameters() if n.endswith('.mask')}
            model = pruner.prune([rate, frames['train'], None, 2, True, single.OUTPUT_SLICES,
                labels, args.para_batch, args.impt, args.stop_batch, args.delta_p, 'mobilenetv2', args.sensitive_group], True)
            if args.impt == 6:
                validate_original_masks(model, rate, previous)
            freeze_masks(model)
            model.single_validation_metrics = None
            if args.save_mask:
                (model_root / 'pruning').mkdir(parents=True, exist_ok=True)
                torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith('.mask')},
                           model_root / 'pruning' / f'impt{args.impt}_{args.target_attr}_official_masks_{i+1:03d}.pt')
            # Per-round epochs are independent of the final-retraining switch.
            if args.retrain_iter > 0:
                model = fit(model, args, frames, loaders, [args.retrain_lr], [args.retrain_iter], rate)
            model.single_validation_metrics = evaluate_val(model, 'prune', i+1)
            single.save_checkpoint(model, checkpoint_path('prune', model.single_validation_metrics, i+1),
                                   metadata, iteration=i+1, source=source, settings=info)
            pruner.update_model(model)
            pruner.update_forward_pass()
        # Match impt=2: --no_retrain only skips this final extra schedule.
        if not args.no_retrain:
            config.glo_train_stage = 'final_retrain'
            model = fit(model, args, frames, loaders, rates, epochs, args.prune_rate)
    final_stage = 'final' if iterations else 'dense'
    if iterations:
        model.single_validation_metrics = evaluate_val(model, final_stage, iterations)
    final_path = checkpoint_path(final_stage, model.single_validation_metrics)
    if iterations:
        single.save_checkpoint(model, final_path, metadata, iteration=iterations, source=source, settings=info)
    # No Test loader exists until training, pruning and Validation selection have finished.
    if not args.defer_test:
        _, test_loader = make_datasets(frames['test'], frames['test'], True, args.batch, labels)
        evaluate_stage(model, test_loader, final_stage, iterations, 'test')
    (root / 'complete.json').write_text(json.dumps({'status': 'complete', 'checkpoint': str(final_path),
                                                  'test_evaluated': not args.defer_test}))
    return model
