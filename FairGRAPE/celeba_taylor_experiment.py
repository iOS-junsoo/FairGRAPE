"""impt=7: online Taylor-FO-BN with shared CelebA single-task artifacts."""
import datetime
import json
import math
import shlex
import sys
from pathlib import Path

import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights, resnet18, ResNet18_Weights

import config
import celeba_single as single
import celeba_single_experiment as legacy
import celeba_single_reporting as reporting
from dataset import make_datasets
from taylor_pruning import (METHOD, TaylorPruner, attach_gates, conv_bn_pairs,
                            statistics, trainable_weights, validate_masks)
from util import safe_forward_with_cudnn_fallback, setseed
from taylor_scoring import (scoring_options, accumulate_frozen_scores, guarded_prune,
                            PruningCollapseError)


def build_model(network, *, pretrained=False):
    if network == 'resnet18':
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
    elif network == 'mobilenetv2':
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError(f'Unsupported Taylor network: {network}')
    return model


def validate_args(args):
    if ((args.impt, args.dataset, args.prune_type) != (7, 'CelebA', 'FairGRAPE')
            or args.network not in ('mobilenetv2', 'resnet18')):
        raise ValueError('impt=7 requires CelebA / mobilenetv2 or resnet18 / FairGRAPE')
    if not args.target_attr or args.target_attr == args.sensitive_group or args.sensitive_group != 'gender':
        raise ValueError('impt=7 requires --target_attr and --sensitive_group gender (distinct labels)')
    conflicts = {'--keep_per_iter', '--prune_iter', '--retrain_iter', '--retrain_lr', '--no_retrain',
                 '--lr_decay_iter', '--para_batch', '--delta_p', '--stop_batch', '--fg_scope', '--fg_mode',
                 '--impt2_alpha', '--impt2_layer_max_remove_ratio', '--save_model_iter'}
    supplied = conflicts.intersection(getattr(args, '_explicit_options', []))
    if supplied:
        raise ValueError(f'impt=7 uses --taylor_* options; incompatible options: {sorted(supplied)}')
    for key in ('no_pretrained', 'drop_race', 'race_binary', 'skew_beta', 'init_pruned',
                'score_only', 'save_impt', 'perf_only', 'phi_analysis', 'phi_gap_only', 'no_retrain'):
        if getattr(args, key, False):
            raise ValueError(f'impt=7 does not support --{key}')
    for key in ('fg_mode', 'fg_scope', 'impt2_alpha', 'impt2_layer_max_remove_ratio', 'prune_iter'):
        if getattr(args, key, None) is not None:
            raise ValueError(f'impt=7 does not support --{key}')
    if args.no_init_train and not args.checkpoint:
        raise ValueError('--no_init_train requires a validated dense checkpoint')
    if args.single_attr_eval_only and (not args.checkpoint or args.defer_test):
        raise ValueError('--single_attr_eval_only requires --checkpoint and conflicts with --defer_test')
    if args.batch < 1 or args.taylor_epochs < 1 or args.taylor_frequency < 1:
        raise ValueError('Batch, Taylor epochs and frequency must be positive')
    if scoring_options(args)[1] and args.batch < 2:
        raise ValueError('Taylor collapse guard requires --batch >= 2')
    for name, value, lower, upper in (
            ('prune_rate', args.prune_rate, 0, 1),
            ('taylor_lr', args.taylor_lr, 0, math.inf),
            ('taylor_prune_fraction', args.taylor_prune_fraction, 0, 1)):
        if not math.isfinite(value) or value <= lower or value > upper or (name == 'prune_rate' and value == 1):
            raise ValueError(f'Invalid --{name}: {value}')
    if not math.isfinite(args.taylor_ema) or not 0 <= args.taylor_ema < 1:
        raise ValueError('--taylor_ema must be finite and in [0,1)')
    if args.single_attr_smoke_samples < 0:
        raise ValueError('--single_attr_smoke_samples must be nonnegative')
    args.no_grl = True


def load_model(path, metadata, device, *, allow_pruned=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = single.validate_checkpoint(checkpoint, metadata, allow_pruned=allow_pruned, impt_type=7)
    model = build_model(metadata.get('network', 'mobilenetv2'))
    if checkpoint['stage'] == 'pruned':
        attach_gates(model)
    model.load_state_dict(state, strict=True)
    if checkpoint['stage'] == 'pruned':
        validate_masks(model)
        actual = statistics(model)
        if checkpoint.get('pruning_statistics') != actual:
            raise ValueError('Saved Taylor sparsity statistics do not match masks')
        model.taylor_online_state = dict(checkpoint.get('online_state', {}))
    model.single_validation_metrics = checkpoint.get('validation_metrics')
    return model.to(device), checkpoint


def evaluate_preserving_mode(model, loader, directory, args, split):
    modes = [(module, module.training) for module in model.modules()]
    try:
        # DataLoader iterator creation can consume torch RNG, even without
        # stochastic validation transforms. Keep it out of the training stream.
        devices = [next(model.parameters()).device.index or 0] if next(model.parameters()).is_cuda else []
        with torch.random.fork_rng(devices=devices):
            return single.evaluate(model, loader, directory, args.target_attr, args.sensitive_group, split)
    finally:
        for module, training in modes:
            module.training = training


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def online_train(model, train_loader, val_loader, args, root, model_root, log_root, metadata, info, source):
    """Train continuously; validation never rolls back the active training model."""
    pruner = TaylorPruner(model, args.prune_rate, args.taylor_prune_fraction, args.taylor_ema)
    optimizer = torch.optim.SGD(trainable_weights(model), lr=args.taylor_lr, momentum=.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=.1)
    device = next(model.parameters()).device
    best_state = best_metrics = best_online = None
    global_step = 0
    score_mode, collapse_guard = scoring_options(args)
    probe_images = None
    config.glo_train_stage = 'prune'

    def update_online(epoch):
        model.taylor_online_state = dict(global_epoch=epoch, global_step=global_step,
            iteration=pruner.iteration, frequency=args.taylor_frequency, ema=args.taylor_ema,
            per_round=pruner.per_round, target=args.prune_rate, target_reached=pruner.reached,
            score_mode=score_mode, collapse_guard=collapse_guard)

    def consider(metrics):
        nonlocal best_state, best_metrics, best_online
        if pruner.reached and (best_metrics is None or single.selection_key(metrics) > single.selection_key(best_metrics)):
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = dict(metrics)
            best_online = dict(model.taylor_online_state)

    for epoch in range(1, args.taylor_epochs + 1):
        model.train()
        loss_sum = sample_count = 0
        truth, predicted, groups = [], [], []
        for images, labels in train_loader:
            model.zero_grad(set_to_none=True)  # Includes gates, which are not in the optimizer.
            images = images.to(device, dtype=torch.float32)
            target = labels[:, 0].to(device).long()
            if collapse_guard and probe_images is None:
                probe_images = images[:16].detach().clone()
            if not pruner.reached and score_mode == 'frozen_bn':
                accumulate_frozen_scores(pruner, images, target)
            outputs = safe_forward_with_cudnn_fallback(model, images)
            loss = torch.nn.functional.cross_entropy(outputs, target)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Non-finite online training loss')
            loss.backward()
            if not pruner.reached and score_mode == 'train':
                pruner.accumulate()
            pruner.mask_gradients()
            optimizer.step()
            pruner.enforce(optimizer)
            global_step += 1
            loss_sum += loss.item() * len(target)
            sample_count += len(target)
            truth.extend(target.detach().cpu().tolist())
            predicted.extend(outputs.detach().argmax(1).cpu().tolist())
            groups.extend(labels[:, -1].tolist())
            if not pruner.reached and pruner.batches == args.taylor_frequency:
                try:
                    event = (guarded_prune(pruner, optimizer, probe_images) if collapse_guard
                             else pruner.prune(optimizer))
                except PruningCollapseError as exc:
                    update_online(epoch)
                    rejected = exc.diagnostics['attempted_iteration']
                    _json(log_root / f'rejected_iter_{rejected:02d}.json',
                          dict(exc.diagnostics, online_state=model.taylor_online_state))
                    model.single_validation_metrics = None
                    rollback_path = model_root / 'pruning' / f'ROLLBACK_before_prune_{rejected:02d}.pt'
                    rollback_path.parent.mkdir(parents=True, exist_ok=True)
                    single.save_checkpoint(model, rollback_path, metadata, iteration=pruner.iteration,
                                           source=source, settings=info)
                    raise
                if pruner.reached:
                    for state in optimizer.state.values():
                        if 'momentum_buffer' in state:
                            state['momentum_buffer'].zero_()
                update_online(epoch)
                config.glo_prune_iter = pruner.iteration - 1
                event.update(model.taylor_online_state)
                _json(log_root / f'iter_{pruner.iteration:02d}.json', event)
                scores = {name: score.detach().cpu() for (name, _, _, _), score in zip(pruner.pairs, pruner.scores)}
                torch.save(scores, log_root / f'iter_{pruner.iteration:02d}_scores.pt')
                lines = [f'Taylor-FO-BN pruning iteration {pruner.iteration}',
                         f'Global epoch/step: {epoch}/{global_step}',
                         f'Conv/Linear weight sparsity: {event["after"]["weight_sparsity"]:.8f}',
                         'layer neuron importance removed_weights']
                lines += [f'{x["layer"]} {x["neuron"]} {x["importance"]:.12e} {x["removed_weights"]}'
                          for x in event['selected']]
                (log_root / f'iter_{pruner.iteration:02d}.txt').write_text('\n'.join(lines) + '\n')
                directory = reporting.stage_directory(root, 'prune', pruner.iteration)
                metrics = evaluate_preserving_mode(model, val_loader, directory / 'val', args, 'val')
                metrics.update(global_epoch=epoch, global_step=global_step)
                model.single_validation_metrics = metrics
                reporting.write_summary(directory, model, metrics, args, stage='prune', iteration=pruner.iteration,
                                        alpha=None, norm=None, split='val')
                reporting.append_metrics(root, metrics, 'prune', pruner.iteration, 'val', args.target_attr)
                path = reporting.checkpoint_path(model_root, 'prune', metrics, pruner.iteration)
                single.save_checkpoint(model, path, metadata, iteration=pruner.iteration, source=source, settings=info)
                if args.save_mask:
                    masks = {key: value.detach().cpu() for key, value in model.state_dict().items()
                             if key.endswith(('.mask', '.taylor_gate'))}
                    torch.save(masks, model_root / 'pruning' /
                               f'impt7_{args.target_attr}_official_masks_{pruner.iteration:03d}.pt')
                consider(metrics)
                print(f'Taylor iter={pruner.iteration} step={global_step} '
                      f'removed_neurons={len(event["selected"])} sparsity={event["after"]["weight_sparsity"]:.6f}', flush=True)
        if not sample_count:
            raise ValueError('Cannot train on an empty Train split')
        update_online(epoch)
        stage = 'final' if pruner.reached else 'prune'
        directory = reporting.stage_directory(root, stage, pruner.iteration)
        # Epoch validation has a separate detail directory: it must not overwrite
        # the removal-event val/ predictions associated with BEST_MODEL_prune_XX.
        metrics = evaluate_preserving_mode(model, val_loader, directory / f'epoch_{epoch:02d}' / 'val', args, 'val')
        metrics.update(epoch=epoch, global_epoch=epoch, global_step=global_step)
        train_metrics = single.binary_metrics(truth, predicted, groups, loss_sum / sample_count)
        with (directory / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            for phase, values in [('train', train_metrics), ('val', metrics)]:
                record = dict(values, phase=phase, stage=stage, iteration=pruner.iteration,
                              epoch=epoch, global_epoch=epoch, global_step=global_step,
                              lr=optimizer.param_groups[0]['lr'], impt_type=7, target=args.target_attr,
                              weight_sparsity=statistics(model)['weight_sparsity'])
                stream.write(json.dumps(record, allow_nan=False) + '\n')
        text = reporting.report_text(metrics, target=args.target_attr, stage=stage, iteration=pruner.iteration,
                    epoch=epoch, total_epochs=args.taylor_epochs, seed=args.seed, impt_type=7,
                    pruning=reporting.pruning_statistics(model, 7), taylor=model.taylor_online_state,
                    network=args.network)
        (directory / f'Total_info_online_epoch_{epoch:02d}_val.txt').write_text(text, encoding='utf-8')
        consider(metrics)
        scheduler.step()
        print(f'Taylor epoch={epoch}/{args.taylor_epochs} val_acc={metrics["accuracy"]:.6f} '
              f'sparsity={statistics(model)["weight_sparsity"]:.6f}', flush=True)
    if not pruner.reached or best_state is None:
        raise RuntimeError(f'Taylor target not reached: {statistics(model)["weight_sparsity"]:.6f} < {args.prune_rate}; '
                           'increase --taylor_epochs or reduce --taylor_frequency')
    model.load_state_dict(best_state, strict=True)
    model.single_validation_metrics = best_metrics
    model.taylor_online_state = best_online
    validate_masks(model)
    return model, pruner.iteration


def run(args):
    validate_args(args)
    setseed(args.seed)
    frames, metadata = single.load_frames(args.celeba_csv, args.celeba_image_root,
        args.celeba_partition_file, args.target_attr, args.sensitive_group,
        check_readable=not args.skip_readable_check, smoke_samples=args.single_attr_smoke_samples)
    metadata['grl_enabled'] = False
    if args.network == 'resnet18':
        metadata.update(network='resnet18', initialization='ResNet18_Weights.IMAGENET1K_V1')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    checkpoint = None
    if args.checkpoint:
        model, checkpoint = load_model(args.checkpoint, metadata, device, allow_pruned=args.single_attr_eval_only)
        if args.single_attr_eval_only and checkpoint['stage'] != 'pruned':
            raise ValueError('impt=7 evaluation-only requires a pruned Taylor checkpoint')
    else:
        model = build_model(args.network, pretrained=True).to(device)
    pairs = conv_bn_pairs(model)
    expected_counts = {'mobilenetv2': (52, 17056), 'resnet18': (20, 4800)}
    counts = (len(pairs), sum(conv.out_channels for _, conv, _, _ in pairs))
    if counts != expected_counts[args.network]:
        raise ValueError(f'Unexpected {args.network} backbone: {counts}, expected {expected_counts[args.network]}')
    # Check feasibility before creating directories or dense training. Do not add
    # gates to the dense model until its checkpoint has been saved.
    total = statistics(model)['total_weights']
    maximum = sum((conv.out_channels - 1) * conv.weight[0].numel() for _, conv, _, _ in pairs)
    if math.ceil(args.prune_rate * total) > maximum:
        raise ValueError('Requested sparsity is unreachable with one neuron per convolution')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    tag = '_smoke' if args.single_attr_smoke_samples else ''
    root, model_root, log_root = reporting.run_paths(args, stamp, tag, 'taylor_fo_bn')
    legacy.configure(args, root, metadata, None, None, model_root, log_root)
    epochs = [1] if args.single_attr_smoke_samples else [8, 1, 1]
    rates = [1e-4] if args.single_attr_smoke_samples else [1e-4, 1e-5, 1e-6]
    info = dict(arguments=vars(args), experiment_metadata=metadata, impt_type=7,
                pruning_method=METHOD, pruning_unit='neuron', scope='all_backbone_convs',
                dense_lrs=rates, dense_epochs=epochs, alpha=None, norm=None,
                importance_split='train', selection_split='val', final_evaluation_split='test',
                target_sparsity=args.prune_rate, sparsity_denominator='all_conv_linear_weights',
                architecture_variant=('torchvision_resnet18_post_activation' if args.network == 'resnet18'
                                      else 'torchvision_mobilenet_v2'),
                gate_placement='independent_after_each_conv_bn_before_residual_addition',
                conv_layers=counts[0], initial_neurons=counts[1],
                neurons_per_round=math.ceil(counts[1] * args.taylor_prune_fraction),
                optimizer='SGD', momentum=.9, weight_decay=0., lr_milestones=[10, 20], lr_gamma=.1,
                torch=str(torch.__version__), device=str(device),
                paths=dict(results=str(root), models=str(model_root), channel_logs=str(log_root)))
    score_mode, collapse_guard = scoring_options(args)
    if args.single_attr_eval_only:
        saved_online = checkpoint.get('online_state', {})
        score_mode = saved_online.get('score_mode', 'train')
        collapse_guard = saved_online.get('collapse_guard', False)
    info.update(implementation_version=2, score_mode=score_mode, collapse_guard=collapse_guard,
                scoring_split='train', scoring_batch='same_as_training',
                collapse_probe_samples=16 if collapse_guard else 0)
    for split, frame in frames.items():
        frame[['image_name', 'split']].to_csv(root / f'{split}_manifest.csv', index=False)
    _json(root / 'run.json', info)
    run_text = '실행 명령어: ' + shlex.join([sys.executable] + sys.argv) + '\n' + json.dumps(info, indent=2, ensure_ascii=False)
    for directory in (root, model_root, log_root):
        (directory / '0_run_info.txt').write_text(run_text + '\n', encoding='utf-8')
    labels = [args.target_attr, args.sensitive_group]
    try:
        if args.single_attr_eval_only:
            # The evaluation contract follows the saved run's target, not CLI defaults.
            saved_target = checkpoint['settings']['target_sparsity']
            if statistics(model)['removed_weights'] < math.ceil(saved_target * total):
                raise ValueError('Evaluation checkpoint has not reached its saved target sparsity')
            _, test_loader = make_datasets(frames['test'], frames['test'], True, args.batch, labels)
            directory = reporting.stage_directory(root, 'final')
            result = evaluate_preserving_mode(model, test_loader, directory / 'test', args, 'test')
            reporting.write_summary(directory, model, result, args, stage='final',
                                    iteration=checkpoint['prune_iteration'], alpha=None, norm=None, split='test')
            reporting.append_metrics(root, result, 'final', checkpoint['prune_iteration'], 'test', args.target_attr)
            _json(root / 'complete.json', dict(status='evaluation_only', checkpoint=args.checkpoint, test_evaluated=True))
            return model
        train_loader, val_loader = make_datasets(frames['train'], frames['val'], True, args.batch, labels)
        loaders = {'train': train_loader, 'test': val_loader}
        if checkpoint is None:
            model = legacy.fit(model, args, frames, loaders, rates, epochs, 0.)
        directory = reporting.stage_directory(root, 'dense')
        dense_metrics = evaluate_preserving_mode(model, val_loader, directory / 'val', args, 'val')
        model.single_validation_metrics = dense_metrics
        reporting.write_summary(directory, model, dense_metrics, args, stage='dense', iteration=0, alpha=None, norm=None, split='val')
        reporting.append_metrics(root, dense_metrics, 'dense', 0, 'val', args.target_attr)
        dense_path = reporting.checkpoint_path(model_root, 'dense', dense_metrics)
        single.save_checkpoint(model, dense_path, metadata, settings=info)
        origin = Path(args.checkpoint).resolve() if args.checkpoint else dense_path
        source = dict(path=str(origin), sha256=single.file_hash(origin))
        # Exercise the same dense loading contract used across separate invocations.
        model, _ = load_model(dense_path, metadata, device)
        model, iteration = online_train(model, train_loader, val_loader, args, root, model_root, log_root,
                                        metadata, info, source)
        directory = reporting.stage_directory(root, 'final')
        selected = dict(model.single_validation_metrics)
        final_metrics = evaluate_preserving_mode(model, val_loader, directory / 'val', args, 'val')
        final_metrics['selected_epoch'] = selected.get('global_epoch')
        final_metrics['selected_step'] = selected.get('global_step')
        model.single_validation_metrics = final_metrics
        reporting.write_summary(directory, model, final_metrics, args, stage='final', iteration=iteration,
                                alpha=None, norm=None, split='val')
        reporting.append_metrics(root, final_metrics, 'final', iteration, 'val', args.target_attr)
        final_path = reporting.checkpoint_path(model_root, 'final', final_metrics)
        single.save_checkpoint(model, final_path, metadata, iteration=iteration, source=source, settings=info)
        if not args.defer_test:
            _, test_loader = make_datasets(frames['test'], frames['test'], True, args.batch, labels)
            result = evaluate_preserving_mode(model, test_loader, directory / 'test', args, 'test')
            reporting.write_summary(directory, model, result, args, stage='final', iteration=iteration,
                                    alpha=None, norm=None, split='test')
            reporting.append_metrics(root, result, 'final', iteration, 'test', args.target_attr)
        _json(root / 'complete.json', dict(status='complete', checkpoint=str(final_path),
              test_evaluated=not args.defer_test, pruning_statistics=statistics(model)))
        return model
    except Exception as exc:
        _json(root / 'failed.json', dict(status='failed', error=str(exc), error_type=type(exc).__name__))
        raise
