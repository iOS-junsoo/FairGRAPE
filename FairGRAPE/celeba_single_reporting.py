"""Human-readable single-attribute reports and isolated dense/pruning output paths."""
import json
import csv
from pathlib import Path


def run_paths(args, stamp, tag, model_tag, *, original=False):
    """Shared impt=5/6/7 output layout; never reuse an existing result root."""
    network_tag = '_resnet18' if getattr(args, 'network', None) == 'resnet18' else ''
    run_name = f'CelebA_impt{args.impt}{network_tag}_{args.target_attr}_official_seed{args.seed}{tag}_{stamp}'
    root = Path(args.single_attr_run_dir or Path('retrain_epoch_results') / '임시 저장소' / run_name).resolve()
    root.mkdir(parents=True, exist_ok=False)
    model_root = (root / 'models' if args.single_attr_run_dir else Path('save_models') /
                  f'{stamp}_CelebA_{model_tag}_impt{args.impt}{network_tag}_{args.target_attr}_official_seed{args.seed}{tag}').resolve()
    log_root = (root / ('fg_original' if original else 'channel_pruning') if args.single_attr_run_dir else
                Path('fg_orig_pruning_logs' if original else 'channel_pruning_logs') / run_name).resolve()
    return root, model_root, log_root


def checkpoint_path(model_root, stage, metrics, iteration=0):
    directory = Path(model_root) / ('dense' if stage == 'dense' else 'pruning')
    directory.mkdir(parents=True, exist_ok=True)
    label = f'prune_{iteration:02d}' if stage == 'prune' else stage
    return directory / (f'BEST_MODEL_{label}_val_acc_{metrics["accuracy"]:.4f}'
                        f'_eo_{format_value(metrics.get("eo"))}.pt')


def append_metrics(root, result, stage, iteration, split, attribute):
    columns = ['stage', 'iteration', 'split', 'attribute', 'accuracy', 'loss', 'ba', 'eo', 'eopp', 'dp', 'samples']
    path = Path(root) / 'metrics.csv'
    new = not path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if new:
            writer.writeheader()
        row = dict(result, stage=stage, iteration=iteration, split=split, attribute=attribute)
        writer.writerow({key: row.get(key) for key in columns})


def stage_directory(root, stage, iteration=0):
    root = Path(root)
    if stage == 'dense':
        return root / 'dense'
    if stage == 'prune':
        return root / 'pruning' / f'iter_{iteration:02d}'
    return root / 'pruning' / 'final'


def format_value(value):
    return 'N/A' if value is None else f'{value:.4f}'


def report_text(metrics, *, target, sensitive='gender', split='val', stage='prune',
                iteration=0, epoch=None, total_epochs=None, seed=None, alpha=None,
                norm=None, grl=False, title='EPOCH EVALUATION', pruning=None, impt_type=5, taylor=None,
                network='mobilenetv2'):
    lines = ['=' * 80, title, '=' * 80,
             'Dataset: CelebA', f'Network: {network}', f'Impt Type: {impt_type}',
             f'Target Attribute: {target}', f'Sensitive Group: {sensitive}',
             'Split Protocol: official',
             f'Evaluation Split: {dict(train="Train", val="Validation", test="Test")[split]}',
             f'Samples: {metrics.get("samples", "N/A")}', f'Stage: {stage}',
             f'Prune Iteration: {iteration}', f'Seed: {seed}',
             f'Alpha: {alpha}', f'Norm: {norm}', f'GRL Enabled: {grl}']
    if impt_type == 6:
        lines += ['FG Mode: original', 'FG Scope: all', 'Pruning Unit: weight',
                  'Reference: prune_orig_ref (b677eb9)']
    if impt_type == 7:
        online = taylor or {}
        scope = 'conv1, layer1~4 including downsample convolutions' if network == 'resnet18' else 'features.0~18'
        lines += ['Pruning Method: Taylor-FO-BN', 'Pruning Unit: neuron (feature map)',
                  f'Pruning Scope: all_backbone_convs ({scope})',
                  'Classifier: protected (weights, bias and both outputs)',
                  'Training Mode: online',
                  f'Importance Scoring: {online.get("score_mode", "not applicable (dense)" if stage == "dense" else "train (legacy)")}',
                  f'Collapse Guard: {online.get("collapse_guard", False)}',
                  f'Global Epoch: {online.get("global_epoch", 0)}',
                  f'Global Step: {online.get("global_step", 0)}',
                  f'Pruning Frequency: {online.get("frequency", 30)} mini-batches',
                  f'Importance EMA: {online.get("ema", 0.9)}',
                  f'Max Neurons Per Round: {online.get("per_round", "N/A")}',
                  f'Target Sparsity (Conv/Linear weights): {100 * online.get("target", .7):.4f}%']
    if epoch is not None:
        lines.append(f'Epoch: {epoch} / {total_epochs}')
    if metrics.get('selected_epoch') is not None:
        lines.append(f'Selected Epoch: {metrics["selected_epoch"]}')
    lines += ['', '-' * 80, f'{target} 성능 요약', '-' * 80]
    for label, key in [('Task Loss', 'loss'), ('Accuracy', 'accuracy'), ('BA', 'ba'),
                       ('EO (Equalized Odds gap)', 'eo'),
                       ('EOPP (Equal Opportunity gap)', 'eopp'),
                       ('DP (Demographic Parity gap)', 'dp')]:
        lines.append(f'{label}: {format_value(metrics.get(key))}')
    if 'auxiliary_loss' in metrics:
        lines += [f'Total Loss: {format_value(metrics["total_loss"])}',
                  f'GRL Auxiliary Loss (unweighted): {format_value(metrics["auxiliary_loss"])}']
    groups = metrics.get('groups', {})
    if groups:
        lines += ['', '-' * 80, f'{target} 그룹 × 정답별 정확도', '-' * 80]
        for g, c in groups.items():
            for y, correct, total in ((0, c['N'] - c['FP'], c['N']), (1, c['TP'], c['P'])):
                lines.append(f'{sensitive}={g}, {target}={y}: {correct} / {total} = '
                             f'{format_value(correct / total if total else None)}')
        lines += ['', '-' * 80, f'{target} 공정성 상세', '-' * 80]
        for g, c in groups.items():
            lines += [f'{sensitive}={g} TPR: {c["TP"]} / {c["P"]} = {format_value(c["tpr"])}',
                      f'{sensitive}={g} FPR: {c["FP"]} / {c["N"]} = {format_value(c["fpr"])}',
                      f'{sensitive}={g} Positive Prediction Rate: '
                      f'{c["TP"] + c["FP"]} / {c["P"] + c["N"]} = '
                      f'{format_value(c["positive_prediction_rate"])}']
    lines += ['', 'EO = (|TPR_0 - TPR_1| + |FPR_0 - FPR_1|) / 2',
              'EOPP = |TPR_0 - TPR_1|',
              'DP = |P(pred=1 | group=0) - P(pred=1 | group=1)|',
              'EO / EOPP / DP: 낮을수록 그룹 간 차이가 작음; 분모가 없으면 N/A']
    if pruning:
        lines += ['', '-' * 80, 'Pruning Statistics (실제 마스크 기준)', '-' * 80]
        for name, value in pruning.items():
            lines.append(f'{name}: {value}')
    return '\n'.join(lines) + '\n'


def pruning_statistics(model, impt_type=5):
    if impt_type == 7:
        from taylor_pruning import statistics
        s = statistics(model)
        removed = s['removed_weights']
        return {
            '프루닝된 파라미터 / 전체 파라미터': f'{removed:,} / {s["original_parameters"]:,} ({100 * removed / s["original_parameters"]:.4f}%)',
            '프루닝된 파라미터 / 프루닝 대상 파라미터': f'{removed:,} / {s["target_weights"]:,} ({100 * removed / s["target_weights"]:.4f}%)',
            'Current Sparsity (Conv/Linear weights)': f'{100 * s["weight_sparsity"]:.4f}%',
            'Removed Conv/Linear Weights': f'{removed:,} / {s["total_weights"]:,}',
            'Candidate Convolutions': s['conv_layers'],
            'Removed Neurons / Initial Neurons': f'{s["removed_neurons"]:,} / {s["initial_neurons"]:,}',
            'Live Neurons': s['live_neurons'],
        }
    from prune import _get_impt_type2_block_layer_names, _get_fg_scope_layers
    modules = dict(model.named_modules())
    target_names = {name for b in range(1, 18)
                    for name in _get_impt_type2_block_layer_names(f'features.{b}')
                    if name in modules}
    if impt_type == 6:
        target_names = set(_get_fg_scope_layers(model, 'all'))
    total = sum(p.numel() for n, p in model.named_parameters() if not n.endswith('.mask'))
    removed = target_total = target_removed = 0
    for name, layer in modules.items():
        weight, mask = getattr(layer, 'weight', None), getattr(layer, 'mask', None)
        if weight is None:
            continue
        zeros = int((mask == 0).sum().item()) if mask is not None and mask.shape == weight.shape else 0
        removed += zeros
        if name in target_names:
            target_total += weight.numel()
            target_removed += zeros
    return {'프루닝된 파라미터 / 전체 파라미터':
            f'{removed:,} / {total:,} ({100 * removed / total if total else 0:.2f}%)',
            '프루닝된 파라미터 / 프루닝 대상 파라미터':
            f'{target_removed:,} / {target_total:,} ({100 * target_removed / target_total if target_total else 0:.2f}%)'}


def write_summary(directory, model, metrics, args, *, stage, iteration, alpha, norm, split='val'):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    label = 'dense' if stage == 'dense' else f'prune_{iteration:02d}' if stage == 'prune' else 'final'
    filename = f'BEST_MODEL_SUMMARY_{label}_val.txt' if split == 'val' else 'FINAL_TEST_SUMMARY.txt'
    text = report_text(metrics, target=args.target_attr, sensitive=args.sensitive_group,
                       split=split, stage=stage, iteration=iteration, seed=args.seed,
                       alpha=alpha, norm=norm, grl=not args.no_grl, impt_type=args.impt,
                       title='BEST MODEL SUMMARY' if split == 'val' else 'FINAL TEST SUMMARY',
                       pruning=pruning_statistics(model, args.impt),
                       taylor=getattr(model, 'taylor_online_state', None),
                       network=getattr(args, 'network', 'mobilenetv2'))
    (directory / filename).write_text(text, encoding='utf-8')


def record_epoch(metrics, *, phase, lr, total_loss, auxiliary_loss):
    import config
    stage, iteration = config.glo_train_stage, config.glo_prune_iter + 1
    epoch = config.glo_single_epoch
    metrics.update(epoch=epoch, lr=lr)
    record = dict(metrics, phase='val' if phase == 'test' else 'train',
                  total_loss=float(total_loss), auxiliary_loss=float(auxiliary_loss),
                  stage=stage, iteration=iteration, target=config.glo_experiment_metadata['targets'][0],
                  impt_type=config.glo_impt_type)
    directory = Path(config.glo_results_run_dir)
    with (directory / 'epochs.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')
    if phase != 'test':
        return
    label = f'dense_epoch_{epoch:02d}' if stage == 'dense' else (
        f'prune_{iteration:02d}_retrain_{epoch:02d}' if stage == 'prune' else f'final_retrain_{epoch:02d}')
    text = report_text(record, target=record['target'], stage=stage, iteration=iteration,
                       epoch=epoch, total_epochs=config.glo_single_total_epochs,
                       seed=config.glo_seed, alpha=config.glo_imp_rate,
                       norm=config.glo_single_norm, grl=config.glo_use_grl, impt_type=config.glo_impt_type,
                       network=config.glo_experiment_metadata.get('network', 'mobilenetv2'))
    (directory / f'Total_info_{label}_val.txt').write_text(text, encoding='utf-8')
