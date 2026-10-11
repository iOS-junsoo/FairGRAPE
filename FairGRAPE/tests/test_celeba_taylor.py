"""Online impt=7 integration using real MobileNetV2 and tiny in-memory splits."""
import copy
import json
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch
from torchvision.models import mobilenet_v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import celeba_single as single
import celeba_single_reporting as reporting
import celeba_taylor_experiment as runner
from main_test import parse_args, experiment
from taylor_pruning import METHOD, statistics


@pytest.fixture(params=['mobilenetv2', 'resnet18'])
def setup_run(tmp_path, monkeypatch, request):
    torch.manual_seed(10)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    before = {key: value for key, value in vars(config).items() if key.startswith('glo_')}
    network = request.param
    metadata = dict(targets=['Smiling'], grl_enabled=False, smoke_samples=4, network=network,
                    initialization=(single.IMAGENET_WEIGHTS if network == 'mobilenetv2'
                                    else 'ResNet18_Weights.IMAGENET1K_V1'), split_protocol='official')
    frames = {split: pd.DataFrame(dict(image_name=[f'{split}_{i}.jpg' for i in range(4)],
                                     split=[split] * 4)) for split in ('train', 'val', 'test')}
    images = torch.randn(4, 3, 32, 32)
    labels = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
    events = []

    def make_loaders(train, val, *args):
        loaders = []
        for frame in (train, val):
            split = frame.iloc[0]['split']
            events.append(('loader', split))
            dataset = torch.utils.data.TensorDataset(images, labels)
            dataset.data_frame = frame
            loaders.append(torch.utils.data.DataLoader(dataset, batch_size=4))
        return tuple(loaders)

    monkeypatch.setattr(single, 'load_frames', lambda *a, **kw: (frames, copy.deepcopy(metadata)))
    monkeypatch.setattr(runner, 'make_datasets', make_loaders)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    dense = tmp_path / 'dense.pt'
    single.save_checkpoint(runner.build_model(network), dense, metadata)
    args = parse_args(['--impt', '7', '--dataset', 'CelebA', '--network', network,
                      '--target_attr', 'Smiling', '--checkpoint', str(dense), '--no_init_train',
                      '--single_attr_run_dir', str(tmp_path / 'run'), '--batch', '4',
                      '--taylor_epochs', '2', '--taylor_frequency', '1',
                      '--taylor_prune_fraction', '1', '--single_attr_smoke_samples', '4', '--save_mask',
                      '--taylor_collapse_guard', 'off'])
    yield args, metadata, events, images
    torch.set_num_threads(threads)
    for key in list(vars(config)):
        if key.startswith('glo_') and key not in before:
            delattr(config, key)
    for key, value in before.items():
        setattr(config, key, value)


def test_real_online_70_percent_artifacts_and_eval_only(setup_run, tmp_path, monkeypatch):
    args, metadata, events, inputs = setup_run
    real_evaluate = single.evaluate

    def evaluate(*a, **kw):
        split = a[-1]
        events.append(('evaluate', split))
        if split == 'test':
            assert list((Path(args.single_attr_run_dir) / 'models' / 'pruning').glob('BEST_MODEL_final_*.pt'))
        return real_evaluate(*a, **kw)

    monkeypatch.setattr(single, 'evaluate', evaluate)
    model = experiment(args)
    root = Path(args.single_attr_run_dir)
    assert .7 <= statistics(model)['weight_sparsity'] < .70044
    assert events.count(('evaluate', 'test')) == 1
    event_report = (root / 'pruning/iter_01/BEST_MODEL_SUMMARY_prune_01_val.txt').read_text()
    convs, neurons = (52, '17,056') if args.network == 'mobilenetv2' else (20, '4,800')
    for text in ('Target Attribute: Smiling', f'Network: {args.network}', f'Candidate Convolutions: {convs}', neurons,
                 'all_backbone_convs', 'Classifier: protected', 'EOPP', 'DP', 'Current Sparsity'):
        assert text in event_report
    assert (root / 'pruning/final/Total_info_online_epoch_02_val.txt').exists()
    assert (root / 'pruning/final/FINAL_TEST_SUMMARY.txt').exists()
    assert (root / 'channel_pruning/iter_01_scores.pt').exists()
    assert (root / 'models/pruning/impt7_Smiling_official_masks_001.pt').exists()
    for filename in ('run.json', '0_run_info.txt', 'train_manifest.csv', 'val_manifest.csv',
                     'test_manifest.csv', 'metrics.csv', 'complete.json'):
        assert (root / filename).exists()
    complete = json.loads((root / 'complete.json').read_text())
    path = Path(complete['checkpoint'])
    restored, checkpoint = runner.load_model(path, metadata, 'cpu', allow_pruned=True)
    assert checkpoint['pruning_method'] == METHOD
    assert checkpoint['online_state']['score_mode'] == ('frozen_bn' if args.network == 'mobilenetv2' else 'train')
    assert checkpoint['settings']['implementation_version'] == 2
    model.eval(); restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), restored(inputs), rtol=0, atol=0)
    # The event's val/ remains the exact metrics/predictions for its saved model.
    event_path = next((root / 'models/pruning').glob('BEST_MODEL_prune_01_*.pt'))
    event_model, event_checkpoint = runner.load_model(event_path, metadata, 'cpu', allow_pruned=True)
    assert json.loads((root / 'pruning/iter_01/val/metrics.json').read_text())['accuracy'] == event_checkpoint['validation_metrics']['accuracy']
    assert event_checkpoint['validation_metrics']['global_step'] == 1
    args.checkpoint = str(path)
    args.single_attr_eval_only = True
    args.single_attr_run_dir = str(tmp_path / 'evaluation')
    args.taylor_score_mode = 'train' if checkpoint['online_state']['score_mode'] == 'frozen_bn' else 'frozen_bn'
    # For eval-only the final model already exists in the input run.
    monkeypatch.setattr(single, 'evaluate', real_evaluate)
    monkeypatch.setattr(runner, 'online_train', lambda *a, **kw: pytest.fail('evaluation trained'))
    runner.run(args)
    assert json.loads((tmp_path / 'evaluation/complete.json').read_text())['status'] == 'evaluation_only'
    assert json.loads((tmp_path / 'evaluation/run.json').read_text())['score_mode'] == checkpoint['online_state']['score_mode']
    assert not (tmp_path / 'evaluation/models/pruning').exists()
    with pytest.raises(ValueError, match='Start pruning from a dense'):
        runner.load_model(path, metadata, 'cpu')
    corrupted = torch.load(path, weights_only=True)
    gate_key = 'features.0.1.taylor_gate' if args.network == 'mobilenetv2' else 'bn1.taylor_gate'
    corrupted['model_state'][gate_key][0] = .5
    torch.save(corrupted, tmp_path / 'corrupt.pt')
    with pytest.raises(ValueError, match='gate'):
        runner.load_model(tmp_path / 'corrupt.pt', metadata, 'cpu', allow_pruned=True)


def test_no_checkpoint_runs_dense_training_before_pruning(setup_run, monkeypatch):
    args, metadata, events, _ = setup_run
    args.checkpoint = None
    args.no_init_train = False
    args.defer_test = True
    calls = []
    actual_build = runner.build_model
    def build(network, *, pretrained=False):
        calls.append((network, pretrained))
        # Unit test avoids a network download; the real dense trainer still runs.
        return actual_build(network, pretrained=False)
    monkeypatch.setattr(runner, 'build_model', build)
    runner.run(args)
    root = Path(args.single_attr_run_dir)
    dense = torch.load(next((root / 'models/dense').glob('*.pt')), weights_only=True)
    assert calls[0] == (args.network, True)
    assert dense['experiment_metadata'] == metadata
    assert dense['stage'] == 'dense'
    text = (root / 'dense/Total_info_dense_epoch_01_val.txt').read_text()
    assert f'Network: {args.network}' in text
    assert (root / 'complete.json').exists()
    assert ('evaluate', 'test') not in events


def test_accumulation_crosses_epochs_and_target_failure_is_not_success(setup_run, tmp_path):
    args, _, events, _ = setup_run
    args.prune_rate = .00001
    args.taylor_frequency = 2
    args.defer_test = True
    runner.run(args)
    root = Path(args.single_attr_run_dir)
    event = json.loads((root / 'channel_pruning/iter_01.json').read_text())
    assert event['scoring_batches'] == 2
    assert event['global_epoch'] == 2 and event['global_step'] == 2
    assert ('evaluate', 'test') not in events
    args.single_attr_run_dir = str(tmp_path / 'failure')
    args.taylor_frequency = 3
    with pytest.raises(RuntimeError, match='target not reached'):
        runner.run(args)
    assert (tmp_path / 'failure/failed.json').exists()
    assert not (tmp_path / 'failure/complete.json').exists()
    assert not list((tmp_path / 'failure/models/pruning').glob('BEST_MODEL_final_*.pt'))


def test_guard_rejection_writes_failure_and_restored_checkpoint(setup_run, monkeypatch):
    import taylor_scoring
    args, metadata, _, _ = setup_run
    args.taylor_collapse_guard = 'on'
    args.defer_test = True
    calls = []
    def probe(model, images):
        calls.append(True)
        return {'rejected': len(calls) == 2, 'samples': len(images)}
    monkeypatch.setattr(taylor_scoring, 'probe_information_flow', probe)
    with pytest.raises(taylor_scoring.PruningCollapseError):
        runner.run(args)
    root = Path(args.single_attr_run_dir)
    assert (root / 'failed.json').exists()
    assert not (root / 'complete.json').exists()
    assert not (root / 'pruning/iter_01/BEST_MODEL_SUMMARY_prune_01_val.txt').exists()
    event = json.loads((root / 'channel_pruning/rejected_iter_01.json').read_text())
    assert event['rolled_back'] is True
    restored, checkpoint = runner.load_model(root / 'models/pruning/ROLLBACK_before_prune_01.pt',
                                             metadata, 'cpu', allow_pruned=True)
    assert statistics(restored)['removed_weights'] == 0
    assert checkpoint['online_state']['iteration'] == 0
    assert checkpoint['validation_metrics'] is None


def test_validation_preserves_training_state_and_rng(setup_run, tmp_path):
    args, _, _, _ = setup_run
    model = mobilenet_v2(weights=None, num_classes=2).train()
    model.features[0].eval()  # Preserve mixed module modes too.
    modes = [m.training for m in model.modules()]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    frame = pd.DataFrame(dict(image_name=['val.jpg']*4, split=['val']*4))
    _, loader = runner.make_datasets(frame, frame)
    rng = torch.get_rng_state().clone()
    runner.evaluate_preserving_mode(model, loader, tmp_path / 'eval', args, 'val')
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [m.training for m in model.modules()]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_cli_defaults_conflicts_and_shared_default_paths(tmp_path, monkeypatch):
    legacy = parse_args([])
    args = parse_args(['--impt', '7', '--dataset', 'CelebA', '--network', 'mobilenetv2', '--target_attr', 'Smiling'])
    assert (legacy.batch, legacy.prune_rate) == (64, .9)
    assert (args.batch, args.prune_rate, args.no_grl) == (256, .7, True)
    runner.validate_args(args)
    for option in ('--keep_per_iter=.9', '--keep_per_it=.9', '--retrain_iter=3', '--stop_batch=10000'):
        bad = parse_args(['--impt', '7', '--dataset', 'CelebA', '--network', 'mobilenetv2', '--target_attr', 'Smiling', option])
        with pytest.raises(ValueError, match='incompatible'):
            runner.validate_args(bad)
    for key, value in [('target_attr', None), ('target_attr', 'gender'), ('taylor_lr', float('nan')),
                       ('taylor_ema', 1.), ('prune_rate', 1.), ('taylor_frequency', 0)]:
        bad = copy.copy(args); setattr(bad, key, value)
        with pytest.raises(ValueError):
            runner.validate_args(bad)
    monkeypatch.chdir(tmp_path)
    root, models, logs = reporting.run_paths(args, 'STAMP', '', 'taylor_fo_bn')
    assert root == tmp_path / 'retrain_epoch_results/임시 저장소/CelebA_impt7_Smiling_official_seed42_STAMP'
    assert models == tmp_path / 'save_models/STAMP_CelebA_taylor_fo_bn_impt7_Smiling_official_seed42'
    assert logs == tmp_path / 'channel_pruning_logs/CelebA_impt7_Smiling_official_seed42_STAMP'
    with pytest.raises(FileExistsError):
        reporting.run_paths(args, 'STAMP', '', 'taylor_fo_bn')
