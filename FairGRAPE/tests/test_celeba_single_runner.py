"""Check orchestration boundaries without Test inference or full training."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tempfile

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import celeba_single as single
import celeba_single_experiment as runner

REAL_FIT = runner.fit
REAL_EVALUATE = single.evaluate


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Linear(1280, 2))

    def forward(self, x):
        return self.classifier(x)


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.events = []
        self.metrics = dict(accuracy=.75, loss=.2, eo=.1, ba=.75, eopp=.15, dp=.2, samples=4)
        self.metadata = dict(dataset='CelebA', network='mobilenetv2', targets=['Smiling'],
                             initialization=single.IMAGENET_WEIGHTS, split_protocol='official', smoke_samples=0)
        self.frames = {}
        for split in ('train','val','test'):
            f = pd.DataFrame({'image_name':[split+'.jpg'], 'split':[split]})
            f.attrs['split'] = split; self.frames[split] = f
        old = {k:v for k,v in vars(config).items() if k.startswith('glo_')}
        def restore():
            for k in list(vars(config)):
                if k.startswith('glo_') and k not in old: delattr(config,k)
            for k,v in old.items(): setattr(config,k,v)
        self.addCleanup(restore)
        self.addCleanup(patch.stopall)
        patch.object(runner, 'setseed').start()
        patch.object(runner.torch.cuda, 'is_available', return_value=False).start()
        self.initializer = patch.object(runner, 'mobilenet_v2', side_effect=lambda **kw:TinyModel()).start()
        patch.object(runner, 'make_model', side_effect=lambda **kw:TinyModel()).start()
        patch.object(single, 'load_frames', side_effect=lambda *a,**kw:(self.frames, copy.deepcopy(self.metadata))).start()
        def loaders(train, val, *a):
            self.events.append(('loader', train.attrs['split'], val.attrs['split']))
            return SimpleNamespace(split=train.attrs['split']), SimpleNamespace(split=val.attrs['split'])
        patch.object(runner, 'make_datasets', side_effect=loaders).start()
        def evaluate(model, loader, directory, target, sensitive, split):
            self.assertEqual(loader.split, split)
            self.events.append(('eval', split))
            return dict(self.metrics)
        self.evaluator = patch.object(single, 'evaluate', side_effect=evaluate).start()
        def fit(model, args, frames, loaders, rates, epochs, rate):
            self.assertEqual(loaders['train'].split,'train')
            self.assertEqual(loaders['test'].split,'val')
            self.events.append(('fit',rate))
            if rate == 0: self.assertFalse(any(n.endswith('.mask') for n in model.state_dict()))
            model.single_validation_metrics = dict(self.metrics)
            return model
        self.fit = patch.object(runner, 'fit', side_effect=fit).start()
        test = self
        class FakePruner:
            def __init__(self, model, criterion, loader, *args):
                test.assertEqual(loader.split,'train'); self.model=model
            def init_mask(self):
                l=self.model.classifier[1]
                l.mask=torch.nn.Parameter(torch.ones_like(l.weight),requires_grad=config.glo_impt_type == 6)
            def prune(self,cfgs,*args):
                test.assertEqual(cfgs[1].attrs['split'],'train')
                test.assertEqual(cfgs[8],config.glo_impt_type)
                if cfgs[8] == 6:
                    layer = self.model.classifier[1]
                    active = layer.mask.detach().flatten().nonzero().flatten()
                    mask = torch.zeros_like(layer.weight).flatten()
                    mask[active[:int(layer.weight.numel() * (1 - cfgs[0]))]] = 1
                    layer.mask = torch.nn.Parameter(mask.view_as(layer.weight), requires_grad=False)
                    with torch.no_grad(): layer.weight.mul_(layer.mask)
                test.events.append(('prune',))
                return self.model
            def get_mask(self,cfgs):
                test.assertEqual(cfgs[1].attrs['split'],'train')
                test.events.append(('score',))
            def update_model(self,model): self.model=model
            def get_model(self): return self.model
            def update_forward_pass(self): pass
        patch.object(runner.prune,'FairGRAPE',FakePruner).start()
        real_save=single.save_checkpoint
        def save(model,path,*args,**kwargs):
            self.events.append(('save',Path(path).name))
            return real_save(model,path,*args,**kwargs)
        patch.object(single,'save_checkpoint',side_effect=save).start()

    def args(self, name='run', **changes):
        values=dict(dataset='CelebA',network='mobilenetv2',prune_type='FairGRAPE',impt=5,
            target_attr='Smiling',sensitive_group='gender',no_pretrained=False,no_init_train=False,
            checkpoint=None,score_only=False,drop_race=False,race_binary=False,skew_beta=0,init_pruned=0,
            fg_mode=None,save_impt=False,perf_only=False,phi_analysis=False,phi_gap_only=False,
            keep_per_iter=.99,prune_rate=.01,prune_iter=1,batch=4,stop_batch=1,no_retrain=False,
            retrain_iter=1,retrain_lr=1e-5,single_attr_smoke_samples=0,impt2_alpha=None,
            impt2_layer_max_remove_ratio=None,seed=42,no_grl=True,exp_idx=0,save_mask=False,
            celeba_csv='unused',celeba_image_root='unused',celeba_partition_file='unused',
            skip_readable_check=True,single_attr_run_dir=str(self.root/name),para_batch=1,delta_p=.5,
            defer_test=False,lr_decay_iter=15)
        values.update(changes);return SimpleNamespace(**values)

    def dense(self):
        args=self.args('dense',prune_iter=0,defer_test=True)
        runner.run(args)
        return next((self.root/'dense'/'models'/'dense').glob('BEST_MODEL_dense_*.pt'))

    def test_test_only_after_final_save_and_all_training_uses_train_val(self):
        runner.run(self.args())
        self.initializer.assert_called_once_with(weights=runner.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.assertEqual(sum(e==('eval','test') for e in self.events),1)
        loader_pos=self.events.index(('loader','test','test'))
        self.assertTrue(any(e[0]=='save' and e[1].startswith('BEST_MODEL_final_') for e in self.events[:loader_pos]))
        self.assertEqual(self.events[-1],('eval','test'))
        self.assertEqual(self.events[0],('loader','train','val'))

    def test_checkpoint_skips_initial_fit_and_both_switches_skip_all_retraining(self):
        dense=self.dense();self.fit.reset_mock();self.initializer.reset_mock();self.events.clear()
        runner.run(self.args('reload',checkpoint=str(dense),retrain_iter=0,no_retrain=True))
        self.fit.assert_not_called();self.initializer.assert_not_called()
        self.assertIn(('prune',),self.events)

    def test_round_epochs_and_final_retraining_are_independent(self):
        dense=self.dense()
        for epochs in (0, 5):
            for skip_final in (False, True):
                with self.subTest(retrain_iter=epochs, no_retrain=skip_final):
                    self.fit.reset_mock(); self.events.clear()
                    runner.run(self.args(f'rounds_{epochs}_{skip_final}', checkpoint=str(dense),
                                         prune_iter=2, retrain_iter=epochs,
                                         no_retrain=skip_final, defer_test=True))
                    expected = [[epochs], [epochs]] if epochs else []
                    if not skip_final:
                        expected.append([8, 1, 1])
                    self.assertEqual([call.args[5] for call in self.fit.call_args_list], expected)
                    self.assertEqual(sum(e == ('prune',) for e in self.events), 2)
                    self.assertNotIn(('loader', 'test', 'test'), self.events)

    def test_score_only_validates_pruned_source_and_never_trains_or_evaluates(self):
        runner.run(self.args())
        path=next((self.root/'run'/'models'/'pruning').glob('BEST_MODEL_final_*.pt'))
        self.fit.reset_mock();self.events.clear()
        runner.run(self.args('scores',checkpoint=str(path),score_only=True))
        self.fit.assert_not_called()
        self.assertIn(('score',),self.events)
        self.assertFalse(any(e[0] in ('eval','save','prune') for e in self.events))
        ck=torch.load(path,weights_only=True);ck.pop('source_checkpoint')
        torch.save(ck,path)
        with self.assertRaises(ValueError):runner.run(self.args('bad_scores',checkpoint=str(path),score_only=True))

    def test_eval_only_uses_test_without_training_or_pruning(self):
        runner.run(self.args())
        path=next((self.root/'run'/'models'/'pruning').glob('BEST_MODEL_final_*.pt'))
        self.fit.reset_mock();self.events.clear()
        runner.run(self.args('evaluation',checkpoint=str(path),single_attr_eval_only=True))
        self.fit.assert_not_called()
        self.assertEqual(self.events,[('loader','test','test'),('eval','test')])

    def test_actual_epoch_reports_and_selected_summaries_are_separated(self):
        self.fit.side_effect = REAL_FIT
        self.evaluator.side_effect = REAL_EVALUATE
        images = torch.zeros(4, 1280)
        labels = torch.tensor([[0,0], [1,0], [0,1], [1,1]])
        def loaders(train, val, *args):
            result = []
            for frame in (train, val):
                dataset = torch.utils.data.TensorDataset(images, labels)
                dataset.data_frame = pd.DataFrame({'image_name': [f'{frame.attrs["split"]}_{i}.jpg' for i in range(4)]})
                loader = torch.utils.data.DataLoader(dataset, batch_size=4)
                loader.split = frame.attrs['split']; result.append(loader)
            return tuple(result)
        with patch.object(runner, 'make_datasets', side_effect=loaders):
            runner.run(self.args('reports', no_retrain=True, retrain_iter=2))
        root = self.root/'reports'
        dense = root/'dense'
        prune = root/'pruning'/'iter_01'
        self.assertEqual(len(list(dense.glob('Total_info*.txt'))), 10)
        self.assertEqual(len(list(dense.glob('BEST_MODEL_SUMMARY*.txt'))), 1)
        self.assertEqual(len(list(prune.glob('Total_info*.txt'))), 2)
        report = (prune/'Total_info_prune_01_retrain_02_val.txt').read_text()
        for text in ('Target Attribute: Smiling', 'Evaluation Split: Validation',
                     'EOPP (Equal Opportunity gap)', 'DP (Demographic Parity gap)',
                     'Positive Prediction Rate', 'Epoch: 2 / 2'):
            self.assertIn(text, report)
        self.assertNotIn('전체 클래스 평균', report)
        records = [json.loads(line) for line in (dense/'epochs.jsonl').read_text().splitlines()]
        vals = [r for r in records if r['phase']=='val']
        self.assertEqual([r['epoch'] for r in vals], list(range(1,11)))
        self.assertTrue(all(r['eopp']==0 and r['dp']==0 for r in vals))
        for directory in (dense/'val', prune/'val', root/'pruning'/'final'/'test'):
            saved = json.loads((directory/'metrics.json').read_text())
            self.assertEqual(saved['eopp'], 0.)
            self.assertEqual(saved['dp'], 0.)
            self.assertTrue((directory/'predictions.csv').exists())
        final_report = (root/'pruning'/'final'/'FINAL_TEST_SUMMARY.txt').read_text()
        self.assertIn('Evaluation Split: Test', final_report)
        self.assertIn('DP (Demographic Parity gap): 0.0000', final_report)
        checkpoint = torch.load(next((root/'models'/'pruning').glob('BEST_MODEL_prune_01_*.pt')), weights_only=True)
        self.assertEqual(checkpoint['validation_metrics']['dp'], 0.)
        summary = pd.read_csv(root/'metrics.csv')
        self.assertEqual(summary['split'].tolist(), ['val', 'val', 'val', 'test'])
        self.assertTrue((summary['eopp']==0).all())
        self.assertTrue((summary['dp']==0).all())
        self.assertFalse(list(root.glob('*.pt')))
        self.assertTrue(list((root/'models'/'dense').glob('*.pt')))
        self.assertTrue(list((root/'models'/'pruning').glob('*.pt')))

    def test_invalid_cli_combinations(self):
        for changes in (dict(no_pretrained=True),dict(no_init_train=True),dict(target_attr=None),
                        dict(dataset='LFWA'),dict(network='resnet34'),dict(score_only=True),
                        dict(perf_only=True,phi_analysis=True),dict(single_attr_eval_only=True)):
            with self.subTest(changes=changes),self.assertRaises(ValueError): runner.validate_args(self.args(**changes))


if __name__=='__main__': unittest.main()
