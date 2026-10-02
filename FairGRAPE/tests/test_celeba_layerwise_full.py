"""Full-run guards and exact checkpoint continuation of the original trainer."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import celeba_layerwise_full as full
from celeba_layerwise_full import torch, np, pd
from torch.utils.data import DataLoader, TensorDataset


class FullTests(unittest.TestCase):
    def test_no_sample_limit_and_fixed_protocol(self):
        p=full.parser()
        args=p.parse_args(['--output-dir','/tmp/full_test'])
        self.assertEqual(args.rounds,3)
        extended=p.parse_args(['--output-dir','/tmp/full22','--rounds','22','--no_retrain'])
        self.assertEqual(extended.rounds,22)
        self.assertEqual(extended.rounds*extended.retrain_epochs,110)
        self.assertEqual(args.retrain_epochs,5)
        self.assertEqual(args.methods,list(full.METHODS))
        for flag in ('--no_retrain','--no-retrain'):
            explicit=p.parse_args(['--output-dir','/tmp/full_test',flag])
            self.assertTrue(explicit.no_retrain)
            self.assertEqual(explicit.retrain_epochs,5)
            self.assertEqual(explicit.rounds,3)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            p.parse_args(['--output-dir','/tmp/full_test','--train-samples','8192'])

    def test_tracked_loader_has_no_truncation_and_keeps_tail(self):
        x=torch.arange(93.).reshape(31,3)
        y=torch.zeros(31,40,dtype=torch.long)
        class Progress:
            def update(self,**kw): pass
        loader=full.TrackedLoader(DataLoader(TensorDataset(x,y),batch_size=8),Progress(),'test')
        batches=list(loader)
        self.assertEqual([len(a) for a,b in batches],[8,8,8,7])
        torch.testing.assert_close(torch.cat([a for a,b in batches]),x)
        self.assertEqual(loader.history[0]['samples'],31)
        self.assertEqual(loader.history[0]['batches'],4)

    @unittest.skipUnless(torch.cuda.is_available(),'original trainer requires cuda:0')
    def test_original_trainer_five_epochs_and_exact_resume(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        attrs=[f'Attr{i}' for i in range(39)]
        x=torch.randn(8,3,2,2)
        y=torch.randint(0,2,(8,40))
        y[:,39]=torch.tensor([0,0,0,0,1,1,1,1])
        def make_model():
            m=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Dropout(.2),torch.nn.Linear(12,78))
            return m.to('cuda:0')
        initial=make_model().state_dict()
        def fresh():
            m=make_model()
            m.load_state_dict(initial)
            full.base.install_masks(m)
            with torch.no_grad():
                m[2].weight[0,0]=0
                m[2].mask[0,0]=0
                m[2].bias[0]=0  # Original trainer freezes pre-existing zero parameters too.
            return m
        frames={k:pd.DataFrame({'id':range(8)}) for k in ('train','val','test')}
        def loaders(train,val,cols,progress,offset=0):
            return {phase:full.TrackedLoader(DataLoader(TensorDataset(x,y),batch_size=4),progress,phase,offset)
                    for phase in ('train','test')}
        args=SimpleNamespace(seed=42,device='cuda:0',image_dir=Path('/tmp'),no_retrain=True)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('continuous','resumed'):
                (root/name).mkdir()
            uninterrupted=fresh()
            with full.Progress(root/'continuous') as progress, \
                    patch.object(full,'legacy_loaders',side_effect=loaders), \
                    contextlib.redirect_stdout(io.StringIO()):
                a=full.original_retrain(uninterrupted,frames,attrs,args,'test',1,root/'continuous',progress,'identity')
            interrupted=fresh()
            def interrupt_after_two(history,best):
                if len(history)==2:
                    raise RuntimeError('simulated interruption')
            with full.Progress(root/'resumed') as progress, \
                    patch.object(full,'legacy_loaders',side_effect=loaders), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    full.original_retrain(interrupted,frames,attrs,args,'test',1,root/'resumed',progress,'identity',interrupt_after_two)
                b=full.original_retrain(fresh(),frames,attrs,args,'test',1,root/'resumed',progress,'identity')
            self.assertEqual(len(a['epochs']),5)
            self.assertEqual(a['total_optimizer_steps'],10)
            self.assertEqual(a['best_epoch'],b['best_epoch'])
            ca=full.load_own(root/'continuous/test_round1_latest_epoch.pt')
            cb=full.load_own(root/'resumed/test_round1_latest_epoch.pt')
            for key in ca['model_state']:
                torch.testing.assert_close(ca['model_state'][key],cb['model_state'][key],rtol=0,atol=0)
                torch.testing.assert_close(ca['best_model_state'][key],cb['best_model_state'][key],rtol=0,atol=0)
            self.assertEqual(float(ca['model_state']['2.bias'][0]),0.)
            self.assertEqual(float(ca['model_state']['2.weight'][0,0]),0.)
            for ha,hb in zip(a['epochs'],b['epochs']):
                self.assertEqual(ha['validation'],hb['validation'])
                self.assertEqual(ha['first_batch_sha256'],hb['first_batch_sha256'])
                self.assertEqual(ha['samples'],8)
                self.assertEqual(ha['optimizer_steps'],2)


if __name__=='__main__':
    unittest.main()
