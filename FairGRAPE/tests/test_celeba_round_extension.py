"""Continuation preserves checkpoint tensors/RNG and rejects a changed protocol."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import celeba_layerwise_full as full
from celeba_layerwise_full import torch


class ExtensionTests(unittest.TestCase):
    def test_saved_epoch_and_cache_keep_tensor_state_and_reject_changed_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source_run'; out=root/'extended'
            (source/'source').mkdir(parents=True); (source/'cache').mkdir(); out.mkdir()
            source_hash={}
            for name in full.SOURCES:
                text=(full.ROOT/name).read_bytes()
                if name=='celeba_layerwise_experiment.py':
                    text=text.replace(b'if n < 1 or pool_n < remove_n or protected_n + pool_n > n:',
                                      b'if n < 10 or pool_n < remove_n or protected_n + pool_n > n:')
                (source/'source'/name).write_bytes(text)
                source_hash[name]=full.base.sha(source/'source'/name)
            common={k:'same' for k in ('checkpoint_sha256','csv_sha256','annotation','split_sizes','sample_sizes','analysis_layers','environment')}
            common['sample_files']={k:{'sha256':'same'} for k in ('selection','retrain','validation','evaluation')}
            old=dict(copy.deepcopy(common),fingerprint='old',source_sha256=source_hash,
                     arguments=dict(seed=42,rounds=3,output_dir=str(source),methods=['A_activation'],retrain_epochs=5,no_retrain=True))
            entry=dict(method='A_activation',round=1,layers={'layer':{'live_before':32}},epochs_completed=3)
            (source/'results.json').write_text(json.dumps(dict(status='interrupted',metadata=old,rounds=[entry],evaluations=[],elapsed_seconds=100.)))
            state=dict(fingerprint='old:dense_round1A_activation',method='A_activation',round=1,best_epoch=3,
                       history=[{'epoch':i} for i in range(1,5)],model_state={'w':torch.tensor([1.,2.])},
                       optimizer_state={'moment':torch.tensor([3.,4.])},rng_state={'torch':torch.Generator().manual_seed(4).get_state()},
                       round_zero_masks={'w':torch.tensor([0.,1.])})
            path=source/'A_activation_round1_latest_epoch.pt'; full.experiment.atomic_save(state,path)
            original_hash=full.base.sha(path)
            full.experiment.atomic_save({'fingerprint':'old:dense_round1','data':{'x':torch.tensor([7.])}},source/'cache/dense_round1_signals.pt')
            new=dict(copy.deepcopy(common),fingerprint='new',source_sha256={n:full.base.sha(full.ROOT/n) for n in full.SOURCES},
                     arguments=dict(old['arguments'],rounds=22,output_dir=str(out),continue_from=str(source)))
            result=dict(metadata=new,rounds=[],evaluations=[])
            elapsed=full.continue_previous_run(source,out,result,'new')
            self.assertEqual(elapsed,100.)
            self.assertEqual(result['metadata']['continuation']['completed_epochs_preserved'],4)
            self.assertEqual(result['rounds'][0]['epochs_completed'],4)
            self.assertEqual(full.base.sha(path),original_hash)
            saved=full.load_own(out/path.name)
            self.assertEqual(saved['fingerprint'],'new:dense_round1A_activation')
            for key,name in [('model_state','w'),('optimizer_state','moment'),('rng_state','torch'),('round_zero_masks','w')]:
                torch.testing.assert_close(saved[key][name],state[key][name],rtol=0,atol=0)
            self.assertEqual(full.load_own(out/'cache/dense_round1_signals.pt')['fingerprint'],'new:dense_round1')
            bad=copy.deepcopy(result);bad['metadata']['arguments']['seed']=43
            with self.assertRaisesRegex(AssertionError,'argument mismatch'):
                full.continue_previous_run(source,root/'reject',bad,'bad')


if __name__=='__main__':unittest.main()
