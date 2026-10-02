#!/usr/bin/env python3
"""Run reviewed max/gamma conditions using the unchanged five-epoch trainer."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_lfwa_channel_cap_diagnostic as runner
from build_lfwa_gamma_screen import selection


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--screen-dir',type=Path,required=True);parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    root=args.screen_dir.resolve();manifest=json.loads((root/'manifest.json').read_text())
    for f,h in manifest['source_sha256'].items():
        if runner.sha256(ROOT/f)!=h:raise ValueError('Reviewed source changed: '+f)
    if args.dry_run:os.environ['CUDA_VISIBLE_DEVICES']=''
    original=runner.configure
    def configure(plan,directory,metadata):
        original(plan,directory,metadata)
        import prune
        prune.IMPT2_PROTECTION_RATIO=plan['protection_ratio']
        assert prune.IMPT2_NORM=='max' and prune.IMPT_TYPE2_ALPHA==plan['alpha']
    runner.configure=configure
    outcomes=[]
    for condition in manifest['conditions']:
        name=condition['name'];plan_path=Path(condition['plan'])
        assert runner.sha256(plan_path)==condition['plan_sha256']
        plan=runner.read_plan(plan_path)
        assert plan['alpha'] in [.5,.7] and plan['protection_ratio'] in [.005,.05,.15]
        assert plan['retrain_lr']==1e-4 and plan['retrain_epochs']==5 and plan['seed']==42
        reuse=condition['reuse']
        if reuse:
            if 'run_dir' in reuse:
                assert runner.sha256(Path(reuse['run_dir'])/reuse['arm']/'retrained.pt')==reuse['checkpoint_sha256']
            outcomes.append({'condition':name,'status':'reused','reuse':reuse});continue
        run=root/('dry_runs' if args.dry_run else 'runs')/name
        complete=run/('dry_run.json' if args.dry_run else 'complete.json')
        if complete.exists():
            meta=json.loads((run/('dry_run_metadata.json' if args.dry_run else 'diagnostic_run.json')).read_text())
            assert meta['plan']==json.loads(plan_path.read_text()), 'Completed condition plan changed'
            for source_file,digest in meta['source_sha256'].items():
                assert digest==manifest['source_sha256'][source_file]
            assert meta['plan']['alpha']==plan['alpha'] and meta['plan']['protection_ratio']==plan['protection_ratio']
            outcomes.append({'condition':name,'status':'already_complete','run_dir':str(run)});continue
        if run.exists():raise FileExistsError('Unfinished run requires review: '+str(run))
        if not args.dry_run:
            assert (root/'dry_runs'/name/'dry_run.json').exists()
            check=selection(plan,device='cuda:0')
            assert check['selected_channels']==plan['arms'][0]['selected_channels']
            assert check['expected_removed_weights']==plan['arms'][0]['expected_removed_weights']
            assert set(check['protected_channels'])==set(map(tuple,plan['protected_channel_ids']))
            runner.write_json(root/'gpu_preflight'/f'{name}.json',{'status':'passed',**check})
        runner.write_json(root/'status.json',{'status':'running','condition':name,'dry_run':args.dry_run})
        sys.argv=[sys.argv[0],'--plan',str(plan_path),'--run-dir',str(run)]+(['--dry-run'] if args.dry_run else [])
        runner.main()
        outcomes.append({'condition':name,'status':'complete','run_dir':str(run)})
        runner.write_json(root/('dry_run_progress.json' if args.dry_run else 'progress.json'),outcomes)
    runner.write_json(root/('dry_run_complete.json' if args.dry_run else 'complete.json'),{'status':'complete','outcomes':outcomes})
    runner.write_json(root/'status.json',{'status':'complete','dry_run':args.dry_run})

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('SCREEN_FAILED',repr(error),flush=True);raise
