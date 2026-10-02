"""Report existing CelebA results using mean-gap EO, EOpp and DP only.

CPU/standard-library reporting; never changes training, source results or checkpoints.
A watcher refreshes the report after each atomic source-results update.
"""
import argparse
import copy
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

FORMULAS = '''FAIRNESS DEFINITIONS / 적용 계산식
For attribute t and Male group g:
TPR[t,g] = TP/(TP+FN); FPR[t,g] = FP/(FP+TN); PPR[t,g] = (TP+FP)/(TP+FN+FP+TN).
EO[t] = (abs(TPR[t,0]-TPR[t,1]) + abs(FPR[t,0]-FPR[t,1])) / 2.
EqualOpportunity[t] = abs(TPR[t,0]-TPR[t,1]).
DemographicParity[t] = abs(PPR[t,0]-PPR[t,1]).
EO is the original mean-gap definition. All fairness gaps are lower-is-better.
Per-metric macro = arithmetic mean over the SAME valid attributes across all states.
Zero denominator -> N/A; never zero-imputed. Report metric-specific n/39 and names.
ACC=(TP+TN)/N; BA=(global TPR+global TNR)/2; Male is sensitive only, never a task.

ACTIVATION / 활성화 및 선택 정의
Layers: depthwise Conv outputs before BN in features.1..17.
a[i,l,c] = mean_HW(A[i,l,c,h,w]), retaining SIGN; group means are per IMAGE.
d[l,c,condition] = mean_Male0(a)-mean_Male1(a).
D[l,K,condition] = abs(sum over kept c of d[l,c,condition]).
A_activation uses ONE global unconditional score, never duplicated over 39 tasks.
B_activation uses mean_t max(D[l,K,y_t=1],D[l,K,y_t=0]); each gap is reported.
Evaluation activation gaps are arithmetic means across the 17 layers.
Performance: original batch average (weight * task-loss gradient)^2 and original
channel aggregation. Task loss averages 39 target losses excluding Male.
Each layer: protect ceil(.10*N); bottom floor(.30*N) is eligible; others fixed.
Greedily ADD eligible channels to fixed keep until N-floor(.10*N) remain.
Tie: objective rounded to 12 decimals, higher performance importance, lower channel ID.
Round-start frozen statistics guide selection; actual pruned models are evaluated.
Every configured round recomputes statistics after retraining. Five full epochs per round.
Best validation: ACC, then original mean-gap EO, then task loss.
Deferred prediction-contribution metrics are N/A (n=0/39), separately from empty cells.
'''
FAIRNESS = ('EO', 'EqualOpportunity', 'DemographicParity')


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def divide(n, d):
    return n / d if d else None


def gap(a, b):
    return abs(a-b) if finite(a) and finite(b) else None


def attribute_metrics(row):
    """Recompute all three fairness gaps from saved group confusion counts."""
    rates = []
    for g in (0, 1):
        tp, fn, fp, tn = [row[f'Male{g}_{k}'] for k in ('TP','FN','FP','TN')]
        assert all(isinstance(x, int) and x >= 0 for x in (tp,fn,fp,tn))
        rates.append((divide(tp,tp+fn), divide(fp,fp+tn), divide(tp+fp,tp+fn+fp+tn)))
    dt, df, dp = [gap(rates[0][i], rates[1][i]) for i in range(3)]
    return dict(EO=(dt+df)/2 if dt is not None and df is not None else None,
                EqualOpportunity=dt, DemographicParity=dp)


def transform(raw):
    result = copy.deepcopy(raw)
    records = result['evaluations']
    names = [a['attribute'] for a in records[0]['attributes']] if records else []
    if names:
        assert len(names) == len(set(names)) == 39 and 'Male' not in names
    for ev in records:
        assert [a['attribute'] for a in ev['attributes']] == names
        for row in ev['attributes']:
            computed = attribute_metrics(row)
            # Check against the original stored metric, never silently reinterpret it.
            for old, new in [('legacy_EO_mean_gap','EO'),('EqualOpportunity','EqualOpportunity'),('DemographicParity','DemographicParity')]:
                if old in row:
                    x,y = row[old], computed[new]
                    assert (x is None and y is None) or (finite(x) and finite(y) and abs(x-y)<1e-12), (row['attribute'],old,x,y)
            row.pop('EqualizedOdds',None)
            row.pop('legacy_EO_mean_gap',None)
            row.update(computed)
        ev.pop('macro',None)
    if records:
        keys = [k for k in records[0]['attributes'][0] if k not in ('attribute','activation_all_global_ref')
                and not k.endswith(('_TP','_TN','_FP','_FN'))]
        masks = {k:[i for i in range(39) if all(finite(e['attributes'][i][k]) for e in records)] for k in keys}
        for ev in records:
            ev['macro'] = {k:dict(value=sum(ev['attributes'][i][k] for i in ids)/len(ids) if ids else None,
                                  n_attributes=len(ids),attributes=[names[i] for i in ids]) for k,ids in masks.items()}
    result['reporting'] = dict(fairness_metrics=list(FAIRNESS),EO_definition='mean of absolute TPR and FPR group gaps',
                               aggregation='common valid attribute arithmetic mean per metric',
                               evaluation_repeated=False,training_changed=False)
    return result


def fmt(x):
    if isinstance(x,str): return x
    return f'{x:.6f}' if finite(x) else 'N/A'


def table(rows, columns):
    lines = [[str(k) for k in columns]] + [[fmt(r.get(k)) for k in columns] for r in rows]
    widths = [max(len(row[i]) for row in lines) for i in range(len(columns))]
    return '\n'.join('  '.join(cell.ljust(widths[i]) for i,cell in enumerate(row)) for row in lines)


def comparison(records, final_only=False, final_round=3):
    rows=[]
    for ev in records:
        if final_only and not (ev['method']=='dense' or (ev['round']==final_round and ev['stage']=='post')):
            continue
        row = dict(state=f'{ev["method"]} r{ev["round"]} {ev["stage"]}',activation_global=ev['global_activation_gap'])
        for k in ('ACC','BA')+FAIRNESS+('activation_positive','activation_negative'):
            m=ev['macro'][k]
            row[k]=f'{fmt(m["value"])} ({m["n_attributes"]}/39)'
        rows.append(row)
    return table(rows,['state','ACC','BA','EO','EqualOpportunity','DemographicParity','activation_global','activation_positive','activation_negative'])


def render(result):
    records=result['evaluations']
    rounds=result.get('metadata',{}).get('arguments',{}).get('rounds',3)
    lines=['CELEBA / 기존 평균 EO · Equal Opportunity · Demographic Parity',
           'STATUS: '+result['status'],FORMULAS,
           'CONFIGURATION / 실행 인자·데이터·설정',json.dumps(result['metadata'],ensure_ascii=False,indent=2),
           'REPORT PROVENANCE / 원본 결과 및 보고 설정',json.dumps(result['reporting'],ensure_ascii=False,indent=2),
           'AVAILABLE EVALUATIONS / 현재 완료된 평가 (pre=재학습 전, post=재학습 후)',comparison(records),
           f'FINAL {rounds}-ROUND COMPARISON / 완료된 {rounds}회 종료 모델만 포함',comparison(records,True,final_round=rounds)]
    if records:
        lines += ['COMMON VALID ATTRIBUTE SETS / 지표별 동일 유효 속성']
        for k,m in records[0]['macro'].items():
            lines.append(f'{k}: {m["n_attributes"]}/39: '+', '.join(m['attributes']))
    lines += ['ROUND SELECTIONS AND TIMINGS / 모든 회차별 채널·시간',json.dumps(result['rounds'],ensure_ascii=False,indent=2)]
    for ev in records:
        lines += ['='*100,f'{ev["method"]} round={ev["round"]} {ev["stage"]} seconds={ev["seconds"]:.3f}',
                  f'GLOBAL activation gap (one value): {ev["global_activation_gap"]:.9f}','MACRO MEANS / 속성 산술평균']
        for k,m in ev['macro'].items():
            lines.append(f'{k:28s} {fmt(m["value"])} n={m["n_attributes"]}/39')
        attrs=ev['attributes']
        for columns in [['attribute','ACC','BA'],['attribute']+[k for k in attrs[0] if k.startswith('Male')],
                        ['attribute']+list(FAIRNESS),['attribute']+[k for k in attrs[0] if k.startswith(('activation_','contribution_'))]]:
            lines.append(table(attrs,columns))
    if records:
        dense=records[0]
        lines.append('REGRESSIONS VERSUS DENSE / 덴스 대비 변화 및 모든 속성·그룹 악화')
        for ev in records[1:]:
            da=ev['global_activation_gap']-dense['global_activation_gap']
            lines += [f'{ev["method"]} round={ev["round"]} {ev["stage"]}',f'global activation delta={da:+.9f}']
            for k in ('ACC','BA')+FAIRNESS:
                a,b=dense['macro'][k]['value'],ev['macro'][k]['value']
                if finite(a) and finite(b): lines.append(f'macro {k}: {a:.7f} -> {b:.7f} delta={b-a:+.7f}')
            for a,b in zip(dense['attributes'],ev['attributes']):
                changes=[]
                lower_bad={'ACC','BA','Male0_ACC','Male1_ACC','Male0_TPR','Male1_TPR'}
                for k in ('ACC','BA','Male0_ACC','Male1_ACC','Male0_TPR','Male1_TPR','Male0_FPR','Male1_FPR')+FAIRNESS:
                    if finite(a[k]) and finite(b[k]):
                        delta=b[k]-a[k]
                        if (delta < -1e-12 if k in lower_bad else delta > 1e-12): changes.append(f'{k}:{a[k]:.6f}->{b[k]:.6f}')
                if changes: lines.append(b['attribute']+': '+'; '.join(changes))
    lines.append(f'TOTAL ELAPSED SECONDS (at latest training report): {result.get("elapsed_seconds",0):.3f}')
    if result.get('error'): lines.append('ERROR: '+result['error'])
    return '\n\n'.join(lines)+'\n'


def atomic_write(path,text):
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix='.'+path.name,delete=False) as f:
        f.write(text)
        name=f.name
    os.replace(name,path)


def update(source, out, payload):
    result=transform(json.loads(payload))
    result['reporting'].update(source=str(source),source_sha256=hashlib.sha256(payload).hexdigest(),
                               generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_write(out/'results.json',json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    atomic_write(out/'results.txt',render(result))
    atomic_write(out/'comparison.txt','STATUS: '+result['status']+'\n'+comparison(result['evaluations'],True,final_round=result.get('metadata',{}).get('arguments',{}).get('rounds',3))+'\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-output',type=Path,required=True)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--follow-pid',type=int)
    args=p.parse_args(argv)
    root=args.source_output.resolve()
    out=root/'reports'
    # Wait for the trainer to create its directory; creating it would break new-run checks.
    def alive():
        if args.follow_pid is None: return True
        try: os.kill(args.follow_pid,0); return True
        except ProcessLookupError: return False
    while not root.exists():
        if not args.watch or not alive(): raise FileNotFoundError(root)
        time.sleep(2)
    out.mkdir(exist_ok=True)
    with (out/'report.lock').open('a') as lock:
        try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Report watcher already running:',out,flush=True)
            return
        source=root/'results.json'
        previous=None
        while True:
            if source.exists():
                payload=source.read_bytes()
                digest=hashlib.sha256(payload).digest()
                if digest!=previous:
                    result=update(source,out,payload)
                    previous=digest
                    print(f'EO REPORT UPDATED status={result["status"]} evaluations={len(result["evaluations"])} file={out/"results.txt"}',flush=True)
                    print(comparison(result['evaluations']),flush=True)
            if not args.watch: break
            status_path=root/'status.json'
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get('status')=='complete' or not alive():
                # Final source read covers the interval between update and trainer exit.
                if source.exists():
                    result=update(source,out,source.read_bytes())
                    print('FINAL REPORT / EO · EqualOpportunity · DemographicParity\n'+comparison(result['evaluations'],True,final_round=result.get('metadata',{}).get('arguments',{}).get('rounds',3)),flush=True)
                break
            time.sleep(2)


if __name__=='__main__':
    main()
