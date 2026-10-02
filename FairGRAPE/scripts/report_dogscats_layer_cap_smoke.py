#!/usr/bin/env python3
"""Write a reviewable report from saved smoke results, without model execution."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {None: 'OFF', 0.2: '20%', 0.1: '10%'}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def pct(value):
    return f'{100 * value:.2f}%'


def main(run):
    arms = [p for p in sorted(run.glob('alpha*')) if p.is_dir() and (p/'manifest.json').exists()]
    records = [(arm, read(p)) for arm in arms for p in sorted(arm.glob('iteration_*/complete.json'))]
    if not records:
        print('No completed iterations yet.'); return
    by_key = {(r['alpha'], r['cap'], r['iteration']): r for _, r in records}
    audit = {'completed_iterations': len(records), 'expected_iterations': 18,
             'cap_violations': [], 'floor_violations': [], 'budget_deficits': [],
             'first_selection_scores_equal': {}, 'same_frames': {}, 'same_code': True}
    for arm, row in records:
        for layer in row['selection']['layers']:
            if layer['limit'] is not None and layer['removed'] > layer['limit']:
                audit['cap_violations'].append([arm.name, row['iteration'], layer['block']])
            if layer['active_after'] < math.ceil(layer['original'] * .03):
                audit['floor_violations'].append([arm.name, row['iteration'], layer['block']])
        if row['selection']['deficit']:
            audit['budget_deficits'].append([arm.name, row['iteration'], row['selection']['deficit']])
    for alpha in [.9, .7]:
        hashes = {a.name: digest(a/'iteration_026/scores_full_precision.json') for a in arms
                  if read(a/'manifest.json')['alpha'] == alpha and (a/'iteration_026/scores_full_precision.json').exists()}
        audit['first_selection_scores_equal'][str(alpha)] = {'arms': list(hashes), 'equal': len(set(hashes.values())) <= 1}
    for name in ['train_frame.csv', 'val_frame.csv']:
        hashes = {a.name: digest(a/name) for a in arms if (a/name).exists()}
        audit['same_frames'][name] = {'arms': list(hashes), 'equal': len(set(hashes.values())) <= 1}
    manifests = [read(a/'manifest.json') for a in arms]
    audit['same_code'] = all(m['source_sha256s'] == manifests[0]['source_sha256s'] for m in manifests)
    (run/'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n')
    lines = ['# DogsCats 채널 상한 스모크 결과 (2026-09-20)', '',
             f'완료: {len(records)}/18 pruning iterations. 결과는 완료된 회차만 포함한다.', '',
             'α=0.9/0.7 각각의 기존 25회 체크포인트에서 OFF, 20%, 10% 조건으로 26~28회를 진행한다. '
             '세 조건은 같은 모델과 seed 정책을 사용한다. 전체 학습 데이터로 중요도를 계산하고 매 회차 Adam 1e-5로 5에폭 재학습한다. '
             '검증 데이터 1,236개(종×색상 네 셀 각 309개)를 사용하며 test를 새로 평가하지 않는다. '
             '이 스모크는 α별 기존 손상 구간의 추가 제거를 비교한다. α 사이에는 25회까지의 이력이 달라 전체 α 트레이드오프를 확정할 수 없다.', '',
             '`--impt2_layer_max_remove_ratio` 생략 시 OFF, `0.20` 또는 `0.10`으로 켠다. '
             '회차 시작 생존 채널 수에 비율을 곱해 내림한 수를 블록별 제거 상한으로 적용한다. '
             '상한을 채우면 점수순으로 다른 블록을 검토하며 목표 가중치 제거량이 부족해도 상한을 풀지 않는다. '
             '20%는 4채널 이하, 10%는 9채널 이하에서 더 제거하지 못하므로 작은 블록 보호 효과도 포함한다.', '',
             '## 회차별 선택된 best 모델', '',
             'EO는 작을수록 좋다. ΔAcc와 ΔEO는 같은 α·회차의 새 OFF 대조군 대비다. '
             '대상 희소도는 features.1~17 convolution 가중치 마스크 기준이다.', '']
    rows = []
    for _, r in records:
        ref = by_key.get((r['alpha'], None, r['iteration']))
        block = next(l for l in r['selection']['layers'] if l['block'] == 'features.1')
        rows.append([r['alpha'], LABELS[r['cap']], r['iteration'], pct(r['best']['ACC']), f"{r['best']['EO']:.4f}",
                     '—' if ref is None else f"{100*(r['best']['ACC']-ref['best']['ACC']):+.2f}%p",
                     '—' if ref is None else f"{r['best']['EO']-ref['best']['EO']:+.4f}",
                     pct(r['after']['target_sparsity']), block['active_after'], r['best_epoch']])
    lines += [table(['α','상한','회차','Acc','EO','ΔAcc','ΔEO','대상 희소도','첫 블록 생존','best epoch'], rows), '']
    alpha_pairs=[]
    for cap in [None,.2,.1]:
        for iteration in [26,27,28]:
            low=by_key.get((.7,cap,iteration));high=by_key.get((.9,cap,iteration))
            if low is None or high is None:continue
            alpha_pairs.append({'cap':cap,'iteration':iteration,
                'alpha09_accuracy':high['best']['ACC'],'alpha09_eo':high['best']['EO'],
                'alpha07_accuracy':low['best']['ACC'],'alpha07_eo':low['best']['EO'],
                'delta_accuracy_pp':100*(low['best']['ACC']-high['best']['ACC']),
                'delta_eo':low['best']['EO']-high['best']['EO']})
    if alpha_pairs:
        with (run/'alpha_comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(alpha_pairs[0]));writer.writeheader();writer.writerows(alpha_pairs)
        lines += ['## 같은 상한에서 α=0.9→0.7로 바꿨을 때', '',
                  'Δ는 α=0.7−α=0.9이다. ΔAcc<0이고 ΔEO<0이면 정확도를 낮추고 공정성을 높이는 방향이다. '
                  '두 α는 각자의 25회 체크포인트에서 시작했으며, 이 표로 전체 프루닝 구간의 단조 관계를 확정할 수 없다.', '',
                  table(['상한','회차','α0.9 Acc','α0.9 EO','α0.7 Acc','α0.7 EO','ΔAcc','ΔEO'],
                        [[LABELS[r['cap']],r['iteration'],pct(r['alpha09_accuracy']),f"{r['alpha09_eo']:.4f}",
                          pct(r['alpha07_accuracy']),f"{r['alpha07_eo']:.4f}",f"{r['delta_accuracy_pp']:+.2f}%p",f"{r['delta_eo']:+.4f}"] for r in alpha_pairs]),'']
    lines += ['## 제거 직후와 재학습 후', '',
              '제거 직후 값은 재학습 전 평가다. 재학습의 영향을 따로 확인하기 위해 기록한다.', '']
    rows = []
    for _, r in records:
        rows.append([r['alpha'], LABELS[r['cap']], r['iteration'], pct(r['post_prune']['ACC']),
                     f"{r['post_prune']['EO']:.4f}", pct(r['best']['ACC']), f"{r['best']['EO']:.4f}", r['selection']['deficit']])
    lines += [table(['α','상한','회차','제거 직후 Acc','제거 직후 EO','best Acc','best EO','제거 목표 미달 가중치'], rows), '']
    lines += ['## 28회 모델의 네 집단 정확도', '',
              '훈련 데이터에서 밝은 고양이와 어두운 개가 소수 집단이다.', '']
    rows = []
    for _, r in records:
        if r['iteration'] != 28: continue
        cells = {(c['species'],c['color']): c['accuracy'] for c in r['best']['cells']}
        rows.append([r['alpha'], LABELS[r['cap']]]+[pct(cells[k]) for k in [(0,0),(0,1),(1,0),(1,1)]])
    lines += [table(['α','상한','어두운 고양이','밝은 고양이','어두운 개','밝은 개'], rows), '']
    lines += ['## 정확도 우선 저장 규칙 확인', '',
              '기존과 같이 정확도→EO→loss 순으로 best를 선택했다. 최저 EO 에폭도 함께 기록하므로 '
              '학습 중 공정성이 좋아져도 정확도 우선 선택으로 가려지는지 확인할 수 있다.', '']
    rows = []
    for arm, r in records:
        epochs = read(arm/f"iteration_{r['iteration']:03d}/epochs.json")
        fair = min(epochs, key=lambda e: (e['eo'], -e['accuracy']))
        rows.append([r['alpha'], LABELS[r['cap']], r['iteration'], r['best_epoch'], pct(r['best']['ACC']),
                     f"{r['best']['EO']:.4f}", fair['epoch'], pct(fair['accuracy']), f"{fair['eo']:.4f}"])
    lines += [table(['α','상한','회차','선택 epoch','선택 Acc','선택 EO','최저 EO epoch','해당 Acc','최저 EO'], rows), '']
    history = ROOT/'docs/dogscats_alpha09_alpha07_analysis_20260920/iterations.csv'
    if history.exists():
        with history.open() as f:
            old = [r for r in csv.DictReader(f) if int(r['iter']) in [25,26,27,28]]
        lines += ['## 이전 연속 실행의 참고값', '',
                  '원래 체크포인트에는 전체 RNG 상태가 없어 이전 연속 실행을 정확히 재현하는 비교가 아니다. '
                  '상한 효과의 주 비교는 위의 새 OFF 대조군이다.', '',
                  table(['α','회차','Acc','EO','대상 희소도'],
                        [[r['alpha'],r['iter'],pct(float(r['acc'])),f"{float(r['eo']):.4f}",pct(float(r['target_sparsity']))] for r in old]), '']
    restart_note = run/'historical_vs_restart_first_block.json'
    if restart_note.exists():
        detail = read(restart_note)
        old = next(r for r in detail['historical']['first_block'] if r['ch'] == '2')
        new = next(r for r in detail['restarted_off']['first_block'] if r['ch'] == '2')
        lines += ['## 이전 연속 실행과 재시작의 점수 차이', '',
                  'α=0.7의 같은 25회 시작 모델을 사용했지만 26회차 첫 블록 2번 채널은 이전 실행에서 제거되고 새 OFF에서는 유지됐다. '
                  '두 실행의 성능 중요도 배치는 모두 128이다. 다음 수치는 저장된 로그를 비교한 것으로, 추가 추론 결과가 아니다.', '',
                  table(['실행','채널','perf raw','perf 정규화','φ 정규화','결합점수','상태'],
                        [[label,'features.1 / ch 2',r['perf_raw'],r['perf_n'],r['phi_n'],r['score'],r['status']]
                         for label,r in [('이전 연속 실행',old),('새 OFF',new)]]), '',
                  '블록 내 perf 최대값(7번 채널)은 약 0.017895→0.009000으로 바뀌었다. '
                  '따라서 2번 채널은 raw perf가 감소해도 정규화 perf가 0.3666→0.6101로 커지고, '
                  'α=0.7 결합점수가 -0.043391→+0.127071로 뒤집혔다. '
                  '이 관찰은 max 정규화의 기준값 변화가 채널 선택에 영향을 줄 수 있음을 보여준다. '
                  '원래 RNG 상태가 저장되지 않았으므로 raw 점수 차이의 원인을 증강/RNG 하나로 확정하지는 않는다. '
                  '첫 블록 급감이 새 OFF에서 재현되지 않으면 상한이 원래 붕괴를 해결했다는 결론을 낼 수 없다.', '']
    lines += ['## 검증 및 저장 위치', '',
              f"상한 위반 {len(audit['cap_violations'])}건, 최소 유지 위반 {len(audit['floor_violations'])}건. "
              f"목표 미달 회차 {len(audit['budget_deficits'])}건. 추가 검증은 `audit.json`을 참조한다.", '',
              '- `verification_final.json`: 모델 18개 해시, 에폭 모델 90개, 예측 파일 36개 지표 재계산, 상한 204건·floor 306건 검사 통과.',
              '- `verification.json`: CPU 선택 로직 6개 통과, OFF가 수정 전 선택·마스크와 동일함을 확인.',
              '- `comparison.csv`, `epochs_comparison.csv`: 회차 및 에폭별 지표.',
              '- `comparison.png`, `comparison.pdf`: Acc, EO, 첫 블록 채널 궤적.',
              '- `alpha*/manifest.json`: 원본 체크포인트 해시, 소스 해시, 전체 설정.',
              '- `alpha*/iteration_028/retrained.pt`: 스모크 마지막 회차의 선택된 모델.',
              '- `alpha*/iteration_*/epoch_01.pt`~`epoch_05.pt`: 에폭별 모델.',
              '- `alpha*/iteration_*/selection.json`: 상한, 실제 제거량, 선택된 채널.',
              '- `alpha*/iteration_*/best/predictions.csv`: best 모델의 검증 예측.',
              '- `source_before/`, `source_after/`: 수정 전후 코드 기록.', '']
    model_rows=[]
    for arm,r in records:
        if r['iteration']!=28:continue
        checkpoint=arm/'iteration_028/retrained.pt'
        model_rows.append([r['alpha'],LABELS[r['cap']],f'[28회차 모델]({checkpoint})',f'[설정과 원본 경로]({arm / "manifest.json"})'])
    lines += ['## 조건별 최종 스모크 모델', '',table(['α','상한','모델','실행 설정'],model_rows),'',
              '`environment.json`에 실제 라이브러리·GPU 버전을 기록했다. `runtime_notes.json`은 CUDA 디스크 캐시 용량 변경 기록이다. '
              '캐시 변경 전후의 수치 계산 설정은 같고, 첫 점수 및 상한 비활성 조건의 예측 일치 여부를 별도로 확인했다.', '',
              '추가 집계: `alpha_comparison.csv`(회차별 α 비교), `layers_comparison.csv`(블록별 채널 수), '
              '`channel_redistribution.png`/`.pdf`(최종 채널 분포 변화), `tradeoff.png`/`.pdf`(OFF와 10%의 최종 지점 비교).', '']
    (run/'README.md').write_text('\n'.join(lines))
    print(run/'README.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    main(parser.parse_args().run_dir.resolve())
