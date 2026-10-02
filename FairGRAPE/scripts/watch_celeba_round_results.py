"""Continuously summarize completed evaluations from an existing official report."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

METRICS = ('ACC', 'BA', 'EO', 'EqualOpportunity', 'DemographicParity')


def render(data):
    args = data['metadata']['arguments']
    lines = [
        'A/B CelebA 회차별 테스트 결과 / 완료된 평가만 표시',
        f"STATUS: {data['status']} | 목표 {args['rounds']}회 × {args['retrain_epochs']}에폭",
        '동일한 테스트 자료의 속성별 지표 산술평균. n/39는 계산 가능한 속성 수.',
        'ACC·BA는 %, 공정성 격차·활성화 격차는 원래 척도. N/A를 0으로 채우지 않음.',
        'EO=(|ΔTPR|+|ΔFPR|)/2; EqualOpportunity=|ΔTPR|; DP=|ΔPPR|.',
        '재학습 완료 후는 회차 내 검증 best 모델 평가. 활성화 격차는 전역 점수 하나.',
        '',
        '| 상태 | 정확도 ↑ | 균형 정확도 ↑ | 기존 EO ↓ | Equal Opportunity ↓ | DP ↓ | 활성화 격차 ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    rounds = {(r['method'], r['round']): r for r in data.get('rounds', [])}
    for e in data['evaluations']:
        if e['method'] == 'dense':
            label = '초기 덴스 모델'
        else:
            label = f"{e['method']} {e['round']}회차 "
            if e['stage'] == 'pre':
                label += '프루닝 직후'
            elif e['stage'] == 'post':
                r = rounds.get((e['method'], e['round']), {})
                best = r.get('current_best_epoch') or r.get('retraining', {}).get('best_epoch')
                label += '재학습 완료 후' + (f' (best {best}에폭)' if best else '')
            else:
                label += e['stage']
        cells = [label]
        for key in METRICS:
            m = e['macro'][key]
            v = m['value']
            value = 'N/A' if v is None else f'{v*100:.3f}%' if key in ('ACC', 'BA') else f'{v:.6f}'
            cells.append(f"{value} ({m['n_attributes']}/39)")
        gap = e['global_activation_gap']
        cells.append('N/A' if gap is None else f'{gap:.6f}')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '전체 설정·39개 속성별 결과: results.txt',
              '원본 보고서 갱신 시각: ' + data.get('reporting', {}).get('generated_utc', 'unknown')]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-output', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--follow-pid', type=int)
    args = parser.parse_args()
    directory = args.source_output.resolve() / 'reports'
    source = directory / 'results.json'
    target = directory / 'round_progress.txt'
    # The experiment and existing official reporter own directory creation.
    assert directory.is_dir(), directory
    with (directory / 'round_progress.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_stat = None
        previous_evaluations = None
        stopped_polls = 0
        while True:
            if source.exists():
                st = source.stat()
                stamp = (st.st_mtime_ns, st.st_size)
                if stamp != previous_stat:
                    data = json.loads(source.read_text())
                    table = render(data)
                    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory,
                                                     prefix='.round_progress.', delete=False) as tmp:
                        tmp.write(table)
                        temporary = tmp.name
                    os.replace(temporary, target)
                    evaluations = (data['status'], json.dumps(data['evaluations'], sort_keys=True))
                    if evaluations != previous_evaluations:
                        print('UPDATED ' + datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
                        print(table, flush=True)
                        previous_evaluations = evaluations
                    previous_stat = stamp
                    if data['status'] == 'complete':
                        break
            if not args.watch:
                break
            if args.follow_pid is not None:
                try:
                    os.kill(args.follow_pid, 0)
                    stopped_polls = 0
                except ProcessLookupError:
                    stopped_polls += 1
                    if stopped_polls >= 3:
                        print('Trainer stopped; summary watcher exiting.', flush=True)
                        break
            time.sleep(2)


if __name__ == '__main__':
    main()
