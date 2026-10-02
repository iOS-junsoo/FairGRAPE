"""Produce a short Korean reading guide from completed experiment results."""
import argparse
import itertools
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output_dir', type=Path)
    args = p.parse_args()
    out = args.output_dir
    r = json.loads((out/'results.json').read_text())
    assert r['status'] == 'complete'
    dense = r['evaluations'][0]
    finals = [x for x in r['evaluations'] if x['round'] == 3 and x['stage'] == 'post']
    lines = ['CelebA 레이어별 3회 프루닝·회차당 5 epoch 재학습: 스모크 결과',
             '원본 전체 train 분할에 대한 본실험이 아니라, 아래 표본을 사용한 실행 검증이다.',
             '표본 수: '+str(r['metadata']['sample_sizes']),
             '과제: Male을 제외한 39속성. 민감속성: Male. 동일한 평가 자료 사용.',
             'features.1~17 각 층: 상위 10% 보호 + 중간 고정, 하위 30%에서만 10% 제거.',
             '회차별 제거 708 / 634 / 569, 총 1,911 / 7,136 (26.78%), 최종 5,225채널.',
             '마스크를 이용한 논리적 제거이며, 모델 텐서 크기나 실행 속도 감소를 의미하지 않는다.',
             '활성화는 부호 유지 공간 평균. 기여는 양성 로짓의 그래디언트×활성화.',
             '각 층에서 채널의 부호 있는 그룹 평균 차이를 더한 뒤 절댓값, 층간 산술평균.',
             '활성화 전체 격차는 전역 값 하나이며, 39속성의 중복 평균에 포함하지 않는다.',
             '최종 분류는 원래 2-class softmax 예측. N/A 지표를 0으로 채우지 않는다.', '']
    metrics = ['ACC','BA','EqualOpportunity','EqualizedOdds','DemographicParity']
    lines.append('모델                       ACC(%)   BA(%)    EOpp     EOmax    DP       전체활성화격차')
    for e in [dense]+finals:
        v = [e['macro'][k]['value'] for k in metrics]
        lines.append(f'{e["method"]:25s} {v[0]*100:8.3f} {v[1]*100:8.3f} {v[2]:8.5f} {v[3]:8.5f} {v[4]:8.5f} {e["global_activation_gap"]:12.7f}')
    lines.append('유효 속성: '+', '.join(f'{k}={dense["macro"][k]["n_attributes"]}/39' for k in metrics))
    lines.append('정확도와 BA는 클수록, EOpp/EOmax/DP는 작을수록 좋다. 신호 격차 감소만으로 공정성 개선을 단정하지 않는다.')
    lines.append('\n최종 모델의 덴스 대비 변화와 악화 사례 (pp=퍼센트포인트)')
    for e in finals:
        lines.append('\n'+e['method'])
        delta = e['global_activation_gap']-dense['global_activation_gap']
        lines.append(f'  전역 활성화 격차 변화: {delta:+.7f}')
        for k in metrics+['activation_positive','activation_negative','contribution_all','contribution_positive','contribution_negative']:
            v, b = e['macro'][k], dense['macro'][k]
            d = v['value']-b['value']
            lines.append(f'  {k}: {b["value"]:.7f} -> {v["value"]:.7f}, 변화={d:+.7f}, n={v["n_attributes"]}/39')
        for k in ['ACC','BA','Male0_ACC','Male1_ACC','Male0_TPR','Male1_TPR','EqualizedOdds']:
            changes = []
            for x,b in zip(e['attributes'], dense['attributes']):
                if x[k] is None or b[k] is None:
                    continue
                d = x[k]-b[k]
                if (d > 1e-12 if k == 'EqualizedOdds' else d < -1e-12):
                    changes.append((abs(d),x['attribute'],b[k],x[k]))
            changes.sort(reverse=True)
            lines.append(f'  {k} 악화 속성 수: {len(changes)}; 최대 3개: '+ '; '.join(
                f'{name} {b:.4f}->{v:.4f} ({(v-b)*100:+.2f}pp)' for _,name,b,v in changes[:3]))
    lines.append('\n방법별 최종 제거 집합의 차이')
    masks = {x['method']:set(map(tuple,x['removed_cumulative'])) for x in r['rounds'] if x['round']==3}
    for a,b in itertools.combinations(masks,2):
        intersect, union = len(masks[a]&masks[b]), len(masks[a]|masks[b])
        lines.append(f'  {a} / {b}: 공통 제거 {intersect}, Jaccard={intersect/union:.4f}')
    lines.append('\n재학습 검증: 매회 5 epoch, epoch마다 전체 지정 표본 사용; 네 방법 간 증강 첫 배치 해시 동일.')
    lines.append('각 회차는 validation ACC, 기존 EO(mean gap), loss 순으로 고른 best epoch를 복원한다.')
    nsel = r['metadata']['sample_sizes']['selection']
    ibs = r['metadata']['arguments']['importance_batch_size']
    nb, tail = divmod(nsel, ibs)
    lines.append(f'성능 중요도 표본: {ibs}장짜리 {nb}배치'+(f' + 마지막 {tail}장 배치' if tail else '')+'. 원본대로 배치별 중요도를 동등 평균한다.')
    lines.append('선택 자료에는 희귀 정답×Male 조합을 보충했다. 일부 셀은 표본이 매우 적으며, 마지막 작은 배치도 동일 가중치이므로 선택 중요도의 대표성에 한계가 있다.')
    lines.append('모든 39속성·그룹 지표, 누락 셀, 계산식, 제거 채널 및 회차별 시간은 results.txt 참조.')
    lines.append(f'실험 경과 시간: {r["elapsed_seconds"]:.3f}초 ({r["elapsed_seconds"]/60:.2f}분); Python import 전 대기/별도 감사 시간 제외.')
    lines.append('\n정확한 재실행 명령 (output-dir은 새 경로로 변경):\ncd /workspace/FairGRAPE/FairGRAPE\n'+r['metadata']['command'])
    text='\n'.join(lines)+'\n'
    (out/'summary_ko.txt').write_text(text,encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
