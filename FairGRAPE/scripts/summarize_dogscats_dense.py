# ~/dogscats_dense_scratch_b<β>_s42.log 들에서 train 구성, val best, test acc/EO 를 표로 모은다.
# 실행: python scripts/summarize_dogscats_dense.py   (pandas 불필요)
import glob, os, re
rows = []
for f in sorted(glob.glob(os.path.expanduser('~/dogscats_dense_scratch_b*_s42.log')),
                key=lambda p: int(re.search(r'_b(\d+)_', p).group(1))):
    beta = int(re.search(r'_b(\d+)_', f).group(1)); s = open(f, errors='ignore').read()
    tr = re.findall(r"\[skew β=[\d.]+\] species=(\d): color=0 \d+→(\d+), color=1 \d+→(\d+)", s)
    if tr:
        cells = {int(g): (int(d), int(b)) for g, d, b in tr}
        train = f"dc {cells[0][0]} / bc {cells[0][1]} / dd {cells[1][0]} / bd {cells[1][1]}"
        n_tr = sum(cells[0]) + sum(cells[1])
    else:
        m = re.search(r"(\d+) images: (\d+) training", s); train = "natural (no skew)"; n_tr = int(m.group(2)) if m else -1
    val = re.findall(r"최고 정확도, 손실 갱신: tensor\(([\d.]+)", s)
    best = re.findall(r"BEST_MODEL_prune_01_acc_([\d.]+)_eo_([\d.]+)", s)
    test_eo = re.findall(r"전체 평균 EO_diff = ([\d.]+)", s)
    test_acc = re.findall(r"전체 클래스 평균 정확도: ([\d.]+)", s)
    done = '완료' if '추론 결과 CSV 저장 완료' in s else '진행중'
    rows.append((beta, n_tr, train, best[0][0] if best else '-', best[0][1] if best else '-',
                 test_eo[-1] if test_eo else '-', done))
print(f"{'β':>4} {'n_train':>7}  {'train cells (dark cat / bright cat / dark dog / bright dog)':60s} {'val acc':>8} {'val EO':>7} {'test EO':>8}  상태")
for r in rows:
    print(f"{r[0]:>4} {r[1]:>7}  {r[2]:60s} {r[3]:>8} {r[4]:>7} {r[5]:>8}  {r[6]}")
print("\nval acc/EO = 첫 학습 단계(에폭 40) best 모델 기준. test EO = 최종 모델의 균형 test EO_diff.")
