"""DogsCats 분할 프로토콜 검증 (이미지 불필요, LNL 라벨 리스트만 사용).

main_test.py의 DogsCats 경로와 동일하게
  make_frame(stratify=True, strat_cols=['species','color'])  → (species×color) 층화 8:1:1
  apply_fscl_skew(beta=5, group_col='species', skew_col='color') → cat dark:bright=5:1, dog 1:5
  make_balanced_eval(cols=('species','color'))                 → val/test 4셀 균형
을 실제 dataset.py 함수로 실행해 셀 크기가 기대값과 일치하는지 확인한다.

기대값은 LNL 리스트 셀 크기(dark cat 4,748 / bright cat 3,086 / bright dog 5,299 / dark dog 3,652)에서
8:1:1 → β=5 → 최소셀 균형을 손으로 계산한 값이다.

실행: python tests/test_dogscats_split.py   또는   python -m pytest tests/test_dogscats_split.py -q
"""
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import make_frame, apply_fscl_skew, make_balanced_eval  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTS = {'bright': os.path.join(ROOT, 'csv/dogscats_list_bright.txt'),
         'dark': os.path.join(ROOT, 'csv/dogscats_list_dark.txt')}


def _build_label_csv(path):
    rows = []
    for color, fn in LISTS.items():
        for line in open(fn):
            name = line.strip()
            if name:
                rows.append({'image_name': name, 'face_name_align': name,
                             'species': 0 if name.startswith('cat') else 1,
                             'color': 1 if color == 'bright' else 0})
    pd.DataFrame(rows).to_csv(path, index=False)


def _cells(frame):
    return {k: int(v) for k, v in frame.groupby(['species', 'color']).size().items()}


def test_dogscats_split_skew_balance():
    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, 'DogsCats.csv')
        _build_label_csv(csv)
        frames = make_frame(csv, None, seven_races=False, stratify=True,
                            strat_cols=['species', 'color'],
                            split_csv_out=os.path.join(d, 'split.csv'))

    assert frames['all'].shape[0] == 16785
    # (species, color): 0=cat/1=dog, 0=dark/1=bright
    assert _cells(frames['all']) == {(0, 0): 4748, (0, 1): 3086, (1, 0): 3652, (1, 1): 5299}
    assert _cells(frames['train']) == {(0, 0): 3798, (0, 1): 2468, (1, 0): 2921, (1, 1): 4239}
    assert _cells(frames['val']) == {(0, 0): 475, (0, 1): 309, (1, 0): 365, (1, 1): 530}
    assert _cells(frames['test']) == {(0, 0): 475, (0, 1): 309, (1, 0): 366, (1, 1): 530}

    # 세 분할이 서로 겹치지 않고 전체를 덮는지
    names = [set(frames[s]['image_name']) for s in ('train', 'val', 'test')]
    assert not (names[0] & names[1]) and not (names[0] & names[2]) and not (names[1] & names[2])
    assert len(names[0] | names[1] | names[2]) == 16785

    train = apply_fscl_skew(frames['train'], 5, skew_col='color', group_col='species')
    # cat: dark 3798 유지, bright 3798/5=760 / dog: bright 4239 유지, dark 4239/5=848
    assert _cells(train) == {(0, 0): 3798, (0, 1): 760, (1, 0): 848, (1, 1): 4239}
    assert train.shape[0] == 9645

    val = make_balanced_eval(frames['val'], cols=('species', 'color'))
    test = make_balanced_eval(frames['test'], cols=('species', 'color'))
    assert _cells(val) == {(0, 0): 309, (0, 1): 309, (1, 0): 309, (1, 1): 309}
    assert _cells(test) == {(0, 0): 309, (0, 1): 309, (1, 0): 309, (1, 1): 309}

    # 라벨 dtype이 정수인지 (텐서 변환 안전성)
    for c in ('species', 'color'):
        assert str(train[c].dtype).startswith('int')


def test_utkface_default_path_unchanged():
    """strat_cols/group_col 기본값이 기존 UTKFace 동작(gender × race, race 그룹)을 유지하는지."""
    import numpy as np
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame({'image_name': [f'{i}.jpg' for i in range(n)],
                       'face_name_align': [f'{i}.jpg' for i in range(n)],
                       'race': rng.randint(0, 2, n), 'gender': rng.randint(0, 2, n)})
    out = apply_fscl_skew(df, 3)  # 기본 group_col='race', skew_col='gender'
    c = _cells_rg = {k: int(v) for k, v in out.groupby(['race', 'gender']).size().items()}
    assert abs(c[(0, 0)] / c[(0, 1)] - 3) < 0.15 and abs(c[(1, 1)] / c[(1, 0)] - 3) < 0.15


if __name__ == '__main__':
    test_dogscats_split_skew_balance()
    test_utkface_default_path_unchanged()
    print('OK')
