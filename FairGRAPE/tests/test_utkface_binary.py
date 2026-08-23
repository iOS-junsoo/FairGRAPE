"""UTKFace White vs Non-White 이진 실험 경로 유닛테스트.

대상: dataset.relabel(race_binary), make_frame(stratify), apply_fscl_skew, make_balanced_eval.
GPU·이미지 불필요 (make_frame은 new_face_dir=None으로 호출해 파일 존재 검사를 건너뜀).

실행: python tests/test_utkface_binary.py   또는   python -m pytest tests/test_utkface_binary.py -q
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# imgaug는 학습 시 augmentation에만 쓰이고 라벨/분할 로직과 무관하다.
# numpy 2.x 환경에서는 import가 깨지므로 없으면 스텁으로 대체해 dataset을 import한다.
try:
    import imgaug  # noqa: F401
except Exception:
    _imgaug = types.ModuleType('imgaug')
    _augmenters = types.ModuleType('imgaug.augmenters')
    _imgaug.augmenters = _augmenters
    sys.modules['imgaug'] = _imgaug
    sys.modules['imgaug.augmenters'] = _augmenters

import numpy as np
import pandas as pd

from dataset import relabel, make_frame, apply_fscl_skew, make_balanced_eval

FULL_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'csv', 'UTKFace_labels_full.csv')


def _toy_frame():
    races = ['White', 'Black', 'East Asian', 'Indian', 'Others'] * 4
    genders = (['Male'] * 5 + ['Female'] * 5) * 2
    return pd.DataFrame({
        'age': ['20-29'] * 20,
        'gender': genders,
        'race': races,
        'image_name': [f'img_{i}.jpg' for i in range(20)],
        'face_name_align': [f'img_{i}.jpg' for i in range(20)],
    })


def test_relabel_race_binary():
    frame = relabel(_toy_frame(), seven_races=False, race_binary=True)
    # White만 0, 나머지(Black/Asian/Indian/Others)는 전부 1
    assert set(frame['race'].astype(int)) == {0, 1}
    assert all(int(r) == (0 if o == 0 else 1)
               for r, o in zip(frame['race'], frame['race_orig']))
    # 병합 전 5범주가 race_orig에 보존 (문자열 맵: White0/Black1/EastAsian3/Indian5/Others7)
    assert set(frame['race_orig'].astype(int)) == {0, 1, 3, 5, 7}
    # raceAndgender는 이진 race 기준 (n_races=2)
    assert set(frame['raceAndgender'].astype(int)) <= {0, 1, 2, 3}
    print('test_relabel_race_binary ok')


def test_race_binary_drop_race_conflict():
    try:
        relabel(_toy_frame(), seven_races=False, drop_race=[3], race_binary=True)
    except ValueError:
        print('test_race_binary_drop_race_conflict ok')
    else:
        raise AssertionError('race_binary + drop_race 동시 지정이 허용됨')


def test_stratified_split():
    frames = make_frame(FULL_CSV, None, seven_races=False, race_binary=True, stratify=True)
    train, val, test = frames['train'], frames['val'], frames['test']
    n = sum(f.shape[0] for f in (train, val, test))
    assert n == frames['all'].shape[0]

    # 8:1:1 (셀별 반올림 오차 허용)
    assert abs(train.shape[0] / n - 0.8) < 0.005, train.shape[0] / n
    assert abs(val.shape[0] / n - 0.1) < 0.005, val.shape[0] / n
    assert abs(test.shape[0] / n - 0.1) < 0.005, test.shape[0] / n

    # 분할 간 중복 없음
    names = [set(f['image_name']) for f in (train, val, test)]
    assert not (names[0] & names[1]) and not (names[0] & names[2]) and not (names[1] & names[2])

    # 층화: (gender × race_orig) 셀 비율이 각 분할에서 전체와 ±1%p 이내
    overall = frames['all'].groupby(['gender', 'race_orig']).size() / n
    for f in (train, val, test):
        share = f.groupby(['gender', 'race_orig']).size() / f.shape[0]
        assert (share - overall).abs().max() < 0.01, (share - overall).abs().max()

    # race는 이진, Others(race_orig=7) 포함
    assert set(train['race']) == {0, 1}
    assert 7 in set(train['race_orig'])

    # 재현성: 같은 호출 → 같은 분할
    frames2 = make_frame(FULL_CSV, None, seven_races=False, race_binary=True, stratify=True)
    assert list(frames2['train']['image_name']) == list(train['image_name'])
    print('test_stratified_split ok')


def test_apply_fscl_skew():
    rng = np.random.RandomState(0)
    frame = pd.DataFrame({
        'race': rng.randint(0, 2, 4000),
        'gender': rng.randint(0, 2, 4000),
        'image_name': [f'i{i}.jpg' for i in range(4000)],
        'face_name_align': [f'i{i}.jpg' for i in range(4000)],
    })
    beta = 3
    out = apply_fscl_skew(frame, beta)
    for r, target in [(0, beta), (1, 1.0 / beta)]:
        n_m = ((out['race'] == r) & (out['gender'] == 0)).sum()
        n_f = ((out['race'] == r) & (out['gender'] == 1)).sum()
        assert abs(n_m / n_f - target) / target < 0.01, (r, n_m, n_f)
    assert list(out.index) == list(range(out.shape[0]))  # FaceDataset의 .loc[idx] 접근용

    for bad_beta in (1, 0.5):
        try:
            apply_fscl_skew(frame, bad_beta)
        except ValueError:
            pass
        else:
            raise AssertionError(f'beta={bad_beta} 가 허용됨')

    try:
        apply_fscl_skew(frame.assign(race=frame['race'] + 1), beta)  # race ∈ {1,2}
    except ValueError:
        pass
    else:
        raise AssertionError('비이진 race 프레임이 허용됨')
    print('test_apply_fscl_skew ok')


def test_make_balanced_eval():
    rng = np.random.RandomState(1)
    frame = pd.DataFrame({
        'race': rng.choice([0, 1], 1000, p=[0.7, 0.3]),
        'gender': rng.randint(0, 2, 1000),
        'image_name': [f'i{i}.jpg' for i in range(1000)],
        'face_name_align': [f'i{i}.jpg' for i in range(1000)],
    })
    out = make_balanced_eval(frame)
    sizes = out.groupby(['race', 'gender']).size()
    assert len(sizes) == 4 and sizes.nunique() == 1
    assert sizes.iloc[0] == frame.groupby(['race', 'gender']).size().min()
    assert list(out.index) == list(range(out.shape[0]))
    print('test_make_balanced_eval ok')


if __name__ == '__main__':
    test_relabel_race_binary()
    test_race_binary_drop_race_conflict()
    test_stratified_split()
    test_apply_fscl_skew()
    test_make_balanced_eval()
    print('\n모든 테스트 통과')
