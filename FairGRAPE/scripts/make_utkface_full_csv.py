# Images/UTKFace 파일명(age_gender_race_date.jpg.chip.jpg)에서 라벨을 파싱해
# csv/UTKFace_labels_full.csv 를 생성한다.
#
# 배경: 기존 csv/UTKFace_labels.csv(22,013행)에는 Others 인종(race digit 4, 1,692장)이
# 빠져 있다. White vs Non-White 이진 실험은 Others를 포함(FSCL 'Caucasian or not' 정의)
# 하므로 이미지 폴더 전체(23,708장)에서 CSV를 재생성한다.
# 기존 UTKFace_labels.csv는 과거 4인종 실험 재현용으로 그대로 둔다.
#
# 실행: python scripts/make_utkface_full_csv.py
import os
import re
import sys

import pandas as pd

IMAGE_DIR = 'Images/UTKFace'
OLD_CSV = 'csv/UTKFace_labels.csv'
OUT_CSV = 'csv/UTKFace_labels_full.csv'

RACE_NAMES = {0: 'White', 1: 'Black', 2: 'East Asian', 3: 'Indian', 4: 'Others'}
GENDER_NAMES = {0: 'Male', 1: 'Female'}

# 기존 CSV와 동일한 age 구간 표기
AGE_BINS = [(0, 2, '0-2'), (3, 9, '3-9'), (10, 19, '10-19'), (20, 29, '20-29'),
            (30, 39, '30-39'), (40, 49, '40-49'), (50, 59, '50-59'), (60, 69, '60-69')]


def age_to_bin(age):
    for lo, hi, name in AGE_BINS:
        if lo <= age <= hi:
            return name
    return 'more than 70'


def main():
    pattern = re.compile(r'^(\d+)_([01])_([0-4])_\d+\.jpg\.chip\.jpg$')
    rows, skipped = [], []
    for fn in sorted(os.listdir(IMAGE_DIR)):
        m = pattern.match(fn)
        if not m:
            skipped.append(fn)
            continue
        age, gender, race = int(m.group(1)), int(m.group(2)), int(m.group(3))
        rows.append({'age': age_to_bin(age), 'gender': GENDER_NAMES[gender],
                     'race': RACE_NAMES[race], 'image_name': fn, 'face_name_align': fn})

    df = pd.DataFrame(rows)
    print(f'파싱 성공 {len(df)}장 / 스킵 {len(skipped)}장')
    if skipped:
        print('스킵된 파일(라벨 필드 누락/형식 불일치):', skipped)
    print('race 분포:\n', df['race'].value_counts().to_string())
    print('gender 분포:\n', df['gender'].value_counts().to_string())

    # 기존 CSV와 겹치는 파일의 라벨이 일치하는지 교차 검증
    old = pd.read_csv(OLD_CSV)
    old['basename'] = old['image_name'].apply(lambda p: p.split('/')[-1])
    merged = df.merge(old[['basename', 'age', 'gender', 'race']],
                      left_on='image_name', right_on='basename', suffixes=('', '_old'))
    mismatch = merged[(merged['age'] != merged['age_old'])
                      | (merged['gender'] != merged['gender_old'])
                      | (merged['race'] != merged['race_old'])]
    print(f'기존 CSV 교차검증: 겹침 {len(merged)}행, 라벨 불일치 {len(mismatch)}행')
    if len(mismatch):
        print(mismatch.head(20).to_string())
        sys.exit('라벨 불일치 발견 — 매핑 규칙을 확인하세요.')

    df.to_csv(OUT_CSV)
    print(f'저장 완료: {OUT_CSV} ({len(df)}행)')


if __name__ == '__main__':
    main()
