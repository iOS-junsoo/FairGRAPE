# Kaggle Dogs vs. Cats(train 25,000장) + LNL(Kim et al., CVPR'19) 색상 라벨로
# csv/DogsCats.csv 를 생성한다.
#
# 배경: 원본에는 종(species) 라벨(파일명 접두어 cat./dog.)만 있고 색상 라벨이 없다.
# LNL(https://github.com/feidfoe/learning-not-to-learn/tree/master/dataset/dogs_and_cats)이
# 16,785장에 bright/dark 색상 라벨을 달아 두었으므로, 그 두 리스트에 있는 이미지만 사용한다.
# (리스트에 없는 8,215장은 LNL과 동일하게 실험에서 제외)
#
# 라벨 코딩 (main_test / dataset.apply_fscl_skew 와 일치해야 함):
#   species: cat=0, dog=1     (타깃 속성, --loss_type species)
#   color  : dark=0, bright=1 (민감 속성, --sensitive_group color)
#   → --skew_beta 5 시 cat은 dark:bright=5:1, dog는 1:5 (FSCL 설정: 검은 고양이 5배, 흰 개 5배)
#
# 필요 파일:
#   Images/DogsCats/cat.N.jpg, dog.N.jpg  (Kaggle train.zip 을 평평하게 풀어둔 것)
#   csv/dogscats_list_bright.txt, csv/dogscats_list_dark.txt  (LNL 리스트 원본 그대로)
#
# 실행: python scripts/make_dogscats_csv.py
import os
import re
import sys

import pandas as pd

IMAGE_DIR = 'Images/DogsCats'
LIST_FILES = {'bright': 'csv/dogscats_list_bright.txt', 'dark': 'csv/dogscats_list_dark.txt'}
OUT_CSV = 'csv/DogsCats.csv'

SPECIES_CODE = {'cat': 0, 'dog': 1}
COLOR_CODE = {'dark': 0, 'bright': 1}


def main():
    pattern = re.compile(r'^(cat|dog)\.(\d+)\.jpg$')
    rows, bad = [], []
    seen = {}
    for color, path in LIST_FILES.items():
        with open(path) as f:
            for line in f:
                fn = line.strip()
                if not fn:
                    continue
                m = pattern.match(fn)
                if not m:
                    bad.append(fn)
                    continue
                if fn in seen:
                    sys.exit(f'중복 파일명 발견: {fn} ({seen[fn]} / {color})')
                seen[fn] = color
                rows.append({'image_name': fn, 'face_name_align': fn,
                             'species': SPECIES_CODE[m.group(1)], 'color': COLOR_CODE[color]})
    if bad:
        sys.exit(f'형식 불일치 파일명 {len(bad)}개: {bad[:10]}')

    df = pd.DataFrame(rows).sort_values('image_name').reset_index(drop=True)

    # 이미지 폴더 존재 여부 교차 검증
    files = set(os.listdir(IMAGE_DIR)) if os.path.isdir(IMAGE_DIR) else set()
    missing = df[~df['image_name'].isin(files)]
    print(f'라벨 {len(df)}장 / 이미지 폴더 {len(files)}장 / 라벨은 있으나 이미지 없음 {len(missing)}장')
    if len(missing):
        print(missing['image_name'].head(10).to_string(index=False))
        sys.exit('이미지 누락 — Images/DogsCats 에 Kaggle train.zip 이 평평하게 풀려 있는지 확인하세요.')

    table = df.groupby(['species', 'color']).size().unstack()
    table.index = table.index.map({0: 'cat', 1: 'dog'})
    table.columns = table.columns.map({0: 'dark', 1: 'bright'})
    print('(species × color) 분포:\n' + table.to_string())

    df.to_csv(OUT_CSV, index=False)
    print(f'저장 완료: {OUT_CSV} ({len(df)}행)')


if __name__ == '__main__':
    main()
