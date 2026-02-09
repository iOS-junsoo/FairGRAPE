import PIL
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import datasets, transforms
import copy
from PIL import Image
from pathlib import Path
from collections import Counter
import imgaug as ia
from imgaug import augmenters as iaa
import dlib
import time
import os
from torch.utils.data import Dataset, DataLoader
import json

# 경고 무시
import warnings
warnings.filterwarnings("ignore")

# 한 이미지 내의 모든 얼굴이 동일한 세트에 속하도록 train/validation/test 세트로 분할
def split_image_name(val):
    return val.split('/')[-1]

# ImageNet person 서브트리에 대한 인구통계 레이블은 ImageNet 데이터베이스를 통해 획득해야 합니다.
# 아래 함수는 다음의 파일/디렉터리가 필요합니다:
# 1, csv/synsets_safety.csv: https://image-net.org/data/synsets_safety.csv
# 2, csv/imageability_scores.csv: https://image-net.org/data/imageability_scores.csv
# 3, csv/ImageNet demographics annotations.json: ImageNet 팀에게 요청해야 합니다.
# 4, Images/Imagenet: ImageNet demographics annotations.json에 나열된 모든 이미지를 포함하는 폴더를 생성
# 위의 파일들은 결합되어 csv에 지정된 경로에 어노테이션 파일을 생성합니다.
def prepare_ImageNet(csv):
    if os.path.exists(csv):
        print("ImageNet annotation exist!")
        return

    safety = pd.read_csv("csv/synsets_safety.csv")
    safeid = set(safety['wnid'][safety['safety'] == "safe"])

    imageable = pd.read_csv("csv/imageability_scores.csv")
    imageableid = imageable['wnid'][imageable['imageability.score'] >= 4]

    annot_df = {'image_name': [], 'classes': [], "gender": []}
    with open('csv/ImageNet demographics annotations.json', 'r') as annt_file:
        labels = json.load(annt_file)

    for i, label in enumerate(labels):
        url = label['url']
        wnid = url.split("_")[0]
        if wnid not in safeid or wnid not in imageableid:
            continue
        if len(label['gender']) > 1 or label['gender'][0] == "unsure":
            continue
        gender = 0 if label['gender'][0] == 'male' else 1
        annot_df['image_name'].append(url)
        annot_df['classes'].append(wnid)
        annot_df['gender'].append(gender)

    annot_df = pd.DataFrame(annot_df)
    n_per_class = Counter(annot_df['classes'])
    classes_enough_sample = np.array([n_per_class[wnid] >= 10 for wnid in annot_df['classes']])
    annot_df = annot_df[classes_enough_sample].reset_index(drop=True)

    annot_df.to_csv(csv)

# races 라벨을 재정의하는 함수
def relabel(frame, seven_races=True, drop_race=False):

    # race 라벨 재할당
    if 'race' in frame.columns:
        frame.loc[frame['race'] == 'White', 'race'] = 0
        frame.loc[frame['race'] == 'Black', 'race'] = 1
        frame.loc[frame['race'] == 'Latino_Hispanic', 'race'] = 2
        frame.loc[frame['race'] == 'East Asian', 'race'] = 3
        frame.loc[frame['race'] == 'Southeast Asian', 'race'] = 4
        frame.loc[frame['race'] == 'Indian', 'race'] = 5
        frame.loc[frame['race'] == 'Middle Eastern', 'race'] = 6

    # gender 라벨
    if 'gender' in frame.columns:
        frame.loc[frame['gender'] == 'Male', 'gender'] = 0
        frame.loc[frame['gender'] == 'Female', 'gender'] = 1

    # 7개 인종이 아닌 경우, race 라벨을 합치거나 제거
    if not seven_races and 'race' in frame.columns:
        # 여기서는 LFWA+(Indian, Latino 없음) & UTK(Latino 없음)과 비교하기 위해
        # White 0, Black 1, Asian 2, Indian 3 으로 설정
        # Latino (2)와 Middle Eastern (6) 제거
        frame.loc[frame['race'].isin([2, 6]), 'race'] = -1
        # East Asian(3)과 Southeast Asian(4)을 합쳐서 Asian(2)으로
        frame.loc[frame['race'].isin([3, 4]), 'race'] = 2
        # Indian(5)를 Indian(3)으로
        frame.loc[frame['race'].isin([5]), 'race'] = 3
        # Latino 제거
        frame = frame[frame['race'] != -1]
        # Indian 제거 (LFWA+와의 비교 위해서만 필요한 경우)
        #frame = frame[frame['race'] != 3]
        Counter(frame.race)
        
    # drop_race 파라미터의 동작:
    # =0: 아무것도 안 함
    # =1/2/3/4: White/Black/Asian/Indian 제거
    # =11/12/13/14: White/Black/Asian/Indian 만 남김
    if drop_race and 'race' in frame.columns:
        if drop_race in [1, 2, 3, 4]:
            drop_race -= 1
            frame = frame[frame['race'] != drop_race].reset_index(drop=True)
        elif drop_race in [11, 12, 13, 14]:
            drop_race -= 11
            frame = frame[frame['race'] == drop_race].reset_index(drop=True)

    # race와 gender가 모두 있으면, raceAndgender 라벨 생성
    if 'race' in frame.columns and 'gender' in frame.columns:
        n_races = max(frame['race']) + 1
        frame['raceAndgender'] = frame['race'] + frame['gender'] * n_races

    return frame

# 데이터셋에 불균형을 인위적으로 추가하는 함수
def add_imbalance(frame):
    if not 'race' in frame.columns:
        print("No race label found! Imbalance not added.")
        return frame
    
    nrow = frame.shape[0]

    np.random.seed(42)
    choice = np.random.rand(nrow)
    # White가 아닌 이미지 중 80%를 제거
    frame = frame[((np.array(frame['race'] == 0))) | ((np.array(frame['race'] != 0)) & (choice >= 0.8))]
    frame = frame.reset_index(drop=True)
    return frame

# csv 파일을 불러와서 new_face_dir 에 저장된 이미지와 매핑, 그리고 train/val/test로 분할
def make_frame(csv, new_face_dir, train_pct=0.8, seven_races=True, drop_race=False, imbalance=False):
    device = torch.device('cuda:0')
    frame = pd.read_csv(csv)
    frame.head()

    frame = relabel(frame, seven_races, drop_race)

    if imbalance:
        frame = add_imbalance(frame)

    # 새로운 디렉토리 경로가 주어졌다면, 해당 디렉토리에 이미지가 존재하는지 확인하고 경로를 업데이트
    if new_face_dir:
        initial_rows = frame.shape[0]
        faces = set(os.listdir(new_face_dir))
        faces_found = 0
        new_face_name = []
        face_found_mask = []
        for i in range(frame.shape[0]):
            face_name_align = split_image_name(frame['face_name_align'][i])
            face_found_mask.append(face_name_align in faces)
            if face_name_align in faces:
                new_path = os.path.join(new_face_dir, face_name_align)
                try:
                    faces_found += 1
                    new_face_name.append(new_path)
                except:
                    continue
        frame = frame[face_found_mask].reset_index(drop=True)
        frame['face_name_align'] = new_face_name
        print("{} out of {} faces are found in new dir!".format(faces_found, initial_rows))

    image_name_frame = frame['image_name'].apply(split_image_name)
    image_names = image_name_frame.unique()
    np.random.seed(42)
    image_names = np.random.permutation(image_names)

    n_images = len(image_names)
    n_train = int(train_pct * n_images)
    n_val = int((n_images - n_train) / 2)
    n_test = n_images - n_train - n_val

    image_names_train = image_names[0:n_train]
    image_names_val = image_names[n_train:n_train + n_val]
    image_names_test = image_names[n_train + n_val:]

    print("{} images: {} training, {} validation, {} test".format(n_images, len(image_names_train), len(image_names_val), len(image_names_test)))

    train_frame = frame[image_name_frame.isin(image_names_train)].reset_index(drop=True)
    val_frame = frame[image_name_frame.isin(image_names_val)].reset_index(drop=True)
    test_frame = frame[image_name_frame.isin(image_names_test)].reset_index(drop=True)

    return {'train': train_frame, "val": val_frame, "test": test_frame, "all": frame}


# Sometimes(0.5, ...)는 전체 케이스 중 50%에 대해 주어진 augmenter를 적용
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# 연속된 augmentation 단계를 정의하고, 모든 이미지를 대상으로 적용
class ImgAugTransform:
    
    def __init__(self):
        self.aug = iaa.Sequential(
            [
                # 50% 확률로 이미지를 좌우 반전
                iaa.Fliplr(0.5),

                # 0~5% 범위 내에서 이미지를 임의로 Crop
                sometimes(iaa.Crop(percent=(0, 0.05))),

                # Affine 변환 일부 적용
                iaa.Affine(
                    scale={"x": (1, 1.1), "y": (1, 1.1)},  # x, y축에 대해 각각 80-120% 범위 내에서 확장
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # -10% ~ +10% 범위 내 이동
                    rotate=(-15, 15),  # -15도 ~ +15도 범위로 회전
                    shear=(-8, 8),  # -8도 ~ +8도 범위로 Shear
                    order=[0, 1],  # 최근접 이웃 또는 bilinear 보간
                    mode=['edge']  # 변환 영역 밖의 픽셀은 'edge' 모드 사용
                ),

                # 0~5개의 작은 augmentation을 추가로 적용
                iaa.SomeOf((0, 5),
                    [
                        # Superpixel로 일부 이미지를 변환 (50~200개 세그먼트)
                        sometimes(
                            iaa.Superpixels(
                                p_replace=(0, 0.1),
                                n_segments=(50, 200)
                            )
                        ),

                        # Blur 기법을 적용
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)),
                            iaa.AverageBlur(k=(2, 7)),
                            iaa.MedianBlur(k=(3, 11)),
                        ]),

                        # Sharpen 효과를 적용
                        iaa.Sharpen(alpha=(0, 0.3), lightness=(0.75, 1.5)),

                        # Emboss 효과를 적용
                        iaa.Emboss(alpha=(0, 0.3), strength=(0, 2)),

                        # Edges 검출 후 일부 이미지는 검출된 에지를 흑백으로 만들어 원본과 합성
                        sometimes(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0, 0.3)),
                            iaa.DirectedEdgeDetect(
                                alpha=(0, 0.3), direction=(0.0, 1.0)
                            ),
                        ])),

                        # 가우시안 노이즈 추가
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                        ),

                        # 일부 픽셀을 랜덤 드롭 (검정색으로 설정)
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.02), per_channel=0.5),
                        ]),

                        # 각 픽셀 값에 -15 ~ +15 범위로 임의값 추가
                        iaa.Add((-15, 15), per_channel=0.5),

                        # 밝기를 0.75배 ~ 1.25배로 변경
                        iaa.Multiply((0.75, 1.25), per_channel=0.5),

                        # 명암비를 0.75배 ~ 1.75배로 변경
                        iaa.ContrastNormalization((0.75, 1.75), per_channel=0.5),

                        # 회색조 변환 후 원본과 합성
                        iaa.Grayscale(alpha=(0.0, 1.0)),

                        # PiecewiseAffine 적용
                        sometimes(iaa.PiecewiseAffine(scale=(0.005, 0.01)))
                    ],
                    # 위의 모든 augmentation을 무작위 순서로 적용
                    random_order=True
                )
            ],
            # 전체 변환들을 무작위 순서로 적용
            random_order=True
        )

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

# 커스텀 FaceDataset 정의
class FaceDataset(Dataset):

    def __init__(self, data_frame, transform=None, col_used=['race','gender','age']):

        self.data_frame = data_frame
        self.transform = transform
        # 앞쪽 두 열이 이미지 이름이라는 가정
        # TODO: 모든 이미지 이름이 첫 두 열에 존재하도록 변경할 수 있음
        if col_used is None:
            col_used = [2, data_frame.shape[1]]
        if isinstance(col_used[0], str):
            self.col_used = col_used
        else:
            self.col_used = [self.data_frame.columns[i] for i in range(col_used[0], col_used[1])]

    def __len__(self):
        # 데이터셋의 총 샘플 수
        return len(self.data_frame)

    def __getitem__(self, idx):
        # idx는 데이터셋으로부터의 인덱스
        # data_frame으로부터 모델의 입력에 필요한 정보를 매핑
        img_name = self.data_frame.loc[idx, 'face_name_align']
        labels = []
        for col in self.col_used:
            labels.append(self.data_frame.loc[idx, col])

        # 이미지를 ndarray 형태(H*W*C)로 읽기
        image = dlib.load_rgb_image(img_name) 
        
        if self.transform:
            image = self.transform(image)
        
        # label을 텐서로 변환
        return (image, torch.from_numpy(np.asarray(labels)))

# 학습용 데이터로더와 테스트용 데이터로더를 만드는 함수
def make_datasets(train_frame, val_frame, give_dataloader=True, batch_size=64, col_used=None):
    device = torch.device('cuda:0')
    transform_train_data = transforms.Compose([
        ImgAugTransform(),
        lambda x: PIL.Image.fromarray(x),
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transformed_train_dataset = FaceDataset(data_frame=train_frame,
                                            transform=transform_train_data,
                                            col_used=col_used
                                           )

    #train_dataloader = DataLoader(transformed_train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    train_dataloader = DataLoader(transformed_train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    transform_test_data = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transformed_test_dataset = FaceDataset(data_frame=val_frame,
                                           transform=transform_test_data,
                                           col_used=col_used
                                          )

    test_dataloader = DataLoader(transformed_test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=8)
    #test_dataloader = DataLoader(transformed_test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    if give_dataloader:
        return train_dataloader, test_dataloader
    else:
        return transformed_train_dataset, transformed_test_dataset
