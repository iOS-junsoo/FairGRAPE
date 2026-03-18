import os
import shutil
import zipfile
import tarfile
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from statistics import stdev
from collections import defaultdict
from tqdm import tqdm
from torchvision import transforms, models
import torchvision

from dataset import split_image_name


def is_cudnn_runtime_error(error):
    err = str(error)
    return (
        "CUDNN_STATUS_EXECUTION_FAILED" in err
        or "Unable to find a valid cuDNN algorithm" in err
        or "CUDNN_STATUS_NOT_SUPPORTED" in err
        or "cuDNN error" in err
    )


def safe_forward_with_cudnn_fallback(forward_fn, *args, **kwargs):
    try:
        return forward_fn(*args, **kwargs)
    except RuntimeError as e:
        if not is_cudnn_runtime_error(e):
            raise
        with torch.backends.cudnn.flags(enabled=False):
            return forward_fn(*args, **kwargs)


# -----------------------------------------------------------------------------
# 1) 사전 학습(Pretrained) 모델 생성 함수
#    - pruning 인자를 통해 (추가적인 로직이 있다면) 분기할 수도 있음
#    - network: 'resnet34' 또는 'mobilenetv2'
#    - dataset: (사용하지 않지만, 필요 시 활용 가능)
#    - n_classes: 최종 분류 클래스 수
# -----------------------------------------------------------------------------
def make_model(pruning=True, network="resnet34", dataset="FairFace", n_classes=18):
    """
    pretrained=True 옵션으로 사전 학습된 모델을 불러온 뒤,
    최종 분류 레이어만 n_classes에 맞게 교체합니다.
    """
    if network == "resnet34":
        # pruning 인자에 따라 로직을 분기할 수도 있음. (현재는 동일하게 동작)
        model_conv = models.resnet34(pretrained=True) if pruning else torchvision.models.resnet34(pretrained=True)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, n_classes)
    elif network == 'mobilenetv2':
        model_conv = models.mobilenet_v2(pretrained=True)
        num_ftrs = model_conv.classifier[1].in_features
        model_conv.classifier[1] = nn.Linear(num_ftrs, n_classes)
    else:
        raise NotImplementedError(f"{network} 은(는) 아직 구현되지 않았습니다.")

    return model_conv


# -----------------------------------------------------------------------------
# 2) 이미지 경로/파일 검증 함수
#    - DataFrame(frame) 내에 기록된 이미지 이름이 실제 디렉토리(new_img_dir)에 있는지 확인
#    - 존재하는 이미지만 남긴 새 DataFrame을 반환
# -----------------------------------------------------------------------------
def check_imgs(frame, new_img_dir):
    initial_rows = frame.shape[0]
    faces = set(os.listdir(new_img_dir))
    faces_found = 0
    new_face_name = []
    face_found_mask = []

    for i in range(frame.shape[0]):
        # 예: UTKFace.jpg → UTKFace.jpg (단순 변환)
        face_name_align = split_image_name(frame['face_name_align'][i])
        is_found = face_name_align in faces
        face_found_mask.append(is_found)
        if is_found:
            faces_found += 1
            new_face_name.append(os.path.join(new_img_dir, face_name_align))

    # 실제로 존재하는 이미지만 필터링
    frame = frame[face_found_mask].reset_index(drop=True)
    frame['face_name_align'] = new_face_name
    print(f"새 디렉토리에서 {faces_found} / {initial_rows} 개 이미지 파일을 찾았습니다.")
    return frame


def filter_readable_images(frame, image_col='face_name_align', loader='dlib'):
    filtered_frame = frame.copy().reset_index(drop=True)
    total_rows = filtered_frame.shape[0]
    valid_mask = []
    bad_files = []

    if total_rows == 0:
        print("읽기 가능한 이미지 확인 완료: 0 / 0")
        return filtered_frame

    dlib_module = None
    if loader == 'dlib':
        import dlib as dlib_module

    for image_path in filtered_frame[image_col]:
        try:
            if loader == 'dlib':
                dlib_module.load_rgb_image(image_path)
            else:
                with Image.open(image_path) as image:
                    image.convert('RGB')
            valid_mask.append(True)
        except Exception:
            valid_mask.append(False)
            bad_files.append(image_path)

    filtered_frame = filtered_frame[np.asarray(valid_mask, dtype=bool)].reset_index(drop=True)

    removed_count = total_rows - filtered_frame.shape[0]
    if removed_count > 0:
        print(f"읽을 수 없는 이미지 {removed_count}개를 제외했습니다.")
        preview_count = min(10, len(bad_files))
        for image_path in bad_files[:preview_count]:
            print(f"  - 제외됨: {image_path}")
        if len(bad_files) > preview_count:
            print(f"  - 추가 제외 파일 {len(bad_files) - preview_count}개")
    else:
        print(f"읽기 가능한 이미지 확인 완료: {total_rows} / {total_rows}")

    return filtered_frame


def write_unreadable_image_report(report_path, split_name, total_rows, removed_count, bad_files):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write(f"split: {split_name}\n")
        handle.write(f"total_rows: {total_rows}\n")
        handle.write(f"removed_rows: {removed_count}\n")
        handle.write("unreadable_files:\n")
        for image_path in bad_files:
            handle.write(f"{image_path}\n")


def filter_readable_images_in_frames(frames, image_col='face_name_align', loader='dlib', report_dir=None, report_prefix='excluded_images'):
    filtered_frames = {}
    for split_name, frame in frames.items():
        print(f"[{split_name}] 이미지 가독성 검사 시작")
        total_rows = frame.shape[0]
        filtered_frame = filter_readable_images(
            frame,
            image_col=image_col,
            loader=loader,
        )
        filtered_frames[split_name] = filtered_frame

        if report_dir is not None:
            filtered_paths = set(filtered_frame[image_col].tolist())
            bad_files = [image_path for image_path in frame[image_col].tolist() if image_path not in filtered_paths]
            report_path = os.path.join(report_dir, f"{report_prefix}_{split_name}.txt")
            write_unreadable_image_report(
                report_path,
                split_name,
                total_rows,
                total_rows - filtered_frame.shape[0],
                bad_files,
            )
    return filtered_frames


# -----------------------------------------------------------------------------
# 3) 공정성(fairness) 등 평가 목적으로 모델 추론 결과를 DataFrame으로 만드는 함수
#    - test_frame: 평가할 이미지 목록과 라벨이 담긴 DataFrame
#    - new_img_dir: 실제 이미지들이 저장된 디렉토리 (None이면 frame 내 경로 그대로 사용)
#    - n_classes, arch, col_used, output_cols_each_task, sensitive_group 등
#      다양한 설정값을 받아서, 예측 확률/정답/민감 속성 등을 DataFrame으로 저장
# -----------------------------------------------------------------------------
import config
def check_fairness0(trained_model,
                    test_frame,
                    new_img_dir,
                    n_classes=7,
                    arch="resnet34",
                    col_used=['race'],
                    output_cols_each_task=[(0, 7)],
                    sensitive_group="gender"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # (A) 모델 로드: 직접 모델 객체를 받았는지, 파일 경로(str)를 받았는지 구분
    if isinstance(trained_model, str):
        if arch == 'resnet34':
            model = models.resnet34()
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        elif arch == 'mobilenetv2':
            model = models.mobilenet_v2()
            model.classifier = nn.Linear(model.classifier.in_features, n_classes)
        else:
            raise NotImplementedError(f"{arch} 은(는) 아직 구현되지 않았습니다.")
        model.load_state_dict(torch.load(trained_model))
    else:
        model = trained_model

    model = model.to(device)
    model.eval()

    # (B) 전처리 파이프라인(transform)
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    face_names = []
    scores = defaultdict(list)
    preds = defaultdict(list)
    truths = defaultdict(list)
    outputs_raw = defaultdict(list)

    # col_used가 None이면, test_frame의 2열 이후를 모두 사용 (지금은 사용 X)
    if col_used is None:
        col_used = [test_frame.columns[i] for i in range(2, len(test_frame.columns))]

    # (C) 실제 디렉토리와 맞춰 파일 검증
    if new_img_dir and config.glo_is_checked == False: # config.glo_is_checked를 조건에 넣은 이유는 재학습마다 체크이미지가 필요없기 때문이다.
        test_frame = check_imgs(test_frame, new_img_dir)
        config.glo_is_checked = True  # 체크 완료 상태로 변경


    # (D) 각 샘플별 모델 추론
    for _, row in tqdm(test_frame.iterrows(), total=test_frame.shape[0]):
        image_name = row['face_name_align']

        # 이미지 열기 (RGB 변환)
        image = Image.open(image_name).convert('RGB')
        image = trans(image)
        image = image.view(1, *image.shape).to(device)

        output_i = safe_forward_with_cudnn_fallback(model, image).cpu().detach().numpy()  # [1, n_class] 형태
        outputs = np.squeeze(output_i)                  # [n_class]

        for col, (start_idx, end_idx) in zip(col_used, output_cols_each_task):
            # (i) 해당 col에 해당하는 출력 범위(start_idx:end_idx)
            outputs_this_col = outputs[start_idx:end_idx]
            outputs_raw[col].append(outputs_this_col)

            # (ii) softmax 확률 계산
            exps = np.exp(outputs_this_col)
            score_this_col = exps / np.sum(exps)
            scores[col].append(score_this_col)

            # (iii) 예측 결과(가장 확률 높은 클래스)
            preds_this_cols = np.argmax(score_this_col)
            preds[col].append(preds_this_cols)

            # (iv) 실제 라벨(정답)
            #truths[col].append(row[col])
            truths[col].append(pd.to_numeric(row[col], errors='coerce'))

        face_names.append(image_name)

    # (E) 결과를 DataFrame으로 변환
    fair_df_cols = [
        face_names,
        list(test_frame[sensitive_group])
    ]
    fair_df_col_names = [
        'face_name_align',
        sensitive_group
    ]

    # scores, outputs, preds, truth 순으로 값과 컬럼이름을 확장
    fair_df_cols += [scores[col] for col in col_used] # 모델이 해당 속성에 대해 소프트맥스 확률 분포로 계산한 값
    fair_df_cols += [outputs_raw[col] for col in col_used] # 소프트맥스 적용 전, 모델의 마지막 레이어레서 나온 원시 출력 값
    fair_df_cols += [preds[col] for col in col_used] # 모델의 최종 예측 결과
    fair_df_cols += [truths[col] for col in col_used] # 실제 정답

    fair_df_col_names += [col + "_scores_fair" for col in col_used]
    fair_df_col_names += [col + "_outputs" for col in col_used]
    fair_df_col_names += [col + "_preds_fair" for col in col_used]
    fair_df_col_names += col_used

    fair_test = pd.DataFrame(fair_df_cols).T
    fair_test.columns = fair_df_col_names

    return fair_test


# -----------------------------------------------------------------------------
# 4) 정확도(Accuracy) 계산 함수
#    - col_used: 예측해야 하는 여러 컬럼(라벨)이 있을 때 각각의 정확도
#    - sensitive_group: 그룹별 정확도도 계산
# -----------------------------------------------------------------------------
def print_acc_scores(fair_df, col_used=['race'], sensitive_group="gender"):
    """
    전체 정확도와, 민감 속성별 정확도를 계산해 반환합니다.
    반환 형식: {"acc": overall_accuracy, "acc_each_group": [...]}
    """
    unq_groups = list(set(fair_df[sensitive_group]))
    # 타입을 통일해서 정렬 (모두 문자열로 변환)
    unq_groups = [str(g) for g in unq_groups]
    unq_groups.sort()  # 순서 고정(출력 편의)
    
    n_row = fair_df.shape[0]
    
    # (A) 각 태스크별 정확도
    acc_each_task = []
    for col in col_used:
        # 'col'과 'col+"_preds_fair"'이 동일한 비율 (numpy sum 사용)
        acc_val = np.sum(fair_df[col].values == fair_df[col + "_preds_fair"].values) / n_row
        acc_each_task.append(acc_val)
    
    # (B) 민감 그룹별 정확도
    acc_each_group = {}
    for g in unq_groups:
        # 원래 데이터프레임에서 해당 그룹만 필터링
        group_df = fair_df[fair_df[sensitive_group].astype(str) == g]
        n_g = len(group_df)
        
        group_accs = []
        for col in col_used:
            # 그룹별 정확도 계산
            sub_acc = np.sum(group_df[col].values == group_df[col + "_preds_fair"].values) / n_g
            group_accs.append(sub_acc)
        
        acc_each_group[g] = np.mean(group_accs)  # 여러 태스크 평균
    
    overall_acc = np.mean(acc_each_task)
    
    return {"acc": overall_acc, "acc_each_group": acc_each_group}


# -----------------------------------------------------------------------------
# 5) 가지치기 전/후 모델의 정확도를 비교하는 예시 함수
#    - 주의: 함수 내부에서 'prune_type', 'full_fair_df' 등이 언급되나,
#      인자로 전달되지 않아 실제로 동작 안 할 수 있음 → 아래 주석 처리
# -----------------------------------------------------------------------------
def show_acc_df(fair_df, fair_df_full, col_used, sensitive_group, prune_type="Unknown"):
    """
    가지치기 전/후 데이터프레임을 받아 정확도를 비교 출력합니다.
    단, 이 코드에서는 full_fair_df와 prune_type을 함수 외부에서 가져온다고 가정하고 있어
    실제 동작하려면 함수 시그니처나 내부 로직을 수정해야 합니다.
    """
    acc_scores = print_acc_scores(fair_df, col_used, sensitive_group)
    acc_scores_full = print_acc_scores(fair_df_full, col_used, sensitive_group)

    # overall
    acc_pruned = acc_scores['acc']
    acc_full = acc_scores_full['acc']
    diff = acc_pruned - acc_full

    # 그룹별
    # acc_each_group는 dict이므로, 그룹 순서나 개수를 맞춰야 함
    # 예시로 키 기준으로 정렬
    sorted_groups = sorted(acc_scores['acc_each_group'].keys())
    group_acc_pruned = [acc_scores['acc_each_group'][k] for k in sorted_groups]
    group_acc_full = [acc_scores_full['acc_each_group'][k] for k in sorted_groups]
    group_diff = [p - f for p, f in zip(group_acc_pruned, group_acc_full)]

    print(f"[가지치기 전/후 정확도 비교] (Prune Type: {prune_type})")
    print(f"전체 정확도(가지치기 후): {acc_pruned:.4f}")
    print(f"전체 정확도(가지치기 전): {acc_full:.4f}")
    print(f"차이(Difference): {diff:.4f}")
    print(f"그룹별 정확도(가지치기 후): {group_acc_pruned}")
    print(f"그룹별 정확도(가지치기 전): {group_acc_full}")
    print(f"그룹별 차이: {group_diff}")
    if len(group_diff) > 1:
        print(f"그룹 차이 표준편차: {stdev(group_diff):.4f}")
    else:
        print("그룹이 하나뿐이라, 표준편차 계산 불가.")


# -----------------------------------------------------------------------------
# 6) 모델 저장 함수
#    - cfgs: (prune_type, dataset, prune_type, loss_type, sensitive_group, network, prune_rate, exp_idx)
#    - 내부에서 model_path를 구성하고, 이미 동일 파일명이 있으면 exp_idx를 증가
# -----------------------------------------------------------------------------
def save_model(best_model, cfgs):
    prune_type, dataset, prune_type, loss_type, sensitive_group, network, prune_rate, exp_idx = cfgs

    # 첫 번째 prune_type과 세 번째 prune_type2가 중복됨 → 필요하다면 수정 가능
    # 여기서는 그대로 사용하되, 파일명엔 세 번째 값(prune_type2)을 이용
    model_dir = f"trained_model/{prune_type}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    model_path = f"{model_dir}/{dataset}_{prune_type}_{loss_type}_by{sensitive_group}_{network}_{prune_rate}_{exp_idx}.pt"

    # 같은 이름 파일이 존재하면 exp_idx를 증가시키면서 고유 파일명 확보
    while os.path.exists(model_path):
        exp_idx += 1
        model_path = f"{model_dir}/{dataset}_{prune_type}_{loss_type}_by{sensitive_group}_{network}_{prune_rate}_{exp_idx}.pt"

    torch.save(best_model.state_dict(), model_path)
    print(f"모델이 저장되었습니다: {model_path}")


# -----------------------------------------------------------------------------
# 7) 모델 추론 결과를 CSV로 저장
#    - cfgs: (dataset, prune_type, loss_type, prune_rate, test_frame, face_dir, total_classes,
#             network, col_used, output_cols_each_task, sensitive_group, exp_idx)
# -----------------------------------------------------------------------------
def save_output(best_model, cfgs, csv_savedir="fair_dfs", save_file=True):
    (dataset, prune_type, loss_type, prune_rate, test_frame, face_dir,
     total_classes, network, col_used, output_cols_each_task,
     sensitive_group, exp_idx) = cfgs

    # 추론 결과를 DataFrame으로
    fair_df = check_fairness0(best_model,
                              test_frame,
                              face_dir,
                              total_classes,
                              network,
                              col_used,
                              output_cols_each_task,
                              sensitive_group)
    import config

    def calc_equalized_odds(
            df: pd.DataFrame,
            label_cols: list[str],
            group_col: str = 'gender',
            pred_suffix: str = '_preds_fair'
        ) -> float:


        # ── (1) 중복 컬럼 제거
        df = df.loc[:, ~df.columns.duplicated()]

        config.glo_fairness.clear()

        eo_gaps = []  # 속성별 EO_gap 저장

        for truth_col in label_cols:
            pred_col = truth_col + pred_suffix
            if pred_col not in df.columns:
                print(f"[경고] {pred_col} 컬럼이 없음 → 건너뜀")
                continue

            # ── (2) 집단별 TPR / FPR 계산
            tpr, fpr = {}, {}
            for g in df[group_col].unique():
                sub = df[df[group_col] == g]

                TP = ((sub[truth_col] == 1) & (sub[pred_col] == 1)).sum()
                FN = ((sub[truth_col] == 1) & (sub[pred_col] == 0)).sum()
                FP = ((sub[truth_col] == 0) & (sub[pred_col] == 1)).sum()
                TN = ((sub[truth_col] == 0) & (sub[pred_col] == 0)).sum()

                pos = TP + FN      # 양성 표본 수
                neg = FP + TN      # 음성 표본 수

                tpr[g] = TP / pos if pos else np.nan
                fpr[g] = FP / neg if neg else np.nan

                config.glo_fairness.setdefault(truth_col, {})[g] = {
                    'TP': TP, 'FN': FN, 'FP': FP, 'TN': TN,
                    'TPR': tpr[g], 'FPR': fpr[g]
                }

            # ── (3) 두 집단 모두 값이 있어야 EO 계산
            if len(tpr) != 2 or len(fpr) != 2 or np.any(pd.isna(list(tpr.values()))) \
                                                or np.any(pd.isna(list(fpr.values()))):
                print(f"[정보] {truth_col} → 표본 부족으로 EO 미계산")
                continue

            g0, g1 = sorted(tpr.keys())     # 집단 키 정렬(보통 0,1)

            tpr_diff = abs(tpr[g0] - tpr[g1])
            fpr_diff = abs(fpr[g0] - fpr[g1])
            eo_diff  = (tpr_diff + fpr_diff) / 2

            print(f"{truth_col:<25} EO_diff = {eo_diff:.3f}")
            eo_gaps.append(eo_diff)

        # ── (4) 평균 EO
        avg_eo = np.nanmean(eo_gaps) if eo_gaps else 0.0
        print("-" * 40)
        print(f"전체 평균 EO_diff = {avg_eo:.3f}")

        return avg_eo

    

    total_EO_diff = calc_equalized_odds(fair_df, col_used[:-1])

    
    config.glo_EO = total_EO_diff

    if not os.path.exists(csv_savedir):
        os.makedirs(csv_savedir, exist_ok=True)

    df_path = f"{csv_savedir}/{dataset}_{prune_type}_{loss_type}_by{sensitive_group}_{network}_{prune_rate}_{exp_idx}.csv"

    # 같은 이름 파일이 존재하면 exp_idx 증가
    while os.path.exists(df_path):
        exp_idx += 1
        df_path = f"{csv_savedir}/{dataset}_{prune_type}_{loss_type}_by{sensitive_group}_{network}_{prune_rate}_{exp_idx}.csv"

    if save_file:
        fair_df.to_csv(df_path, index=False)
        print(f"추론 결과 CSV 저장 완료: {df_path}")

    return fair_df

# -----------------------------------------------------------------------------
# 8) 맞춤(custom) forward 함수 예시
#    - 가지치기 시, weight에 mask를 곱하여 실제 연산에 반영
# -----------------------------------------------------------------------------
def custom_forward_conv2d(self, x):
    # print(f"Conv2d 레이어에 전달: mask.shape = {self.mask.shape}")
    masked_weight = self.weight * self.mask
    return safe_forward_with_cudnn_fallback(
        F.conv2d,
        x,
        masked_weight,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups,
    )

def custom_forward_conv1d(self, x):
    return safe_forward_with_cudnn_fallback(F.conv1d, x, self.weight * self.mask, self.bias)

def custom_forward_linear(self, x):
    return F.linear(x, self.weight * self.mask, self.bias)


# -----------------------------------------------------------------------------
# 9) 데이터셋 자동 다운로드(단, Imagenet은 수작업이 필요하다는 예외 처리)
#    - gdown 라이브러리 이용
# -----------------------------------------------------------------------------
def download_dataset(dataset, img_dir):
    """
    img_dir 경로에 이미 데이터셋이 있으면 패스,
    없으면 gdown으로 다운로드 후 압축 해제, 내부 폴더 구조 정리.
    """
    # (A) 이미 다운로드된 경우
    if os.path.isdir(img_dir):
        print("이미 해당 데이터셋이 존재합니다.")
        return

    # (B) Imagenet은 직접 준비해야 함
    if dataset == "Imagenet":
        raise FileNotFoundError("Imagenet은 별도로 다운로드해야 합니다. person subtree 이미지를 Images/Imagenet에 배치하세요.")

    # (C) 나머지 데이터셋(FairFace, UTKFace, CelebA)에 대해 gdown으로 다운로드 시도
    urls = {
        "FairFace": "https://drive.google.com/u/0/uc?id=1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86&export=download",
        "UTKFace": "https://drive.google.com/file/d/0BxYys69jI14kYVM3aVhKS1VhRUk/view?usp=sharing&resourcekey=0-dabpv_3J0C0cditpiAfhAw",
        "CelebA": "https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ"
    }

    if dataset not in urls:
        raise ValueError(f"'{dataset}'에 대한 다운로드 URL이 정의되지 않았습니다.")

    url = urls[dataset]
    output_zip = f"Images/{dataset}.zip"

    print(f"{dataset} 데이터셋 다운로드 중...")
    gdown.download(url, output_zip, quiet=False, fuzzy=True)

    # (D) 압축 해제
    print(f"{dataset} 압축 해제 중...")
    if dataset == 'UTKFace':
        # UTKFace는 tar 형식이므로 tarfile 사용
        with tarfile.open(output_zip) as tar:
            tar.extractall(img_dir)
    else:
        # 나머지는 zipfile 사용
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(img_dir)

    # (E) 폴더 구조 정리: 이미지들을 상위 폴더로 옮기고 서브폴더 제거
    subfolders = [x[0] for x in os.walk(img_dir)][1:]  # 첫 번째는 img_dir 자기 자신
    for subfolder in subfolders:
        files = os.listdir(subfolder)
        for file in files:
            src = os.path.join(subfolder, file)
            if dataset == 'FairFace':
                # FairFace는 train/val 폴더가 따로 있으므로, 파일 이름에 prefix를 붙임
                split = subfolder.split(os.sep)[-1]  # 'train' or 'val'
                dst_name = f"{split}_{file}"
                dst = os.path.join(img_dir, dst_name)
            else:
                dst = os.path.join(img_dir, file)

            shutil.move(src, dst)
        os.rmdir(subfolder)

    print(f"{dataset} 데이터셋 다운로드 및 정리가 완료되었습니다.")


# -----------------------------------------------------------------------------
# 10) 시드 설정 함수
# -----------------------------------------------------------------------------
def setseed(seed):
    """
    랜덤 시드를 고정해 재현성(reproducibility)을 높입니다.
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


# -----------------------------------------------------------------------------
# 10) 가지치기 전 모델 저장장
# -----------------------------------------------------------------------------

def save_unpruned_model(model, optimizer, scheduler, cfgs):
    """
    unpruned 모델을 (가중치뿐 아니라 optimizer, scheduler 상태도 함께) 저장하기 위한 함수.
    cfgs는 (dataset, loss_type, sensitive_group, network, exp_idx) 등을 담은 튜플(또는 리스트)라고 가정합니다.
    prune_type, prune_rate 등은 사용하지 않습니다.
    """

    # cfgs 예시:
    # (dataset, loss_type, sensitive_group, network, exp_idx)
    # 사용자 코드에 맞춰 실제로는 cfgs 구조를 알맞게 변형하세요.
    dataset, loss_type, sensitive_group, network, exp_idx = cfgs

    # unpruned 모델 저장 경로
    model_dir = "trained_model/unpruned"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # 파일명에 "unpruned" 표시
    model_path = f"{model_dir}/{dataset}_unpruned_{loss_type}_by{sensitive_group}_{network}_{exp_idx}.pt"

    # 같은 이름의 파일이 존재하면 exp_idx를 증가시키면서 고유 파일명 확보
    while os.path.exists(model_path):
        exp_idx += 1
        model_path = f"{model_dir}/{dataset}_unpruned_{loss_type}_by{sensitive_group}_{network}_{exp_idx}.pt"

    # Optimizer, Scheduler등도 함께 저장
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict() if optimizer else None,
        'scheduler_state': scheduler.state_dict() if scheduler else None
    }, model_path)

    print(f"가지치기 전 모델이 저장되었습니다: {model_path}")