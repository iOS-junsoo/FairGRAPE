# FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face Attribute Classification

This repo presents an official implementation of FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face Attribute Classification

![FairGRAPE_concept2](https://user-images.githubusercontent.com/60991389/177890025-4599bd0f-176d-4f5f-aff8-73df9c963a6e.png)

## Dependencies

The code has been tested on the following environment:

```
python 3.9
pytorch 1.11.0
dlib 19.22.0
opencv2 4.5.5
```

Use Anaconda and the following command to replicate the full environment:

```
conda env create -f environment.yml
```

## Datasets

This code automatically downloads the following datasets for trianing, cross validation and testing: [FairFace](https://github.com/joojs/fairface), [UTKFace](https://susanqq.github.io/UTKFace/) and [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Request demographics labels of the Imagenet person subtree through the offical [database](https://image-net.org/), save the annotation file under /csv and images under /Images/Imagenet.


## Example Usage:

UTKFace experiments:
```
python main_test.py --sensitive_group 'gender' --loss_type 'race' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'UTKFace'  --prune_rate 0.9 --keep_per_iter 0.975
```

FairFace experiments
```
python main_test.py --sensitive_group race--loss_type 'race' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'FairFace'  --prune_rate 0.99
```

CelebA experiments
```
python main_test.py --sensitive_group 'gender' --loss_type 'classes' --prune_type 'FairGRAPE' --network 'mobilenetv2' --dataset 'CelebA' --prune_rate 0.9
```

Imagenet experiments:
```
python main_test.py --sensitive_group 'gender' --loss_type 'classes' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'Imagenet'  --prune_rate 0.5
```

Download trained models [here](https://www.dropbox.com/sh/rk362mypuikeklh/AADF93dWPQo3rPTUhyaLBn3Ga?dl=0).

Loading a trained model:
```
python main_test.py --sensitive_group 'gender' --loss_type 'race' --prune_type 'FairGRAPE' --network 'mobilenetv2'--dataset 'UTKFace'  --prune_rate 0.9 --keep_per_iter 0.975 --checkpoint "UTKFace_FairGRAPE_race_bygender_resnet34_0.9_0.pt" --init_train 0 --retrain 0 --print_acc 1
```


## Online Taylor neuron pruning (impt=7)

`impt=7` implements Taylor-FO-BN from [Molchanov et al., CVPR 2019](https://arxiv.org/abs/1906.10771)
for one CLI-selected CelebA attribute. It uses the official Train split for online
SGD and importance, Validation for selection, and Test only after final selection.

```bash
python main_test.py --dataset CelebA --network mobilenetv2 \
  --prune_type FairGRAPE --impt 7 --target_attr Smiling \
  --sensitive_group gender --prune_rate 0.70 --batch 256 --seed 42 \
  --no_grl --save_mask
```

Replace `Smiling` with any supported prediction attribute. Add
`--checkpoint PATH --no_init_train` to reuse a compatible dense single-attribute
checkpoint (same attribute, official split metadata and GRL disabled). Without
a checkpoint, the existing ImageNet-initialized dense training schedule is used.
Pruned checkpoints are accepted only with `--single_attr_eval_only`; training
resume is not supported. `--defer_test` postpones final Test evaluation.

MobileNetV2's 52 backbone convolutions (`features.0~18`, 17,056 output neurons) have independent
gates immediately after BN. `--network resnet18` uses torchvision's post-activation
ResNet-18: 20 convolutions / 4,800 output neurons, including the stem and three
downsample projections. Its `fc` classifier is protected; gates precede residual
addition and do not gate the summed block output. This is an adaptation, not the
paper's CIFAR-10 pre-activation ResNet-18 or its shared ResNet skip-gate design.
The classifier is trained but never pruned. Removing
a neuron zeros its gate and its own Conv output filter; it does **not** remove a
downstream input column, another Conv's neuron, or the residual identity path.
Tensor shapes stay unchanged; this is masked pruning, not a physically compact
model or a guarantee of inference speedup.

Importance is the average of **per-minibatch squared gate gradients** (mean task
Cross-Entropy), with inter-event EMA and no per-layer normalization. Gates are
excluded from SGD. Zeroed filter gradients, weights and momentum are masked.

For MobileNetV2, importance now uses a **separate evaluation-mode pass** on the
same Train minibatch, before the ordinary training update. BN running statistics
are fixed and dropout is disabled for scoring; `autograd.grad` requests only gate
gradients and does not write training `.grad` buffers. The ordinary SGD pass still
trains with minibatch BN statistics and dropout. This avoids the depthwise/BN
scale cancellation observed in expansion-gate gradients. It is an experimental
adaptation of the paper, not a guaranteed accuracy recovery. After reaching the
target, the extra scoring pass stops. ResNet-18 retains its training-pass scoring.

| Option | impt=7 default |
| --- | --- |
| `--prune_rate` | 0.70, measured over all Conv/Linear weights including the protected classifier |
| `--batch` | 256 |
| `--taylor_epochs` | 25 total online pruning + fixed-mask fine-tuning epochs |
| `--taylor_lr` | 0.001; SGD momentum 0.9, weight decay 0; LR divided by 10 after epochs 10 and 20 |
| `--taylor_frequency` | 30 training minibatches, accumulation continues across epochs |
| `--taylor_prune_fraction` | 0.02 of initial neurons, rounded up (MobileNetV2: 342; ResNet-18: 96 neurons/event) |
| `--taylor_ema` | 0.9 |
| `--taylor_score_mode` | `auto`: `frozen_bn` for MobileNetV2, `train` for ResNet-18; either can be selected explicitly |
| `--taylor_collapse_guard` | `auto`: enabled for MobileNetV2, disabled for ResNet-18; explicit `on` / `off` available |

The collapse guard caches up to 16 distinct-input samples from the first Train
batch and checks for exactly input-independent block outputs/logits before and
after removal. If eval-mode outputs collapse, it repeats the check using minibatch
BN statistics; all BN buffers and module modes are restored afterward. A removal
is rejected when both modes collapse, or non-finite outputs occur. Constant class
predictions alone are not a rejection criterion. On rejection, model, masks, gates,
optimizer momentum and pruning score state are restored, a `rejected_iter_XX.json`
and `ROLLBACK_before_prune_XX.pt` snapshot are saved, and the run stops with
`failed.json` instead of a successful final model. A small probe cannot guarantee
that every image retains a useful representation. No layer-wise cap or BN
recalibration is added. Guarded runs require a batch size of at least two.

Each Conv retains at least one neuron. Selection stops at the first neuron that
reaches the weight target: at 70%, the achieved sparsity is in `[70%, 70.044%)`.
SGD momentum is reset once on reaching the target. The winner is selected only
among target-reaching models by Validation accuracy, then smaller EO, then loss.
An unreachable or unmet target produces `failed.json`, never `complete.json`.
Legacy per-round settings (`--keep_per_iter`, `--retrain_iter`, `--prune_iter`,
etc.) conflict with this online schedule and are rejected when explicitly supplied.

The paper's gate criterion and online procedure are retained. MobileNet gate
placement, independent filter masks, the one-neuron floor, weight-based stopping,
342-neuron batches, frozen-BN MobileNet scoring, collapse checks, deterministic
tie handling, and EO tie-breaking are adaptations;
this is not a reproduction of its ImageNet experiment settings.

ResNet-18, Attractive, without a task checkpoint (ImageNet initialization,
then dense training for 8/1/1 epochs at Adam learning rates 1e-4/1e-5/1e-6,
then 25 total online pruning/fine-tuning epochs):

```bash
python main_test.py --dataset CelebA --network resnet18 \
  --prune_type FairGRAPE --impt 7 --target_attr Attractive \
  --sensitive_group gender --prune_rate 0.70 --batch 256 --seed 42 \
  --taylor_epochs 25 --taylor_lr 0.001 --taylor_frequency 30 \
  --taylor_prune_fraction 0.02 --taylor_ema 0.9 --no_grl --save_mask
```

Default ResNet-18 run/model/log names include `_resnet18`; directory structure
and report formats remain shared. Dense and pruned checkpoints are validated
against the network, two-class head, attribute and official split metadata.
Changing the architecture does not guarantee accuracy at 70% weight sparsity;
the existing one-neuron floor and global ranking are retained. No BN
recalibration or per-layer pruning cap is added by this network option.

### Results and checkpoints

The layout and report functions are shared with impt=5/6:

- Results: `retrain_epoch_results/임시 저장소/CelebA_impt7_{attribute}_official_seed{seed}_{timestamp}/`
- Models: `save_models/{timestamp}_CelebA_taylor_fo_bn_impt7_{attribute}_official_seed{seed}/`
- Neuron logs/scores: `channel_pruning_logs/{run_name}/`
- With `--single_attr_run_dir PATH`, these become `PATH`, `PATH/models`, and `PATH/channel_pruning`.

Every removal event gets `pruning/iter_XX/val/`, a
`BEST_MODEL_SUMMARY_prune_XX_val.txt`, a `metrics.csv` row, and a model named
`BEST_MODEL_prune_XX_val_acc_…_eo_….pt`. Validation runs in evaluation mode without
advancing the training importance or BN statistics. Event predictions are never
overwritten by later epoch evaluations: those go in `epoch_XX/val/`, alongside
`epochs.jsonl` and `Total_info_online_epoch_XX_val.txt`. After reaching the target,
epoch records and the final selected report go in `pruning/final/`.

Checkpoints always contain gates, masks, the dense source hash, method/scope,
actual sparsity and selection metadata. `--save_mask` also exports
`impt7_{attribute}_official_masks_XXX.pt`. Existing group metrics (Accuracy, BA,
EO, EOPP, DP), manifests, run information and `complete.json` are retained.

For a small end-to-end validation run, add:

```bash
--single_attr_smoke_samples 4 --batch 4 --taylor_epochs 2 \
--taylor_frequency 1 --taylor_prune_fraction 1 --single_attr_run_dir /tmp/taylor-smoke
```

This accelerated smoke configuration is for implementation validation, not an
accuracy experiment; its metadata cannot be used as a production dense checkpoint.

## Acknowledgement 
Parts of code were borrowed from [SNIP](https://github.com/mil-ad/snip), [WS](https://github.com/mightydeveloper/Deep-Compression-PyTorch), [GraSP](https://github.com/alecwangcq/GraSP)
