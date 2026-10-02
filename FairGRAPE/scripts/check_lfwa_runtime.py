"""Preflight with four real LFWA training images and the production GPU model."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    print('LFWA runtime preflight: loading runtime', flush=True)
    import torch
    import lfwa
    from dataset import make_datasets
    from train_and_val import loss_multi_tasks
    from util import make_model, setseed
    torch.set_num_threads(2)
    setseed(42)
    frames, metadata = lfwa.load_frames()
    train_loader, _ = make_datasets(frames['train'].iloc[:4], frames['val'].iloc[:4], True, 4, lfwa.LABEL_COLUMNS)
    images, labels = next(iter(train_loader))
    assert list(images.shape) == [4, 3, 224, 224] and list(labels.shape) == [4, 40]
    print('LFWA runtime preflight: real images and labels loaded; creating pretrained model', flush=True)
    model = make_model(network='mobilenetv2', n_classes=78, pretrained=True)
    print('LFWA runtime preflight: moving model to CUDA', flush=True)
    model.cuda()
    images, labels = images.cuda(), labels.cuda()
    print('LFWA runtime preflight: forward', flush=True)
    torch.backends.cudnn.enabled = False
    logits = model(images)
    assert list(logits.shape) == [4, 78]
    criterion = torch.nn.CrossEntropyLoss()
    loss = loss_multi_tasks(logits, labels, criterion, lfwa.OUTPUT_SLICES)
    changed = labels.clone()
    changed[:, -1] = 1 - changed[:, -1]
    assert torch.equal(loss, loss_multi_tasks(logits, changed, criterion, lfwa.OUTPUT_SLICES))
    print('LFWA runtime preflight: backward and Adam update', flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(loss):
        raise AssertionError('Non-finite preflight loss')
    report = {'status': 'passed', 'images': list(images.shape), 'labels': list(labels.shape),
              'outputs': list(logits.shape), 'young_loss_invariance': True,
              'backward_and_adam': True, 'loss': float(loss),
              'peak_gpu_mib': torch.cuda.max_memory_allocated() / 1024 ** 2,
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'manifest_sha256': metadata['manifest_sha256'],
              'seconds': time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
