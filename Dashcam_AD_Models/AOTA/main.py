"""
Train MYModel (model.py) on the synthetic DoTA/DADA-style dataset produced by
process_dataset.py.

Usage:
    python process_dataset.py            # generate the dataset first
    python main.py                       # train with defaults
    python main.py --eval_only --resume checkpoints/best_model.pth.tar
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model import MYModel, load_checkpoint

FPS = 10.0


class SyntheticAccidentDataset(Dataset):
    def __init__(self, data_root, split):
        self.files = sorted(glob.glob(os.path.join(data_root, split, '*.npz')))
        if not self.files:
            raise FileNotFoundError(
                f'No .npz files found in {os.path.join(data_root, split)}. '
                f'Run "python process_dataset.py" first.')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        features = torch.from_numpy(data['features'].astype(np.float32))   # [T, 20, 512]
        detection = torch.from_numpy(data['detection'].astype(np.float32)) # [T, 19, 6]
        depth = torch.from_numpy(data['depth'])                            # [T, 72, 128] uint8
        label = torch.from_numpy(data['label'].astype(np.float32))         # [2] one-hot
        toa = torch.tensor(float(data['toa']), dtype=torch.float32)
        return features, detection, depth, label, toa


def average_precision(labels, scores):
    """Video-level AP without external dependencies."""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    precision = tp / (np.arange(len(labels)) + 1)
    return float((precision * labels).sum() / labels.sum())


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    total_loss, num_batches = 0.0, 0
    video_labels, video_scores, ttas = [], [], []

    with torch.no_grad():
        for features, detection, depth, label, toa in loader:
            losses, out = model(features, label.to(device), toa.to(device),
                                detection, depth.numpy())
            total_loss += losses['cross_entropy'].item()
            num_batches += 1

            prob = torch.softmax(out, dim=-1)[:, :, 1].cpu().numpy()  # [B, T]
            label_np = label.numpy()
            toa_np = toa.numpy()

            for i in range(prob.shape[0]):
                is_positive = label_np[i, 1] > 0.5
                video_labels.append(1.0 if is_positive else 0.0)
                video_scores.append(float(prob[i].max()))
                if is_positive:
                    alarms = np.where(prob[i] >= threshold)[0]
                    alarms = alarms[alarms < toa_np[i]]
                    tta = (toa_np[i] - alarms[0]) / FPS if len(alarms) > 0 else 0.0
                    ttas.append(float(tta))

    ap = average_precision(video_labels, video_scores)
    preds = (np.asarray(video_scores) >= threshold).astype(np.float64)
    acc = float((preds == np.asarray(video_labels)).mean())
    mtta = float(np.mean(ttas)) if ttas else 0.0
    avg_loss = total_loss / max(num_batches, 1)
    return {'loss': avg_loss, 'AP': ap, 'accuracy': acc, 'mTTA': mtta}


def train(args):
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_set = SyntheticAccidentDataset(args.data_root, 'train')
    test_set = SyntheticAccidentDataset(args.data_root, 'test')
    # AdaptiveAdjacencyMatrix in model.py is built for a fixed batch size of 10,
    # so both loaders must deliver full batches of exactly that size.
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=True)
    print(f'Train videos: {len(train_set)} | Test videos: {len(test_set)}')

    model = MYModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, args.resume,
                                                        isTraining=not args.eval_only)

    if args.eval_only:
        metrics = evaluate(model, test_loader, device)
        print(f"[eval] loss {metrics['loss']:.4f} | AP {metrics['AP']:.4f} | "
              f"accuracy {metrics['accuracy']:.4f} | mTTA {metrics['mTTA']:.2f}s")
        return

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_ap = -1.0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, num_batches = 0.0, 0
        t0 = time.time()

        for step, (features, detection, depth, label, toa) in enumerate(train_loader):
            losses, _ = model(features, label.to(device), toa.to(device),
                              detection, depth.numpy())
            loss = losses['cross_entropy']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            if (step + 1) % args.log_interval == 0:
                print(f'  epoch {epoch + 1} step {step + 1}/{len(train_loader)} '
                      f'loss {loss.item():.4f}')

        avg_loss = epoch_loss / max(num_batches, 1)
        metrics = evaluate(model, test_loader, device)
        print(f'Epoch {epoch + 1}/{args.epochs} | train loss {avg_loss:.4f} | '
              f"test loss {metrics['loss']:.4f} | AP {metrics['AP']:.4f} | "
              f"accuracy {metrics['accuracy']:.4f} | mTTA {metrics['mTTA']:.2f}s | "
              f'{time.time() - t0:.1f}s')

        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'epoch': epoch + 1}
        torch.save(state, os.path.join(args.ckpt_dir, 'last_model.pth.tar'))
        if metrics['AP'] > best_ap:
            best_ap = metrics['AP']
            torch.save(state, os.path.join(args.ckpt_dir, 'best_model.pth.tar'))
            print(f'  new best AP {best_ap:.4f} -> saved best_model.pth.tar')

    print(f'Training finished. Best test AP: {best_ap:.4f}')


def main():
    parser = argparse.ArgumentParser(description='Train accident anticipation model on synthetic data')
    parser.add_argument('--data_root', type=str, default='data/synthetic')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=10,
                        help='must stay 10: AdaptiveAdjacencyMatrix in model.py is built for batch size 10')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default='',
                        help='path to a checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--log_interval', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
