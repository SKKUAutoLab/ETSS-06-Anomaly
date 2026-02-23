import torch
from torch.utils.data import DataLoader
from models.model import AMNet
from dataloader import MyDataset
import argparse
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve, auc
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def evaluation(all_pred, all_labels):
    fpr, tpr, thresholds = roc_curve(np.array(all_labels), np.array(all_pred), pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def average_losses(losses_all):
    cross_entropy = 0
    for losses in losses_all:
        cross_entropy += losses['cross_entropy']
    losses_mean = cross_entropy/len(losses_all)
    return losses_mean

def _load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar'):
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
    else:
        print("No checkpoint found at '{}'".format(filename))
    return model, optimizer, start_epoch

def train_eval(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_data = MyDataset(args.data_path, 'train', toTensor=True, device=device)
    test_data = MyDataset(args.data_path, 'val', toTensor=True, device=device)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    n_frames = 100
    model_file = args.ckpt_file
    model = AMNet(args.x_dim, args.h_dim, n_frames)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    model = model.to(device)
    model.train()
    if args.resume:
        model, optimizer, start_epoch = _load_checkpoint(model, optimizer=optimizer, filename=model_file)
    auc_max = 0
    ap_max = 0

    for k in range(args.epoch):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (batch_xs, batch_det, batch_toas, batch_flow) in loop:
            optimizer.zero_grad()
            losses, all_outputs, all_labels = model(batch_xs, batch_det, batch_flow)
            losses['cross_entropy'].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            loop.set_description(f"Epoch [{k}/{args.epoch}]")
            loop.set_postfix(loss=losses['cross_entropy'].item())
        print("Start evaluation")
        model.eval()
        losses_all, all_pred, all_labels = test_all(test_loader, model)
        loss_val = average_losses(losses_all)
        fpr, tpr, roc_auc = evaluation(all_pred, all_labels)
        ap = average_precision_score(np.array(all_labels), np.array(all_pred))
        print('Epoch:', k)
        print('Loss:', loss_val.item())
        print(f"AUC: {roc_auc:.4f}")
        print(f"AP: {ap:.4f}")
        model.train()
        model_file = os.path.join(args.output_dir, 'model_%02d.pth' % k)
        torch.save({'epoch': k, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_file)
        if roc_auc > auc_max:
            auc_max = roc_auc
            model_file = os.path.join(args.output_dir, 'best_auc.pth')
            torch.save({'epoch': k, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_file)
            print('Best AUC Model has been saved to: %s' % model_file)
        elif ap > ap_max:
            ap_max = ap
            model_file = os.path.join(args.output_dir, 'best_ap.pth')
            torch.save({'epoch': k, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_file)
            print('Best AP Model has been saved to: %s' % model_file)
        scheduler.step(losses['cross_entropy'])

def test_all(test_loader, model):
    all_pred = []
    all_labels = []
    losses_all = []
    with torch.no_grad():
        for i, (batch_xs, batch_det, batch_toas, batch_flow) in enumerate(test_loader):
            losses, all_outputs, labels = model(batch_xs, batch_det, batch_flow)
            losses_all.append(losses)
            for t in range(100):
                frame = all_outputs[t]
                if len(frame) == 0:
                    continue
                else:
                    for j in range(len(frame)):
                        score = np.exp(frame[j][:, 1])/np.sum(np.exp(frame[j]), axis=1)
                        all_pred.append(score)
                        all_labels.append(labels[t][j] + 0) # added zero to convert array to scalar
    return losses_all, all_pred, all_labels

def test_eval(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_data = MyDataset(args.data_path, 'val', toTensor=True, device=device)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
    n_frames = 100
    model_file = args.ckpt_file
    model = AMNet(args.x_dim, args.h_dim, n_frames)
    model = model.to(device)
    model.eval()
    model, _, _ = _load_checkpoint(model, filename=model_file)
    print('Load checkpoint successfully')
    losses_all, all_pred, all_labels = test_all(test_loader, model)
    fpr, tpr, roc_auc = evaluation(all_pred, all_labels)
    ap = average_precision_score(np.array(all_labels), np.array(all_pred))
    print(f"AUC: {roc_auc:.4f}")
    print(f"AP: {ap:.4f}")
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='feat_extract/feature/ROL')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--h_dim', type=int, default=256, help='hidden dimension of the gru. Default: 256')
    parser.add_argument('--x_dim', type=int, default=2048, help='dimension of the resnet')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--ckpt_file', type=str, default='checkpoints/best_auc.pth')
    parser.add_argument('--resume', action='store_true', help='If to resume the training. Default: False')
    parser.add_argument('--tl', action='store_true', help='If want transfer learning. Default: False')
    args = parser.parse_args()
    if args.phase == 'train':
        print('Start training')
        train_eval(args)
    else:
        print('Start testing')
        test_eval(args)