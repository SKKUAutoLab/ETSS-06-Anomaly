import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import AccidentXai
from dataloader import MyDataset, MySampler
from tqdm import tqdm
import os
import glob
import numpy as np
from eval import evaluation_P_R80
import argparse
from natsort import natsorted

parser = argparse.ArgumentParser()
parser.add_argument('--fps', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--extractor', type=str, default='resnet50')
parser.add_argument('--loss_type', type=str, default='exponential', choices=['exponential', 'crossentropy'])
parser.add_argument('--network_type', type=str, default='gru', choices=['cnn', 'gru'])
parser.add_argument('--train_dir', type=str, default='data/train')
parser.add_argument('--test_dir', type=str, default='data/test')
parser.add_argument('--demo_dir', type=str, default='demo/000605')
parser.add_argument('--model_dir', type=str, default='snapshot')
args = parser.parse_args()

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
device = ("cuda" if torch.cuda.is_available() else "cpu")
# train loader
train_class_paths = [d.path for d in os.scandir(args.train_dir) if d.is_dir] # ['data/train/positive', 'data/train/negative']
train_class_image_paths = []
train_end_idx = []
for c, class_path in enumerate(train_class_paths):
    for d in os.scandir(class_path):
        if d.is_dir:
            paths = natsorted(glob.glob(os.path.join(d.path, '*.jpg')))
            paths = [(p, c) for p in paths] # c = 0/1
            train_class_image_paths.extend(paths)
            train_end_idx.extend([len(paths)])
train_end_idx = [0, *train_end_idx]
train_end_idx = torch.cumsum(torch.tensor(train_end_idx), 0)
seq_length = 99
train_sampler = MySampler(train_end_idx, seq_length)
train_data = MyDataset(image_paths=train_class_image_paths, transform=transform, length=len(train_sampler))
train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, sampler=train_sampler)
# test loader
test_class_paths = [d.path for d in os.scandir(args.test_dir) if d.is_dir]
test_class_image_paths = []
test_end_idx = []
for c, class_path in enumerate(test_class_paths):
    for d in os.scandir(class_path):
        if d.is_dir:
            paths = natsorted(glob.glob(os.path.join(d.path, '*.jpg')))
            paths = [(p, c) for p in paths] # c = 0/1
            test_class_image_paths.extend(paths)
            test_end_idx.extend([len(paths)])
test_end_idx = [0, *test_end_idx]
test_end_idx = torch.cumsum(torch.tensor(test_end_idx), 0)
seq_length = 99
test_sampler = MySampler(test_end_idx, seq_length)
test_data = MyDataset(image_paths=test_class_image_paths, transform=transform, length=len(test_sampler))
test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, sampler=test_sampler)

def test(test_dataloader, model):
    all_pred = []
    all_labels = []
    losses_all = []
    all_toas = []
    with torch.no_grad():
        loop = tqdm(test_dataloader, total=len(test_dataloader), leave=True)
        for imgs, labels, toa in loop:
            imgs = imgs.to(device)
            labels = torch.squeeze(labels).to(device)
            loss, outputs = model(imgs, labels, toa)
            loss = loss['total_loss'].item()
            losses_all.append(loss)
            num_frames = imgs.size()[1]
            batch_size = imgs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            for t in range(num_frames):
                pred = outputs[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1])/np.sum(np.exp(pred), axis=1)
            all_pred.append(pred_frames)
            label_onehot = labels.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)
            toas = np.squeeze(toa.cpu().numpy()).astype(np.int32)
            all_toas.append(toas)
            loop.set_postfix(val_loss=np.mean(losses_all))
    all_pred = np.vstack((np.vstack(all_pred[0][:-1]), all_pred[0][-1]))
    all_labels = np.hstack((np.hstack(all_labels[0][:-1]), all_labels[0][-1]))
    all_toas = np.hstack((np.hstack(all_toas[0][:-1]), all_toas[0][-1]))
    return all_pred, all_labels, all_toas, losses_all

def train():
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model = AccidentXai(args.h_dim, args.n_layers, args.extractor, args.loss_type, args.network_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.extractor == 'resnet50':
        for name, param in model.features.named_parameters():
            if "fc.0.weight" in name or "fc.0.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif args.extractor == 'vgg16':
        # VGG16
        for name, param in model.features.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # GRU
        for name, param in model.gru_net.named_parameters():
            if 'gru.weight' in name or 'gru.bias' in name:
                param.requires_grad = True
            elif 'dense1' in name or 'dense2' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        raise NotImplementedError
    model.train()
    loss_best = 100
    for epoch in range(args.epoch):
        loop = tqdm(train_dataloader, total=len(train_dataloader), leave=True)
        for imgs, labels, toa in loop:
            loop.set_description(f"Epoch [{epoch + 1}/{args.epoch}]")
            imgs = imgs.to(device)
            labels = torch.squeeze(labels).to(device)
            loss, outputs = model(imgs, labels, toa)
            optimizer.zero_grad()
            loss['total_loss'].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            loop.set_description(f"Epoch [{epoch + 1}/{args.epoch}]")
            loop.set_postfix(loss=loss['total_loss'].item())
        print('Train loss:', loss['total_loss'].item())
        print('Start testing')
        model.eval()
        all_pred, all_labels, all_toas, losses_all = test(test_dataloader, model)
        total_loss = np.mean(losses_all)
        metrics = {}
        metrics['AP'], metrics['mTTA'], metrics['TTA_R80'], metrics['PR80'] = evaluation_P_R80(all_pred, all_labels, all_toas, args.fps)
        print('Loss test:', total_loss)
        print('AP:', metrics['AP'])
        print('PR80:', metrics['PR80'])
        print('mTTA:', metrics['mTTA'])
        print('TTA_R80', metrics['TTA_R80'])
        model.train()
        # save best model
        best_model_file = os.path.join(args.model_dir, 'best_model.pth')
        model_file = os.path.join(args.model_dir, 'saved_model_%02d.pth' % epoch)
        torch.save(model.state_dict(), model_file)
        if total_loss < loss_best:
            loss_best = total_loss
            torch.save(model.state_dict(), best_model_file)

if __name__ == "__main__":
    train()