import numpy as np
import torch
import os
import argparse
import shutil
from torch.utils.data import DataLoader
from src.Models import DSTA
from src.eval import evaluation_P_R80
from src.DataLoader import DADDataset, CrashDataset
from tqdm import tqdm
from sklearn.metrics import average_precision_score
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

def average_losses(losses_all):
    total_loss, cross_entropy, aux_loss = 0, 0, 0
    losses_mean = {}
    for losses in losses_all:
        total_loss += losses['total_loss']
        cross_entropy += losses['cross_entropy']
        aux_loss += losses['auxloss']
    losses_mean['total_loss'] = total_loss / len(losses_all)
    losses_mean['cross_entropy'] = cross_entropy / len(losses_all)
    losses_mean['auxloss'] = aux_loss / len(losses_all)
    return losses_mean

def test_all(testdata_loader, model):
    all_pred = []
    all_labels = []
    all_toas = []
    losses_all = []
    with torch.no_grad():
        for i, (batch_xs, batch_ys, batch_toas) in enumerate(testdata_loader):
            losses, all_outputs, hiddens, alphas = model(batch_xs, batch_ys, batch_toas, hidden_in=None)
            losses['total_loss'] = losses['cross_entropy']
            losses['total_loss'] += args.loss_beta * losses['auxloss']
            losses_all.append(losses)
            num_frames = batch_xs.size()[1]
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            for t in range(num_frames):
                pred = all_outputs[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
            all_pred.append(pred_frames)
            label_onehot = batch_ys.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)
            toas = np.squeeze(batch_toas.cpu().numpy()).astype(np.int32)
            all_toas.append(toas)
    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))
    return all_pred, all_labels, all_toas, losses_all

def test_all_vis(testdata_loader, model, vis=True):
    model = model.cuda()
    model.eval()
    all_pred = []
    all_labels = []
    all_toas = []
    vis_data = []
    with torch.no_grad():
        for i, (batch_xs, batch_ys, batch_toas, detections, video_ids) in tqdm(enumerate(testdata_loader), desc="batch progress", total=len(testdata_loader)):
            losses, all_outputs, hiddens, alphas = model(batch_xs, batch_ys, batch_toas, hidden_in=None)
            num_frames = batch_xs.size()[1]
            batch_size = batch_xs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            for t in range(num_frames):
                pred = all_outputs[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
            all_pred.append(pred_frames)
            label_onehot = batch_ys.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size,])
            all_labels.append(label)
            toas = np.squeeze(batch_toas.cpu().numpy()).astype(np.int32)
            all_toas.append(toas)
            if vis:
                vis_data.append({'pred_frames': pred_frames, 'label': label, 'toa': toas, 'detections': detections, 'video_ids': video_ids})
    all_pred = np.vstack((np.vstack(all_pred[:-1]), all_pred[-1]))
    all_labels = np.hstack((np.hstack(all_labels[:-1]), all_labels[-1]))
    all_toas = np.hstack((np.hstack(all_toas[:-1]), all_toas[-1]))
    return all_pred, all_labels, all_toas, vis_data

def update_final_model(src_file, dest_file):
    assert os.path.exists(src_file), "src file does not exist!"
    if os.path.exists(dest_file):
        if not os.path.samefile(src_file, dest_file):
            os.remove(dest_file)
    shutil.copyfile(src_file, dest_file)

def load_checkpoint(model, optimizer=None, filename='final_model.pth', isTraining=True):
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        if isTraining:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint {}".format(filename))
    else:
        print("No checkpoint found at '{}'".format(filename))
    return model, optimizer, start_epoch

def train_eval():
    data_path = os.path.join(args.data_path, args.dataset)
    model_dir = os.path.join('output', args.dataset, 'snapshot')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.dataset == 'dad':
        train_data = DADDataset(data_path, args.feature_name, 'training', toTensor=True, device=device)
        test_data = DADDataset(data_path, args.feature_name, 'testing', toTensor=True, device=device)
    elif args.dataset == 'crash':
        train_data = CrashDataset(data_path, args.feature_name, 'train', toTensor=True, device=device)
        test_data = CrashDataset(data_path, args.feature_name, 'test', toTensor=True, device=device)
    else:
        raise NotImplementedError
    traindata_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    testdata_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
    model = DSTA(train_data.dim_feature, args.hidden_dim, args.latent_dim, n_layers=args.num_rnn, n_obj=train_data.n_obj, n_frames=train_data.n_frames, fps=train_data.fps, with_saa=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    model = model.cuda()
    model.train()
    start_epoch = -1
    if args.resume:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer=optimizer, filename=args.model_file)
    iter_cur = 0
    best_metric = 0
    metrics = {}
    metrics['AP'] = 0
    for k in range(args.epoch):
        loop = tqdm(enumerate(traindata_loader), total=len(traindata_loader))
        if k <= start_epoch:
            iter_cur += len(traindata_loader)
            continue
        for i, (batch_xs, batch_ys, batch_toas) in loop:
            optimizer.zero_grad()
            losses, all_outputs, hidden_st, alphas = model(batch_xs, batch_ys, batch_toas)
            losses['total_loss'] = losses['cross_entropy']
            losses['total_loss'] += args.loss_beta * losses['auxloss']
            losses['total_loss'].mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            loop.set_description(f"Epoch [{k}/{args.epoch}]")
            loop.set_postfix(loss= losses['total_loss'].item())
            print("Total training loss:", losses['total_loss'].mean().item())
            print("Cross entropy loss:", losses['cross_entropy'].mean().item())
            print("Aux loss:", losses['auxloss'].mean().item())
            iter_cur += 1
            # eval
            if iter_cur % args.test_iter == 0:
                model.eval()
                all_pred, all_labels, all_toas, losses_all = test_all(testdata_loader, model)
                model.train()
                print("Start evaluation")
                metrics = {}
                metrics['AP'], metrics['mTTA'], metrics['TTA_R80'], metrics['P_R80'] = evaluation_P_R80(all_pred, all_labels, all_toas, fps=test_data.fps)
                print("Total testing loss:", losses['total_loss'].mean().item())
                print('Cross entropy loss:', losses['cross_entropy'].mean().item())
                print("AP:", metrics['AP'])
                print('P_R80:', metrics['P_R80'])
                print("mTTA:", metrics['mTTA'])
                print('TTA_R80:', metrics['TTA_R80'])
        model_file = os.path.join(model_dir, 'bayesian_gcrnn_model_%02d.pth'% k)
        torch.save({'epoch': k, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_file)
        if metrics['AP'] > best_metric:
            best_metric = metrics['AP']
            update_final_model(model_file, os.path.join(model_dir, 'final_model.pth'))
        print('Best model has been saved to: %s' % model_file)
        scheduler.step(losses['total_loss'])

def test_eval():
    data_path = os.path.join(args.data_path, args.dataset)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.dataset == 'dad':
        test_data = DADDataset(data_path, args.feature_name, 'testing', toTensor=True, device=device, vis=True)
    elif args.dataset == 'crash':
        test_data = CrashDataset(data_path, args.feature_name, 'test', toTensor=True, device=device, vis=True)
    else:
        raise NotImplementedError
    testdata_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, drop_last=True)
    model = DSTA(test_data.dim_feature, args.hidden_dim, args.latent_dim, n_layers=args.num_rnn, n_obj=test_data.n_obj, n_frames=test_data.n_frames, fps=test_data.fps, with_saa=True)
    result_file = os.path.join("output/pred_res.npz")
    model, _, _ = load_checkpoint(model, filename=args.model_file, isTraining=False)
    all_pred, all_labels, all_toas, vis_data = test_all_vis(testdata_loader, model, vis=True)
    np.savez(result_file[:-4], pred=all_pred, label=all_labels, toas=all_toas)
    all_vid_scores = [max(pred[:int(toa)]) for toa, pred in zip(all_toas, all_pred)]
    AP_video = average_precision_score(all_labels, all_vid_scores)
    print("AP = %.4f" % AP_video)
    AP, mTTA, TTA_R80, P_R80 = evaluation_P_R80(all_pred, all_labels, all_toas, fps=test_data.fps)
    print("AP = %.4f" % AP)
    print("mTTA = %.4f" % mTTA)
    print("TTA_R80 = %.4f" % TTA_R80)
    print("P_R80 = %.4f" % P_R80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dad', choices=['dad', 'crash'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_rnn', type=int, default=1)
    parser.add_argument('--feature_name', type=str, default='vgg16')
    parser.add_argument('--test_iter', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--loss_beta', type=float, default=15)
    parser.add_argument('--phase', type=str, choices=['train', 'test'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model_file', type=str, default='output/DSTA/vgg16/dad/snapshot/final_model.pth')
    args = parser.parse_args()
    if args.phase == 'train':
        train_eval()
    else:
        test_eval()