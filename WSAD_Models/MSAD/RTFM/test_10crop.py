import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
from tqdm import tqdm

def test(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)
        for i, inputs in tqdm(enumerate(dataloader)):
            input = inputs.to(device)
            if len(input.size()) == 4:
                input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))
        if args.dataset == 'shanghai':
            gt = np.load('list/gt-sh2.npy')
        if args.dataset == 'ped2':
            gt = np.load('list/gt-ped2.npy')
        if args.dataset == 'ucf':
            gt = np.load('list/gt-ucf.npy')
        if args.dataset == 'msad':
            gt = np.load('list/gt-MSAD-WS-new.npy')
        if args.dataset == 'cuhk':
            gt = np.load('list/gt-cuhk.npy')
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))
        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        return rec_auc