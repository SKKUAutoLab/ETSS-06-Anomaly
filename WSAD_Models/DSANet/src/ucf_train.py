import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random

from model import DSANet
from ucf_test import test
from utils.dataset import UCFDataset
from utils.tools import get_prompt_text, get_batch_label
from utils.StableAdamW import StableAdamW
import ucf_option
import os
import sys

def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLASM_EVENT(logits, labels, lengths, device, epsilon=0.1):
    num_classes = logits.shape[2]
    instance_logits = torch.zeros(0).to(device)

    labels_sum = labels.sum(dim=1, keepdim=True).clamp(min=1e-6)
    labels_sm = (1 - epsilon) * (labels / labels_sum) + epsilon / num_classes
    labels_sm = labels_sm.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(1),#int(lengths[i] / 16 + 1), 
                            largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels_sm * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLASM_BKG(logits, labels, lengths, device, epsilon=0.1):
    num_classes = logits.shape[2]
    instance_logits = torch.zeros(0).to(device)

    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)

    labels2 = torch.full(labels.shape, 0.01, device=labels.device)
    labels2[:, 0] = 1
    labels2_sum = labels2.sum(dim=1, keepdim=True).clamp(min=1e-6)
    labels2 = (1 - epsilon) * (labels2 / labels2_sum) + epsilon / num_classes
    labels2 = labels2.to(device)

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(1),#int(lengths[i] / 16 + 1), 
                            largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)

    milloss = -torch.mean(torch.sum(labels2 * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss

def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])

    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)

    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss

from torch.optim.lr_scheduler import _LRScheduler
class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]

class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, logits1, original_features, reconstructed_features, lengths):
        recon_error_score = 1.0 - F.cosine_similarity(
            original_features, 
            reconstructed_features, 
            dim=-1
        )
        recon_error_score = recon_error_score / 2.0
        classifier_prob_score = torch.sigmoid(logits1.squeeze(-1))

        B, N = logits1.shape[0], logits1.shape[1]
        mask = torch.arange(N, device=logits1.device)[None, :] < lengths[:, None]

        valid_recon_scores = recon_error_score[mask]
        valid_classifier_scores = classifier_prob_score[mask]
        
        consistency_loss = self.mse_loss(valid_classifier_scores, valid_recon_scores)

        return consistency_loss

consistency_loss_fn = ConsistencyLoss()

def train(model, normal_loader, anomaly_loader, testloader, args, label_map, device):
    model.to(device)
    gt = np.load(args.gt_path)
    gtsegments = np.load(args.gt_segment_path, allow_pickle=True)
    gtlabels = np.load(args.gt_label_path, allow_pickle=True)

    refiner_params = []
    main_model_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'video_anomaly_refiner' in name:
            refiner_params.append(param)
        else:
            main_model_params.append(param)
    optimizer_refiner = StableAdamW(
        [{'params': refiner_params}],
        lr=args.lr, 
        betas=(0.9, 0.999), 
        weight_decay=1e-4, 
        amsgrad=True, 
        eps=1e-10
    )
    total_epochs = args.max_epoch
    num_batches_per_epoch = len(normal_loader) + len(anomaly_loader)
    total_iters_refiner = total_epochs * num_batches_per_epoch
    scheduler_refiner = WarmCosineScheduler(
        optimizer_refiner, 
        base_value=args.lr, 
        final_value=args.lr * 0.1, 
        total_iters=total_iters_refiner,
        warmup_iters=100
    )
    optimizer_main = torch.optim.AdamW(
        [{'params': main_model_params}], 
        lr=args.lr
    )
    scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_main, 
        T_max=args.max_epoch
    )

    prompt_text = get_prompt_text(label_map)
    ap_best = 0
    epoch = 0

    if args.use_checkpoint == True:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_main.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        ap_best = checkpoint['ap']
        print("checkpoint info:")
        print("epoch:", epoch+1, " ap:", ap_best)

    for e in range(args.max_epoch):
        DNP_use = args.DNP_use
        model.train()
        loss_total1 = 0
        loss_total2 = 0
        loss_total4 = 0
        loss_total5 = 0
        normal_iter = iter(normal_loader)
        anomaly_iter = iter(anomaly_loader)
        for i in range(min(len(normal_loader), len(anomaly_loader))):
            step = 0
            normal_features, normal_label, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_label, anomaly_lengths = next(anomaly_iter)

            visual_features = torch.cat([normal_features, anomaly_features], dim=0).to(device)
            text_labels = list(normal_label) + list(anomaly_label)
            feat_lengths = torch.cat([normal_lengths, anomaly_lengths], dim=0).to(device)
            text_labels = get_batch_label(text_labels, prompt_text, label_map).to(device)

            if DNP_use == True:
                text_features, logits1, logits2, logits3, logits4, DNP = model(visual_features, None, prompt_text, feat_lengths, DNP_use)
            else:
                text_features, logits1, logits2, logits3, logits4 = model(visual_features, None, prompt_text, feat_lengths, DNP_use)
            #loss1
            loss1 = CLAS2(logits1, text_labels, feat_lengths, device) 
            loss_total1 += loss1.item()
            #loss2
            loss2 = CLASM(logits2, text_labels, feat_lengths, device)
            loss_total2 += loss2.item()
 
            if DNP_use == True:
                consistency_loss = consistency_loss_fn(
                    logits1=logits1,
                    original_features=DNP['original_features'],
                    reconstructed_features=DNP['reconstructed_features'],
                    lengths=feat_lengths
                )
                g_loss = DNP['g_loss']
            
            #loss4
            loss4 = CLASM_EVENT(logits3, text_labels, feat_lengths, device)
            loss_total4 += loss4.item()
            #loss5
            loss5 = CLASM_BKG(logits4, text_labels, feat_lengths, device)
            loss_total5 += loss5.item()
            #loss3
            loss3 = torch.zeros(1).to(device)
            text_feature_normal = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
            for j in range(1, text_features.shape[0]):
                text_feature_abr = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                loss3 += torch.abs(text_feature_normal @ text_feature_abr)
            loss3 = loss3 / 13
            if DNP_use == True:
                loss = loss1 + loss2 * args.loss2_weight + loss3 + loss4 + loss5 + consistency_loss + g_loss
            else:
                loss = loss1 + loss2 + loss3 + loss4 + loss5

            optimizer_main.zero_grad()
            optimizer_refiner.zero_grad()
            loss.backward()
            optimizer_main.step()
            optimizer_refiner.step()
            scheduler_refiner.step()
            step += i * normal_loader.batch_size * 2
            if step % 1280 == 0 and step != 0:
                log_items = [
                    f"epoch: {e+1}",
                    f"step: {step}",
                    f"loss1: {loss_total1 / (i+1):.4f}",
                    f"loss2: {loss_total2 / (i+1):.4f}",
                    f"loss3: {loss3.item():.4f}",
                    f"loss4: {loss_total4 / (i+1):.4f}",
                    f"loss5: {loss_total5 / (i+1):.4f}",
                ]
                if DNP_use:
                    log_items.append(f"consistency_loss: {consistency_loss.item():.4f}")
                    log_items.append(f"g_loss: {g_loss.item():.4f}")
                
                print(" | ".join(log_items), flush=True)
                sys.stdout.flush()
                AUC, AP = test(model, testloader, args.visual_length, prompt_text, gt, gtsegments, gtlabels, DNP_use, device, args)
                AP = AUC

                model.train()

                if AP > ap_best:
                    ap_best = AP 
                    checkpoint = {
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer_main.state_dict(),
                        'ap': ap_best}
                    os.makedirs('model', exist_ok=True)
                    torch.save(checkpoint, args.checkpoint_path)
                
        scheduler_main.step()
        
        torch.save(model.state_dict(), 'model/model_cur.pth')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    checkpoint = torch.load(args.checkpoint_path)
    torch.save(checkpoint['model_state_dict'], args.model_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = ucf_option.parser.parse_args()
    setup_seed(args.seed)

    label_map = dict({'Normal': 'normal', 'Abuse': 'abuse', 'Arrest': 'arrest', 'Arson': 'arson', 'Assault': 'assault', 'Burglary': 'burglary', 'Explosion': 'explosion', 'Fighting': 'fighting', 'RoadAccidents': 'roadAccidents', 'Robbery': 'robbery', 'Shooting': 'shooting', 'Shoplifting': 'shoplifting', 'Stealing': 'stealing', 'Vandalism': 'vandalism'})

    normal_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, True)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    anomaly_dataset = UCFDataset(args.visual_length, args.train_list, False, label_map, False)
    anomaly_loader = DataLoader(anomaly_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = UCFDataset(args.visual_length, args.test_list, True, label_map)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = DSANet(args.classes_num, args.embed_dim, args.visual_length, args.visual_width, args.visual_head, args.visual_layers, args.attn_window, args.prompt_prefix, args.prompt_postfix, args, device)
    train(model, normal_loader, anomaly_loader, test_loader, args, label_map, device)
