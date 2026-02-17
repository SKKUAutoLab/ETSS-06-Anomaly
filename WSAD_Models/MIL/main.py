# https://github.com/seominseok0429/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch
from torch.utils.data import DataLoader
from loss import *
from dataset import *
import os
from sklearn import metrics
import argparse
from net import Net
import warnings
warnings.filterwarnings("ignore") # disable warnings

parser = argparse.ArgumentParser(description='MIL implementation')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--w', default=0.0010000000474974513, type=float, help='weight_decay')
parser.add_argument('--modality', default='TWO', type=str, choices=['RGB', 'FLOW', 'C3D', 'TWO'], help='modality') # RGB, Flow or RGB + Flow
parser.add_argument('--input_dim', default=2048, type=int, help='input_dim')
parser.add_argument('--num_epochs', default=75, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=30, type=int)
args = parser.parse_args()

if args.modality == 'RGB':
    print('Training MIL with RGB features...')
elif args.modality == 'FLOW':
    print('Training MIL with Optical Flow features')
elif args.modality == 'C3D':
    print('Training MIL with C3D features...')
else:
    print('Training MIL with RGB + Flow features...')

best_auc = 0
normal_train_dataset = Normal_Loader(is_train=1, modality=args.modality)
normal_test_dataset = Normal_Loader(is_train=0, modality=args.modality)
anomaly_train_dataset = Anomaly_Loader(is_train=1, modality=args.modality)
anomaly_test_dataset = Anomaly_Loader(is_train=0, modality=args.modality)

normal_train_loader = DataLoader(normal_train_dataset, batch_size=args.batch_size, shuffle=True)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=False)
anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=args.batch_size, shuffle=True)
anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=False)

model = Net(input_dim=args.input_dim).cuda()
optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.w)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
criterion = MIL

def train(epoch):
    print('Epoch: %d' % epoch)
    model.train()
    train_loss = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1) # [30, 64, 2048]
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).cuda()
        outputs = model(inputs) # [1920, 1]
        loss = criterion(outputs, batch_size, 0, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('Loss = {}'.format(train_loss/len(normal_train_loader)))
    scheduler.step()

def test_abnormal():
    model.eval()
    global best_auc
    auc = 0
    cnt = 0
    with torch.no_grad():
        for i, (anomaly_test, normal_test) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            # positive bag
            inputs, gts, frames = anomaly_test
            inputs = inputs.view(-1, inputs.size(-1)).cuda()
            score = model(inputs).cpu().detach().numpy() # [32, 1]
            score_list = np.zeros(frames[0]) # frame -> type tensor
            step = np.round(np.linspace(0, frames[0]//16, 33)) # range [0, frames[0]//16] with len arr = 33
            for j in range(32):
                score_list[int(step[j])*16:(int(step[j + 1]))*16] = score[j]
            gt_list = np.zeros(frames[0])
            for k in range(len(gts)//2):
                s = gts[k*2] # start anomaly
                e = min(gts[k*2+1], frames) # end anomaly
                gt_list[s-1:e] = 1 # mark anomaly idx as 1
            # negative bag
            inputs2, gts2, frames2 = normal_test
            inputs2 = inputs2.view(-1, inputs2.size(-1)).cuda()
            score2 = model(inputs2).cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, frames2[0]//16, 33))
            for kk in range(32):
                score_list2[int(step2[kk])*16:(int(step2[kk + 1]))*16] = score2[kk]
            gt_list2 = np.zeros(frames2[0]) # all idx are 0
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)
            fpr, tpr, thresholds = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)
            cnt += 1
        print('AUC = {}'.format(auc/cnt)) # cnt: 140 for UCF-Crime -> 140 is total testing
        if best_auc < (auc/cnt):
            print('Saving best model')
            state = {'net': model.state_dict()}
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, 'checkpoint/ckpt.pth')
            best_auc = auc/cnt
    print('Best AUC:', best_auc)

if __name__ == "__main__":
    for epoch in range(0, args.num_epochs):
        train(epoch)
        test_abnormal()