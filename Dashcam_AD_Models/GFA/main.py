import torch
from RNN import *
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import glob
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

batch_size = 10
n_input = 4096
n_classes = 2
n_hidden = 512
learning_rate = 1e-3
lambda_loss_amount = 0.5
num_epochs = 50
display_iter = 10
save_path = './models/'
n_frames = 100
n_objects = 10

train_path = "./datasets/DAD/features/training/"
test_path = "./datasets/DAD/features/testing/"

train_num = len(glob.glob(os.path.join(train_path, "batch_*.npz")))
test_num = len(glob.glob(os.path.join(test_path, "batch_*.npz")))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('yes', 'true', 't', '1')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='accident_LSTM')
    parser.add_argument('--mode', dest='mode', help='train or test or viz', default='viz')
    parser.add_argument('--model', dest='model', default='./models/')
    parser.add_argument('--gpu', dest='gpu', default='0')
    parser.add_argument('--restore', type=str2bool, default=False)
    args = parser.parse_args()

    return args


def build_model():
    model = LSTM_RNN(n_input, n_frames, n_objects, batch_size, n_classes)
    model = model.to(device)
    return model


def find_checkpoint(model_path):
    """Return a checkpoint file from a path (file or directory with model_*.pth)."""
    if os.path.isfile(model_path):
        return model_path

    checkpoints = glob.glob(os.path.join(model_path, "model_*.pth"))
    if len(checkpoints) == 0:
        return None

    checkpoints.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]))
    return checkpoints[-1]


def train(args):
    print('..training..')
    print(device)

    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs('output', exist_ok=True)
    previous_runs = [s for s in os.listdir('output') if s.startswith('run_')]
    if len(previous_runs) == 0:
        run_number = 1
    else:
        run_number = max([int(s.split('run_')[1]) for s in previous_runs]) + 1

    logdir = 'run_%02d' % run_number
    writer = SummaryWriter(os.path.join('output', logdir))

    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)

    if (args.restore == True):
        checkpoint_path = find_checkpoint(args.model)
        assert checkpoint_path is not None, "no checkpoint found in " + args.model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        qq = checkpoint['epoch']
        print("model restored from", checkpoint_path, "at epoch", qq)

    else:
        qq = 0

    i = int(qq)
    for epoch in range(int(qq), num_epochs):
        tStart_epoch = time.time()

        model.train()

        # To keep track of training's performance
        epoch_loss = np.zeros((train_num, 1), dtype=float)

        for step in range(1, train_num + 1):

            file_name = '%03d' % (step)
            batch_data = np.load(train_path + 'batch_' + file_name + '.npz')

            batch = batch_data["data"]
            batch_xs = batch[:, :, 0:n_objects, :]
            batch_y = batch_data["labels"]

            if (epoch <= 10):
                lr = 0.0001

            elif (epoch > 10 and epoch < 20):

                lr = 0.0001

            elif (epoch >= 20):
                lr = 0.0001

            for group in optimizer.param_groups:
                group['lr'] = lr

            x = torch.from_numpy(batch_xs).float().to(device)
            y = torch.from_numpy(batch_y).float().to(device)

            loss, soft_pred, zt = model(x, y, keep=0.5)

            optimizer.zero_grad()
            (loss / n_frames).backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss[step - 1] = batch_loss / batch_size

            print("Batches done", step, " out of", int(train_num), "Epoch is ", epoch)

        epochloss = np.mean(epoch_loss)
        # print one epoch
        print("Epoch:", epoch + 1, " done. Loss:", epochloss)
        writer.add_scalar("train_loss", epochloss, i)
        i += 1
        tStop_epoch = time.time()
        print("Epoch Time Cost:", round(tStop_epoch - tStart_epoch, 2), "s")
        sys.stdout.flush()
        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch + 1,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()},
                       save_path + "model_" + str(epoch + 1) + ".pth")

        if (epoch + 1) % 10 == 0:
            print("Testing")
            test_all(model, train=False)
    print("Optimization Finished!")
    torch.save({'epoch': num_epochs,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()},
               save_path + "final_model.pth")


def test_all(model, train=True):

    model.eval()

    total_loss = 0.0
    acc = 0
    all_pred = []
    all_labels = []

    for num_batch in range(1, test_num + 1):
        print(acc)
        acc = acc + 1

        file_name = '%03d' % (num_batch)
        test_all_data = np.load(test_path + 'batch_' + file_name + '.npz')

        batch = test_all_data["data"]
        batch_xs = batch[:, :, 0:n_objects, :]
        test_labels = test_all_data["labels"]

        x = torch.from_numpy(batch_xs).float().to(device)
        y = torch.from_numpy(test_labels).float().to(device)

        with torch.no_grad():
            temp_loss, pred, zt = model(x, y, keep=0.0)

        pred = pred.cpu().numpy()

        if num_batch <= 1:
            all_pred = pred[:, 0:90]
            all_labels = np.reshape(test_labels[:, 1], [batch_size, 1])

        else:
            all_pred = np.vstack((all_pred, pred[:, 0:90]))
            all_labels = np.vstack((all_labels, np.reshape(test_labels[:, 1], [batch_size, 1])))

    evaluation(all_pred, all_labels)


def evaluation(all_pred, all_labels, total_time=90, vis=False, length=None):
    ### input: all_pred (N x total_time) , all_label (N,)
    ### where N = number of videos, fps = 20 , time of accident = total_time
    ### output: AP & Time to Accident
    print(len(all_pred))
    length = []
    for i in range(len(all_pred)):
        a = len(all_pred[i])
        length.append(a)

    flat_list = [item for sublist in all_pred for item in sublist]
    temp_shape = len(flat_list)

    Precision = np.zeros((temp_shape))
    Recall = np.zeros((temp_shape))
    Time = np.zeros((temp_shape))
    cnt = 0
    AP = 0.0

    flat_list = [item for sublist in all_pred for item in sublist]
    a = 0
    for Th in sorted(flat_list):
        print(a, "out of ", len(flat_list))
        a += 1
        if length is not None and Th == 0:
            continue
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0
        Fn = 0.0
        F_P = []
        Fp = 0.0

        for i in range(len(all_pred)):

            j = np.array(all_pred[i])
            tp = np.where(j * all_labels[i] >= Th)

            if (all_labels[i] == 1):
                Fn += float(len((np.where(j <= Th)[0] > 0)))

            if (all_labels[i] == 0):
                Fp = float(len((np.where(j >= Th)[0] > 0)))
                F_P.append(Fp)

            Tp += float(len(tp[0] > 0))

            if float(len(tp[0] > 0)) > 0:
                time += tp[0][0] / float(length[i])
                counter = counter + 1
            Tp_Fp += float(len(np.where(j >= Th)[0] > 0))
        if Tp_Fp == 0:
            Precision[cnt] = np.nan
        else:
            Precision[cnt] = Tp / Tp_Fp

        if np.sum(all_labels) == 0:
            Recall[cnt] = np.nan
        else:
            Recall[cnt] = Tp / (Tp + Fn)

        if counter == 0:
            Time[cnt] = np.nan
        else:

            Time[cnt] = (1 - time / counter) * 4.5
        cnt += 1

    np.save("Precision_Dashcam_LSTM.npy", Precision)
    np.save("Recall_Dashcam_LSTM.npy", Recall)
    np.save("Time.npy", Time)
    index = np.argsort(Recall)
    TT = Time[index]
    RR = Recall[index]

    plt.figure()
    plt.plot(RR, TT)
    plt.xlim(0, 1)
    plt.ylim(0, 4.5)
    plt.ylabel('TTA')
    plt.xlabel('Recall')
    name = "TTA-Recall.jpg"
    plt.savefig(name)

    print("Mean Precision is ", np.mean(Precision))
    print("Mean Recall is ", np.mean(Recall))
    print("Mean Time is ", np.mean(Time))

    np.save("Precision_Dashcam_LSTM.npy", Precision)
    np.save("Recall_Dashcam_LSTM.npy", Recall)

    a += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _, rep_index = np.unique(Recall, return_index=1)
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index) - 1):
        new_Time[i] = np.max(Time[rep_index[i]:rep_index[i + 1]])
        new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i + 1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[~np.isnan(new_Precision)]
    new_Recall = new_Recall[~np.isnan(new_Precision)]
    new_Precision = new_Precision[~np.isnan(new_Precision)]

    if new_Recall[0] != 0:
        AP += new_Precision[0] * (new_Recall[0] - 0)
    for i in range(1, len(new_Precision)):
        AP += (new_Precision[i - 1] + new_Precision[i]) * (new_Recall[i] - new_Recall[i - 1]) / 2

    print("Average Precision= " + "{:.4f}".format(AP) + " ,mean Time to accident= " + "{:.4}".format(
        np.mean(new_Time) * 4.5))


def test(model_path):
    # load model
    model = build_model()
    checkpoint_path = find_checkpoint(model_path)
    assert checkpoint_path is not None, "no checkpoint found in " + model_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    print("model restore!!!")
    print("Testing")
    test_all(model, train=False)


def viz(args):

    pathh = "./figs_all/"
    os.makedirs(pathh, exist_ok=True)

    model = build_model()
    checkpoint_path = find_checkpoint(args.model)
    assert checkpoint_path is not None, "no checkpoint found in " + args.model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    path = test_path
    # load data
    for num_batch in range(1, test_num):
        print(num_batch)
        file_name = '%03d' % num_batch
        all_data = np.load(path + 'batch_' + file_name + '.npz', allow_pickle=True)
        data = all_data['data']
        data = data[:, :, 0:10, :]
        labels = all_data['labels']
        ID = all_data['ID']

        # run result
        x = torch.from_numpy(data).float().to(device)
        y = torch.from_numpy(labels).float().to(device)

        with torch.no_grad():
            all_loss, pred, zt = model(x, y, keep=0.0)

        pred = pred.cpu().numpy()

        for i in range(len(ID)):
            time = 0.0

            plt.figure(figsize=(14, 5))
            prediction = pred[i, 0:90]
            j = np.array(prediction)
            tp = np.where(j * labels[i][1] >= 0.8)
            if float(len(tp[0] > 0)) > 0:
                time = (1 - (tp[0][0] / float(90))) * 4.5

            print(time)
            plt.plot(prediction, linewidth=3.0)
            yy = 0.8 * np.ones(90)
            plt.plot(yy, color='r', linestyle='--')
            plt.ylim(0, 1)
            plt.xlim(0, 90)
            plt.ylabel('Probability')
            plt.xlabel('Frame')
            plt.xticks(np.arange(10, 100, 10))
            vid = ID[i]
            if isinstance(vid, bytes):
                vid = vid.decode()
            video_name = os.path.splitext(os.path.basename(str(vid)))[0]
            name = pathh + video_name + "-" + str(labels[i][1]) + "-" + str(time) + ".jpg"
            plt.savefig(name)
            plt.close()


if __name__ == '__main__':

    args = parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args.model)
    elif args.mode == 'viz':
        viz(args)
