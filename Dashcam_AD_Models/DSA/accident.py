import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import os
import pdb
import time
import matplotlib.pyplot as plt
import sys

############### Global Parameters ###############
# path
train_path = './dataset/features/training/'
test_path = './dataset/features/testing/'
demo_path = './dataset/features/testing/'
default_model_path = './model/demo_model'
save_path = './model/'
video_path = './dataset/videos/testing/positive/'
# batch_number
train_num = 126
test_num = 46


############## Train Parameters #################

# Parameters
learning_rate = 0.0001
n_epochs = 30
batch_size = 10
display_step = 10

# Network Parameters
n_input = 4096 # fc6 or fc7(1*4096)
n_detection = 20 # number of object of each image (include image features)
n_hidden = 512 # hidden layer num of LSTM
n_img_hidden = 256 # embedding image features
n_att_hidden = 256 # embedding object features
n_classes = 2 # has accident or not
n_frames = 100 # number of frame in each video
##################################################

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='accident_LSTM')
    parser.add_argument('--mode',dest = 'mode',help='train or test',default = 'demo')
    parser.add_argument('--model',dest = 'model',default= default_model_path)
    parser.add_argument('--gpu',dest = 'gpu',default= '0')
    args = parser.parse_args()

    return args


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PeepholeLSTMCell(nn.Module):
    """Equivalent of tf.contrib.rnn.LSTMCell(n_hidden, use_peepholes=True,
    state_is_tuple=False) with random_normal(mean=0.0, stddev=0.01) initializer.
    State is the concatenation [c, h] (axis 1), gate order is i, j, f, o and
    forget_bias = 1.0, matching the TF implementation."""

    def __init__(self, input_size, hidden_size, forget_bias=1.0):
        super(PeepholeLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.kernel = nn.Parameter(torch.randn(input_size + hidden_size, 4 * hidden_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
        self.w_i_diag = nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.w_f_diag = nn.Parameter(torch.randn(hidden_size) * 0.01)
        self.w_o_diag = nn.Parameter(torch.randn(hidden_size) * 0.01)

    @property
    def state_size(self):
        return 2 * self.hidden_size

    def forward(self, inputs, state):
        c, h = torch.split(state, self.hidden_size, dim=1)
        lstm_matrix = torch.matmul(torch.cat([inputs, h], 1), self.kernel) + self.bias
        i, j, f, o = torch.split(lstm_matrix, self.hidden_size, dim=1)
        c_new = (torch.sigmoid(f + self.forget_bias + self.w_f_diag * c) * c +
                 torch.sigmoid(i + self.w_i_diag * c) * torch.tanh(j))
        h_new = torch.sigmoid(o + self.w_o_diag * c_new) * torch.tanh(c_new)
        return h_new, torch.cat([c_new, h_new], 1)


class AccidentLSTM(nn.Module):
    def __init__(self):
        super(AccidentLSTM, self).__init__()
        # Define weights
        self.weights = nn.ParameterDict({
            'em_obj': nn.Parameter(torch.randn(n_input, n_att_hidden) * 0.01),
            'em_img': nn.Parameter(torch.randn(n_input, n_img_hidden) * 0.01),
            'att_w': nn.Parameter(torch.randn(n_att_hidden, 1) * 0.01),
            'att_wa': nn.Parameter(torch.randn(n_hidden, n_att_hidden) * 0.01),
            'att_ua': nn.Parameter(torch.randn(n_att_hidden, n_att_hidden) * 0.01),
            'out': nn.Parameter(torch.randn(n_hidden, n_classes) * 0.01)
        })
        self.biases = nn.ParameterDict({
            'em_obj': nn.Parameter(torch.randn(n_att_hidden) * 0.01),
            'em_img': nn.Parameter(torch.randn(n_img_hidden) * 0.01),
            'att_ba': nn.Parameter(torch.zeros(n_att_hidden)),
            'out': nn.Parameter(torch.randn(n_classes) * 0.01)
        })
        # a lstm cell with peepholes (input = concat of image & attention embeddings)
        self.lstm_cell = PeepholeLSTMCell(n_img_hidden + n_att_hidden, n_hidden)

    def forward(self, x, y, keep):
        # x: (batch, n_frames, n_detection, n_input), y: (batch, n_classes)
        # keep: dropout probability applied to the LSTM output
        # (feed 0.5 for training, 0.0 for testing, same as the keep placeholder)
        device = x.device
        batch = x.shape[0]
        # init LSTM parameters
        istate = torch.zeros(batch, self.lstm_cell.state_size, device=device)
        h_prev = torch.zeros(batch, n_hidden, device=device)
        # init loss
        loss = 0.0
        # Mask
        zeros_object = (x[:, :, 1:n_detection, :].permute(1, 2, 0, 3).sum(3) != 0).float() # frame x n x b

        soft_pred = None
        all_alphas = None
        for i in range(n_frames):
            # input features (Faster-RCNN fc7)
            X = x[:, i, :, :].permute(1, 0, 2)  # permute n_steps and batch_size (n x b x h)
            # frame embedded
            image = torch.matmul(X[0, :, :], self.weights['em_img']) + self.biases['em_img'] # 1 x b x h
            # object embedded
            n_object = X[1:n_detection, :, :].reshape(-1, n_input) # (n_steps*batch_size, n_input)
            n_object = torch.matmul(n_object, self.weights['em_obj']) + self.biases['em_obj'] # (n x b) x h
            n_object = n_object.reshape(n_detection - 1, batch, n_att_hidden) # n-1 x b x h
            n_object = n_object * zeros_object[i].unsqueeze(2)

            # object attention
            image_part = torch.matmul(n_object, self.weights['att_ua']) + self.biases['att_ba'] # n x b x h
            e = torch.tanh(torch.matmul(h_prev, self.weights['att_wa']) + image_part) # n x b x h
            # the probability of each object
            alphas = F.softmax(torch.matmul(e, self.weights['att_w']).sum(2), dim=0) * zeros_object[i]
            # weighting sum
            attention_list = alphas.unsqueeze(2) * n_object
            attention = attention_list.sum(0) # b x h
            # concat frame & object
            fusion = torch.cat([image, attention], 1)

            outputs, istate = self.lstm_cell(fusion, istate)
            # dropout on the output of LSTM (state is kept intact, as in DropoutWrapper)
            outputs = F.dropout(outputs, p=keep, training=True)
            # save prev hidden state of LSTM
            h_prev = outputs
            # FC to output
            pred = torch.matmul(outputs, self.weights['out']) + self.biases['out'] # b x n_classes
            # save the predict of each time step
            if i == 0:
                soft_pred = F.softmax(pred, dim=1)[:, 1].reshape(batch, 1)
                all_alphas = alphas.unsqueeze(0)
            else:
                temp_soft_pred = F.softmax(pred, dim=1)[:, 1].reshape(batch, 1)
                soft_pred = torch.cat([soft_pred, temp_soft_pred], 1)
                temp_alphas = alphas.unsqueeze(0)
                all_alphas = torch.cat([all_alphas, temp_alphas], 0)

            # softmax cross entropy with logits (labels may be soft)
            cross_entropy = -(y * F.log_softmax(pred, dim=1)).sum(1)
            # positive example (exp_loss)
            pos_loss = np.exp(-(n_frames - i - 1) / 20.0) * cross_entropy
            # negative example
            neg_loss = cross_entropy # Softmax loss

            temp_loss = (pos_loss * y[:, 1] + neg_loss * y[:, 0]).mean()
            loss = loss + temp_loss

        return loss, soft_pred, all_alphas


def train():
    device = get_device()
    # build model
    model = AccidentLSTM().to(device)
    # Define loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam Optimizer
    # mkdir folder for saving model
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path)
    # Keep training until reach max iterations
    # start training
    for epoch in range(n_epochs):
         # random chose batch.npz
         epoch_loss = np.zeros((train_num,1),dtype = float)
         n_batchs = np.arange(1,train_num+1)
         np.random.shuffle(n_batchs)
         tStart_epoch = time.time()
         model.train()
         for batch in n_batchs:
             file_name = '%03d' %batch
             batch_data = np.load(train_path+'batch_'+file_name+'.npz')
             batch_xs = torch.from_numpy(batch_data['data']).float().to(device)
             batch_ys = torch.from_numpy(batch_data['labels']).float().to(device)
             optimizer.zero_grad()
             batch_loss, _, _ = model(batch_xs, batch_ys, keep=0.5)
             (batch_loss / n_frames).backward()
             optimizer.step()
             epoch_loss[batch-1] = batch_loss.item()/batch_size
         # print one epoch
         print("Epoch:", epoch+1, " done. Loss:", np.mean(epoch_loss))
         tStop_epoch = time.time()
         print("Epoch Time Cost:", round(tStop_epoch - tStart_epoch,2), "s")
         sys.stdout.flush()
         if (epoch+1) %5 == 0:
            torch.save(model.state_dict(), save_path+"model-"+str(epoch+1)+".pth")
            print("Training")
            test_all(model,train_num,train_path,device)
            print("Testing")
            test_all(model,test_num,test_path,device)
    print("Optimization Finished!")
    torch.save(model.state_dict(), save_path+"final_model.pth")

def test_all(model,num,path,device):
    total_loss = 0.0

    for num_batch in range(1,num+1):
         # load test_data
         file_name = '%03d' %num_batch
         test_all_data = np.load(path+'batch_'+file_name+'.npz')
         test_data = torch.from_numpy(test_all_data['data']).float().to(device)
         test_labels = test_all_data['labels']
         labels = torch.from_numpy(test_labels).float().to(device)
         with torch.no_grad():
             temp_loss, pred, _ = model(test_data, labels, keep=0.0)
         pred = pred.cpu().numpy()

         total_loss += temp_loss.item()/batch_size

         if num_batch <= 1:
             all_pred = pred[:,0:90]
             all_labels = np.reshape(test_labels[:,1],[batch_size,1])
         else:
             all_pred = np.vstack((all_pred,pred[:,0:90]))
             all_labels = np.vstack((all_labels,np.reshape(test_labels[:,1],[batch_size,1])))

    evaluation(all_pred,all_labels)


def evaluation(all_pred,all_labels, total_time = 90, vis = False, length = None):
    ### input: all_pred (N x total_time) , all_label (N,)
    ### where N = number of videos, fps = 20 , time of accident = total_time
    ### output: AP & Time to Accident

    if length is not None:
        all_pred_tmp = np.zeros(all_pred.shape)
        for idx, vid in enumerate(length):
                all_pred_tmp[idx,total_time-vid:] = all_pred[idx,total_time-vid:]
        all_pred = np.array(all_pred_tmp)
        temp_shape = sum(length)
    else:
        length = [total_time] * all_pred.shape[0]
        temp_shape = all_pred.shape[0]*total_time
    Precision = np.zeros((temp_shape))
    Recall = np.zeros((temp_shape))
    Time = np.zeros((temp_shape))
    cnt = 0
    AP = 0.0
    for Th in sorted(all_pred.flatten()):
        if length is not None and Th == 0:
                continue
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0
        for i in range(len(all_pred)):
            tp =  np.where(all_pred[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                time += tp[0][0] / float(length[i])
                counter = counter+1
            Tp_Fp += float(len(np.where(all_pred[i]>=Th)[0])>0)
        if Tp_Fp == 0:
            Precision[cnt] = np.nan
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0:
            Recall[cnt] = np.nan
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            Time[cnt] = np.nan
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1

    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    _,rep_index = np.unique(Recall,return_index=1)
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])

    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    new_Time = new_Time[~np.isnan(new_Precision)]
    new_Recall = new_Recall[~np.isnan(new_Precision)]
    new_Precision = new_Precision[~np.isnan(new_Precision)]

    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    print("Average Precision= " + "{:.4f}".format(AP) + " ,mean Time to accident= " +"{:.4}".format(np.mean(new_Time) * 5))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    print("Recall@80%, Time to accident= " +"{:.4}".format(sort_time[np.argmin(np.abs(sort_recall-0.8))] * 5))

    ### visualize

    if vis:
        plt.plot(new_Recall, new_Precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall example: AUC={0:0.2f}'.format(AP))
        plt.show()
        plt.clf()
        plt.plot(new_Recall, new_Time, label='Recall-mean_time curve')
        plt.xlabel('Recall')
        plt.ylabel('time')
        plt.ylim([0.0, 5])
        plt.xlim([0.0, 1.0])
        plt.title('Recall-mean_time' )
        plt.show()


def vis(model_path):
    device = get_device()
    # build model
    model = AccidentLSTM().to(device)
    # restore model
    model.load_state_dict(torch.load(model_path, map_location=device))
    # load data
    for num_batch in range(1,test_num):
        file_name = '%03d' %num_batch
        all_data = np.load(demo_path+'batch_'+file_name+'.npz')
        data = all_data['data']
        labels = all_data['labels']
        det = all_data['det']
        ID = all_data['ID']
        # run result
        with torch.no_grad():
            all_loss, pred, weight = model(torch.from_numpy(data).float().to(device),
                                           torch.from_numpy(labels).float().to(device), keep=0.0)
        pred = pred.cpu().numpy()
        weight = weight.cpu().numpy()
        file_list = sorted(os.listdir(video_path))
        for i in range(len(ID)):
            if labels[i][1] == 1 :
                plt.figure(figsize=(14,5))
                plt.plot(pred[i,0:90],linewidth=3.0)
                plt.ylim(0, 1)
                plt.ylabel('Probability')
                plt.xlabel('Frame')
                plt.show()
                file_name = ID[i]
                if isinstance(file_name, bytes):
                    file_name = file_name.decode()
                bboxes = det[i]
                new_weight = weight[:,:,i]*255
                counter = 0
                cap = cv2.VideoCapture(video_path+file_name+'.mp4')
                ret, frame = cap.read()
                font = cv2.FONT_HERSHEY_SIMPLEX
                while(ret):
                    attention_frame = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.uint8)
                    now_weight = new_weight[counter,:]
                    new_bboxes = bboxes[counter,:,:]
                    index = np.argsort(now_weight)
                    for num_box in index:
                        if now_weight[num_box]/255.0>0.4:
                            cv2.rectangle(frame,(int(new_bboxes[num_box,0]),int(new_bboxes[num_box,1])),(int(new_bboxes[num_box,2]),int(new_bboxes[num_box,3])),(0,255,0),3)
                        else:
                            cv2.rectangle(frame,(int(new_bboxes[num_box,0]),int(new_bboxes[num_box,1])),(int(new_bboxes[num_box,2]),int(new_bboxes[num_box,3])),(255,0,0),2)
                        cv2.putText(frame,str(round(now_weight[num_box]/255.0*10000)/10000),(int(new_bboxes[num_box,0]),int(new_bboxes[num_box,1])), font, 0.5,(0,0,255),1,cv2.LINE_AA)
                        attention_frame[int(new_bboxes[num_box,1]):int(new_bboxes[num_box,3]),int(new_bboxes[num_box,0]):int(new_bboxes[num_box,2])] = now_weight[num_box]

                    attention_frame = cv2.applyColorMap(attention_frame, cv2.COLORMAP_HOT)
                    dst = cv2.addWeighted(frame,0.6,attention_frame,0.4,0)
                    cv2.putText(dst,str(counter+1),(10,30), font, 1,(255,255,255),3)
                    cv2.imshow('result',dst)
                    c = cv2.waitKey(50)
                    ret, frame = cap.read()
                    if c == ord('q') and c == 27 and ret:
                        break;
                    counter += 1

            cv2.destroyAllWindows()



def test(model_path):
    device = get_device()
    # load model
    model = AccidentLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("model restore!!!")
    print("Training")
    test_all(model,train_num,train_path,device)
    print("Testing")
    test_all(model,test_num,test_path,device)



if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if args.mode == 'train':
           train()
    elif args.mode == 'test':
           test(args.model)
    elif args.mode == 'demo':
           vis(args.model)
