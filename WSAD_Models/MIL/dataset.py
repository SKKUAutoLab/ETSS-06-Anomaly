from torch.utils.data import Dataset
import numpy as np
import os
import random

class Normal_Loader(Dataset):
    # is_train = 1 --> train, 0 --> test
    def __init__(self, is_train=1, path='dataset/UCF-Crime/', modality='TWO'):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines() # total training = 950
        else:
            data_list = os.path.join(path, 'test_normal.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10] # normal dataset have no last 10 videos, total testing = 950

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', self.data_list[idx][:-1] + '.npy')) # [32, 1024]
            flow_npy = np.load(os.path.join(self.path + 'all_flows', self.data_list[idx][:-1] + '.npy')) # [32, 1024]
            c3d_npy = np.load(os.path.join(self.path + 'all_c3d', self.data_list[idx][:-1] + '.npy')).astype(np.float32) # [32, 4096]
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)  # [32, 2048]
            if self.modality == 'RGB':
                return rgb_npy
            elif self.modality == 'FLOW':
                return flow_npy
            elif self.modality == 'C3D':
                return c3d_npy
            else:
                return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split(' ')[0], int(self.data_list[idx].split(' ')[1]), int(self.data_list[idx].split(' ')[2][:-1])
            rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', name + '.npy')) # [32, 1024]
            flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy')) # [32, 1024]
            c3d_npy = np.load(os.path.join(self.path + 'all_c3d', name + '.npy')).astype(np.float32) # [32, 4096]
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)  # [32, 2048]
            if self.modality == 'RGB':
                return rgb_npy, gts, frames
            elif self.modality == 'FLOW':
                return flow_npy, gts, frames
            elif self.modality == 'C3D':
                return c3d_npy, gts, frames
            else:
                return concat_npy, gts, frames

class Anomaly_Loader(Dataset):
    def __init__(self, is_train=1, path='dataset/UCF-Crime/', modality='TWO'):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.modality = modality
        self.path = path
        if self.is_train == 1:
            data_list = os.path.join(path, 'train_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()
        else:
            data_list = os.path.join(path, 'test_anomaly.txt')
            with open(data_list, 'r') as f:
                self.data_list = f.readlines()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.is_train == 1:
            rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', self.data_list[idx][:-1] + '.npy')) # [32, 1024]
            flow_npy = np.load(os.path.join(self.path + 'all_flows', self.data_list[idx][:-1] + '.npy')) # [32, 1024]
            c3d_npy = np.load(os.path.join(self.path + 'all_c3d', self.data_list[idx][:-1] + '.npy')).astype(np.float32) # [32, 4096]
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)  # [32, 2048]
            if self.modality == 'RGB':
                return rgb_npy
            elif self.modality == 'FLOW':
                return flow_npy
            elif self.modality == 'C3D':
                return c3d_npy
            else:
                return concat_npy
        else:
            name, frames, gts = self.data_list[idx].split('|')[0], int(self.data_list[idx].split('|')[1]), self.data_list[idx].split('|')[2][1:-2].split(',')
            gts = [int(i) for i in gts]
            rgb_npy = np.load(os.path.join(self.path + 'all_rgbs', name + '.npy')) # [32, 1024]
            flow_npy = np.load(os.path.join(self.path + 'all_flows', name + '.npy')) # [32, 1024]
            c3d_npy = np.load(os.path.join(self.path + 'all_c3d', name + '.npy')).astype(np.float32) # [32, 4096]
            concat_npy = np.concatenate([rgb_npy, flow_npy], axis=1)  # [32, 2048]
            if self.modality == 'RGB':
                return rgb_npy, gts, frames
            elif self.modality == 'FLOW':
                return flow_npy, gts, frames
            elif self.modality == 'C3D':
                return c3d_npy, gts, frames
            else:
                return concat_npy, gts, frames