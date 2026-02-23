import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path, phase, toTensor=False, device=torch.device('cuda')):
        self.data_path = data_path
        self.phase = phase
        self.toTensor = toTensor
        self.device = device
        filepath = os.path.join(self.data_path, phase)
        self.files_list = self.get_filelist(filepath)

    def __len__(self):
        return len(self.files_list)

    def get_filelist(self, filepath):
        assert os.path.exists(filepath), "Directory does not exist: %s" % filepath
        file_list = []
        for filename in sorted(os.listdir(filepath)):
            file_list.append(filename)
        return file_list

    def __getitem__(self, index):
        data_file = os.path.join(self.data_path, self.phase, self.files_list[index])
        assert os.path.exists(data_file)
        try:
            data = np.load(data_file)
            features = data['feature'] # [100, 31, 2048]
            toa = [data['toa'] + 0] # [1]
            detection = data['detection'] # [100, 30, 6]
            flow = data['flow_feat'] # [100, 31, 2048]
        except:
            raise IOError('Load data error! File: %s' % data_file)
        if self.toTensor:
            features = torch.Tensor(features).to(self.device)
            detection = torch.Tensor(detection).to(self.device)
            toa = torch.Tensor(toa).to(self.device)
            flow = torch.Tensor(flow).to(self.device)
        return features, detection, toa, flow