import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class MySampler(torch.utils.data.Sampler):
    def __init__(self, end_idx, seq_length):
        indices = []
        for i in range(len(end_idx) - 1):
            start = end_idx[i]
            end = end_idx[i + 1] - seq_length
            indices.append(torch.arange(start, end))
        indices = torch.cat(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)

class MyDataset(Dataset):
    def __init__(self, image_paths, transform, length, device=("cuda" if torch.cuda.is_available() else "cpu")):
        self.image_paths = image_paths
        self.transform = transform
        self.length = length
        self.device = device
        self.n_frames = 50
        self.toa_dict = self.get_toa_all('ccd/Crash-1500.txt')

    def get_toa_all(self, annofile):
        toa_dict = {}
        annoData = self.read_anno_file(annofile)
        for anno in annoData:
            labels = np.array(anno['label'], dtype=np.int32)
            toa = np.where(labels==1)[0][0]
            toa = min(max(1, toa), self.n_frames - 1)
            toa_dict[anno['vid']] = toa
        return toa_dict

    def read_anno_file(self, anno_file):
        assert os.path.exists(anno_file), "Annotation file does not exist! %s" % anno_file
        result = []
        with open(anno_file, 'r') as f:
            for line in f.readlines():
                items = {}
                items['vid'] = line.strip().split(',[')[0]
                labels = line.strip().split(',[')[1].split('],')[0]
                items['label'] = [int(val) for val in labels.split(',')]
                assert sum(items['label']) > 0, 'invalid accident annotation!'
                others = line.strip().split(',[')[1].split('],')[1].split(',')
                items['startframe'], items['vid_ytb'], items['lighting'], items['weather'], items['ego_involve'] = others
                result.append(items)
        f.close()
        return result

    def __getitem__(self, index):
        start = index
        end = index + self.n_frames
        indices = list(range(start, end))
        images = []
        for i in indices:
            image_path = self.image_paths[i][0]
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        video_id = self.image_paths[i][0]
        vid = video_id.split('/')[-2]
        x = torch.stack(images).to(self.device)
        if self.image_paths[start][1] == 1:
            label = 0, 1
            try:
                toa = [self.toa_dict[vid]]
            except:
                toa = [self.n_frames + 1]
        else:
            label = 1, 0
            toa = [self.n_frames + 1]
        y = torch.tensor([label], dtype=torch.float32).to(self.device)
        toa = torch.Tensor(toa).to(self.device)
        return x, y, toa

    def __len__(self):
        return self.length