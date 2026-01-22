import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
torch.set_default_tensor_type('torch.FloatTensor')

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = './list/shanghai-i3d-test-10crop.list'
            else:
                self.rgb_list_file = './list/shanghai-i3d-train-10crop.list'
        if self.dataset == 'ped2':
            if test_mode:
                self.rgb_list_file = './list/ped2-i3d-test.list'
            else:
                self.rgb_list_file = './list/ped2-i3d-train.list'
        if self.dataset == 'ucf':
            if test_mode:
                self.rgb_list_file = 'list/ucf-i3d-test.list'
            else:
                self.rgb_list_file = 'list/ucf-i3d.list'
        if self.dataset == 'msad':
            if test_mode:
                self.rgb_list_file = 'list/msad-i3d-test.list'
            else:
                self.rgb_list_file = 'list/msad-i3d.list'
        if self.dataset == 'cuhk':
            if test_mode:
                self.rgb_list_file = 'list/cuhk-i3d-test.list'
            else:
                self.rgb_list_file = None
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                else:
                    self.list = self.list[:63]
            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                else:
                    self.list = self.list[:810]
            elif self.dataset == 'msad':
                if self.is_normal:
                    self.list = self.list[120:]
                else:
                    self.list = self.list[:120]
            elif self.dataset == 'ped2':
                if self.is_normal:
                    self.list = self.list[6:]
                else:
                    self.list = self.list[:6]

    def __getitem__(self, index):
        label = self.get_label()
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)
        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            features = features.transpose(1, 0, 2)
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)
            return divided_features, label

    def get_label(self):
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame