import math
import os
import os.path
from pathlib import Path
from typing import Any, List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src import utils

log = utils.get_pylogger(__name__)

def round_to_nearest(number: float, X: int) -> int:
    return math.ceil(number / X) * X

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), length + 1, dtype=np.int)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i] : r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat

class VideoRecord:
    def __init__(self, row, root_datapath, spatialannotationdir_path=None):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])
        if spatialannotationdir_path:
            filename = row[0].split("/")[1].replace("_x264", "")
            self._spatialannotationdir_path = Path(spatialannotationdir_path, filename).with_suffix(".txt")
            self._spatialannotationdir_path = self._spatialannotationdir_path if self._spatialannotationdir_path.is_file() else None
        else:
            self._spatialannotationdir_path = None

    @property
    def path(self) -> str:
        return self._path + ".npy"

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1 # +1 because end frame is inclusive

    @property
    def start_frame(self) -> int:
        return int(self._data[1])

    @property
    def end_frame(self) -> int:
        return int(self._data[2])

    @property
    def label(self) -> Union[int, List[int]]:
        if len(self._data) == 4:
            return int(self._data[3])
        else:
            return [int(label_id) for label_id in self._data[3:]]

    @property
    def tbox(self) -> List[Tuple[int]]:
        if self._spatialannotationdir_path:
            anno_df = pd.read_csv(self._spatialannotationdir_path, delim_whitespace=True, header=None)
            anno_df.columns = ["Track", "xmin", "ymin", "xmax", "ymax", "frame", "lost", "occluded", "generated", "label"]
            anno_df = anno_df.loc[(anno_df["frame"] >= self.start_frame) & (anno_df["frame"] <= self.end_frame)]
            anomaly_frames = (1 - anno_df["lost"].values).tolist()
        else:
            anomaly_frames = [0] * self.num_frames
        return anomaly_frames

class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: str, annotationfile_path: str, normal_id: int, num_segments: int = 32, frames_per_segment: int = 16, imagefile_template: str = "{:06d}.jpg",
                 transform=None, test_mode: bool = False, val_mode: bool = False, ncrops: int = 1, temporal_annotation_file: str = None, labels_file: str = None, stride: int = 1, spatialannotationdir_path: str = None):
        super().__init__()
        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.normal_id = normal_id
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.test_mode = test_mode
        self.val_mode = val_mode
        self.ncrops = ncrops
        self.temporal_annotation_file = temporal_annotation_file
        self.labels_file = labels_file
        self.stride = stride
        self.spatialannotationdir_path = spatialannotationdir_path
        self._parse_labelsfile()
        self._parse_annotationfile()
        if self.test_mode or self.val_mode:
            self.annotations = self._temporal_testing_annotations()

    def _load_image(self, directory: str, idx: int) -> Image.Image:
        return Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert("RGB")

    def _parse_labelsfile(self):
        self.labels = pd.read_csv(self.labels_file) if self.labels_file else None

    def _parse_annotationfile(self):
        self.video_list = [VideoRecord(x.strip().split(), self.root_path, self.spatialannotationdir_path) for x in open(self.annotationfile_path)]

    def _temporal_testing_annotations(self):
        annotations = {}
        if self.temporal_annotation_file:
            with open(self.temporal_annotation_file) as annotations_f:
                lines = annotations_f.readlines()
                annotations = {str(Path(line.strip().split()[0]).stem): line.strip().split()[2:] for line in lines}
        return annotations

    def _get_start_indices(self, record: VideoRecord) -> "np.ndarray[int]":
        if self.test_mode:
            end_frame = round_to_nearest(record.num_frames, self.num_segments * self.frames_per_segment * self.stride)
            start_indices = np.arange(end_frame / (self.frames_per_segment * self.stride)) * (self.frames_per_segment * self.stride)
        else:
            lower_bound = self.num_segments * self.frames_per_segment * self.stride
            if record.num_frames >= lower_bound:
                distance_between_indices = (record.num_frames - self.frames_per_segment + 1) // self.num_segments
            else:
                distance_between_indices = (lower_bound - self.frames_per_segment + 1) // self.num_segments
            start_indices = np.multiply(list(range(self.num_segments)), distance_between_indices) + np.random.randint((distance_between_indices + 1) - self.frames_per_segment + 1, size=self.num_segments)
        return start_indices

    def __getitem__(self, idx: int) -> Union[Tuple[List[Image.Image], Union[int, List[int]]], Tuple["torch.Tensor[num_frames, channels, height, width]", Union[int, List[int]]], Tuple[Any, Union[int, List[int]]]]:
        record: VideoRecord = self.video_list[idx]
        frame_start_indices: "np.ndarray[int]" = self._get_start_indices(record)
        return self._get(record, frame_start_indices)

    def _get(self, record: VideoRecord, frame_start_indices: "np.ndarray[int]") -> Union[Tuple[List[Image.Image], Union[int, List[int]]],
             Tuple["torch.Tensor[num_frames, channels, height, width]", Union[int, List[int]]], Tuple[Any, Union[int, List[int]]]]:
        im_feature = np.load(record.path, allow_pickle=True)
        im_feature = torch.tensor(im_feature)
        if self.test_mode or self.val_mode:
            labels = list()
            video_name = Path(record.path).stem
            if self.annotations:
                start_indices = self.annotations[video_name][::2]
                stop_indices = self.annotations[video_name][1::2]
            else:
                start_indices = []
                stop_indices = []
            for i in range(im_feature.shape[0] // self.ncrops):
                label = self.normal_id
                for start_idx, end_idx in zip(start_indices, stop_indices):
                    if int(start_idx) <= i + record.start_frame <= int(end_idx):
                        label = record.label
                labels.append(label)
        im_feature = im_feature.view(-1, self.ncrops, im_feature.shape[-1])
        im_feature = torch.transpose(im_feature, 0, 1)
        im_feature = torch.permute(im_feature, (1, 0, 2))
        if self.transform is not None:
            im_feature = self.transform(im_feature)
        features = list()
        val_labels = list()
        for start_index in frame_start_indices:
            for i in range(self.frames_per_segment):
                frame_index = (int(start_index) + i * self.stride) % im_feature.shape[0]
                feature = im_feature[frame_index]
                features.append(feature)
                if self.val_mode:
                    val_labels.append(labels[frame_index])
        features = torch.cat(features)
        features = features.view(-1, self.ncrops, features.shape[-1])
        features = torch.permute(features, (1, 0, 2))
        if self.test_mode:
            segment_size = len(frame_start_indices) // self.num_segments
            return features, np.asarray(labels), record.label, segment_size, record.path
        elif self.val_mode:
            return features, record.label, np.asarray(val_labels)
        else:
            return features, record.label

    def __len__(self):
        return len(self.video_list)