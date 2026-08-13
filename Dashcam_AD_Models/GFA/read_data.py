from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import numpy as np

# DAD (Dashcam Accident Dataset): every video has exactly 100 frames and, for
# positive (accident) videos, the accident happens in the last 10 frames
# (i.e. starting at frame index 90).
N_FRAMES = 100
ACCIDENT_FRAME = 90


class Dashcam_data():
    def __init__(self, dir='./datasets/DAD/', batch_size=10, frame_size=[112, 112],
                 train=True, mean_file='mean_file.npy'):
        self.dir = dir
        self.batch_size = batch_size
        self.frame_size = frame_size
        self.train = train
        self.mean_file = mean_file
        self.im_names = []
        self.im_pointer = 0
        self.batch = []

        self.paths = []
        self.labels = []
        self.frame_labels = []
        self.features = []
        self.feature_pointer = 0

        self.annotation_path = os.path.join(self.dir, "annotation")

        if (self.train == True):
            self.cat_path_normal = os.path.join(self.dir, "videos", "training", "negative")
            self.cat_path_abnormal = os.path.join(self.dir, "videos", "training", "positive")
            self.cat_path_features = os.path.join(self.dir, "features", "training")

        elif (self.train == False):
            self.cat_path_normal = os.path.join(self.dir, "videos", "testing", "negative")
            self.cat_path_abnormal = os.path.join(self.dir, "videos", "testing", "positive")
            self.cat_path_features = os.path.join(self.dir, "features", "testing")

        for file in sorted(os.listdir(self.cat_path_normal)):

            path_normal = os.path.join(self.cat_path_normal, file)
            self.paths.append(path_normal)
            q = self.one_hot(0)
            self.labels.append(q)
            l = np.zeros((N_FRAMES))
            self.frame_labels.append(l)

        for file in sorted(os.listdir(self.cat_path_abnormal)):

            path_abnormal = os.path.join(self.cat_path_abnormal, file)
            self.paths.append(path_abnormal)
            l = np.zeros((N_FRAMES))
            l[ACCIDENT_FRAME:] = 1
            self.frame_labels.append(l)
            e = self.one_hot(1)
            self.labels.append(e)

        if os.path.isdir(self.cat_path_features):
            for file in sorted(os.listdir(self.cat_path_features)):
                path = os.path.join(self.cat_path_features, file)
                self.features.append(path)

        self.total_features = len(self.features)
        self.feature_ind = list(range(self.total_features))
        self.total_videos = len(self.paths)

        self.vid_ind = list(range(len(self.paths)))

    def one_hot(self, y_, n_classes=2):
        # Function to encode neural one-hot output labels from number indexes
        # e.g.:
        # one_hot(y_=[[5], [0], [3]], n_classes=6):
        #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
        y_ = [int(y_)]

        return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

    def read_annotation(self, video_path):
        """Read the DAD object annotation of a (positive) video.

        Each annotation line is:
        frame  track_id  class  x1  y1  x2  y2  accident_flag
        with pixel coordinates on the original frame.

        :return: per-frame list of boxes [y1, x1, y2, x2], or None when the
                 video has no annotation file (negative videos).
        """
        video_path = str(video_path)

        # annotations exist only for positive videos; negative videos reuse the
        # same file numbering, so match on the folder instead of the file name
        if "positive" not in video_path.split(os.sep):
            return None

        video_id = os.path.splitext(os.path.basename(video_path))[0]
        ann_file = os.path.join(self.annotation_path, video_id + ".txt")

        if not os.path.isfile(ann_file):
            return None

        boxes = [[] for _ in range(N_FRAMES)]
        with open(ann_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 8:
                    continue
                frame = int(parts[0]) - 1
                x1, y1, x2, y2 = int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6])
                if 0 <= frame < N_FRAMES:
                    boxes[frame].append([y1, x1, y2, x2])

        return boxes

    def read_video(self, video_path):
        """Read all frames of a video as an RGB uint8 array (n_frames, H, W, 3)."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        return np.array(frames)

    def get_next_batch(self):

        if (self.im_pointer == 0):
            np.random.shuffle(self.vid_ind)

        self.l = []
        self.batch = []
        self.fl = []

        for idx in range(self.batch_size):
            self.batch.append(self.paths[self.vid_ind[self.im_pointer]])
            self.l.append(self.labels[self.vid_ind[self.im_pointer]])
            self.fl.append(self.frame_labels[self.vid_ind[self.im_pointer]])

            self.im_pointer += 1
            if (self.im_pointer == len(self.paths)):
                self.im_pointer = 0
                np.random.shuffle(self.vid_ind)

        self.batch = np.array(self.batch)
        self.l = np.array(self.l)

        return self.batch, self.l, self.fl

    def get_next_batch_features(self):
        """Load the next pre-extracted feature batch file.

        Each feature file holds a full batch: data (batch_size, 100, 20, 4096),
        labels (batch_size, 2) and the video paths (ID).
        `start` is the accident frame for positive videos (0 for negative ones)
        and `end` marks the valid frames (always all 100 for DAD).
        """

        if (self.feature_pointer == 0):
            np.random.shuffle(self.feature_ind)

        f = self.features[self.feature_ind[self.feature_pointer]]
        batch = np.load(f, allow_pickle=True)

        data = batch['data']
        labels = batch['labels']
        paths = list(batch['ID'])

        start = np.zeros((self.batch_size))
        end = np.ones((self.batch_size, N_FRAMES))

        for idx in range(self.batch_size):
            # labels[idx][0] == 0 means the one-hot label is [0, 1] -> positive video
            if labels[idx][0] == 0:
                start[idx] = ACCIDENT_FRAME
            else:
                start[idx] = 0

        self.feature_pointer += 1
        if (self.feature_pointer == len(self.features)):
            self.feature_pointer = 0
            np.random.shuffle(self.feature_ind)

        return data, labels, paths, start, end


if __name__ == '__main__':

    dataset = Dashcam_data(train=False)

    im_names = (dataset.total_features)
    tot_batches = int(im_names)
    for i in range(tot_batches):
        print(i, "out of", tot_batches)
        batch, labels, paths, start, end = dataset.get_next_batch_features()
