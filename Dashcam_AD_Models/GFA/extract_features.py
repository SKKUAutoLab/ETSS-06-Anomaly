from read_data import *

import os

import cv2
import numpy as np
import torch
import torchvision

import vgg16

n_classes = 2
batch_size = 10
n_frames = 100
n_boxes = 20  # slot 0: full frame, slots 1..19: object crops
n_objects = n_boxes - 1

dataset_path = "./datasets/DAD/"
features_path = "./datasets/DAD/features/"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Object classes kept by the original code (COCO ids):
# 1: person, 2: bicycle, 3: car, 4: motorcycle, 6: bus, 8: truck
VALID_CLASSES = (1, 2, 3, 4, 6, 8)
DET_SCORE_THRESH = 0.5
DET_BATCH = 5
VGG_BATCH = 100


def build_models(device):
    vgg = vgg16.Vgg16().to(device)
    vgg.eval()

    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    detector = detector.to(device)
    detector.eval()

    return vgg, detector


def detect_objects(detector, frames, device):
    """Run the detector on all frames of one video.

    :param frames: (n_frames, H, W, 3) uint8 RGB
    :return: list (one entry per frame) of up to `n_objects` boxes [y1, x1, y2, x2]
    """
    all_boxes = []

    with torch.no_grad():
        for f in range(0, len(frames), DET_BATCH):
            chunk = frames[f:f + DET_BATCH]
            images = [torch.from_numpy(im).permute(2, 0, 1).float().div(255.0).to(device)
                      for im in chunk]
            outputs = detector(images)

            for output in outputs:
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()

                keep = []
                for b in range(len(boxes)):
                    if scores[b] < DET_SCORE_THRESH:
                        continue
                    if labels[b] not in VALID_CLASSES:
                        continue

                    x1, y1, x2, y2 = boxes[b]
                    keep.append([int(y1), int(x1), int(y2), int(x2)])

                    if len(keep) == n_objects:
                        break

                all_boxes.append(keep)

    return all_boxes


def extract_video_features(vgg, detector, video_path, dashcam, device):
    """Extract fc6 features for one video: (n_frames, n_boxes, 4096).

    Positive videos use the precomputed DAD annotation boxes; negative videos
    (which have no annotations) fall back to the object detector.
    """

    frames = dashcam.read_video(video_path)
    assert len(frames) > 0, "could not read video: " + video_path

    if len(frames) > n_frames:
        frames = frames[:n_frames]

    boxes_per_frame = dashcam.read_annotation(video_path)
    if boxes_per_frame is None:
        boxes_per_frame = detect_objects(detector, frames, device)

    x_scratch = np.zeros((n_frames, n_boxes, 224, 224, 3), dtype=np.float32)

    for file in range(len(frames)):
        imag = frames[file].astype(np.float32) / 255.0
        assert (0 <= imag).all() and (imag <= 1.0).all()

        img_height = imag.shape[0]
        img_width = imag.shape[1]

        resized_img = cv2.resize(imag, (224, 224))
        x_scratch[file, 0, :, :, :] = resized_img

        frame = boxes_per_frame[file]

        if (len(frame) > n_objects):
            frame = frame[0:n_objects]

        for objects in range(len(frame)):
            box = frame[objects]

            box[0] = max(0, min(box[0], img_height - 1))
            box[2] = max(box[0] + 1, min(box[2], img_height))
            box[1] = max(0, min(box[1], img_width - 1))
            box[3] = max(box[1] + 1, min(box[3], img_width))

            cropped_image = imag[box[0]:box[2], box[1]:box[3], :]
            resized_img = cv2.resize(cropped_image, (224, 224))
            x_scratch[file, objects + 1, :, :, :] = resized_img

    x_ = x_scratch.reshape([-1, 224, 224, 3])

    temp1 = []
    with torch.no_grad():
        for f in range(0, n_frames * n_boxes, VGG_BATCH):
            ba = x_[f:f + VGG_BATCH, :, :, :]
            ba = torch.from_numpy(ba).permute(0, 3, 1, 2).to(device)

            prob = vgg(ba)
            temp1.append(prob.cpu().numpy())

    prob = np.concatenate(temp1, axis=0).reshape([n_frames, n_boxes, 4096])

    return prob


def extract_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    vgg, detector = build_models(device)

    for train in [True, False]:
        split = "training" if train else "testing"
        out_dir = os.path.join(features_path, split)
        os.makedirs(out_dir, exist_ok=True)

        dashcam = Dashcam_data(dir=dataset_path, train=train, batch_size=batch_size)
        tot_batches = dashcam.total_videos // batch_size

        for i in range(int(tot_batches)):

            batch, video_labels, frame_labels = dashcam.get_next_batch()

            data = np.zeros((batch_size, n_frames, n_boxes, 4096), dtype=np.float32)

            for k in range(batch_size):
                video_path = batch[k]
                print(video_path)

                data[k] = extract_video_features(vgg, detector, video_path, dashcam, device)

            labels = video_labels.reshape(batch_size, n_classes)

            path = os.path.join(out_dir, "batch_" + str(i + 1).zfill(3) + ".npz")
            np.savez(path, data=data, labels=labels, ID=batch, paths=batch)
            print(split, "batches ", i + 1, "done out of", int(tot_batches))


if __name__ == "__main__":
    extract_features()
