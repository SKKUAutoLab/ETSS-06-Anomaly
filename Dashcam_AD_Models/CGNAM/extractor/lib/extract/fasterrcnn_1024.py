# yolo.py
# Drop-in replacement for fasterrcnn.py using YOLOv8 for detection and ResNet101 (layer3) for 1024-dim feature extraction
# Outputs the same format: res_feat (num_objs+1, 1024) numpy array, res_dect (num_objs+1, 6) numpy array
# with res_dect[:, :4] as xyxy, [:,4] scores, [:,5] class_ids (0-4 for person, bike, motorbike, car, bus)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from ultralytics import YOLO

# Mock cfg for compatibility
class CFG:
    classes = np.asarray(['__background__', 'person', 'bike', 'motorbike', 'car', 'bus'])
    CUDA = torch.cuda.is_available()
    vis = False  # Will be set in init
    TEST = type('Test', (), {})()
    TEST.NMS = 0.3
    TEST.MAX_SIZE = 1000  # Placeholder
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])  # BGR means, but not used here

cfg = CFG()

# Mock other imports if needed, but vis_detections is replaced with custom drawing

def parse_args(arg):
    parser = argparse.ArgumentParser(description='YOLO Detector')
    parser.add_argument('--vis', dest='vis', help='visualization mode', action='store_true')
    args = parser.parse_args(arg)
    return args

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet101(weights='IMAGENET1K_V1')
        # Extract up to layer3 (1024 channels)
        self.resnet_layer3 = nn.Sequential(*list(resnet.children())[:-3])  # Up to layer3 output (batch, 1024, H', W')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # To get 1024-dim vector

    def forward(self, x):
        x = self.resnet_layer3(x)
        x = self.avgpool(x)
        return x.squeeze()  # (1024,)

class FasterRCNN(object):
    def __init__(self, arg=None):
        if arg is not None:
            args = parse_args(arg)
            cfg.vis = args.vis
        else:
            cfg.vis = False

        self.class_names = cfg.classes

        # Load YOLOv8 model (use 'yolov8l.pt' for balance; change to 'yolov8x.pt' for higher accuracy)
        self.model = YOLO('yolov8l.pt')

        # Feature extractor
        self.feature_extractor = FeatureExtractor()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # Image transform for ResNet (RGB, [0,1], normalized)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Class mapping from COCO to custom (coco_id: custom_id)
        # COCO: 0=person, 1=bicycle (bike), 3=motorcycle (motorbike), 2=car, 5=bus
        self.class_map = {0: 0, 1: 1, 3: 2, 2: 3, 5: 4}

        print('Loaded YOLOv8 and ResNet101 (layer3) for detection and 1024-dim features.')

    def __call__(self, ori_im, det_output, use_nms=True):
        im = np.array(ori_im)  # BGR numpy array

        # Detection with YOLOv8
        iou_thres = cfg.TEST.NMS if use_nms else 0.0  # Disable NMS if not use_nms
        results = self.model(im, conf=0.05, iou=iou_thres, verbose=False, max_det=300)

        result = results[0]

        if result.boxes is None:
            boxes = torch.empty((0, 4), device=self.device)
            scores = torch.empty((0,), device=self.device)
            cls_ids = torch.empty((0,), device=self.device)
        else:
            boxes = result.boxes.xyxy.to(self.device)
            scores = result.boxes.conf.to(self.device)
            cls = result.boxes.cls.to(self.device)

            # Filter to relevant classes
            relevant_classes = torch.tensor(list(self.class_map.keys()), device=self.device)
            mask = torch.isin(cls, relevant_classes)
            boxes = boxes[mask]
            scores = scores[mask]
            cls = cls[mask]

            # Map class ids
            for coco_id, custom_id in self.class_map.items():
                cls[cls == coco_id] = custom_id

            cls_ids = cls

        # Prepare res_dect: (1 + num_dets, 6)
        res_dect = torch.zeros((1, 6), device=self.device)
        if boxes.numel() > 0:
            cls_dets = torch.cat((boxes, scores.unsqueeze(1), cls_ids.unsqueeze(1)), dim=1)
            res_dect = torch.cat((res_dect, cls_dets), dim=0)

        # Features
        # Frame feature
        img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            frame_feature = self.feature_extractor(input_tensor)

        # Object features
        obj_feats = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            crop = im[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.feature_extractor(crop_tensor)
            obj_feats.append(feat)

        if obj_feats:
            obj_feats = torch.stack(obj_feats, dim=0)
        else:
            obj_feats = torch.empty((0, frame_feature.shape[0]), device=self.device)

        res_feat = torch.cat((frame_feature.unsqueeze(0), obj_feats), dim=0)

        # Visualization if enabled
        if cfg.vis:
            im2show = np.copy(im)
            for i in range(1, res_dect.shape[0]):  # Skip first row
                bbox = res_dect[i, :4].cpu().numpy().astype(int)
                score = res_dect[i, 4].cpu().numpy()
                cls_id = int(res_dect[i, 5].cpu().numpy())
                class_name = self.class_names[cls_id + 1]  # +1 for bg offset

                cv2.rectangle(im2show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                text = '{} {:.2f}'.format(class_name, score)
                cv2.putText(im2show, text, (bbox[0], max(bbox[1] - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("frame", im2show)
            cv2.waitKey(1)
            if det_output is not None:
                det_output.write(im2show)

        return res_feat.cpu().numpy(), res_dect.cpu().numpy()
