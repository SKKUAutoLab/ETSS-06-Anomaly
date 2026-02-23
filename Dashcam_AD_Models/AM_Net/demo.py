import torch
from models.model import AMNet
import argparse
import os
import numpy as np

def _load_checkpoint(model, optimizer=None, filename='checkpoint.pth.tar'):
    start_epoch = 0
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
    else:
        print("No checkpoint found at '{}'".format(filename))
    return model, optimizer, start_epoch

def init_risky_object_model(model_file, x_dim, h_dim, n_frames):
    model = AMNet(x_dim, h_dim, n_frames)
    model = model.to(device)
    model.eval()
    model, _, _ = _load_checkpoint(model, filename=model_file)
    return model

def load_input_data(feature_file, device=torch.device('cuda')):
    data = np.load(feature_file)
    features = data['feature']
    toa = [data['toa'] + 0]
    detection = data['detection']
    flow = data['flow_feat']
    features = torch.Tensor(features).to(device)
    detection = torch.Tensor(detection).to(device)
    toa = torch.Tensor(toa).to(device)
    flow = torch.Tensor(flow).to(device)
    return features, detection, toa, flow

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_file', type=str, default="checkpoints/best_auc.pth")
    parser.add_argument('--h_dim', type=int, default=256, help='hidden dimension of the gru')
    parser.add_argument('--x_dim', type=int, default=2048, help='dimension of the resnet')
    parser.add_argument('--feature_dir', type=str, help="the path to the feature file.", default="feat_extract/feature/ROL/val")
    parser.add_argument('--output_dir', type=str, default="checkpoints/output/ROL")
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    feature_files = os.listdir(args.feature_dir)
    for file in feature_files:
        feature_file = os.path.join(args.feature_dir, file)
        features, detection, toa, flow = load_input_data(feature_file, device)
        features = features.unsqueeze(0)
        detection = detection.unsqueeze(0)
        flow = flow.unsqueeze(0)
        model = init_risky_object_model(args.ckpt_file, args.x_dim, args.h_dim, 100)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with torch.no_grad():
            losses, all_outputs, all_labels = model(features, detection, flow)
            file_name = os.path.join(args.output_dir, file)
            np.savez_compressed(file_name, output=all_outputs, label=all_labels)
