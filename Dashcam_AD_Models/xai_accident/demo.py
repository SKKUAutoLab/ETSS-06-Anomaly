import numpy as np
import glob
import cv2
import os
import torch
from torchvision import transforms
from PIL import Image
from natsort import natsorted
import argparse
from model import AccidentXai
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

parser = argparse.ArgumentParser()
parser.add_argument('--demo_dir', type=str, default='demo/000057')
parser.add_argument('--h_dim', type=int, default=256)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--model_file', type=str, default='snapshot/best_model.pth')
parser.add_argument('--dest_dir', type=str, default='output_demo')
args = parser.parse_args()

def init_accident_model(model_file, h_dim, n_layers):
    model = AccidentXai(h_dim, n_layers)
    model = model.to(device)
    model.eval()
    model = load_checkpoint(model, model_file)
    return model

def load_checkpoint(model, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
    else:
        print("No checkpoint found at '{}'".format(filename))
    return model

def get_input_video(video_dir, device):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    images = []
    video_path = natsorted(glob.glob(os.path.join(video_dir, '*.jpg')))
    for i in video_path:
        image_path = i
        image = Image.open(image_path)
        image = transform(image)
        images.append(image)
    x = torch.stack(images).to(device) # [50, 3, 224, 224]
    x = torch.unsqueeze(x, 0) # [1, 50, 3, 224, 224]
    return x

def parse_results(all_outputs, batch_size=1, n_frames=50):
    pred_score = np.zeros((batch_size, n_frames), dtype=np.float32)
    for t in range(n_frames):
        pred = all_outputs[t]
        pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
        pred_score[:, t] = np.exp(pred[:, 1])/np.sum(np.exp(pred), axis=1)
    return pred_score

def building_cam(model, methods):
    target_layers = [model.features.resnet.layer4[-1]]
    cam_algorithm = methods['gradcam']
    cam = cam_algorithm(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    return cam

def saliency_map(cam, video_dir, destination_dir):
    video_path = natsorted(glob.glob(os.path.join(video_dir, '*.jpg')))
    target_category = 1
    dim = (512, 384)
    for img in video_path:
        image_path = img
        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] # [224, 224, 3]
        rgb_img = np.float32(rgb_img)/255
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # [1, 3, 224, 224]
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category) # [1, 224, 224]
        grayscale_cam = grayscale_cam[0, :] # [224, 224]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True) # [224, 224, 3]
        file_name = img.split('/')[-1]
        file_save = os.path.join(destination_dir, file_name)
        resized = cv2.resize(cam_image, dim)
        cv2.imwrite(file_save, resized)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_data = get_input_video(args.demo_dir, device)
    model = init_accident_model(args.model_file, args.h_dim, args.n_layers)
    labels = torch.Tensor([[0, 1]]).to(device)
    toa = torch.Tensor([[36]]).to(device)
    with torch.no_grad():
        loss, output = model(input_data, labels, toa)
    pred_score = parse_results(output)
    print('Prediction score:', pred_score)
    methods = {"gradcam": GradCAM, "scorecam": ScoreCAM, "gradcam++": GradCAMPlusPlus, "ablationcam": AblationCAM, "xgradcam": XGradCAM, "eigencam": EigenCAM, "eigengradcam": EigenGradCAM}
    cam = building_cam(model, methods)
    video_name = args.demo_dir.split('/')[-1]
    destination_dir = args.dest_dir + '/' + video_name
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    saliency_map(cam, args.demo_dir, destination_dir)