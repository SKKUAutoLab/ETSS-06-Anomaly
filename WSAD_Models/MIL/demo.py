import glob
import os
import torch.nn.functional as F
from feature_demo.resnext import generate_model
from net import Net
from PIL import Image
import numpy as np
import torch
import time
import cv2
from matplotlib import pyplot as plt
import argparse
try:
    import accimage
except ImportError:
    accimage = None

parser = argparse.ArgumentParser(description='MIL Demo')
parser.add_argument('--folder_test', default='Burglary001_x264', type=str, help='file name for demo')
parser.add_argument('--mil_checkpoint', default='checkpoint/ckpt.pth', type=str, help='file name for MIL model')
parser.add_argument('--cls_checkpoint', default='weight/RGB_Kinetics_16f.pth', type=str, help='file name for feature extraction model')
args = parser.parse_args()
# import torchvision.transforms as transforms
# transform = transforms.Compose([transforms.PILToTensor()])

# https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/demo_transform.py
class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

model = generate_model() # feature extraction model
classifier = Net().cuda() # MIL model
checkpoint = torch.load(args.cls_checkpoint)
model.load_state_dict(checkpoint['state_dict'])
checkpoint = torch.load(args.mil_checkpoint)
classifier.load_state_dict(checkpoint['net'])
model.eval()
classifier.eval()

path = args.folder_test + '/*'
save_path = args.folder_test + '_result'
img = sorted(glob.glob(path))

segment = len(img)//16
x_value = [i for i in range(segment)]
inputs = torch.Tensor(1, 3, 16, 240, 320) # need to resize img before using demo
x_time = [jj for jj in range(len(img))]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 16 frames

for num, i in enumerate(img):
    # first 16 images have fps = pred = 0
    if num < 16:
        inputs[:, :, num, :, :] = ToTensor(1)(Image.open(i)) # [1, 3, 240, 320]
        cv_img = cv2.imread(i)
        h, w, _ = cv_img.shape
        cv_img = cv2.putText(cv_img, 'FPS : 0.0, Pred : 0.0', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 240), 2)
    else:
        inputs[:, :, :15, :, :] = inputs[:, :, 1:, :, :] # [1, 3, 15, 240, 320]
        inputs[:, :, 15, :, :] = ToTensor(1)(Image.open(i))
        inputs = inputs.cuda()
        start = time.time()
        output, feature = model(inputs) # [1, 400] and [1, 2048]
        feature = F.normalize(feature, p=2, dim=1) # normalize feature to make score in range [0...1]
        out = classifier(feature)
        y_pred.append(out.item())
        end = time.time()
        FPS = str(1/(end-start))[:5]
        out_str = str(out.item())[:5] # round score .3f
        cv_img = cv2.imread(i)
        cv_img = cv2.putText(cv_img, 'FPS :' + FPS + ' Pred :' + out_str, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 240), 2)
        print('Processing frame:', i)
        if out.item() > 0.4:
            cv_img = cv2.rectangle(cv_img, (0, 0), (w, h), (0, 0, 255), 3) # visualize red when high anomaly
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    path = './' + save_path + '/' + os.path.basename(i)
    cv2.imwrite(path, cv_img)
os.system('ffmpeg -i "%s" "%s"' % (save_path+'/%05d.jpg', save_path+'.mp4'))
plt.plot(x_time, y_pred)
plt.savefig(save_path + '.png', dpi=600)
plt.cla()