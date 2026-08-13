import time

import torch
import torch.nn as nn
import torchvision

# ImageNet normalization used by the torchvision pretrained VGG16
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]


class Vgg16(nn.Module):
    """VGG16 feature extractor.

    Takes RGB images scaled to [0, 1] with shape (batch, 3, 224, 224) and
    exposes the fc6 layer (4096-d, pre-ReLU), matching the original
    TensorFlow model's `vgg.fc6` output.
    """

    def __init__(self):
        super(Vgg16, self).__init__()

        start_time = time.time()
        print("build model started")

        vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)

        self.features = vgg.features
        self.avgpool = vgg.avgpool
        # classifier[0] is the first fully-connected layer (fc6): 25088 -> 4096
        self.fc6_layer = vgg.classifier[0]

        mean = torch.tensor(VGG_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(VGG_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        print(("build model finished: %ds" % (time.time() - start_time)))

    def forward(self, rgb):
        """
        :param rgb: rgb image tensor (batch, 3, 224, 224), values scaled [0, 1]
        :return: fc6 features (batch, 4096), pre-ReLU
        """
        assert rgb.shape[1:] == (3, 224, 224)

        x = (rgb - self.mean) / self.std

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        self.fc6 = self.fc6_layer(x)

        return self.fc6
