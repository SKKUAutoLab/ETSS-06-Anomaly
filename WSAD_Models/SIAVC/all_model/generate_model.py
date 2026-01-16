import torch
from .models.cnn import cnn3d
from .models import C3DNet, resnet, ResNetV2, ResNeXt, ResNeXtV2, WideResNet, PreActResNet, EfficientNet, DenseNet, ShuffleNet, ShuffleNetV2, SqueezeNet, MobileNet, MobileNetV2
import argparse

def main(cnn_name, model_depth, n_classes, in_channels, sample_size):
    if cnn_name == 'cnn':
        print(cnn_name)
        model = cnn3d()
    elif cnn_name == 'C3D':
        model = C3DNet.get_model(sample_size=sample_size, sample_duration=16, num_classes=n_classes, in_channels=1)
    elif cnn_name == 'resnet':
        model = resnet.generate_model(model_depth=model_depth, n_classes=n_classes, n_input_channels=in_channels, shortcut_type='B', conv1_t_size=7, conv1_t_stride=1,
                                      no_max_pool=False, widen_factor=1.0)
    elif cnn_name == 'ResNetV2':
        model = ResNetV2.generate_model(model_depth=model_depth, n_classes=n_classes, n_input_channels=in_channels, shortcut_type='B', conv1_t_size=7, conv1_t_stride=1,
                                        no_max_pool=False, widen_factor=1.0)
    elif cnn_name == 'ResNeXt':
        model = ResNeXt.generate_model(model_depth=model_depth, n_classes=n_classes, in_channels=in_channels, sample_size=sample_size, sample_duration=16)
    elif cnn_name == 'ResNeXtV2':
        model = ResNeXtV2.generate_model(model_depth=model_depth, n_classes=n_classes, n_input_channels=in_channels)
    elif cnn_name == 'PreActResNet':
        model = PreActResNet.generate_model(model_depth=model_depth, n_classes=n_classes, n_input_channels=in_channels)
    elif cnn_name == 'WideResNet':
        model = WideResNet.generate_model(model_depth=model_depth, n_classes=n_classes, n_input_channels=in_channels)
    elif cnn_name == 'DenseNet':
        model = DenseNet.generate_model(model_depth=model_depth, num_classes=n_classes, n_input_channels=in_channels)
    elif cnn_name == 'SqueezeNet':
        model = SqueezeNet.get_model(version=1.0, sample_size=sample_size, sample_duration=16, num_classes=n_classes, in_channels=in_channels)
    elif cnn_name == 'ShuffleNetV2':
        model = ShuffleNetV2.get_model(sample_size=sample_size, num_classes=n_classes, width_mult=1., in_channels=in_channels)
    elif cnn_name == 'ShuffleNet':
        model = ShuffleNet.get_model(groups=3, num_classes=n_classes, in_channels=in_channels)
    elif cnn_name == 'MobileNet':
        model = MobileNet.get_model(sample_size=sample_size, num_classes=n_classes, in_channels=in_channels)
    elif cnn_name == 'MobileNetV2':
        model = MobileNetV2.get_model(sample_size=sample_size, num_classes=n_classes, in_channels=in_channels)
    elif cnn_name == 'EfficientNet':
        model = EfficientNet.from_name('efficientnet-b4', override_params={'num_classes': n_classes}, in_channels=in_channels)
    if torch.cuda.is_available():
        model.cuda()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_seed', default=1234, type=int, help='Mannual seed')
    parser.add_argument('--cnn_name', default='ResNet', type=str, help='cnn model names')
    parser.add_argument('--model_depth', default=101, type=str, help='model depth (18|34|50|101|152|200)')
    parser.add_argument('--n_classes', default=2, type=str, help='model output classes')
    parser.add_argument('--in_channels', default=1, type=str, help='model input channels (1|3)')
    parser.add_argument('--sample_size', default=128, type=str, help='image size')
    args = parser.parse_args()

    model = main(cnn_name=args.cnn_name, model_depth=args.model_depth, n_classes=args.n_classes, in_channels=args.in_channels, sample_size=args.sample_sizes)