import math
import numbers
import random
import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps

class GroupRandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        w, h = img_group[0].size
        th, tw = self.size
        out_images = list()
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        for img in img_group:
            assert img.size[0] == w and img.size[1] == h
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        return out_images

class GroupCenterCrop:
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupRandomHorizontalFlip:
    def __init__(self, is_sth=False):
        self.is_sth = is_sth

    def __call__(self, img_group, is_sth=False):
        v = random.random()
        if not self.is_sth and v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group

class GroupNormalize1:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = self.mean * (tensor.size()[0] // len(self.mean))
        std = self.std * (tensor.size()[0] // len(self.std))
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        if len(tensor.size()) == 3:
            tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif len(tensor.size()) == 4:
            tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return tensor

class GroupScale:
    def __init__(self, size, interpolation=Image.BICUBIC):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupOverSample:
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):
        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)
        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size
        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)
                if img.mode == "L" and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)
            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group

class GroupFCSample:
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):
        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)
        image_w, image_h = img_group[0].size
        offsets = GroupMultiScaleCrop.fill_fc_fix_offset(image_w, image_h, image_h, image_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + image_h, o_h + image_h))
                normal_group.append(crop)
            oversample_group.extend(normal_group)
        return oversample_group

class GroupMultiScaleCrop:
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):
        im_size = img_group[0].size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]
        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))
        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4
        ret = list()
        ret.append((0, 0))
        ret.append((4 * w_step, 0))
        ret.append((0, 4 * h_step))
        ret.append((4 * w_step, 4 * h_step))
        ret.append((2 * w_step, 2 * h_step))
        if more_fix_crop:
            ret.append((0, 2 * h_step))
            ret.append((4 * w_step, 2 * h_step))
            ret.append((2 * w_step, 4 * h_step))
            ret.append((2 * w_step, 0 * h_step))
            ret.append((1 * w_step, 1 * h_step))
            ret.append((3 * w_step, 1 * h_step))
            ret.append((1 * w_step, 3 * h_step))
            ret.append((3 * w_step, 3 * h_step))
        return ret

    @staticmethod
    def fill_fc_fix_offset(image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 2
        h_step = (image_h - crop_h) // 2
        ret = list()
        ret.append((0, 0))
        ret.append((1 * w_step, 1 * h_step))
        ret.append((2 * w_step, 2 * h_step))
        return ret

class GroupRandomSizedCrop:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3.0 / 4, 4.0 / 3)
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if random.random() < 0.5:
                w, h = h, w
            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0
        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))

class Stack:
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == "L":
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == "RGB":
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                rst = np.concatenate(img_group, axis=2)
                return rst

class Stack1:
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if self.roll:
            return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
        else:
            rst = np.concatenate(img_group, axis=0)
            return torch.from_numpy(rst)

class ToTorchFormatTensor:
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class ToTorchFormatTensor1:
    def __init__(self, div=True):
        self.worker = torchvision.transforms.ToTensor()

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupToTensor:
    def __init__(self):
        self.worker = torchvision.transforms.ToTensor()

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class IdentityTransform:
    def __call__(self, data):
        return data

class GroupRandomColorJitter:
    def __init__(self, p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1):
        self.p = p
        self.worker = torchvision.transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img_group):
        v = random.random()
        if v < self.p:
            ret = [self.worker(img) for img in img_group]
            return ret
        else:
            return img_group

class GroupRandomGrayscale:
    def __init__(self, p=0.2):
        self.p = p
        self.worker = torchvision.transforms.Grayscale(num_output_channels=3)

    def __call__(self, img_group):
        v = random.random()
        if v < self.p:
            ret = [self.worker(img) for img in img_group]
            return ret
        else:
            return img_group

class GroupGaussianBlur:
    def __init__(self, p):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return [img.filter(ImageFilter.GaussianBlur(sigma)) for img in img_group]
        else:
            return img_group


class GroupSolarization:
    def __init__(self, p):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            return [ImageOps.solarize(img) for img in img_group]
        else:
            return img_group

class GroupTenCrop:
    def __init__(self, size):
        self.worker = torchvision.transforms.TenCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupTenCropToTensor:
    def __init__(self):
        self.worker = lambda crops: torch.stack([torchvision.transforms.ToTensor()(crop) * 255 for crop in crops])

    def __call__(self, crops):
        group_ = [self.worker(crop) for crop in crops]
        stack = torch.stack(group_, 1)
        return stack

class GroupTenNormalize:
    def __init__(self, mean, std):
        self.worker = GroupNormalize(mean, std)

    def __call__(self, crops):
        group_ = [self.worker(crop) for crop in crops]
        stack = torch.stack(group_, 0)
        return stack


class GroupNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.worker = torchvision.transforms.Normalize(mean, std)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

class GroupResize:
    def __init__(self, size):
        self.worker = torchvision.transforms.Resize(size)

    def __call__(self, images):
        return [self.worker(image) for image in images]

class ToTensor:
    def __init__(self):
        self.worker = lambda x: torchvision.transforms.functional.to_tensor(x) * 255

    def __call__(self, img_group):
        img_group = [self.worker(img) for img in img_group]
        return torch.stack(img_group, 0)

class LoopPad:
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, tensor):
        length = tensor.shape[0]
        if length >= self.max_len:
            return tensor
        n_pad = self.max_len - length
        pad = [tensor] * (n_pad // length)
        if n_pad % length > 0:
            pad += [tensor[0 : n_pad % length]]
        tensor = torch.cat([tensor] + pad, 0)
        return tensor