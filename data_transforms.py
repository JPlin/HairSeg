import os
import numpy as np
import torch
import math
import random
from torchvision import transforms, utils
from skimage import io, color, exposure, transform


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top:top + new_h, left:left + new_w]
        label = label[top:top + new_h, left:left + new_w]
        return {'image': image, 'label': label}


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        label = transform.resize(label, (new_h, new_w), mode='edge')

        return {'image': img, 'label': label}


class Exposure(object):
    def __init__(self, grey_ratio=0.1, adjust_gamma=True):
        self.grey_ratio = grey_ratio
        self.adjust_gamma = adjust_gamma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.adjust_gamma:
            gamma = math.exp(max(-1.6, min(1.6, random.normalvariate(0, 0.8))))
            image = exposure.adjust_gamma(image, gamma)

        if random.uniform(0, 1) < self.grey_ratio:
            image = color.rgb2gray(image)
            image = np.stack([image] * 3, -1)
        return {"image": image, "label": label}


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose((2, 0, 1))
        return {
            'image': torch.from_numpy(image),
            'label': torch.from_numpy(label)
        }
