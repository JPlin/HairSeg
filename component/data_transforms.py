import os
import numpy as np
import torch
import math
import random
from torchvision import transforms, utils
from skimage import io, color, exposure, transform, img_as_float


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label, x_pos, y_pos, g_map = sample['image'], sample[
            'label'], sample['x_pos'], sample['y_pos'], sample['g_map']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top:top + new_h, left:left + new_w]
        label = label[top:top + new_h, left:left + new_w]
        if x_pos is not None:
            x_pos = x_pos[top:top + new_h, left:left + new_w]
        if y_pos is not None:
            y_pos = y_pos[top:top + new_h, left:left + new_w]
        if g_map is not None:
            g_map = g_map[top:top + new_h, left:left + new_w]
        return {
            'image': image,
            'label': label,
            'x_pos': x_pos,
            'y_pos': y_pos,
            'g_map': g_map
        }


class Rescale(object):
    def __init__(self, output_size, random_scale=0):
        '''
        output_size: the min value between width and height 
        random_scale: the minus and plus range value
        '''
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.random_scale = random_scale

    def __call__(self, sample):
        image, label, x_pos, y_pos, g_map = sample['image'], sample[
            'label'], sample['x_pos'], sample['y_pos'], sample['g_map']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            output_size = max(
                self.output_size + random.randint(-self.random_scale,
                                                  self.random_scale), 520)
            if h > w:
                new_h, new_w = output_size * h / w, output_size
            else:
                new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_shape = (int(new_h), int(new_w))

        image = transform.resize(image, new_shape)
        label = transform.resize(
            label.astype(np.float), new_shape, order=0,
            mode='reflect').astype(np.uint8)
        if x_pos is not None:
            x_pos = transform.resize(x_pos, new_shape)
        if y_pos is not None:
            y_pos = transform.resize(y_pos, new_shape)
        if g_map is not None:
            g_map = transform.resize(g_map, new_shape)
        return {
            'image': image,
            'label': label,
            'x_pos': x_pos,
            'y_pos': y_pos,
            'g_map': g_map
        }


class Exposure(object):
    def __init__(self, grey_ratio=0.1, adjust_gamma=True):
        self.grey_ratio = grey_ratio
        self.adjust_gamma = adjust_gamma

    def __call__(self, sample):
        image = sample['image']

        if self.adjust_gamma:
            gamma = math.exp(max(-1.6, min(1.6, random.normalvariate(0, 0.8))))
            image = exposure.adjust_gamma(image, gamma)

        if random.uniform(0, 1) < self.grey_ratio:
            image = color.rgb2gray(image)
            image = np.stack([image] * 3, -1)
        sample['image'] = image
        return sample


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        if np.max(image) > 1:
            image = img_as_float(image)
        image = (image - self.mean) / self.std
        sample['image'] = image
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, label, x_pos, y_pos, g_map = sample['image'], sample[
            'label'], sample['x_pos'], sample['y_pos'], sample['g_map']

        # control the input image size not too large
        h, w = image.shape[:2]
        if h > 3000 or w > 3000:
            resize_size = (int(h / 2), int(w / 2))
            image = transform.resize(image, resize_size)
            label = transform.resize(
                label.astype(np.float32), resize_size, order=0,
                mode='reflect').astype(np.uint8)
            if x_pos is not None:
                x_pos = transform.resize(x_pos, resize_size)
            if y_pos is not None:
                y_pos = transform.resize(y_pos, resize_size)

        if x_pos is not None and y_pos is not None:
            x_pos = np.expand_dims(x_pos, -1)
            y_pos = np.expand_dims(y_pos, -1)
            image = np.concatenate((image, x_pos, y_pos), -1)
        if g_map is not None:
            g_map = np.expand_dims(g_map, -1)
            image = np.concatenate((image, g_map), -1)
        image = image.transpose((2, 0, 1))
        return {
            'image': torch.from_numpy(image).to(torch.float),
            'label': torch.from_numpy(label).to(torch.long),
        }
