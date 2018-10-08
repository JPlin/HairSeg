import math
import os
import pickle
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import tool_func

from component.data_transforms import (Exposure, Normalize, RandomCrop,
                                       Rescale, ToTensor, Resize_Padding,
                                       ToTensor2)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# where the real image library locate , path is linux style
sys.path.append(os.path.join('/mnt', 'd1p8', 'ming', 'jplin', 'FaceParsing'))
import Parsing as ps


class GeneralDataset(Dataset):
    def __init__(self,
                 options,
                 mode='train',
                 from_to_ratio=None,
                 transform=None):
        super(GeneralDataset, self).__init__()
        self.options = options
        if options:
            self.im_size = options['im_size']
            self.aug_setting_name = options['aug_setting_name']
            self.query_label_names = options['query_label_names']
        else:
            # test data is pre_define
            self.im_size = 512
            self.aug_setting_name = 'aug_512_0.6_multi_person'
            self.query_label_names = ['hair']

        print(self.query_label_names)
        self.transform = transform
        if mode == 'train':
            self.raw_dataset = self.gen_training_data(
                self.query_label_names, self.aug_setting_name,
                options.get('aug_ids', None), options.get('dataset_names', []))
        else:
            self.raw_dataset = self.gen_testing_data(
                self.query_label_names, self.aug_setting_name,
                options.get('dataset_names', []))
        image_list = list(range(len(self.raw_dataset)))
        if from_to_ratio is not None:
            fr = int(from_to_ratio[0] * len(self.raw_dataset))
            to = int(from_to_ratio[1] * len(self.raw_dataset))
            self.image_ids = image_list[fr:to]
        else:
            self.image_ids = image_list[:]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        im = self.raw_dataset.load_image(image_id)
        label = self.raw_dataset.load_labels(image_id)

        im_info = self.raw_dataset[image_id]['image_path']
        x_pos_map = None
        y_pos_map = None
        gaussian_map = None
        pos_map_path = im_info.replace('.jpg', '.pk').replace(
            'images', 'positions')
        if self.options.get('position_map',
                            False) and os.path.exists(pos_map_path):
            pos_map = pickle.load(open(pos_map_path, 'rb'))
            x_pos_map, y_pos_map = pos_map['x_map'], pos_map['y_map']
        elif self.options.get('center_map', False):
            x_pos_map, y_pos_map = self.get_xy_map(self.im_size)
        elif self.options.get('with_gaussian', False):
            gaussian_map = self.get_gaussian_map(self.im_size)

        res = {
            'image': im,
            'label': label,
            'x_pos': x_pos_map,
            'y_pos': y_pos_map,
            'g_map': gaussian_map
        }
        if self.transform:
            res = self.transform(res)
        return res

    def get_info(self, idx):
        image_id = self.image_ids[idx]
        im_info = self.raw_dataset[image_id]
        return im_info

    @staticmethod
    def get_xy_map(im_size):
        '''
            im_size: int or list
            return: x_mesh , y_mesh : HxW , [-0.5 , 0.5] center at center.
        '''
        if type(im_size) == int:
            h, w = im_size, im_size
        elif type(im_size) == list:
            h, w = im_size[0], im_size[1]
        x_center, y_center = w / 2, h / 2
        y_range = np.arange(h)
        x_range = np.arange(w)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        x_mesh = (x_mesh - x_center) / w
        y_mesh = (y_mesh - y_center) / h
        return x_mesh, y_mesh

    @staticmethod
    def get_gaussian_map(im_size):
        '''
        im_size: int or list
        return: H x W gaussian map
        '''
        if type(im_size) == int:
            h, w = im_size, im_size
        elif type(im_size) == list:
            h, w = im_size[0], im_size[1]
        x_center, y_center = w / 2, h / 2
        y_range = np.arange(h)
        x_range = np.arange(w)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)

        def _gaussian(x, y, x_center, y_center, sigma):
            x = np.abs(x - x_center)
            y = np.abs(y - y_center)
            ret = np.exp(-(x**2 + y**2) /
                         (2 * sigma**2)) / (sigma * math.sqrt(2 * math.pi))
            ret = ret / (np.max(ret) - np.min(ret))
            return ret

        gaussian_map = _gaussian(x_mesh, y_mesh, x_center, y_center, 50)
        return gaussian_map

    def gen_training_data(self,
                          query_label_names,
                          aug_setting_name='aug_512_0.8',
                          aug_ids=[0, 1, 2, 3],
                          dataset_names=[]):
        datasets = []
        if len(dataset_names) == 0:
            dataset_names = [
                'HELENRelabeled', 'MultiPIE', 'HangYang', 'Portrait724'
            ]

        for dataset_name in dataset_names:
            datasets.append(
                ps.Dataset(
                    dataset_name,
                    category='train',
                    aug_ids=aug_ids,
                    aug_setting_name=aug_setting_name,
                    query_label_names=query_label_names))
        return ps.CombinedDataset(datasets)

    def gen_testing_data(self,
                         query_label_names,
                         aug_setting_name='aug_512_0.8',
                         dataset_names=[]):
        datasets = []
        if len(dataset_names) == 0:
            dataset_names = [
                'HELENRelabeled', 'MultiPIE', 'HangYang', 'Portrait724'
            ]

        for dataset_name in dataset_names:
            datasets.append(
                ps.Dataset(
                    dataset_name,
                    category='test',
                    aug_ids=[0],
                    aug_setting_name=aug_setting_name,
                    query_label_names=query_label_names))
        return ps.CombinedDataset(datasets)


class AttnSegDataset(Dataset):
    def __init__(self,
                 options,
                 mode='train',
                 from_to_ratio=None,
                 transform=None):
        super(AttnSegDataset, self).__init__()
        self.options = options
        if options:
            self.im_size = options['im_size']
            self.query_label_names = options['query_label_names']
        else:
            # test data is pre_define
            self.im_size = 512
            self.query_label_names = ['hair']

        print('query_label_names', self.query_label_names)
        self.transform = transform

        # create basic dataset instance
        if mode == 'train':
            self.raw_dataset = self.gen_training_data(
                self.query_label_names, options.get('training_dataset', []))
        else:
            self.raw_dataset = self.gen_testing_data(
                self.query_label_names, options.get('test_dataset', []))

        # generate index mapping
        image_list = list(range(len(self.raw_dataset)))
        if from_to_ratio is not None:
            fr = int(from_to_ratio[0] * len(self.raw_dataset))
            to = int(from_to_ratio[1] * len(self.raw_dataset))
            self.image_ids = image_list[fr:to]
        else:
            self.image_ids = image_list[:]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        im = self.raw_dataset.load_image(image_id)
        label_list = self.raw_dataset.load_labels(image_id)
        fa_point_list = self.raw_dataset.load_fa_points(image_id)
        res = {'image': im, 'labels': label_list, 'fa_points': fa_point_list}
        if self.transform:
            res = self.transform(res)
        return res

    def gen_training_data(self, query_label_names, dataset_names):
        ret_dataset = ps.MergeDataset(
            dataset_names,
            category='train',
            query_label_names=query_label_names)
        return ret_dataset

    def gen_testing_data(self, query_label_names, dataset_names):
        ret_dataset = ps.MergeDataset(
            dataset_names,
            category='test',
            query_label_names=query_label_names)
        return ret_dataset


# for evaluate
def get_helen_test_data(query_label_names, aug_setting_name):
    return ps.Dataset(
        'HELENRelabeled_wo_pred',
        category='test',
        aug_ids=[0],
        aug_setting_name=aug_setting_name,
        query_label_names=query_label_names)


# calling general dataset
def gen_transform_data_loader(options,
                              mode='train',
                              batch_size=1,
                              shuffle=True,
                              dataloader=True,
                              use_original=False):
    # define composition of transforms
    transform_list = []
    if mode == 'train':
        transform_list = [
            Exposure(options['grey_ratio']),
            Rescale(options['crop_size'], options.get('random_scale', 0)),
            RandomCrop(options['im_size']),
            Normalize(),
            ToTensor()
        ]
    elif mode == 'test':
        if not use_original:
            transform_list = [
                Rescale(options['crop_size'], options.get('random_scale',
                                                          400)),
                RandomCrop(options['im_size']),
                Normalize(),
                ToTensor()
            ]
        else:
            transform_list = [Normalize(), ToTensor()]
    _transforms = transforms.Compose(transform_list)

    # define pytorch dataset
    ds = GeneralDataset(options, mode=mode, transform=_transforms)
    print("=> generate data loader: mode({0}) , length({1})".format(
        mode, len(ds)))

    # define pytorch dataloader
    ds_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=12)

    if dataloader:
        return ds_loader
    else:
        return ds


# for unit test , test pytorch dataset
def test_dataset(options):
    transform = transforms.Compose([
        Exposure(options['grey_ratio']),
        Rescale(options['crop_size'], options.get('random_scale', 400)),
        RandomCrop(options['im_size']),
        ToTensor()
    ])
    ds = GeneralDataset(options, mode='train', transform=transform)
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample['image'].size(), sample['label'].size())
        image = np.transpose(sample['image'].numpy(), [1, 2, 0])
        fig, axes = plt.subplots(ncols=4)
        axes[0].imshow(image)
        axes[0].set(title='image')
        axes[1].imshow(sample['label'].numpy())
        axes[1].set(title='ground-truth')
        axes[2].imshow(sample['x_pos'].numpy())
        axes[2].set(title='pos_map')
        axes[3].imshow(sample['y_pos'].numpy())
        axes[3].set(title='pos_map')
        plt.show()
        if i == 3:
            break


# for unit test , test pytorch dataloader
def test_dataloader(options):
    transform = transforms.Compose([
        Exposure(options['grey_ratio']),
        Rescale(options['crop_size'], options.get('random_scale', 400)),
        RandomCrop(options['im_size']),
        ToTensor()
    ])
    ds = GeneralDataset(options, mode='train', transform=transform)
    ds_loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=1)

    def _show_batch(sample_batch):
        image_batch, label_batch = sample_batch['image'], sample_batch['label']

        batch_size = len(image_batch)
        grid = utils.make_grid(image_batch)
        plt.figure()
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        grid = label_batch.numpy()
        print(np.unique(grid))
        grids = []
        for i in range(batch_size):
            grids.append(grid[i])
        plt.figure()
        plt.imshow(np.concatenate(grids, 1))

    for i_batch, sample_batch in enumerate(ds_loader):
        print(i_batch, sample_batch['image'].size(),
              sample_batch['label'].size())
        '''
        if i_batch == 3:
            _show_batch(sample_batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
        '''


# calling AttnSegDataset
def gen_transform_data_loader_2(options,
                                mode='train',
                                batch_size=1,
                                shuffle=True,
                                dataLoader=True):
    #  define composition of transforms
    transform_list = []
    if mode == 'train':
        transform_list = [
            Exposure(options['grey_ratio']),
            Resize_Padding(options['im_size']),
            Normalize(),
            ToTensor2()
        ]
    elif mode == 'test':
        transform_list = [
            Resize_Padding(options['im_size']),
            Normalize(),
            ToTensor2()
        ]
    _transforms = transforms.Compose(transform_list)

    # define pytorch dataset
    ds = AttnSegDataset(options, mode=mode, transform=_transforms)
    print("=> generate data loader: mode({0}) , length({1})".format(
        mode, len(ds)))

    # define pytorch dataloader
    ds_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=12)

    if DataLoader:
        return ds_loader
    else:
        return ds


def test_dataloader2(options):
    transform = transforms.Compose([
        Exposure(options['grey_ratio']),
        Resize_Padding(options['im_size']),
        Normalize(),
        ToTensor2()
    ])
    ds = AttnSegDataset(options, mode='train', transform=transform)
    print("=> generate data loader: mode({0}) , length({1})".format(
        'train', len(ds)))
    ds_loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=1)

    def _show_batch(sample_batch):
        image_batch, label_batch, fa_point_batch = sample_batch[
            'image'], sample_batch['labels'], sample_batch['fa_points']

        batch_size = len(image_batch)

        # visualize image
        grid = utils.make_grid(image_batch)
        plt.figure()
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

        # visualize labels
        grid = label_batch[0].numpy()
        print(np.unique(grid))
        grids = []
        for i in range(batch_size):
            grids.append(grid[i])
        plt.figure()
        plt.imshow(np.concatenate(grids, 1))

        # grid = label_batch[1].numpy()
        print(np.unique(grid))
        grids = []
        for i in range(batch_size):
            grids.append(grid[i])
        plt.figure()
        plt.imshow(np.concatenate(grids, 1))

        # visualize fa points
        grid = fa_point_batch[0].numpy()
        tool_func.vis_points(image_batch[0].numpy().transpose((1, 2, 0)),
                             grid[0])
        # tool_func.vis_points(image_batch[0].numpy().transpose((1, 2, 0)),
        #                      grid[0])

    for i_batch, sample_batch in enumerate(ds_loader):
        image = sample_batch['image']
        labels = sample_batch['labels']
        fa_points = sample_batch['fa_points']
        print(image.shape, labels[0].shape, fa_points[0].shape)

        if i_batch == 3:
            _show_batch(sample_batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break


if __name__ == '__main__':
    import yaml
    from torchvision import transforms, utils

    # test dataloader

    # plt.ion()
    # options = yaml.load(
    #     open(
    #         os.path.join(ROOT_DIR, 'options',
    #                      'dfn_hairseg_attention_randomcrop.yaml')))
    # print(options)

    # #test_dataset(options)
    # test_dataloader(options)

    # test dataloader2
    plt.ion()
    options = yaml.load(
        open(os.path.join(ROOT_DIR, 'options', 'attnseg.yaml')))
    test_dataloader2(options)
