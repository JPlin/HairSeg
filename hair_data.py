import os
import sys
import numpy as np

from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# where the real image placement
sys.path.append('\\\\MININT-37Q0T4O\\Datasets\\FaceData')
import Parsing as ps  # parsing data module


class GeneralDataset(Dataset):
    def __init__(self,
                 options,
                 mode='train',
                 from_to_ratio=None,
                 transform=None):
        super(GeneralDataset, self).__init__()
        self.im_size = options['im_size']
        self.aug_setting_name = options['aug_setting_name']
        self.query_label_names = options['query_label_names']
        self.transform = transform
        if mode == 'train':
            self.raw_dataset = self.gen_training_data(self.query_label_names,
                                                      self.aug_setting_name)
        else:
            self.raw_dataset = self.gen_testing_data(self.query_label_names,
                                                     self.aug_setting_name)
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

        res = {'image': im, 'label': label}
        if self.transform:
            res = self.transform(res)
        return res

    def gen_training_data(self,
                          query_label_names,
                          aug_setting_name='aug_512_0.8'):
        datasets = [
            ps.Dataset(
                'HELENRelabeled',
                category='train',
                aug_ids=[0, 1, 2],
                aug_setting_name=aug_setting_name,
                query_label_names=query_label_names),
            ps.Dataset(
                'MultiPIE',
                category='train',
                aug_ids=[0, 1, 2],
                aug_setting_name=aug_setting_name,
                query_label_names=query_label_names),
            ps.Dataset(
                'HangYang',
                category='train',
                aug_ids=[0, 1, 2],
                aug_setting_name=aug_setting_name,
                query_label_names=query_label_names),
            ps.Dataset(
                'Portrait724',
                category='train',
                aug_ids=[0, 1, 2],
                aug_setting_name=aug_setting_name,
                query_label_names=query_label_names),
        ]
        return ps.CombinedDataset(datasets)

    def gen_testing_data(self,
                         query_label_names,
                         aug_setting_name='aug_512_0.8'):
        datasets = [
            ps.Dataset(
                'HELENRelabeled',
                category='test',
                aug_ids=[0],
                aug_setting_name=aug_setting_name,
                query_label_names=query_label_names),
            ps.Dataset(
                'MultiPIE',
                category='test',
                aug_ids=[0],
                aug_setting_name=aug_setting_name,
                query_label_names=query_label_names),
            ps.Dataset(
                'HangYang',
                category='test',
                aug_ids=[0],
                aug_setting_name=aug_setting_name,
                query_label_names=query_label_names),
            ps.Dataset(
                'Portrait724',
                category='test',
                aug_ids=[0],
                aug_setting_name=aug_setting_name,
                query_label_names=query_label_names),
        ]
        return ps.CombinedDataset(datasets)


def test_dataset(options):
    transform = transforms.Compose([
        Exposure(options['grey_ratio']),
        Rescale(options['crop_size']),
        RandomCrop(options['im_size']),
        ToTensor()
    ])
    ds = GeneralDataset(options, mode='train', transform=transform)
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample['image'].size(), sample['label'].size())
        if i == 3:
            break


def test_dataloader(options):
    transform = transforms.Compose([
        Exposure(options['grey_ratio']),
        Rescale(options['crop_size']),
        RandomCrop(options['im_size']),
        ToTensor()
    ])
    ds = GeneralDataset(options, mode='train', transform=transform)
    ds_loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    def _show_batch(sample_batch):
        image_batch, label_batch = sample_batch['image'], sample_batch['label']

        batch_size = len(image_batch)
        grid = utils.make_grid(image_batch)
        plt.figure()
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        grid = label_batch.numpy()
        grids = []
        for i in range(batch_size):
            grids.append(grid[i])
        plt.figure()
        plt.imshow(np.concatenate(grids, 1))

    for i_batch, sample_batch in enumerate(ds_loader):
        print(i_batch, sample_batch['image'].size(),
              sample_batch['label'].size())
        if i_batch == 3:
            _show_batch(sample_batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break


def gen_transform_data_loader(options,
                              mode='train',
                              batch_size=1,
                              shuffle=True):
    ds = GeneralDataset(
        options,
        mode=mode,
        transform=transforms.Compose([
            Exposure(options['grey_ratio']),
            Rescale(options['crop_size']),
            RandomCrop(options['im_size']),
            ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    ds_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    return ds_loader


if __name__ == '__main__':
    import yaml
    from torchvision import transforms, utils
    from data_transforms import Rescale, RandomCrop, Exposure, ToTensor
    plt.ion()
    options = yaml.load(
        open(os.path.join(ROOT_DIR, 'options', 'dfn_hairseg.yaml')))
    print(options)

    #test_dataset(options)
    test_dataloader(options)
