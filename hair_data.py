import os
import sys
import numpy as np

from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from component.data_transforms import Rescale, RandomCrop, Exposure, ToTensor, Normalize

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# where the real image placement
sys.path.append('E:\\haya\\FaceData')
import Parsing as ps


class GeneralDataset(Dataset):
    def __init__(self,
                 options,
                 mode='train',
                 from_to_ratio=None,
                 transform=None):
        super(GeneralDataset, self).__init__()
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


# for evaluate
def get_helen_test_data(query_label_names, aug_setting_name='aug_512_0.8'):
    return ps.Dataset(
        'HELENRelabeled',
        category='test',
        aug_ids=[0],
        aug_setting_name=aug_setting_name,
        query_label_names=query_label_names)


# for unit test , test pytorch dataset
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


# for unit test , test pytorch dataloader
def test_dataloader(options):
    transform = transforms.Compose([
        Exposure(options['grey_ratio']),
        Rescale(options['crop_size']),
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
        if i_batch == 3:
            _show_batch(sample_batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break


def gen_transform_data_loader(options,
                              mode='train',
                              batch_size=1,
                              shuffle=True,
                              dataloader=True):
    ds = GeneralDataset(
        options,
        mode=mode,
        transform=None if mode == 'test' else transforms.Compose([
            Exposure(options['grey_ratio']),
            Rescale(options['crop_size']),
            RandomCrop(options['im_size']),
            Normalize(),
            ToTensor(),
        ]))
    print("=> generate data loader: mode({0}) , length({1})".format(
        mode, len(ds)))
    ds_loader = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    if dataloader:
        return ds_loader
    else:
        return ds


if __name__ == '__main__':
    import yaml
    from torchvision import transforms, utils
    from component.data_transforms import Rescale, RandomCrop, Exposure, ToTensor
    plt.ion()
    options = yaml.load(
        open(os.path.join(ROOT_DIR, 'options', 'dfn_hairseg.yaml')))
    print(options)

    #test_dataset(options)
    test_dataloader(options)
