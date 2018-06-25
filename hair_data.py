import os
import sys
import numpy as np
import pickle

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
                                                      self.aug_setting_name,
                                                      options.get(
                                                          'dataset_names', []))
        else:
            self.raw_dataset = self.gen_testing_data(self.query_label_names,
                                                     self.aug_setting_name,
                                                     options.get(
                                                         'dataset_names', []))
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
        pos_map_path = im_info.replace('.jpg', '.pk').replace(
            'images', 'positions')
        if os.path.exists(pos_map_path):
            pos_map = pickle.load(open(pos_map_path, 'rb'))
            x_pos_map, y_pos_map = pos_map['x_map'], pos_map['y_map']

        res = {
            'image': im,
            'label': label,
            'x_pos': x_pos_map,
            'y_pos': y_pos_map
        }
        if self.transform:
            res = self.transform(res)
        return res

    def get_info(self, idx):
        image_id = self.image_ids[idx]
        im_info = self.raw_dataset[image_id]
        return im_info

    def gen_training_data(self,
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
                    category='train',
                    aug_ids=[0, 1, 2, 3],
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


def gen_transform_data_loader(options,
                              mode='train',
                              batch_size=1,
                              shuffle=True,
                              dataloader=True):
    # define composition of transforms
    if mode == 'train':
        _transforms = transforms.Compose([
            Exposure(options['grey_ratio']),
            Rescale(options['crop_size'], options.get('random_scale', 400)),
            RandomCrop(options['im_size']),
            Normalize(),
            ToTensor(),
        ])
    elif mode == 'test':
        if options.get('position_map', False):
            _transforms = transforms.Compose([
                #Rescale(options['crop_size'], options.get('random_scale',
                #                                          400)),
                #RandomCrop(options['im_size']),
                Normalize(),
                ToTensor(),
            ])
        else:
            _transforms = transforms.Compose([Normalize(), ToTensor()])

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


if __name__ == '__main__':
    import yaml
    from torchvision import transforms, utils
    from component.data_transforms import Rescale, RandomCrop, Exposure, ToTensor
    plt.ion()
    options = yaml.load(
        open(
            os.path.join(ROOT_DIR, 'options',
                         'dfn_hairseg_attention_randomcrop.yaml')))
    print(options)

    #test_dataset(options)
    test_dataloader(options)
