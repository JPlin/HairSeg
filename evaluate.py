import argparse
import os
import sys
import time
import yaml

import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from skimage import io, transform

from hair_data import GeneralDataset, get_helen_test_data, gen_transform_data_loader
from HairNet import DFN
from component.metrics import Acc_score
from tool_func import *

global args, device, save_dir

parser = argparse.ArgumentParser(
    description='Pytorch Hair Segmentation Evaluate')
parser.add_argument(
    'evaluate_name', type=str, help='evaluate name | that is save dir')
parser.add_argument(
    '--model_name', required=True, default='', type=str, metavar='model name')
parser.add_argument('--batch_size', required=True, type=int, help='batch_size')
parser.add_argument(
    '--save',
    type=str2bool,
    nargs='?',
    default=False,
    help='save or visualize')
parser.add_argument(
    '--original',
    type=str2bool,
    nargs='?',
    default=False,
    help='evaluate on the original image')
parser.add_argument('--gpu_ids', type=int, nargs='*')
parser.add_argument(
    '--data_settings', default='aug_512_0.6_multi_person', type=str)
args = parser.parse_args()
device = None
save_dir = None


def main():
    global device, save_dir
    # use the gpu or cpu as specificed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = None
    if args.gpu_ids is None:
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
    else:
        device_ids = args.gpu_ids
        device = torch.device("cuda:{}".format(device_ids[0]))

    # set save dir
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(ROOT_DIR, 'evaluate_' + args.evaluate_name)
    os.makedirs(save_dir, exist_ok=True)

    # set options path
    option_path = os.path.join('logs', args.model_name,
                               args.model_name + '.yaml')
    if not os.path.exists(option_path):
        print('options path {} is not exists.'.format(option_path))
        sys.exit(1)
    options = yaml.load(open(option_path))

    # check model path
    model_path = os.path.join('logs', args.model_name, 'checkpoint.pth')
    if not os.path.exists(model_path):
        print('model path {} is not exists.'.format(model_path))
        sys.exit(1)

    # build the model
    if options['add_fc'] is not None:
        add_fc = options['add_fc']
    else:
        add_fc = False

    if options['self_attention'] is not None:
        self_attention = options['self_attention']
    else:
        self_attention = False

    model = DFN(
        in_channels=5 if options.get('position_map', False) else 3,
        add_fc=add_fc,
        self_attention=self_attention,
        back_bone=options['arch'])
    model = nn.DataParallel(model, device_ids=device_ids)

    # loading checkpoint
    print("=> loading checkpoint '{}'".format(model_path))
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(
        model_path, checkpoint['epoch']))

    if options.get('position_map', False) or args.original:
        test_ds = gen_transform_data_loader(
            options,
            mode='test',
            batch_size=1,
            shuffle=False,
            dataloader=False,
            use_original=args.original)
        evaluate_general_dataset(model, test_ds)
    else:
        test_ds = get_helen_test_data(
            ['hair'], aug_setting_name=args.data_settings)
        evaluate_raw_dataset(model, test_ds)


    # ------ begin evaluate
def evaluate_raw_dataset(model, dataset):
    batch_time = AverageMeter()
    acc_hist_all = Acc_score(['hair'])
    acc_hist_single = Acc_score(['hair'])

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        batch_index = 0
        batch = None
        labels = None
        image_ids = []
        image_names = []
        data_len = len(dataset.image_ids)
        for idx, image_id in enumerate(dataset.image_ids):
            # ------ start iteration
            image = dataset.load_image(image_id)
            if batch_index == 0:
                batch = np.zeros((args.batch_size, image.shape[0],
                                  image.shape[1], 3))
                labels = np.zeros((args.batch_size, image.shape[0],
                                   image.shape[1]))
            batch[batch_index] = (image / 255 - mean) / std
            labels[batch_index] = dataset.load_labels(image_id)
            image_ids.append(image_id)
            batch_index = batch_index + 1
            if batch_index < args.batch_size and idx != data_len - 1:
                continue
            # ------ end iteration

            batch_index = 0
            input = batch.transpose((0, 3, 1, 2))
            input = torch.from_numpy(input).to(torch.float).to(device)

            # get and deal with output
            output = model(input)
            if type(output) == list:
                output = output[0]
            if output.size()[-1] < labels.shape[-1]:
                output = F.upsample(
                    output, size=labels.shape[-2:], mode='bilinear')
            output = torch.argmax(output, dim=1).cpu().detach().numpy()

            # ------ start iteration
            input_images = unmold_input(batch)
            for b in range(input_images.shape[0]):
                # get data
                image_name = os.path.basename(
                    dataset[image_ids[b]]['image_path'])[:-4]
                ori_image = dataset.load_original_image(image_ids[b])
                target = dataset.load_original_labels(image_ids[b])
                tform_params = dataset.load_align_transform(image_ids[b])
                pred = transform.warp(
                    output[b],
                    tform_params,
                    output_shape=target.shape[:2],
                    preserve_range=True)
                pred = pred.astype(np.uint8)

                # calculate result
                acc_hist_all.collect(target, pred)
                acc_hist_single.collect(target, pred)
                f1_result = acc_hist_single.get_f1_results()['hair']

                # visualize result
                print(ori_image.shape, target.shape)
                gt_blended = blend_labels(ori_image, target)
                predict_blended = blend_labels(ori_image, pred)

                fig, axes = plt.subplots(ncols=2)
                axes[0].imshow(predict_blended)
                axes[0].set(title=f'predict:%04f' % (f1_result))
                axes[1].imshow(gt_blended)
                axes[1].set(title='ground-truth')

                if args.save:
                    save_path = os.path.join(save_dir, f'%04f_%s.png' %
                                             (f1_result, image_names))
                    plt.savefig(save_path)
                else:
                    plt.show()
                plt.close(fig)
                acc_hist_single.reset()

            # ------ end iteration
            batch_time.update(time.time() - end)
            end = time.time()
            image_ids = []

        f1_result = acc_hist_all.get_f1_results()['hair']
        print('Valiation: [{0}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Acc of f-score [{1}]'.format(
                  len(dataset), f1_result, batch_time=batch_time))


def evaluate_general_dataset(model, dataset):
    batch_time = AverageMeter()
    acc_hist_all = Acc_score(['hair'])
    acc_hist_single = Acc_score(['hair'])

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        batch_index = 0
        batch = None
        labels = None
        image_names = []
        data_len = len(dataset.image_ids)
        for idx, image_id in enumerate(dataset.image_ids):
            torch_tensor = dataset[idx]
            input, target = torch_tensor['image'].numpy(), torch_tensor[
                'label'].numpy()
            if batch_index == 0:
                batch = np.zeros((args.batch_size, input.shape[0],
                                  input.shape[1], input.shape[2]))
                labels = np.zeros((args.batch_size, target.shape[0],
                                   target.shape[1]))
            batch[batch_index] = input
            labels[batch_index] = target
            image_names.append(
                os.path.basename(dataset.get_info(idx)['image_path'])[:-4])
            batch_index = batch_index + 1
            if batch_index < args.batch_size and idx != data_len - 1:
                continue

            batch_index = 0
            input, target = torch.from_numpy(batch).to(
                torch.float).to(device), torch.from_numpy(labels).to(
                    torch.long).to(device)

            # get and deal with output
            output = model(input)
            if type(output) == list:  # multi scale output
                output = output[0]
            print(
                f'dealing with: input.shape{input.size()} output.shape{output.size()}'
            )
            if output.size()[-1] < target.size()[-1]:
                output = F.upsample(
                    output, size=target.size()[-2:], mode='bilinear')

            target = target.cpu().detach().numpy()
            pred = torch.argmax(output, dim=1).cpu().detach().numpy()
            acc_hist_all.collect(target, pred)
            acc_hist_single.collect(target, pred)
            f1_result = acc_hist_single.get_f1_results()['hair']

            input_images = unmold_input(input, keep_dims=True)
            for b in range(input_images.shape[0]):
                print('deal with', input_images[b].shape, target[b].shape)
                gt_blended = blend_labels(input_images[b], target[b])
                predict_blended = blend_labels(input_images[b], pred[b])

                fig, axes = plt.subplots(ncols=2)
                axes[0].imshow(predict_blended)
                axes[0].set(title=f'predict:%04f' % (f1_result))
                axes[1].imshow(gt_blended)
                axes[1].set(title='ground-truth')

                if args.save:
                    save_path = os.path.join(save_dir, f'%04f_%s.png' %
                                             (f1_result, image_names[b]))
                    plt.savefig(save_path)
                else:
                    plt.show()
                plt.close(fig)
                acc_hist_single.reset()

            batch_time.update(time.time() - end)
            end = time.time()

            image_names = []

        f1_result = acc_hist_all.get_f1_results()['hair']
        print('Valiation: [{0}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Acc of f-score [{1}]'.format(
                  len(dataset), f1_result, batch_time=batch_time))


if __name__ == '__main__':
    main()