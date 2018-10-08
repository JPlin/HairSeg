import argparse
import os
import shutil
import sys
import time
from tqdm import tqdm
from contextlib import closing

import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.models as models
import torchvision.utils as vutils
import yaml
import torch.nn.functional as F

from hair_data import *
from HairNet import DFN
from AttentionSegNet import AttnSegNet
from component.criterion import *
from component.metrics import Acc_score
from tool_func import *


def main(arguments):
    global device, options, writer, best_pick, acc_hist, args
    args = arguments
    global_step = 0
    # use the gpu or cpu as specificed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = None
    if args.gpu_ids is None:
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
    else:
        device_ids = args.gpu_ids
        device = torch.device("cuda:{}".format(device_ids[0]))

    # set logger
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    save_log_dir = os.path.join(ROOT_DIR, args.log_dir, args.options[:-5])
    check_paths(save_log_dir)
    save_evaluate_dir = os.path.join(ROOT_DIR, args.log_dir, args.options[:-5],
                                     "evaluate")
    check_paths(save_evaluate_dir)

    # set other settings
    options = yaml.load(open(os.path.join(ROOT_DIR, 'options', args.options)))
    shutil.copyfile(
        os.path.join(ROOT_DIR, 'options', args.options),
        os.path.join(save_log_dir, args.options))
    start_epoch = 0
    best_pick = 0
    acc_hist = Acc_score(options['query_label_names'])

    # build the model
    add_fc = options.get('add_fc', False)
    self_attention = options.get('self_attention', False)
    attention_plus = options.get('channel_attention', False)

    in_channels = 3
    if options.get('position_map', False):
        in_channels = 5
    elif options.get('center_map', False):
        in_channels = 5
    elif options.get('with_gaussian', False):
        in_channels = 4

    net_name = options.get('net', 'DFN')
    if net_name == 'DFN':
        model = DFN(
            in_channels=in_channels,
            add_fc=add_fc,
            self_attention=self_attention,
            attention_plus=attention_plus,
            debug=args.debug,
            back_bone=options['arch'])
    elif net_name.lower() == 'attnseg':
        model = AttnSegNet(
            len(options['query_label_names']) + 1,
            in_channels,
            debug=args.debug,
            back_bone=options['arch'])
    else:
        raise 'net not implemented error'

    # dummy_input = torch.rand(1, 3, 512, 512)
    # model(dummy_input)
    # exit(1)
    # writer.add_graph(model, (dummy_input))  # add the model
    if options['arch'].startswith('alexnet') or options['arch'].startswith(
            'vgg'):
        model.features = nn.DataParallel(model.features, device_ids=device_ids)
    else:
        model = nn.DataParallel(model, device_ids=device_ids)

    model.to(device)
    if args.debug:
        print(model)

    # set loss function
    if options.get('multi_scale_loss', False):
        criterion = Multi_Scale_CrossEntropyLoss2d().to(device)
    elif options.get('floss', False):
        criterion = Fscore_Loss().to(device)
    else:
        criterion = CrossEntropyLoss2d().to(device)

    if options['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            options['lr_base'],
            momentum=options['momentum'],
            weight_decay=options['weight_decay'])
    elif options['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            options['lr_base'],
            weight_decay=options['weight_decay'])
    elif options['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            options['lr_base'],
            weight_decay=options['weight_decay'])
    else:
        raise ('optimizer not defined')

    if args.resume:
        start_epoch, best_pick, global_step = load_checkpoints(
            args.resume, model, optimizer)

    # define dataset
    if net_name == 'DFN':
        train_loader = gen_transform_data_loader(
            options, mode='train', batch_size=options['batch_size'])
        test_loader = gen_transform_data_loader(
            options,
            mode='test',
            batch_size=options['batch_size'],
            shuffle=False)
    elif net_name.lower() == 'attnseg':
        train_loader = gen_transform_data_loader_2(
            options, mode='train', batch_size=options['batch_size'])
        test_loader = gen_transform_data_loader_2(
            options,
            mode='test',
            batch_size=options['evaluate_batch_size'],
            shuffle=False)

    # start training
    with closing(
            MultiStepStatisticCollector(
                log_dir=save_log_dir,
                comment='DFN event',
                global_step=global_step)) as stat_log:
        for epoch in range(start_epoch, options['epoch']):
            adjust_learning_rate(optimizer, epoch, options['lr_base'],
                                 options['lr_decay'])

            # train for each epoch
            train(train_loader, model, criterion, optimizer, epoch, stat_log)

            # evalute on validation set
            pick_new = validata(test_loader, model, criterion, stat_log)

            # tag best pick and save the checkpoint
            is_best = pick_new > best_pick
            best_pick = max(pick_new, best_pick)
            save_checkpoint({
                'epoch': epoch,
                'arch': options['arch'],
                'state_dict': model.state_dict(),
                'best_pick': best_pick,
                'optimizer': optimizer.state_dict(),
                'global_step': stat_log.current_step()
            }, is_best, save_log_dir)


def train(train_loader, model, criterion, optimizer, epoch, stat_log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    pbar = tqdm(train_loader)
    for i, batch in enumerate(pbar):
        stat_log.next_step()
        input, target = batch['image'].to(device), batch['labels'].to(device)

        fa_points = batch.get('fa_points', None)
        if fa_points is not None:
            fa_points = fa_points.to(device)

        if args.debug:
            print('input.size: {} , target.size: {}'.format(
                input.size(), target.size()))

        # measure data loading time
        data_time.update(time.time() - end)
        # print every 10 step
        freq = 10 if args.debug else 100

        # compute output
        if fa_points is not None:
            #model.apply(set_bn_to_eval)
            loss_list = []
            gathers = []
            outputs = model(input, fa_points)
            target = target.transpose(0, 1)
            merge_targets = target[0]
            merge_outputs = outputs[0]
            if i % freq == 0:
                image_input = unmold_input(input)
                print(image_input.shape)
                stat_log.add_image('train_input', image_input)
                stat_log.add_scalar('train_loss_avg', losses.avg)

            for j, output in enumerate(outputs):
                target_i = target[j]
                output_i = outputs[j]
                loss_list.append(criterion(output_i, target_i))

                merge_targets = torch.where(target_i > 0, target_i,
                                            merge_targets)
                merge_outputs = torch.where(merge_outputs > output_i,
                                            merge_outputs, output_i)

                if i % freq == 0:
                    image_target_i = raw2image(target_i, if_max=False)
                    image_output_i = raw2image(output_i)

                    blend_target_i = blend_labels(image_input , image_target_i[:,:,0])
                    blend_output_i = blend_labels(image_input , image_output_i[:,:,0])
                    stat_log.add_image(f'train_target_{j}', blend_target_i)
                    stat_log.add_image(f'train_pred_{j}', blend_output_i)

            loss_list.append(criterion(merge_outputs, merge_targets))
            loss = sum(loss_list)

        else:
            output = model(input)
            loss = criterion(output, target)

            if i % freq == 0:
                stat_log.add_scalar('train_loss', losses.avg)
                stat_log.add_image('input',
                                   unmold_input(input).astype(np.int64))
                stat_log.add_image('target', raw2image(target, if_max=False))
                stat_log.add_image('pred', raw2image(output[0]))

        if args.debug:
            print('output.size: {}, target.size: {} , loss.size: {}'.format(
                output[0].size() if type(output) == list else output.size(),
                target.size(), loss.size()))

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if i % freq == 0:
            stat_log.add_scalar('train_loss' , losses.val)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_description(
            'Epoch {epoch} Data {data_time.avg:.3f} Loss ({loss.avg:.4f})'.
            format(epoch=epoch, data_time=data_time, loss=losses))

        if options['step_per_epoch'] != -1 and options['step_per_epoch'] <= i:
            print('end of epoch')
            break


def validata(test_loader, model, criterion, stat_log):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_hist.reset()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(tqdm(test_loader)):
            input, target = batch['image'].to(device), batch['labels'].to(
                device)

            fa_points = batch.get('fa_points', None)
            if fa_points is not None:
                fa_points = fa_points.to(device)

            save_name_prefix = f'step_{i}'

            # compute output
            if fa_points is not None:
                #model.apply(set_bn_to_eval)
                loss_list = []
                outputs = model(input, fa_points)
                target = target.transpose(0, 1)
                merge_targets = target[0]
                merge_outputs = outputs[0]
                image_input = unmold_input(input)
                save_image(image_input , save_name_prefix + f'_input')
                for j, output in enumerate(outputs):
                    target_i = target[j]
                    output_i = outputs[j]
                    loss_list.append(criterion(output_i, target_i))
                    merge_targets = torch.where(target_i > 0, target_i,
                                                merge_targets)
                    merge_outputs = torch.where(merge_outputs > output_i,
                                                merge_outputs, output_i)

                    # evaluate the f score
                    pred_i = F.upsample(
                        output_i, size=target.size()[-2:], mode='bilinear')
                    pred_i = torch.argmax(pred_i, dim=1).cpu().detach().numpy()
                    target_i = target_i.cpu().detach().numpy()
                    acc_hist.collect(target_i, pred_i)

                    # visualize and save images
                    image_target_i = raw2image(target_i, if_max=False)
                    image_output_i = raw2image(pred_i, if_max=False)

                    blend_target_i = blend_labels(image_input , image_target_i[:,:,0])
                    blend_output_i = blend_labels(image_input , image_output_i[:,:,0])
                    save_image(blend_target_i, save_name_prefix + f'_person_{j}_gt')
                    save_image(blend_output_i , save_name_prefix + f'_person_{j}_pred')
                    stat_log.add_image(f'evaluate_target_{j}',blend_target_i)
                    stat_log.add_image(f'evaluate_pred_{j}',blend_output_i)

                loss_list.append(criterion(merge_outputs, merge_targets))
                loss = sum(loss_list)
            else:
                output = model(input)
                loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if i >= options['validation_step']:
                break

        f1_result = acc_hist.get_f1_results(options['query_label_names'])
        print('Valiation: [{0}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc of f-score [{1}]'.format(
                  len(test_loader),
                  f1_result['hair'],
                  batch_time=batch_time,
                  loss=losses))

        stat_log.add_scalar('valid_loss', losses.avg)
        stat_log.add_scalar('f1_score', f1_result['hair'])
    return f1_result['hair']


if __name__ == '__main__':
    import warnings
    parser = argparse.ArgumentParser(
        description='Pytorch Hair Segmentation Train')
    parser.add_argument(
        'options', type=str, help='train options name | xxx.yaml')
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--pretrain', default=True)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--gpu_ids', type=int, nargs='*')
    parser.add_argument('--debug', type=str2bool, nargs='?', default=False)
    args = parser.parse_args()
    if not args.debug:
        warnings.filterwarnings("ignore")
    main(args)