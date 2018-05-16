import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.models as models
import torchvision.utils as vutils
import yaml
from tensorboardX import SummaryWriter

from hair_data import GeneralDataset, gen_transform_data_loader
from HairNet import DFN
from component.criterion import CrossEntropyLoss2d
from component.metrics import Acc_score


def main(arguments):
    global device, options, writer, best_pick, acc_hist, args, global_step
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
    save_log_dir = os.path.join(ROOT_DIR, args.log_dir, args.options[:-4])
    check_paths(save_log_dir)
    writer = SummaryWriter(log_dir=save_log_dir, comment='DFN event')

    # set other settings
    options = yaml.load(open(os.path.join(ROOT_DIR, 'options', args.options)))
    start_epoch = 0
    best_pick = 0
    acc_hist = Acc_score(options['query_label_names'])

    # build the model
    model = DFN()
    if options['arch'].startswith('alexnet') or options['arch'].startswith(
            'vgg'):
        model.features = nn.DataParallel(model.features, device_ids=device_ids)
    else:
        model = nn.DataParallel(model, device_ids=device_ids)

    model.to(device)
    if args.debug:
        print(model)
    dummy_input = torch.rand(1, 3, 512, 512)
    #writer.add_graph(model, (dummy_input))  # add the model

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
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_pick = checkpoint['best_pick']
            global_step = checkpoint['global_step']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found as '{}'".format(args.resume))

    # define dataset
    train_loader = gen_transform_data_loader(
        options, mode='train', batch_size=options['batch_size'])
    test_loader = gen_transform_data_loader(
        options, mode='test', batch_size=options['batch_size'], shuffle=False)

    # start training
    for epoch in range(start_epoch, options['epoch']):
        adjust_learning_rate(optimizer, epoch)

        # train for each epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evalute on validation set
        pick_new = validata(test_loader, model, criterion)

        # tag best pick and save the checkpoint
        is_best = pick_new > best_pick
        best_pick = max(pick_new, best_pick)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': options['arch'],
            'state_dict': model.state_dict(),
            'best_pick': best_pick,
            'optimizer': optimizer.state_dict(),
            'global_step': global_step
        }, is_best)

    # close writer
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch):
    global global_step
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        global_step += 1
        input, target = batch['image'].to(device), batch['label'].to(device)

        if args.debug:
            print('input.size: {} , target.size: {}'.format(
                input.size(), target.size()))

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        if args.debug:
            print('output.size: {}, target.size: {} , loss.size: {}'.format(
                output.size(), target.size(), loss.size()))

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print every 10 step
        freq = 10 if args.debug else 100
        if i % freq == 0:
            print('Epoch: [{0}]<--[{1}]/[{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses))

            def _make_numpy_image(tensor, if_max=True):
                p = tensor.cpu().clone()
                p = p.detach().numpy()
                if if_max:
                    p = np.argmax(p, axis=1)
                p = np.expand_dims(p, -1)
                p = np.tile(p, (1, 1, 1, 3))
                print(">> ", p.shape)
                return p[0]

            def _unmold(tensor):
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                p = tensor.cpu().clone()
                p = p.detach().numpy()
                p = np.transpose(p, (0, 2, 3, 1))
                p = p * std + mean
                return p[0]

            writer.add_scalar(
                'train_loss', losses.avg, global_step=global_step)
            writer.add_image('input', _unmold(input), global_step)
            writer.add_image('target', _make_numpy_image(target, if_max=False),
                             global_step)
            writer.add_image('pred', _make_numpy_image(output), global_step)
        if options['step_per_epoch'] != -1 and options['step_per_epoch'] <= i:
            break


def validata(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_hist.reset()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(test_loader):
            input, target = batch['image'].to(device), batch['label'].to(
                device)

            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))
            acc_hist.collect(target.item(), torch.argmax(output, dim=1).item())

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
                  f1_result,
                  batch_time=batch_time,
                  loss=losses))

        writer.add_scalar('valid_loss', losses.avg, global_step=global_step)
        writer.add_scalar('f1-score', f1_result, global_step=global_step)
    return f1_result


def adjust_learning_rate(optimizer, epoch):
    lr = options['lr_base'] * ((1 - options['lr_decay'])**(epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state,
                    is_best,
                    filename=os.path.join('logs', 'checkpoint.pth')):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('.pth', '_best.pth'))


def check_paths(save_dir):
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pytorch Hair Segmentation Train')
    parser.add_argument(
        'options', type=str, help='train options name | xxx.yaml')
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    parser.add_argument('--pretrain', default=True)
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--gpu_ids', type=int, nargs='*')
    args = parser.parse_args()
    main(args)