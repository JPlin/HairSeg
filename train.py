import argparse
import os
import shutil
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torchvision.models as models
import yaml
from logger import Logger

from hair_data import GeneralDataset, gen_transform_data_loader
from HairNet import DFN

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Pytorch Hair Segmentation Train')
parser.add_argument('options', type=str, help='train options name | xxx.yaml')
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--pretrain', default=True)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--gpu_ids', type=int, nargs='*')
args = parser.parse_args()


def main():
    global device, options, best_pick

    # use the gpu or cpu as specificed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = None
    if not args.gpu_ids is None:
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
    else:
        device_ids = args.gpu_ids
    
    # set logger
    save_log_dir = os.path.join(ROOT_DIR, args.log_dir, args.options[:-4])
    check_paths(save_log_dir)
    logger = Logger(save_log_dir)

    # set other settings
    options = yaml.load(open(os.path.join(ROOT_DIR, 'options', args.options)))
    start_epoch = 0
    best_pick = 0


    # build the model
    model = DFN()
    if options['arch'].startswith('alexnet') or options['arch'].startswith(
            'vgg'):
        model.features = nn.DataParallel(model.features , device_ids= device_ids)
    else:
        model = nn.DataParallel(model , device_ids= device_ids)
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
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
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found as '{}'".format(args.resume))

    # define dataset
    train_loader = gen_transform_data_loader(options, mode='train')
    test_loader = gen_transform_data_loader(
        options, mode='test', shuffle=False)

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
            'optimizer': optimizer.state_dict()
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    for i in range(options[step_per_epoch]):
        batch = next(train_loader)
        input , target = batch['image'] , batch['label']
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.to(device)

        # compute output
        output = model(input)
        prediction = torch.argmax(output , dim = 1)
        loss = criterion(output, target)

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
        if i % 100 == 0:
            print('Epoch: [{0}][{1}][{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses))
            # log scalar values
            logger.scalar_summary('loss' , losses.avg , epoch*options[step_per_epoch] + i + 1)

            # log training images
            info={'input_image': input.view(-1,).cpu().numpy(),
                    'ground_truth': target.view(-1,).cpu().numpy(),
                    'prediction': output.view(-1).cpu().numpy()}

def validata(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
             

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
