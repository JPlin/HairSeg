import os
import torch
import numpy as np
import shutil


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


def save_checkpoint(state, is_best, dirname):
    filename = os.path.join(dirname, 'checkpoint.pth')
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


def unmold_input(tensor):
    # tensor: torch tensor
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if tensor.size()[1] > 3:
        tensor = tensor[:, :3]

    if type(tensor) == torch.Tensor:
        p = tensor.cpu().detach().numpy()
        p = np.transpose(p, (0, 2, 3, 1))
        p = p * std + mean
        return p[0]
    else:
        p = tensor * std + mean
        return p


def raw2image(tensor, if_max=True):
    '''
    input:
        tensor: [b,w , h , 2] or [b ,w , h] , means network one-hot output or target
        if_max: if get the max index of axis 1
    return: 
        numpy array: [b , w ,h ,3]
    '''
    if type(tensor) == torch.Tensor:
        p = tensor.cpu().detach().numpy()
    else:
        p = tensor
    if if_max:
        p = np.argmax(p, axis=1)
    p = np.expand_dims(p, -1)
    p = np.tile(p, (1, 1, 1, 3))
    return p[0]


_color_table = [
    np.array((1.0, 1.0, 1.0), np.float32),
    # np.array((20, 20, 255), np.float32) / 255.0, # hair?
    np.array((255, 250, 79), np.float32) / 255.0,  # face
    np.array([255, 125, 138], np.float32) / 255.0,  # lb
    np.array([213, 32, 29], np.float32) / 255.0,  # rb
    np.array([0, 144, 187], np.float32) / 255.0,  # le
    np.array([0, 196, 253], np.float32) / 255.0,  # re
    np.array([255, 129, 54], np.float32) / 255.0,  # nose
    np.array([88, 233, 135], np.float32) / 255.0,  # ulip
    np.array([0, 117, 27], np.float32) / 255.0,  # llip
    np.array([255, 76, 249], np.float32) / 255.0,  # imouth
    np.array((1.0, 0.5, 0.0), np.float32),
    np.array((0.0, 1.0, 0.5), np.float32),
    np.array((1.0, 0.0, 0.5), np.float32),
]


def blend_labels(image, labels):
    '''
    input:
        image: numpy , [h , w , 3]
        labels: numpy , [h , w]
    output:
        blended image: numpy , [h , w,  3]
    '''
    assert len(labels.shape) == 2
    colors = _color_table
    if image is None:
        image = np.zeros([labels.shape[0], labels.shape[1], 3], np.float32)
        alpha = 1.0
    else:
        image = image / np.max(image) * 0.4
        alpha = 0.6
    for i in range(1, np.max(labels) + 1):
        image += alpha * \
            np.tile(
                np.expand_dims(
                    (labels == i).astype(np.float32), -1),
                [1, 1, 3]) * colors[(i) % len(colors)]
    image[np.where(image > 1.0)] = 1.0
    image[np.where(image < 0)] = 0.0
    return image


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')