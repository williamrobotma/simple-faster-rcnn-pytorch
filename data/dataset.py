from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from data.pipe_dataset import PipeDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
import json
from utils.config import opt


def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()

def pipe_normalze(img, mnstds):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=mnstds[0],
                                std=mnstds[1])
    img = normalize(t.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)

def preprocess_pipe(img, min_size=600, max_size=1000, mnstds):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    # img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size

    return pipe_normalze(img, mnstds)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000, mnstds):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale

class Transform_PIPE(object):

    def __init__(self, min_size=600, max_size=1000, mnstds):
        self.min_size = min_size
        self.max_size = max_size

        self.mnstds = mnstds

        self.trans = RandomTranspose(),

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess_pipe(img, self.min_size, self.max_size, self.mnstds)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally and vertically flip
        img, params = util.random_flip(
            img, x_random=True, y_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), y_flip=params['y_flip'], x_flip=params['x_flip'])

        # random transpose
        img, bbox = self.trans(img, bbox)

        return img, bbox, label, scale

class RandomTranspose(object):
    """Randomly transpose sample
    """

    def __call__(self, sample, bbox):
        transpose = random.random() < 0.5
        if transpose:
            # sample is CxHxW
            sample = np.transpose(sample, (0, 2, 1))
            bbox = bbox.copy()

            y_min = bbox[:, 0]
            x_min = bbox[:, 1]
            y_max = bbox[:, 2]            
            x_max = bbox[:, 3]

            bbox[:, 0] = x_min
            bbox[:, 1] = y_min
            bbox[:, 2] = x_max
            bbox[:, 3] = y_max

            return sample, bbox

        else:
            return sample, bbox


class Dataset:
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)

## Classes for the express purpose of training on PIPE 
class Dataset_PIPE:
    def __init__(self, opt, split='train'):
        self.opt = opt
        self.db = PipeDataset(os.path.join(opt.pipe_data_dir, split))

        with open(os.path.join(opt.pipe_data_dir, 'stds.json'), 'r') as f:
            mnstds = json.load(f)

        self.tsf = Transform(opt.min_size, opt.max_size, mnstds)

    def __getitem__(self, idx):
        ori_img, bbox, label= self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset_PIPE:
    def __init__(self, opt, split='test'):
        self.opt = opt
        self.db = PipeDataset(os.path.join(opt.pipe_data_dir, split))
        with open(os.path.join(opt.pipe_data_dir, 'stds.json'), 'r') as f:
            self.mnstds = json.load(f)

    def __getitem__(self, idx):
        ori_img, bbox, label = self.db.get_example(idx)
        img = preprocess_pipe(
            ori_img, self.opt.min_size, self.opt.max_size, self.mnstds)
        return img, ori_img.shape[1:], bbox, label

    def __len__(self):
        return len(self.db)
