import os
import cv2
import torch
import random
import numbers
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from alisuretool.Tools import Tools
from torchvision.transforms import functional as F


class ImageDataTrain(data.Dataset):

    def __init__(self, data_root, data_list):
        self.sal_root = data_root
        self.sal_source = data_list
        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]
        self.sal_num = len(self.sal_list)
        pass

    def __getitem__(self, item):
        # sal data loading
        im_name = self.sal_list[item % self.sal_num].split()[0]
        gt_name = self.sal_list[item % self.sal_num].split()[1]
        sal_image = self.load_image(os.path.join(self.sal_root, im_name))
        sal_label = self.load_sal_label(os.path.join(self.sal_root, gt_name))
        sal_image, sal_label = self.cv_random_flip(sal_image, sal_label)
        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)

        if sal_image.shape[1:] != sal_label.shape[1:]:
            Tools.print('IMAGE ERROR, PASSING {} {}'.format(im_name, gt_name))
            sal_image, sal_label = self.__getitem__(np.random.randint(0, self.sal_num))
            pass
        return sal_image, sal_label

    def __len__(self):
        return self.sal_num

    @staticmethod
    def load_sal_label(path):
        if not os.path.exists(path):
            print('File {} not exists'.format(path))
        im = Image.open(path)
        label = np.array(im, dtype=np.float32)
        if len(label.shape) == 3:
            label = label[:, :, 0]
        label = label / 255.
        label = label[np.newaxis, ...]
        return label

    @staticmethod
    def cv_random_flip(img, label):
        flip_flag = random.randint(0, 1)
        if flip_flag == 1:
            img = img[:, :, ::-1].copy()
            label = label[:, :, ::-1].copy()
        return img, label

    @staticmethod
    def load_image(path):
        if not os.path.exists(path):
            print('File {} not exists'.format(path))
        im = cv2.imread(path)
        in_ = np.array(im, dtype=np.float32)
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        in_ = in_.transpose((2, 0, 1))
        return in_

    @staticmethod
    def load_image_test(path):
        if not os.path.exists(path):
            print('File {} not exists'.format(path))
        im = cv2.imread(path)
        in_ = np.array(im, dtype=np.float32)
        im_size = tuple(in_.shape[:2])
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        in_ = in_.transpose((2, 0, 1))
        return in_, im_size

    pass


class ImageDataTest(data.Dataset):

    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)
        pass

    def __getitem__(self, item):
        image, im_size = self.load_image_test(os.path.join(self.data_root, self.image_list[item]))
        image = torch.Tensor(image)

        return {'image': image, 'name': self.image_list[item % self.image_num], 'size': im_size}

    def __len__(self):
        return self.image_num

    pass
