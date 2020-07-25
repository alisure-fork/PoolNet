import os
import cv2
import math
import time
import torch
import random
import numbers
import numpy as np
from PIL import Image
import scipy.misc as sm
from torch.optim import Adam
from torch.utils import data
from torch.backends import cudnn
from torchvision import transforms
import torchvision.utils as vutils
from alisuretool.Tools import Tools
from collections import OrderedDict
from torch.autograd import Variable
from torch.nn import utils, functional as F
from torchvision.transforms import functional as F
from networks.poolnet import build_model, weights_init


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

        sample = {'sal_image': sal_image, 'sal_label': sal_label}
        return sample

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

    def __init__(self, sal_mode):
        self.data_root, self.data_list = self.get_test_info(sal_mode)
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

    @staticmethod
    def get_test_info(sal_mode='e'):
        if sal_mode == 'e':
            image_root, image_source = './data/ECSSD/Imgs/', './data/ECSSD/test.lst'
        elif sal_mode == 'p':
            image_root, image_source = './data/PASCALS/Imgs/', './data/PASCALS/test.lst'
        elif sal_mode == 'd':
            image_root, image_source = './data/DUTOMRON/Imgs/', './data/DUTOMRON/test.lst'
        elif sal_mode == 'h':
            image_root, image_source = './data/HKU-IS/Imgs/', './data/HKU-IS/test.lst'
        elif sal_mode == 's':
            image_root, image_source = './data/SOD/Imgs/', './data/SOD/test.lst'
        elif sal_mode == 't':
            image_root, image_source = './data/DUTS-TE/Imgs/', './data/DUTS-TE/test.lst'
        elif sal_mode == 'm_r':  # for speed test
            image_root, image_source = './data/MSRA/Imgs_resized/', './data/MSRA/test_resized.lst'
        else:
            raise Exception(".................")
        return image_root, image_source

    pass


class Solver(object):

    def __init__(self, train_loader, epoch, batch_size, iter_size, epoch_save,
                 save_folder, show_every, arch, pretrained_model, lr, wd):
        self.train_loader = train_loader
        self.iter_size = iter_size
        self.epoch = epoch
        self.epoch_save = epoch_save
        self.batch_size = batch_size
        self.show_every = show_every
        self.save_folder = save_folder

        self.pretrained_model = pretrained_model

        self.arch = arch
        self.wd = wd
        self.lr = lr
        self.lr_decay_epoch = [15, ]

        self.net = self.build_model()
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                              lr=self.lr, weight_decay=self.wd)
        pass

    def build_model(self):
        net = build_model(self.arch).cuda()
        if self.pretrained_model:
            net.base.load_pretrained_model(torch.load(self.pretrained_model))
            pass
        self._print_network(net, 'PoolNet Structure')
        return net

    def train(self):
        iter_num = len(self.train_loader.dataset) // self.batch_size
        ave_grad = 0
        for epoch in range(self.epoch):
            r_sal_loss = 0
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label = data_batch['sal_image'], data_batch['sal_label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    Tools.print('IMAGE ERROR, PASSING```')
                    continue
                sal_image, sal_label = torch.tensor(sal_image).cuda(), torch.tensor(sal_label).cuda()

                sal_pred = self.net(sal_image)
                sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')

                sal_loss = sal_loss_fuse / (self.iter_size * self.batch_size)
                r_sal_loss += sal_loss.data

                sal_loss.backward()

                ave_grad += 1

                # accumulate gradients as done in DSS
                if ave_grad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    ave_grad = 0
                    pass

                if i % (self.show_every // self.batch_size) == 0:
                    Tools.print('epoch: [%2d/%2d], iter:[%5d/%5d] || Sal:%10.4f' % (epoch, self.epoch, i,
                                                                                    iter_num, r_sal_loss / (i + 1)))
                    Tools.print('Learning rate: {}'.format(self.lr))
                    r_sal_loss = 0
                    pass
                pass

            if (epoch + 1) % self.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.save_folder, epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                      lr=self.lr, weight_decay=self.wd)
                pass
            pass

        torch.save(self.net.state_dict(), '{}/final.pth'.format(self.save_folder))

        pass

    @staticmethod
    def test(arch, model_path, test_loader, result_fold):
        Tools.print('Loading trained model from {}'.format(model_path))
        net = build_model(arch).cuda()
        net.load_state_dict(torch.load(model_path))
        net.eval()

        mode_name = 'sal_fuse'
        time_s = time.time()
        img_num = len(test_loader)
        for i, data_batch in enumerate(test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            with torch.no_grad():
                images = torch.tensor(images).cuda()
                preds = net(images)
                pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                multi_fuse = 255 * pred
                cv2.imwrite(os.path.join(result_fold, name[:-4] + '_' + mode_name + '.png'), multi_fuse)
        time_e = time.time()
        Tools.print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        Tools.print('Test Done!')
        pass

    @staticmethod
    def _print_network(model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        Tools.print(name)
        Tools.print(model)
        Tools.print("The number of parameters: {}".format(num_params))
        pass

    @staticmethod
    def bce2d(input, target, reduction=None):
        assert (input.size() == target.size())
        pos = torch.eq(target, 1).float()
        neg = torch.eq(target, 0).float()

        num_pos = torch.sum(pos)
        num_neg = torch.sum(neg)
        num_total = num_pos + num_neg

        alpha = num_neg / num_total
        beta = 1.1 * num_pos / num_total
        weights = alpha * pos + beta * neg
        return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

    pass


if __name__ == '__main__':
    is_train = True
    if is_train:
        vgg_path = './pretrained/vgg16_20M.pth'
        resnet_path = './pretrained/resnet50_caffe.pth'
        arch = "resnet"  # vgg
        pretrained_model = resnet_path

        n_color = 3
        lr, wd = 5e-5, 5e-5
        eopch, epoch_save, batch_size, iter_size, show_every = 24, 3, 1, 10, 50
        train_root, train_list = "", ""
        save_folder = Tools.new_dir('./results/run-0')

        dataset = ImageDataTrain(train_root, train_list)
        train_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        train = Solver(train_loader, epoch, batch_size, iter_size, epoch_save,
                       save_folder, show_every, arch, pretrained_model, lr, wd)
        train.train()
    else:
        _sal_mode = "e"
        _arch = "resnet"  # vgg
        _model_path = None
        _result_fold = Tools.new_dir("./results/test")

        _dataset = ImageDataTest(_sal_mode)
        _test_loader = data.DataLoader(dataset=_dataset, batch_size=1, shuffle=False, num_workers=1)
        Solver.test(_arch, _model_path, _test_loader, _result_fold)
        pass

    pass
