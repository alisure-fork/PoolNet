import os
import cv2
import math
import time
import torch
import random
import numbers
import numpy as np
from torch import nn
from PIL import Image
import scipy.misc as sm
from torch.nn import init
from torch.optim import Adam
from torch.utils import data
from torch.backends import cudnn
from torchvision import transforms
import torchvision.utils as vutils
from alisuretool.Tools import Tools
from collections import OrderedDict
from torch.autograd import Variable
from torch.nn import utils, functional as F


class ImageDataTrain(data.Dataset):

    def __init__(self, data_root, data_list):
        self.sal_root = data_root
        self.sal_source = data_list
        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]
        # self.sal_list = self.sal_list[:20]
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

    pass


class ImageDataTest(data.Dataset):

    def __init__(self, sal_mode):
        self.data_source = self.get_test_info(sal_mode)
        self.data_root = self.data_source["image_root"]
        with open(self.data_source["image_source"], 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]
        self.image_num = len(self.image_list)
        pass

    def __getitem__(self, item):
        image, im_size = self.load_image_test(os.path.join(self.data_root, self.image_list[item]))
        return {'image': torch.Tensor(image), 'name': self.image_list[item % self.image_num], 'size': im_size}

    def __len__(self):
        return self.image_num

    @staticmethod
    def get_test_info(sal_mode='e'):
        result = {}
        if sal_mode == 'e':
            result["image_root"] = './data/ECSSD/Imgs/'
            result["image_source"] = './data/ECSSD/test.lst'
        elif sal_mode == 'p':
            image_root, image_source = './data/PASCALS/Imgs/', './data/PASCALS/test.lst'
            result["image_root"] = image_root
            result["image_source"] = image_source
        elif sal_mode == 'd':
            image_root, image_source = './data/DUTOMRON/Imgs/', './data/DUTOMRON/test.lst'
            result["image_root"] = image_root
            result["image_source"] = image_source
        elif sal_mode == 'h':
            image_root, image_source = './data/HKU-IS/Imgs/', './data/HKU-IS/test.lst'
            result["image_root"] = image_root
            result["image_source"] = image_source
        elif sal_mode == 's':
            image_root, image_source = './data/SOD/Imgs/', './data/SOD/test.lst'
            result["image_root"] = image_root
            result["image_source"] = image_source
        elif sal_mode == 't':
            image_root, image_source = './data/DUTS/DUTS-TE/DUTS-TE-Image/', './data/DUTS/DUTS-TE/test.lst'
            mask_root = './data/DUTS/DUTS-TE/DUTS-TE-Mask/'
            result["image_root"] = image_root
            result["mask_root"] = mask_root
            result["image_source"] = image_source
        elif sal_mode == 'm_r':  # for speed test
            image_root, image_source = './data/MSRA/Imgs_resized/', './data/MSRA/test_resized.lst'
            result["image_root"] = image_root
            result["image_source"] = image_source
        else:
            raise Exception(".................")
        return result

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


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.extract = [8, 15, 22, 29]  # [3, 8, 15, 22, 29]
        self.base = nn.ModuleList(self.vgg(self.cfg, 3))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        pass

    @staticmethod
    def vgg(cfg, i, batch_norm=False):
        layers = []
        in_channels = i
        stage = 1
        for v in cfg:
            if v == 'M':
                stage += 1
                if stage == 6:
                    layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
                else:
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            else:
                if stage == 6:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return layers

    def load_pretrained_model(self, model):
        self.base.load_state_dict(model, strict=False)
        pass

    def forward(self, x):
        tmp_x = []
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.extract:
                tmp_x.append(x)
        return tmp_x

    pass


class VGG16Locate(nn.Module):

    def __init__(self):
        super(VGG16Locate, self).__init__()
        self.vgg16 = VGG16()
        self.in_planes = 512
        self.out_planes = [512, 256, 128]

        ppms, infos = [], []
        for ii in [1, 3, 5]:
            ppms.append(nn.Sequential(nn.AdaptiveAvgPool2d(ii), nn.Conv2d(self.in_planes, self.in_planes,
                                                                          1, 1, bias=False), nn.ReLU(inplace=True)))
        self.ppms = nn.ModuleList(ppms)

        self.ppm_cat = nn.Sequential(nn.Conv2d(self.in_planes * 4, self.in_planes, 3, 1, 1, bias=False),
                                     nn.ReLU(inplace=True))
        for ii in self.out_planes:
            infos.append(nn.Sequential(nn.Conv2d(self.in_planes, ii, 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.infos = nn.ModuleList(infos)

        self.weight_init(self.modules())
        pass

    @staticmethod
    def weight_init(modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            pass
        pass

    def load_pretrained_model(self, model):
        self.vgg16.load_pretrained_model(model)
        pass

    def forward(self, x):
        x_size = x.size()[2:]
        xs = self.vgg16(x)

        xls = [xs[-1]]
        for k in range(len(self.ppms)):
            xls.append(F.interpolate(self.ppms[k](xs[-1]), xs[-1].size()[2:], mode='bilinear', align_corners=True))
        xls = self.ppm_cat(torch.cat(xls, dim=1))
        infos = []
        for k in range(len(self.infos)):
            infos.append(self.infos[k](F.interpolate(xls, xs[len(self.infos) - 1 - k].size()[2:], mode='bilinear', align_corners=True)))

        return xs, infos

    pass


class DeepPoolLayer(nn.Module):

    def __init__(self, k, k_out, need_x2, need_fuse):
        super(DeepPoolLayer, self).__init__()
        self.pools_sizes = [2,4,8]
        self.need_x2 = need_x2
        self.need_fuse = need_fuse
        pools, convs = [],[]
        for i in self.pools_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(k, k, 3, 1, 1, bias=False))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.relu = nn.ReLU()
        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.need_fuse:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)
        pass

    def forward(self, x, x2=None, x3=None):
        x_size = x.size()
        resl = x
        for i in range(len(self.pools_sizes)):
            y = self.convs[i](self.pools[i](x))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
        resl = self.relu(resl)
        if self.need_x2:
            resl = F.interpolate(resl, x2.size()[2:], mode='bilinear', align_corners=True)
        resl = self.conv_sum(resl)
        if self.need_fuse:
            resl = self.conv_sum_c(torch.add(torch.add(resl, x2), x3))
        return resl

    pass


class ScoreLayer(nn.Module):

    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k, 1, 1, 1)
        pass

    def forward(self, x, x_size=None):
        x = self.score(x)
        if x_size is not None:
            x = F.interpolate(x, x_size[2:], mode='bilinear', align_corners=True)
        return x

    pass


class PoolNet(nn.Module):

    def __init__(self):
        super(PoolNet, self).__init__()
        self.base = VGG16Locate()

        config = {'deep_pool': [[512, 512, 256, 128], [512, 256, 128, 128],
                                [True, True, True, False], [True, True, True, False]], 'score': 128}
        deep_pool = []
        for i in range(len(config['deep_pool'][0])):
            deep_pool += [DeepPoolLayer(config['deep_pool'][0][i], config['deep_pool'][1][i],
                                        config['deep_pool'][2][i], config['deep_pool'][3][i])]
            pass

        self.deep_pool = nn.ModuleList(deep_pool)
        self.score = ScoreLayer(config['score'])

        VGG16Locate.weight_init(self.modules())
        pass

    def forward(self, x):
        x_size = x.size()
        conv2merge, infos = self.base(x)
        conv2merge = conv2merge[::-1]

        merge = self.deep_pool[0](conv2merge[0], conv2merge[1], infos[0])  # A + F
        for k in range(1, len(conv2merge)-1):
            merge = self.deep_pool[k](merge, conv2merge[k+1], infos[k])  # A + F

        merge = self.deep_pool[-1](merge)  # A
        merge = self.score(merge, x_size)
        return merge

    pass


class Solver(object):

    def __init__(self, train_loader, epoch, batch_size, iter_size, save_folder, show_every, pretrained_model, lr, wd):
        self.train_loader = train_loader
        self.iter_size = iter_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.show_every = show_every
        self.save_folder = save_folder

        self.pretrained_model = pretrained_model

        self.wd = wd
        self.lr = lr
        self.lr_decay_epoch = [15, ]

        self.net = self.build_model()
        self.optimizer = Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        pass

    def build_model(self):
        net = PoolNet().cuda()

        if self.pretrained_model:
            net.base.load_pretrained_model(torch.load(self.pretrained_model))
            pass
        self._print_network(net, 'PoolNet Structure')
        return net

    def train(self):
        self.net.train()
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
                sal_image, sal_label = torch.Tensor(sal_image), torch.Tensor(sal_label)
                if torch.cuda.is_available():
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()

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
                    Tools.print('epoch: [{:2d}/{:2d}], lr={:.6f} iter:[{:5d}/{:5d}] || Sal:{:10.4f}'.format(
                        epoch, self.epoch, self.lr, i, iter_num, r_sal_loss / self.show_every))
                    r_sal_loss = 0
                    pass
                pass

            torch.save(self.net.state_dict(), '{}/epoch_{}.pth'.format(self.save_folder, epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
                pass
            pass

        torch.save(self.net.state_dict(), '{}/final.pth'.format(self.save_folder))
        pass

    @staticmethod
    def test(model_path, test_loader, result_fold):
        Tools.print('Loading trained model from {}'.format(model_path))
        net = PoolNet().cuda()
        net.load_state_dict(torch.load(model_path))
        net.eval()

        time_s = time.time()
        img_num = len(test_loader)
        for i, data_batch in enumerate(test_loader):
            if i % 100 == 0:
                Tools.print("test {} {}".format(i, img_num))
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            with torch.no_grad():
                images = torch.Tensor(images).cuda()
                pred = net(images)
                pred = np.squeeze(torch.sigmoid(pred).cpu().data.numpy()) * 255
                cv2.imwrite(os.path.join(result_fold, name[:-4] + '.png'), pred)
        time_e = time.time()
        Tools.print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        Tools.print('Test Done!')
        pass

    @classmethod
    def eval(cls, label_list, eval_list, th_num=25):
        epoch_mae = 0.0
        epoch_prec = np.zeros(shape=(th_num,)) + 1e-6
        epoch_recall = np.zeros(shape=(th_num,)) + 1e-6
        for i, (label_name, eval_name) in enumerate(zip(label_list, eval_list)):
            # Tools.print("{} {}".format(label_name, eval_name))
            if i % 100 == 0:
                Tools.print("eval {} {}".format(i, len(label_list)))

            im_label = np.asarray(Image.open(label_name).convert("L")) / 255
            im_eval = np.asarray(Image.open(eval_name).convert("L")) / 255

            mae = cls._eval_mae(im_eval, im_label)
            prec, recall = cls._eval_pr(im_eval, im_label, th_num)
            epoch_mae += mae
            epoch_prec += prec
            epoch_recall += recall
            pass

        avg_mae = epoch_mae/len(label_list)
        avg_prec, avg_recall = epoch_prec/len(label_list), epoch_recall/len(label_list)
        score4 = (1 + 0.3) * avg_prec * avg_recall / (0.3 * avg_prec + avg_recall)
        return avg_mae, score4.max(), np.mean(score4)

    @staticmethod
    def _eval_mae(y_pred, y):
        return np.abs(y_pred - y).mean()

    @staticmethod
    def _eval_pr(y_pred, y, th_num=100):
        prec, recall = np.zeros(shape=(th_num,)), np.zeros(shape=(th_num,))
        th_list = np.linspace(0, 1 - 1e-10, th_num)
        for i in range(th_num):
            y_temp = y_pred >= th_list[i]
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / y.sum()
            pass
        return prec, recall

    @staticmethod
    def _print_network(model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        Tools.print(name)
        Tools.print(model)
        Tools.print("The number of parameters: {}".format(num_params))
        pass

    pass


def my_train(run_name="run-6", pretrained_model='./pretrained/vgg16_20M.pth', lr=5e-5, wd=5e-4):
    epoch, batch_size, iter_size, show_every = 24, 1, 10, 50
    train_root, train_list = "./data/DUTS/DUTS-TR", "./data/DUTS/DUTS-TR/train_pair.lst"
    save_folder = Tools.new_dir('./results/{}'.format(run_name))

    dataset = ImageDataTrain(train_root, train_list)
    train_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    train = Solver(train_loader, epoch, batch_size, iter_size, save_folder, show_every, pretrained_model, lr, wd)
    train.train()
    pass


def my_test(run_name="run-6", sal_mode="t", model_path='./results/run-6/epoch_22.pth'):
    result_fold = Tools.new_dir("./results/test/{}/{}".format(run_name, sal_mode))

    dataset = ImageDataTest(sal_mode)
    test_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
    Solver.test(model_path, test_loader, result_fold)

    label_list = [os.path.join(dataset.data_source["mask_root"],
                               "{}.png".format(os.path.splitext(_)[0])) for _ in dataset.image_list]
    eval_list = [os.path.join(result_fold, "{}.png".format(os.path.splitext(_)[0])) for _ in dataset.image_list]
    mae, score_max, score_mean = Solver.eval(label_list, eval_list)
    Tools.print("{} {} {}".format(mae, score_max, score_mean))
    pass


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    _run_name = "run-6"
    # my_train(run_name=_run_name, pretrained_model='./pretrained/vgg16_20M.pth', lr=5e-5, wd=5e-4)
    my_test(run_name=_run_name, sal_mode="t", model_path='./results/{}/epoch_23.pth'.format(_run_name))
    pass
