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


class FixedResize(object):

    def __init__(self, size):
        self.size = (size, size)
        pass

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        return {'image': img, 'label': mask}

    pass


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img, 'label': mask}

    pass


class ImageDataTrain(data.Dataset):

    def __init__(self, data_root, data_list, image_size=None):
        self.sal_root = data_root
        self.sal_source = data_list
        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]
        self.sal_num = len(self.sal_list)

        if image_size is not None:
            self.transform = transforms.Compose([FixedResize(image_size), RandomHorizontalFlip()])
        else:
            self.transform = transforms.Compose([RandomHorizontalFlip()])
            pass
        self.transform2 = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform3 = transforms.Compose([transforms.ToTensor()])
        pass

    def __getitem__(self, item):
        im_name = self.sal_list[item % self.sal_num].split()[0]
        gt_name = self.sal_list[item % self.sal_num].split()[1]
        image = Image.open(os.path.join(self.sal_root, im_name)).convert("RGB")
        label = Image.open(os.path.join(self.sal_root, gt_name)).convert("L")

        if image.size == label.size:
            sample = {'image': image, 'label': label}
            sample = self.transform(sample)
            image, label = sample['image'], sample['label']

            image = self.transform2(image)
            label = self.transform3(label)
        else:
            Tools.print('IMAGE ERROR, PASSING {} {}'.format(im_name, gt_name))
            image, label = self.__getitem__(np.random.randint(0, self.sal_num))
            pass
        return image, label

    def __len__(self):
        return self.sal_num

    pass


class ImageDataTest(data.Dataset):

    def __init__(self, sal_mode):
        self.data_source = self.get_test_info(sal_mode)
        self.data_root = self.data_source["image_root"]
        with open(self.data_source["image_source"], 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]
        self.image_num = len(self.image_list)

        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        pass

    def __getitem__(self, item):
        name = self.image_list[item % self.image_num]
        image = Image.open(os.path.join(self.data_root, name)).convert("RGB")
        image = self.transform(image)
        return image, name

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

    pass


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.extract = [8, 15, 22, 29]  # [3, 8, 15, 22, 29]
        self.features = self.vgg(self.cfg)

        self.weight_init(self.modules())
        pass

    def forward(self, x):
        tmp_x = []
        for k in range(len(self.features)):
            x = self.features[k](x)
            if k in self.extract:
                tmp_x.append(x)
        return tmp_x

    @staticmethod
    def vgg(cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
            pass
        return nn.Sequential(*layers)

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

    def load_pretrained_model(self, pretrained_model="./pretrained/vgg16-397923af.pth"):
        self.load_state_dict(torch.load(pretrained_model), strict=False)
        pass

    pass


class DeepPoolLayer(nn.Module):

    def __init__(self, k, k_out, is_not_last):
        super(DeepPoolLayer, self).__init__()
        self.is_not_last = is_not_last

        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.conv1 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(k, k, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(k, k, 3, 1, 1, bias=False)

        self.conv_sum = nn.Conv2d(k, k_out, 3, 1, 1, bias=False)
        if self.is_not_last:
            self.conv_sum_c = nn.Conv2d(k_out, k_out, 3, 1, 1, bias=False)
        self.relu = nn.ReLU()
        pass

    def forward(self, x, x2=None, x3=None):
        x_size = x.size()

        y1 = self.conv1(self.pool2(x))
        y2 = self.conv2(self.pool4(x))
        y3 = self.conv3(self.pool8(x))
        res = torch.add(x, F.interpolate(y1, x_size[2:], mode='bilinear', align_corners=True))
        res = torch.add(res, F.interpolate(y2, x_size[2:], mode='bilinear', align_corners=True))
        res = torch.add(res, F.interpolate(y3, x_size[2:], mode='bilinear', align_corners=True))
        res = self.relu(res)

        if self.is_not_last:
            res = F.interpolate(res, x2.size()[2:], mode='bilinear', align_corners=True)

        res = self.conv_sum(res)

        if self.is_not_last:
            res = self.conv_sum_c(torch.add(torch.add(res, x2), x3))
        return res

    pass


class PoolNet(nn.Module):

    def __init__(self):
        super(PoolNet, self).__init__()
        # BASE
        self.vgg16 = VGG16()

        # PPM
        ind = 512
        self.ppm1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ind, ind, 1, 1, bias=False), nn.ReLU(inplace=True))
        self.ppm2 = nn.Sequential(nn.AdaptiveAvgPool2d(3), nn.Conv2d(ind, ind, 1, 1, bias=False), nn.ReLU(inplace=True))
        self.ppm3 = nn.Sequential(nn.AdaptiveAvgPool2d(4), nn.Conv2d(ind, ind, 1, 1, bias=False), nn.ReLU(inplace=True))
        self.ppm_cat = nn.Sequential(nn.Conv2d(ind * 4, ind, 3, 1, 1, bias=False), nn.ReLU(inplace=True))

        # INFO
        out_dim = [128, 256, 512]
        self.info1 = nn.Sequential(nn.Conv2d(ind, out_dim[0], 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        self.info2 = nn.Sequential(nn.Conv2d(ind, out_dim[1], 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        self.info3 = nn.Sequential(nn.Conv2d(ind, out_dim[2], 3, 1, 1, bias=False), nn.ReLU(inplace=True))

        # DEEP POOL
        deep_pool = [[512, 512, 256, 128], [512, 256, 128, 128]]
        self.deep_pool4 = DeepPoolLayer(deep_pool[0][0], deep_pool[1][0], True)
        self.deep_pool3 = DeepPoolLayer(deep_pool[0][1], deep_pool[1][1], True)
        self.deep_pool2 = DeepPoolLayer(deep_pool[0][2], deep_pool[1][2], True)
        self.deep_pool1 = DeepPoolLayer(deep_pool[0][3], deep_pool[1][3], False)

        # ScoreLayer
        score = 128
        self.score = nn.Conv2d(score, 1, 1, 1)

        VGG16.weight_init(self.modules())
        pass

    def forward(self, x):
        # BASE
        feature1, feature2, feature3, feature4 = self.vgg16(x)

        x_size = x.size()[2:]
        feature1_size = feature1.size()[2:]
        feature2_size = feature2.size()[2:]
        feature3_size = feature3.size()[2:]
        feature4_size = feature4.size()[2:]

        # PPM
        ppm_list = [feature4,
                    F.interpolate(self.ppm1(feature4), feature4_size, mode='bilinear', align_corners=True),
                    F.interpolate(self.ppm2(feature4), feature4_size, mode='bilinear', align_corners=True),
                    F.interpolate(self.ppm3(feature4), feature4_size, mode='bilinear', align_corners=True)]
        ppm_cat = self.ppm_cat(torch.cat(ppm_list, dim=1))

        # INFO
        info1 = self.info1(F.interpolate(ppm_cat, feature1_size, mode='bilinear', align_corners=True))
        info2 = self.info2(F.interpolate(ppm_cat, feature2_size, mode='bilinear', align_corners=True))
        info3 = self.info3(F.interpolate(ppm_cat, feature3_size, mode='bilinear', align_corners=True))

        # DEEP POOL
        merge = self.deep_pool4(feature4, feature3, info3)  # A + F
        merge = self.deep_pool3(merge, feature2, info2)  # A + F
        merge = self.deep_pool2(merge, feature1, info1)  # A + F
        merge = self.deep_pool1(merge)  # A

        # ScoreLayer
        merge = self.score(merge)
        if x_size is not None:
            merge = F.interpolate(merge, x_size, mode='bilinear', align_corners=True)
        return merge

    pass


class Solver(object):

    def __init__(self, train_loader, epoch, batch_size, iter_size, save_folder, show_every, lr, wd):
        self.train_loader = train_loader
        self.iter_size = iter_size
        self.epoch = epoch
        self.batch_size = batch_size
        self.show_every = show_every
        self.save_folder = save_folder

        self.wd = wd
        self.lr = lr
        self.lr_decay_epoch = [15, ]

        self.net = self.build_model()
        self.optimizer = Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        pass

    def build_model(self):
        net = PoolNet().cuda()
        net.vgg16.load_pretrained_model()
        self._print_network(net, 'PoolNet Structure')
        return net

    def train(self):
        self.net.train()
        iter_num = len(self.train_loader.dataset) // self.batch_size
        ave_grad = 0
        for epoch in range(self.epoch):
            r_sal_loss = 0
            self.net.zero_grad()
            for i, (sal_image, sal_label) in enumerate(self.train_loader):
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
        for i, (images, names) in enumerate(test_loader):
            if i % 100 == 0:
                Tools.print("test {} {}".format(i, img_num))
            with torch.no_grad():
                images = images.cuda()
                pred = net(images)
                pred = np.squeeze(torch.sigmoid(pred).cpu().data.numpy()) * 255
                cv2.imwrite(os.path.join(result_fold, names[0][:-4] + '.png'), pred)
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


def my_train(run_name, lr=5e-5, wd=5e-4, batch_size=1, image_size=None, epoch=24, iter_size=10, show_every=50):
    if batch_size > 1:
        assert image_size is not None

    train_root, train_list = "./data/DUTS/DUTS-TR", "./data/DUTS/DUTS-TR/train_pair.lst"
    save_folder = Tools.new_dir('./results/{}'.format(run_name))

    dataset = ImageDataTrain(train_root, train_list, image_size=image_size)
    train_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    train = Solver(train_loader, epoch, batch_size, iter_size, save_folder, show_every, lr, wd)
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


"""
#  3 2020-07-30 20:02:47 0.10483588910803267 0.8113900467163516 0.6678828142162999
# Max F-measre: 0.819712 # Precision:    0.864221 # Recall:       0.699606 # MAE:          0.104815
# 11 2020-07-30 20:47:48 0.05963944563820783 0.8347614190184267 0.756751045117701
# Max F-measre: 0.848795 # Precision:    0.889131 # Recall:       0.737300 # MAE:          0.0596364
# 12 2020-07-30 21:08:52 0.05539921863422358 0.8515504847969809 0.7707253474111654
# Max F-measre: 0.862992 # Precision:    0.898916 # Recall:       0.761548 # MAE:          0.0553957
# 14 2020-07-30 22:08:11 0.053452942576803254 0.8535626797360707 0.7870600922162752
# Max F-measre: 0.858931 # Precision:    0.891694 # Recall:       0.765213 # MAE:          0.0534508
# 16 2020-07-30 22:59:10 0.05543037568761575 0.8464022109947122 0.7790630094092397
# Max F-measre: 0.851753 # Precision:    0.878842 # Recall:       0.772393 # MAE:          0.0554272
# 19 2020-07-30 23:55:47 0.04404336020844739 0.86804850494036 0.8110303393090958
# Max F-measre: 0.874382 # Precision:    0.899922 # Recall:       0.798812 # MAE:          0.0440422
# 24 2020-07-31 09:43:03 0.042107617182937034 0.870920497128008 0.8176596900025891
# Max F-measre: 0.877171 # Precision:    0.902743 # Recall:       0.801492 # MAE:          0.0421069
"""


"""
11 2020-07-31 17:09:08 0.0536912643030876 0.8581485782839389 0.7777701527814314
11 Max F-measre: 0.86615
11 Precision:    0.898116
11 Recall:       0.774287
11 MAE:          0.0536876
16 2020-07-31 19:02:42 0.048643618717929916 0.8589147415954974 0.7913402526059301

19 2020-07-31 20:19:36 0.041217411982189915 0.8714894999016812 0.816163416733922
19 Max F-measre: 0.881578
19 Precision:    0.908338
19 Recall:       0.802747
19 MAE:          0.0412163

24 2020-07-31 22:28:56 0.04065566856003823 0.8664949516920055 0.8163405015155298
24 Max F-measre: 0.87744
24 Precision:    0.901842
24 Recall:       0.804851
24 MAE:          0.0406549
"""


"""
batch_size=01 320 24 2020-08-01 09:46:14 0.04382734164365161 0.8622201968859436 0.8076379011537488
batch_size=08 320 24 2020-08-01 20:06:48 0.04400467601286176 0.8508594165711362 0.8040955716095500

batch_size=04 480 01 2020-08-01 22:46:08 0.07890319562418711 0.7573049158512178 0.6594011700578953
batch_size=04 480 02 2020-08-01 23:55:38 0.07263538323553173 0.7741063254687782 0.6727149700994606
batch_size=04 480 03 2020-08-02 00:26:07 0.06495515407389885 0.7675775764532387 0.6926043165058576
batch_size=04 480 18 2020-08-02 09:07:45 0.048566140185210216 0.8127042441806771 0.7630705301043974

batch_size=16 256 01 2020-08-02 09:39:17 0.10341842702580475 0.7281378538090353 0.5988118977406801
batch_size=16 256 06 2020-08-02 10:24:33 0.06476589209489125 0.8273255588777589 0.7379714774973788
batch_size=16 256 08 2020-08-02 10:44:09 0.055854637692585134 0.8263596906465597 0.7662890195039139
batch_size=16 256 11 2020-08-02 11:14:27 0.05303567091915015 0.838915426620086 0.782806357546429
"""


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    _batch_size = 16
    _image_size = 256
    _run_name = "run-{}-{}".format(_batch_size, _image_size)
    # my_train(run_name=_run_name, lr=5e-5, wd=5e-4,
    #          batch_size=_batch_size, image_size=_image_size, epoch=24, iter_size=10, show_every=1000)
    my_test(run_name=_run_name, sal_mode="t", model_path='./results/{}/epoch_11.pth'.format(_run_name))
    pass
