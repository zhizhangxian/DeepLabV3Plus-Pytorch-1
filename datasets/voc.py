import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import torch
import numpy as np
import cv2

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from utils import train_ext_transforms as train_et

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}


def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


class VOCSegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    cmap = voc_cmap()

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 num_copy=0):

        is_aug = False
        if year == '2012_aug':
            is_aug = True
            year = '2012'

        self.num_copy = num_copy
        self.root = os.path.expanduser(root)
        self.year = year
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        self.transform = transform

        self.image_set = image_set
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            print(voc_root)
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if is_aug and image_set == 'train':
            mask_dir = os.path.join(voc_root, 'SegmentationClassAug')
            assert os.path.exists(mask_dir), "SegmentationClassAug not found, please refer to README.md and prepare it manually"
            splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, 'train_aug.txt')  # './datasets/data/train_aug.txt'
        else:
            mask_dir = os.path.join(voc_root, 'SegmentationClass')
            splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        __img = np.array(img).copy()
        W, H = img.size
        target = Image.open(self.masks[index])
        if self.transform is not None:
            if self.num_copy == 0:
                img, target = self.transform(img, target)
                return img, target
            elif self.num_copy == 1:
                img, target = self.transform(img, target)
                overlap = torch.zeros((H, W), dtype=torch.bool)
                return img, target, overlap
            elif self.num_copy == 2:
                imgs = []
                targets = []
                transforms = []
                for copy in range(self.num_copy):
                    _img, _target, _transform = self.transform(img, target)
                    imgs.append(_img)
                    targets.append(_target)
                    transforms.append(_transform)
                i1, j1, i2, j2 = transforms[0][0], transforms[0][1], transforms[1][0], transforms[1][1]
                h, w = transforms[0][2], transforms[0][3]
                padding_x, padding_y = transforms[0][-2], transforms[0][-1]

                # # h, w = self.crop_x, self.crop_y
                # overlap = torch.zeros((h, w), dtype=torch.bool)
                # overlap[max(i1, i2, 0):min(i1 + h, i2 + h, H), max(j1, j2, 0):min(j1 + w, j2 + w, W)] = 1
                overlap1 = [max(i2 - i1, 0), max(j2 - j1, 0), min(i2 - i1 + h, h), min(j2 - j1 + h, h)]
                overlap2 = [max(i1 - i2, 0), max(j1 - j2, 0), min(i1 - i2 + h, w), min(j1 - j2 + h, w)]
                overlaps = [overlap1, overlap2]
                return imgs, targets, overlaps
            else:
                raise NotImplementedError
        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)


def collate_fn2(batchs):
    _imgs = []
    _targets = []
    _overlaps = []
    for index, batch in enumerate(batchs):
        _overlaps.append([])
        imgs, targets, overlaps = batch
        for i in range(len(imgs)):
            _imgs.append(torch.unsqueeze(imgs[i], 0))
            _targets.append(torch.unsqueeze(targets[i], 0))
            _overlaps[index].append(overlaps[i])
    return torch.cat(_imgs, dim=0), torch.cat(_targets, dim=0), _overlaps


def main():
    train_transform = train_et.train_ExtCompose([
        # et.ExtResize(size=opts.crop_size),
        # train_et.ExtRandomScale((0.5, 2.0)),
        train_et.ExtRandomCrop(size=(513, 513), pad_if_needed=True),
        # train_et.ExtRandomHorizontalFlip(),
        train_et.ExtToTensor(),
        train_et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    val_transform = train_et.ExtCompose([
        train_et.ExtToTensor(),
        train_et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    train_set = VOCSegmentation(root='/Users/zhangxian/Dataset/', transform=train_transform, num_copy=2, image_set='train')
    val_set = VOCSegmentation(root='/Users/zhangxian/Dataset/', transform=val_transform, image_set='val')

    batch_size = 4
    from torch.utils.data.dataloader import DataLoader
    dl = DataLoader(train_set, collate_fn=collate_fn2, batch_size=batch_size, shuffle=False)
    dl_val = DataLoader(val_set, collate_fn=None, batch_size=1, shuffle=False)

    for diter in dl:
        img, label, overlap = diter[0], diter[1], diter[2]

        # img_1 = img[0, :, overlap[0][0][0]:overlap[0][0][2], overlap[0][0][1]:overlap[0][0][3]]
        # img_2 = img[1, :, overlap[0][1][0]:overlap[0][1][2], overlap[0][1][1]:overlap[0][1][3]]
        for i in range(batch_size // 2):
            img_1 = img[2 * i, :, overlap[i][0][0]:overlap[i][0][2], overlap[i][0][1]:overlap[i][0][3]]
            img_2 = img[2 * i + 1, :, overlap[i][1][0]:overlap[i][1][2], overlap[i][1][1]:overlap[i][1][3]]
            a = (img_1 == img_2)
            print(a)
        break
    for diter in dl_val:
        print(len(diter))
        exit()
        img, label = diter[0], diter[1]
        print(img.shape)
        print(label.shape)
    return dl


def test():
    from matplotlib import pyplot as plt
    train_transform = train_et.ExtCompose([
        # et.ExtResize(size=opts.crop_size),
        # train_et.ExtRandomScale((0.5, 2.0)),
        train_et.ExtRandomCrop(size=(513, 513), pad_if_needed=True),
        # train_et.ExtRandomHorizontalFlip(),
        train_et.ExtToTensor(),
        train_et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])
    dataset = VOCSegmentation(root='/Users/zhangxian/Dataset/', transform=train_transform, num_copy=2)
    _img = [np.array(img) for img in imgs]
    h, w, _ = _img[0].shape
    _img.append(cv2.resize(img0, _img[0].shape[:2], interpolation=1))
    _img = np.hstack(_img).astype(np.uint8)
    # _img = _img[0].astype(np.uint8)

    # overlap = np.expand_dims(overlap, -1).repeat(3, -1)
    img1 = img0.copy()
    img1[np.where(overlap == 0)] = 0
    __img = np.hstack((img0, img1))
    ___img = cv2.resize(__img, (2 * h, w), interpolation=1).astype(np.uint8)
    ___img = np.hstack((___img, np.zeros((h, w, 3), dtype=np.uint8)))

    ____img = np.vstack([_img, ___img])
    cv2.imshow('1', ____img)
    cv2.waitKey(0)
    print(imgs)
    from torch.utils.data.dataloader import DataLoader
    dl = DataLoader(dataset, collate_fn=None)
    return dl


if __name__ == "__main__":
    main()
