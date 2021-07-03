# coding: utf-8
'''
Code is referred from https://github.com/klintan/pytorch-lanenet
delete the one-hot representation for instance output
'''

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np

from torchvision.transforms import ToTensor
from torchvision import datasets, transforms

import random


class TusimpleSet(Dataset):
    def __init__(self, dataset, n_labels=3, transform=None, target_transform=None):
        self._gt_img_list = []
        self._gt_label_binary_list = []
        self._gt_label_instance_list = []
        self.transform = transform
        self.target_transform = target_transform
        self.n_labels = n_labels

        with open(dataset, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()

                self._gt_img_list.append(info_tmp[0])
                self._gt_label_binary_list.append(info_tmp[1])
                self._gt_label_instance_list.append(info_tmp[2])

        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(zip(self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list))
        random.shuffle(c)
        self._gt_img_list, self._gt_label_binary_list, self._gt_label_instance_list = zip(*c)

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) \
               == len(self._gt_img_list)

        # load all

        img = Image.open(self._gt_img_list[idx])
        label_instance_img = cv2.imread(self._gt_label_instance_list[idx], cv2.IMREAD_UNCHANGED)
        label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.IMREAD_COLOR)

        # optional transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label_img = self.target_transform(label_img)
            label_instance_img = self.target_transform(label_instance_img)

        label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        label_binary[mask] = 1

        # we could split the instance label here, each instance in one channel (basically a binary mask for each)
        return img, label_binary, label_instance_img
