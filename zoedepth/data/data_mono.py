# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

# This file is partly inspired from BTS (https://github.com/cleinc/bts/blob/master/pytorch/bts_dataloader.py); author: Jin Han Lee

import itertools
import os
import random

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.utils.data.distributed
from zoedepth.utils.easydict import EasyDict as edict
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from zoedepth.utils.config import change_dataset

from .ddad import get_ddad_loader
from .diml_indoor_test import get_diml_indoor_loader
from .diml_outdoor_test import get_diml_outdoor_loader
from .diode import get_diode_loader
from .hypersim import get_hypersim_loader
from .ibims import get_ibims_loader
from .sun_rgbd_loader import get_sunrgbd_loader
from .vkitti import get_vkitti_loader
from .vkitti2 import get_vkitti2_loader

from .preprocess import CropParams, get_white_border, get_black_border


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode, **kwargs):
    return transforms.Compose([
        ToTensor(mode=mode, **kwargs)
    ])


class DepthDataLoader(object):
    def __init__(self, config, mode, device='cpu', transform=None, **kwargs):
        """
        Data loader for depth datasets

        Args:
            config (dict): Config dictionary. Refer to utils/config.py
            mode (str): "train" or "online_eval"
            device (str, optional): Device to load the data on. Defaults to 'cpu'.
            transform (torchvision.transforms, optional): Transform to apply to the data. Defaults to None.
        """

        self.config = config

        if config.dataset == 'ibims':
            self.data = get_ibims_loader(config, batch_size=1, num_workers=1)
            return

        if config.dataset == 'sunrgbd':
            self.data = get_sunrgbd_loader(
                data_dir_root=config.sunrgbd_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_indoor':
            self.data = get_diml_indoor_loader(
                data_dir_root=config.diml_indoor_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_outdoor':
            self.data = get_diml_outdoor_loader(
                data_dir_root=config.diml_outdoor_root, batch_size=1, num_workers=1)
            return

        if "diode" in config.dataset:
            self.data = get_diode_loader(
                config[config.dataset+"_root"], batch_size=1, num_workers=1)
            return

        if config.dataset == 'hypersim_test':
            self.data = get_hypersim_loader(
                config.hypersim_test_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti':
            self.data = get_vkitti_loader(
                config.vkitti_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'vkitti2':
            self.data = get_vkitti2_loader(
                config.vkitti2_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'ddad':
            self.data = get_ddad_loader(config.ddad_root, resize_shape=(
                352, 1216), batch_size=1, num_workers=1)
            return

        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None

        if transform is None:
            transform = preprocessing_transforms(mode, size=img_size)

        if mode == 'train':

            Dataset = DataLoadPreprocess
            self.training_samples = Dataset(
                config, mode, transform=transform, device=device)

            if config.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples,
                                   batch_size=config.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=config.workers,
                                   pin_memory=True,
                                   persistent_workers=True,
                                #    prefetch_factor=2,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(
                config, mode, transform=transform)
            if config.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=kwargs.get("shuffle_test", False),
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(
                config, mode, transform=transform)
            self.data = DataLoader(self.testing_samples,
                                   1, shuffle=False, num_workers=1)

        else:
            print(
                'mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def repetitive_roundrobin(*iterables):
    """
    cycles through iterables but sample wise
    first yield first sample from first iterable then first sample from second iterable and so on
    then second sample from first iterable then second sample from second iterable and so on

    If one iterable is shorter than the others, it is repeated until all iterables are exhausted
    repetitive_roundrobin('ABC', 'D', 'EF') --> A D E B D F C D E
    """
    # Repetitive roundrobin
    iterables_ = [iter(it) for it in iterables]
    exhausted = [False] * len(iterables)
    while not all(exhausted):
        for i, it in enumerate(iterables_):
            try:
                yield next(it)
            except StopIteration:
                exhausted[i] = True
                iterables_[i] = itertools.cycle(iterables[i])
                # First elements may get repeated if one iterable is shorter than the others
                yield next(iterables_[i])


class RepetitiveRoundRobinDataLoader(object):
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return repetitive_roundrobin(*self.dataloaders)

    def __len__(self):
        # First samples get repeated, thats why the plus one
        return len(self.dataloaders) * (max(len(dl) for dl in self.dataloaders) + 1)


class MixedNYUKITTI(object):
    def __init__(self, config, mode, device='cpu', **kwargs):
        config = edict(config)
        config.workers = config.workers // 2
        self.config = config
        nyu_conf = change_dataset(edict(config), 'nyu')
        kitti_conf = change_dataset(edict(config), 'kitti')

        # make nyu default for testing
        self.config = config = nyu_conf
        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None
        if mode == 'train':
            nyu_loader = DepthDataLoader(
                nyu_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            kitti_loader = DepthDataLoader(
                kitti_conf, mode, device=device, transform=preprocessing_transforms(mode, size=img_size)).data
            # It has been changed to repetitive roundrobin
            self.data = RepetitiveRoundRobinDataLoader(
                nyu_loader, kitti_loader)
        else:
            self.data = DepthDataLoader(nyu_conf, mode, device=device).data


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class CachedReader:
    def __init__(self, shared_dict=None):
        if shared_dict:
            self._cache = shared_dict
        else:
            self._cache = {}

    def open(self, fpath):
        im = self._cache.get(fpath, None)
        if im is None:
            im = self._cache[fpath] = Image.open(fpath)
        return im


class ImReader:
    def __init__(self):
        pass

    # @cache
    def open(self, fpath):
        return Image.open(fpath)


class DataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None, is_for_online_eval=False, **kwargs):
        self.config = config
        if mode == 'online_eval':
            with open(config.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(config.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor(mode)
        self.is_for_online_eval = is_for_online_eval
        if config.use_shared_dict:
            self.reader = CachedReader(config.shared_dict)
        else:
            self.reader = ImReader()

    def postprocess(self, sample):
        return sample

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        sample = {}

        image_path = [None]*4
        depth_path = [None]*4

        if self.mode == 'train':
            if self.config.dataset == 'kitti' and self.config.use_right and random.random() > 0.5:
                image_path = os.path.join(
                    self.config.data_path, remove_leading_slash(sample_path.split()[3]))
                depth_path = os.path.join(
                    self.config.gt_path, remove_leading_slash(sample_path.split()[4]))
            else:
                for i in range(4):
                    image_path[i] = os.path.join(
                        self.config.data_path, remove_leading_slash(sample_path.split()[0].replace('_p', '_p'+str(i+1))))
                    depth_path[i] = os.path.join(
                        self.config.gt_path, remove_leading_slash(sample_path.split()[1].replace('_gt', '_gt'+str(i+1))))
                
            image = [None]*4
            depth_gt = [None]*4
            mask = [None]*4

            for i in range(4):
                image[i] = self.reader.open(image_path[i])
                depth_gt[i] = self.reader.open(depth_path[i])
            w, h = image[0].size

            if self.config.do_kb_crop:    #nyu不干这个
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # Avoid blank boundaries due to pixel registration?
            # Train images have white border. Test images have black border.
            if self.config.dataset == 'nyu' and self.config.avoid_boundary: #nyu不干这个
                # print("Avoiding Blank Boundaries!")
                # We just crop and pad again with reflect padding to original size
                # original_size = image.size
                crop_params = get_white_border(np.array(image, dtype=np.uint8))
                image = image.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))
                depth_gt = depth_gt.crop((crop_params.left, crop_params.top, crop_params.right, crop_params.bottom))

                # Use reflect padding to fill the blank
                image = np.array(image)
                image = np.pad(image, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right), (0, 0)), mode='reflect')
                image = Image.fromarray(image)

                depth_gt = np.array(depth_gt)
                depth_gt = np.pad(depth_gt, ((crop_params.top, h - crop_params.bottom), (crop_params.left, w - crop_params.right)), 'constant', constant_values=0)
                depth_gt = Image.fromarray(depth_gt)


            if self.config.do_random_rotate and (self.config.aug): #nyu不干这个
                random_angle = (random.random() - 0.5) * 2 * self.config.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(
                    depth_gt, random_angle, flag=Image.NEAREST)

            
            for i in range(4):
                image[i] = np.asarray(image[i], dtype=np.float32) / 255.0
                depth_gt[i] = np.asarray(depth_gt[i], dtype=np.float32)
                depth_gt[i] = np.expand_dims(depth_gt[i], axis=2)

                if self.config.dataset == 'nyu':
                    depth_gt[i] = depth_gt[i] / 1000.0
                else:
                    depth_gt[i] = depth_gt[i] / 256.0

                image[i], depth_gt[i] = self.train_preprocess(image[i], depth_gt[i])  #好像也是数据增强

                mask[i] = np.logical_and(depth_gt[i] > self.config.min_depth,
                                    depth_gt[i] < self.config.max_depth).squeeze()[None, ...]
                

            top_image = np.concatenate([image[0],image[2]], axis=1)  # 左上 + 右上
            bottom_image = np.concatenate([image[1],image[3]], axis=1)  # 左下 + 右下
            image4 = np.concatenate([top_image, bottom_image], axis=0)  # 上下拼接

            top_depth = np.concatenate([depth_gt[0],depth_gt[2]], axis=1)  # 左上 + 右上
            bottom_depth = np.concatenate([depth_gt[1],depth_gt[3]], axis=1)  # 左下 + 右下
            depth_gt4 = np.concatenate([top_depth, bottom_depth], axis=0)  # 上下拼接

            top_mask = np.concatenate([mask[0],mask[2]], axis=2)  # 左上 + 右上
            bottom_mask = np.concatenate([mask[1],mask[3]], axis=2)  # 左下 + 右下
            mask4 = np.concatenate([top_mask, bottom_mask], axis=1)  # 上下拼接

            sample = {'image': image4, 'depth': depth_gt4, 'focal': focal,
                        'mask': mask4, **sample}

        else:  #先nmlgbd解决train的事儿
            if self.mode == 'online_eval':
                data_path = self.config.data_path_eval
            else:
                data_path = self.config.data_path

            image = [None]*4
            for i in range(4):
                image_path = os.path.join(
                    data_path, remove_leading_slash(sample_path.split()[0].replace('_p', '_p'+str(i+1))))
                image[i] = np.asarray(self.reader.open(image_path),
                                dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                depth_gt = [None]*4
                mask = [None]*4
                for i in range(4):                
                    gt_path = self.config.gt_path_eval
                    depth_path = os.path.join(
                        gt_path, remove_leading_slash(sample_path.split()[1].replace('_gt', '_gt'+str(i+1))))
                    has_valid_depth = False
                    try:
                        depth_gt[i] = self.reader.open(depth_path)
                        has_valid_depth = True
                    except IOError:
                        depth_gt[i] = False
                        # print('Missing gt for {}'.format(image_path))

                    if has_valid_depth:
                        depth_gt[i] = np.asarray(depth_gt[i], dtype=np.float32)
                        depth_gt[i] = np.expand_dims(depth_gt[i], axis=2)
                        if self.config.dataset == 'nyu':
                            depth_gt[i] = depth_gt[i] / 1000.0
                        else:
                            depth_gt[i] = depth_gt[i] / 256.0

                        mask[i] = np.logical_and(
                            depth_gt[i] >= self.config.min_depth, depth_gt[i] <= self.config.max_depth).squeeze()[None, ...]
                    else:
                        mask[i] = False
                

            # if self.config.do_kb_crop:
            #     height = image.shape[0]
            #     width = image.shape[1]
            #     top_margin = int(height - 352)
            #     left_margin = int((width - 1216) / 2)
            #     image = image[top_margin:top_margin + 352,
            #                   left_margin:left_margin + 1216, :]
            #     if self.mode == 'online_eval' and has_valid_depth:
            #         depth_gt = depth_gt[top_margin:top_margin +
            #                             352, left_margin:left_margin + 1216, :]
            
            top_image = np.concatenate([image[0],image[2]], axis=1)  # 左上 + 右上
            bottom_image = np.concatenate([image[1],image[3]], axis=1)  # 左下 + 右下
            image4 = np.concatenate([top_image, bottom_image], axis=0)  # 上下拼接

            top_depth = np.concatenate([depth_gt[0],depth_gt[2]], axis=1)  # 左上 + 右上
            bottom_depth = np.concatenate([depth_gt[1],depth_gt[3]], axis=1)  # 左下 + 右下
            depth_gt4 = np.concatenate([top_depth, bottom_depth], axis=0)  # 上下拼接

            top_mask = np.concatenate([mask[0],mask[2]], axis=2)  # 左上 + 右上
            bottom_mask = np.concatenate([mask[1],mask[3]], axis=2)  # 左下 + 右下
            mask4 = np.concatenate([top_mask, bottom_mask], axis=1)  # 上下拼接


            if self.mode == 'online_eval':
                sample = {'image': image4, 'depth': depth_gt4, 'focal': focal, 'has_valid_depth': has_valid_depth,
                          'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1],
                          'mask': mask4}
            else:
                sample = {'image': image4, 'focal': focal}

        # if (self.mode == 'train') or ('has_valid_depth' in sample and sample['has_valid_depth']):
        #     mask[i] = np.logical_and(depth_gt[i] > self.config.min_depth,
        #                           depth_gt[i] < self.config.max_depth).squeeze()[None, ...]
            
        #     sample['mask'] = mask4

        if self.transform:
            sample = self.transform(sample)

        sample = self.postprocess(sample)  #没逼用
        sample['dataset'] = self.config.dataset
        sample = {**sample, 'image_path': sample_path.split()[0], 'depth_path': sample_path.split()[1]}

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        return img, depth
    
    def random_translate(self, img, depth, max_t=20):
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        p = self.config.translate_prob
        do_translate = random.random()
        if do_translate > p:
            return img, depth
        x = random.randint(-max_t, max_t)
        y = random.randint(-max_t, max_t)
        M = np.float32([[1, 0, x], [0, 1, y]])
        # print(img.shape, depth.shape)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        depth = cv2.warpAffine(depth, M, (depth.shape[1], depth.shape[0]))
        depth = depth.squeeze()[..., None]  # add channel dim back. Affine warp removes it
        # print("after", img.shape, depth.shape)
        return img, depth

    def train_preprocess(self, image, depth_gt):
        if self.config.aug:
            # Random flipping
            do_flip = random.random()
            if do_flip > 0.5:
                image = (image[:, ::-1, :]).copy()
                depth_gt = (depth_gt[:, ::-1, :]).copy()

            # Random gamma, brightness, color augmentation
            do_augment = random.random()
            if do_augment > 0.5:
                image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.config.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode, do_normalize=False, size=None):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if do_normalize else nn.Identity()
        self.size = size
        if size is not None:
            self.resize = transforms.Resize(size=size)
        else:
            self.resize = nn.Identity()

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)
        image = self.resize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {**sample, 'image': image, 'depth': depth, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            image = self.resize(image)
            return {**sample, 'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth,
                    'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
