import os
import glob
import itertools
import random
import torch
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class NYUDataset(Dataset):
    """NYU dataset."""

    def __init__(self, data_path, is_train_set, normalize):
        """
        :param data_path: string / Path to the NYU data file directory.
        :param is_train_set: boolean / True - train dataset, False - eval dataset.
        :param normalize: boolean / Whether to perform image normalization.
        """
        self.is_train_set = is_train_set
        transform_list = [
            RandomHFlip(.5),
            SwapChannel(.25),
            ResizeDepth((320, 240)),
            ToTensor(normalize)
        ]

        if is_train_set:
            self.data_path_list = glob.glob(os.path.join(data_path, '*', '*.jpg'))
            self.depth_replace = {'old': '.jpg', 'new': '.png'}
            self.transform = transforms.Compose(transform_list[:])
        else:
            self.data_path_list = glob.glob(os.path.join(data_path, '*_colors.png'))
            self.depth_replace = {'old': 'colors.png', 'new': 'depth.png'}
            self.transform = transforms.Compose(transform_list[2:])

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.data_path_list[idx]
        image_name = os.path.sep.join(image_path.split(os.path.sep)[-2 if self.is_train_set else -1:])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth_path = image_path.replace(self.depth_replace['old'], self.depth_replace['new'])
        depth = np.asarray(Image.open(depth_path))
        # depth = 1000. / np.clip(1000*depth/255., 0., 1000.) if self.is_train_set \
        #     else 1000. / (depth / 10.)
        depth = 1. - ((depth - depth.min()) / (depth.max() - depth.min()))

        sample = {'image': image, 'depth': depth, 'name': image_name}
        sample = self.transform(sample)

        return sample


class RandomHFlip:
    """Flip image and depth horizontally with prob probability."""
    def __init__(self, prob=.5):
        """
        :param prob: float / Probability to flip image and depth.
        """
        self.prob = prob

    def __call__(self, pair_dict):
        image, depth = pair_dict['image'], pair_dict['depth']

        if np.random.uniform() < self.prob:
            image = cv2.flip(image, 1)
            depth = cv2.flip(depth, 1)

        pair_dict.update({'image': image, 'depth': depth})
        return pair_dict


class SwapChannel:
    """Change the order of color channels of image."""
    def __init__(self, prob=.25):
        """
        :param prob: float / Probability to swap color channels.
        """
        self.prob = prob
        self.ch_order = [list(perm) for perm in itertools.permutations([0, 1, 2], 3)]
        self.ch_order.remove([0, 1, 2])

    def __call__(self, pair_dict):
        image, depth = pair_dict['image'], pair_dict['depth']

        if np.random.uniform() < self.prob:
            order = random.choice(self.ch_order)
            image = image[..., order]

        pair_dict.update({'image': image, 'depth': depth})
        return pair_dict


class ResizeDepth:
    """Resize depth map."""
    def __init__(self, size):
        """
        :param size: tuple / new size of depth map.
        """
        assert isinstance(size, (list, tuple))
        assert len(size)==2

        self.size = size

    def __call__(self, pair_dict):
        depth = pair_dict['depth']
        depth = cv2.resize(depth, self.size)

        pair_dict.update({'depth': depth})
        return pair_dict


class ToTensor:
    """Convert image and depth map and pass to device."""
    def __init__(self, normalize):
        """
        :param normalize: boolean / Whether to perform image normalization.
        """
        self.normalize = normalize

    def __call__(self, pair_dict):
        image, depth = pair_dict['image'], pair_dict['depth']
        image = image.transpose((2, 0, 1))
        depth = depth[..., None].transpose((2, 0, 1))

        image = torch.tensor(image, dtype=torch.float)
        depth = torch.tensor(depth, dtype=torch.float)

        if self.normalize:
            image.mul_(1 / 255.)

        pair_dict.update({'image': image, 'depth': depth})
        return pair_dict


if __name__ == '__main__':
    nyu_path = '/mnt/hdd/dataset/NYU_Depth_V2/data'
    train_ds = NYUDataset(os.path.join(nyu_path, 'nyu2_train'), True, True)
    eval_ds = NYUDataset(os.path.join(nyu_path, 'nyu2_test'), False, False)

    train_data = train_ds[0]
    eval_data = eval_ds[0]
    print('train data(normalized)        img - shape: {} / min: {} / max: {}'.format(train_data['image'].size(),
                                                                                      train_data['image'].min(),
                                                                                      train_data['image'].max()))
    print('{:28}depth - shape: {} / min: {} / max: {}'.format(train_data['name'],
                                                              train_data['depth'].size(),
                                                              train_data['depth'].min(),
                                                              train_data['depth'].max()))
    print('eval data(unnormalized)       img - shape: {} / min: {} / max: {}'.format(eval_data['image'].size(),
                                                                                     eval_data['image'].min(),
                                                                                     eval_data['image'].max()))
    print('{:28}depth - shape: {} / min: {} / max: {}'.format(eval_data['name'],
                                                              eval_data['depth'].size(),
                                                              eval_data['depth'].min(),
                                                              eval_data['depth'].max()))

    t_max_depth = 0
    t_min_depth = 99999
    for i, sample in enumerate(train_ds):
        if i % 10000 == 0:
            print(i)
        dep = sample['depth']
        max_depth = max(t_max_depth, dep.max().item())
        min_depth = min(t_min_depth, dep.min().item())

    e_max_depth = 0
    e_min_depth = 99999
    for i, sample in enumerate(eval_ds):
        if i % 10000 == 0:
            print(i)
        dep = sample['depth']
        max_depth = max(e_max_depth, dep.max().item())
        min_depth = min(e_min_depth, dep.min().item())

    print('train depth map - max : {} / min : {}'.format(t_max_depth, t_min_depth))
    print('eval  depth map - max : {} / min : {}'.format(e_max_depth, e_min_depth))