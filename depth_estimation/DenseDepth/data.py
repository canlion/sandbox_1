import os
import glob
import itertools
import random
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class NYUDataset(Dataset):
    """NYU dataset."""

    def __init__(self, data_path, is_train_set, device, normalize):
        """
        :param data_path: string / Path to the NYU data file directory.
        :param is_train_set: boolean / True - train dataset, False - eval dataset.
        :param device: string / Name of device for operation.
        :param normalize: boolean / Whether to perform image normalization.
        """
        self.is_train_set = is_train_set
        self.transform = transforms.Compose([
            RandomHFlip(.5),
            SwapChannel(.25),
            ResizeDepth((320, 240)),
            ToTensor(device, normalize)
        ])

        if is_train_set:
            self.data_path_list = glob.glob(os.path.join(data_path, '*', '*.jpg'))
            self.depth_replace = {'old': '.jpg', 'new': '.png'}
        else:
            self.data_path_list = glob.glob(os.path.join(data_path, '*_colors.png'))
            self.depth_replace = {'old': 'colors.png', 'new': 'depth.png'}

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.data_path_list[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth_path = image_path.replace(self.depth_replace['old'], self.depth_replace['new'])
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth = np.clip(depth, .4, 10.)

        sample = {'image': image, 'depth': depth}
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

        return {'image': image, 'depth': depth}


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

        return {'image': image, 'depth': depth}


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
        image, depth = pair_dict['image'], pair_dict['depth']
        depth = cv2.resize(depth, self.size)

        return {'image': image, 'depth': depth}


class ToTensor:
    """Convert image and depth map and pass to device."""
    def __init__(self, device, normalize):
        """
        :param device: torch.device / Device for operation.
        :param normalize: boolean / Whether to perform image normalization.
        """
        self.device = device
        self.normalize = normalize

    def __call__(self, pair_dict):
        image, depth = pair_dict['image'], pair_dict['depth']
        image = image.transpose((2, 0, 1))
        depth = depth[..., None].transpose((2, 0, 1))

        image = torch.tensor(image, dtype=torch.float)
        depth = torch.tensor(depth, dtype=torch.float)
        depth.clamp_(.4, 10.)

        if self.normalize:
            image.mul_(1 / 255.)

        return {'image': image.to(self.device), 'depth': depth.to(self.device)}


if __name__ == '__main__':
    nyu_path = '/mnt/hdd/dataset/NYU_Depth_V2/data'
    train_ds = NYUDataset(os.path.join(nyu_path, 'nyu2_train'), True, 'cpu', True)
    eval_ds = NYUDataset(os.path.join(nyu_path, 'nyu2_test'), False, 'cpu', False)

    train_data = train_ds[0]
    eval_data = eval_ds[0]
    print('train data(normalized)    img - shape: {} / min: {} / max: {}'.format(train_data['image'].size(),
                                                                                 train_data['image'].min(),
                                                                                 train_data['image'].max()))
    print('                        depth - shape: {} / min: {} / max: {}'.format(train_data['depth'].size(),
                                                                                 train_data['depth'].min(),
                                                                                 train_data['depth'].max()))
    print('eval data(unnormalized)   img - shape: {} / min: {} / max: {}'.format(eval_data['image'].size(),
                                                                                 eval_data['image'].min(),
                                                                                 eval_data['image'].max()))
    print('                        depth - shape: {} / min: {} / max: {}'.format(eval_data['depth'].size(),
                                                                                 eval_data['depth'].min(),
                                                                                 eval_data['depth'].max()))
