import os

import torch.utils.data as data
import torchvision
from torchvision import transforms

import numpy as np

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

def train_transform(image_size):
    transform_list = [
        transforms.Resize(size=(image_size*2, image_size*2)),
        transforms.RandomCrop(image_size),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31
    
def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0
            
        
class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = os.listdir(self.root)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

class FlatFolderDatasetPair(data.Dataset):
    def __init__(self, root_1, root_2, transform, identity_radio=5):
        super(FlatFolderDatasetPair, self).__init__()
        self.root_1 = root_1
        self.root_2 = root_2
        self.paths_1 = os.listdir(self.root_1)
        self.paths_2 = os.listdir(self.root_2)
        self.identity_radio = identity_radio
        self.transform = transform

    def __getitem__(self, index):
        path_1 = os.path.join(self.root_1, self.paths_1[index])
        path_2 =  os.path.join(self.root_2, self.paths_2[index])
        img_1 = Image.open(path_1).convert('RGB')
        img_2 = Image.open(path_2).convert('RGB')
        img_1 = self.transform(img_1)
        img_2 = self.transform(img_2)
        if bool(np.random.choice((0, 1), p=[1/self.identity_radio, 1-1/self.identity_radio])):
            return img_1, img_2
        else:
            return img_1, img_1

    def __len__(self):
        if len(self.paths_2)>len(self.paths_1):
            len_ = len(self.paths_1)
        else:
            len_ = len(self.paths_2)
        return len_

    def name(self):
        return 'FlatFolderDatasetPair'
