from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
from skimage.filters import gaussian, median
from skimage.filters.rank import mean_percentile
from skimage.transform import rescale, resize, downscale_local_mean, swirl, radon
from skimage.util import img_as_float, random_noise
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CelebADataset(Dataset):
  

    def __init__(self, data_frame, transform=None,obfuscation=False, sigma=0):

        self.data_frame = data_frame
        self.transform = transform
        self.obfuscation = obfuscation
        self.sigma = sigma

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.obfuscation:
            image = io.imread(self.data_frame.iloc[idx, 1])
#             obfus_image = random_noise(image, var=1)
#             obfus_image = swirl(image, rotation=0, strength=10, radius=120)
#             obfus_image = resize(image, (15, 15),
#                        anti_aliasing=True)
            obfus_image = gaussian(image, sigma=self.sigma, multichannel=True)
            labels = self.data_frame.iloc[idx, 0]
            labels = np.array(labels)
            labels = labels.astype(np.float)
            sample = {'image': image,'obfuscated': obfus_image, 'labels': labels}
           
        else:     
            image = io.imread(self.data_frame.iloc[idx, 1])
            labels = self.data_frame.iloc[idx, 0]
            labels = np.array(labels)
            labels = labels.astype(np.float)
            sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)
        return sample
    


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size_height,output_size_width,obfuscation=False):
        assert isinstance(output_size_height, (int, tuple))
        assert isinstance(output_size_width, (int, tuple))
        self.output_size_height = output_size_height
        self.output_size_width = output_size_width
        self.obfuscation = obfuscation

    def __call__(self, sample):
        
            
              
        new_h, new_w = self.output_size_height,self.output_size_width
        new_h, new_w = int(new_h), int(new_w)
        if self.obfuscation:
            image, obfus_image, labels = sample['image'],sample['obfuscated'], sample['labels']
            obfus_image = transform.resize(obfus_image, (new_h, new_w))
            img = transform.resize(image, (new_h, new_w))
            return {'image': img,'obfuscated': obfus_image, 'labels': labels}
        else:
            image, labels = sample['image'], sample['labels']
            img = transform.resize(image, (new_h, new_w))
            return {'image': img, 'labels': labels}


    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    


    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}

class ObfuscaionToTensor(object):
    """Convert ndarrays in sample to Tensors."""


    def __call__(self, sample):
        image, obfus_image, labels = sample['image'],sample['obfuscated'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        obfus_image = obfus_image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'obfuscated': torch.from_numpy(obfus_image),
                'labels': torch.from_numpy(labels)}
    

#----------------------------------------------------------------------
class CelebADataset_test(Dataset):
  

    def __init__(self, data_frame, transform=None,obfuscation=False, sigma=0):

        self.data_frame = data_frame
        self.transform = transform
        self.obfuscation = obfuscation
        self.sigma = sigma

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.obfuscation:
            # gray scale image
#             image = io.imread(self.data_frame.iloc[idx, 1], as_gray=True)  
            image = io.imread(self.data_frame.iloc[idx, 1]) 
#             obfus_image = gaussian(image, sigma=self.sigma, multichannel=True)
#             obfus_image = random_noise(image, var=1)
            obfus_image = resize(image, (15, 15),
                       anti_aliasing=True)
            original_image=obfus_image
#             image = np.array(image)
#             obfus_image = np.array(obfus_image)
            labels = self.data_frame.iloc[idx, 0]
            labels = np.array(labels)
            labels = labels.astype(np.float)
            sample = {'image': image,'obfuscated': obfus_image,'original':original_image, 'labels': labels}
           
        else:     
            image = io.imread(self.data_frame.iloc[idx, 1])
            labels = self.data_frame.iloc[idx, 0]
            labels = np.array(labels)
            labels = labels.astype(np.float)
            sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)
        return sample
    



class Rescale_test(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size_height,output_size_width,obfuscation=False):
        assert isinstance(output_size_height, (int, tuple))
        assert isinstance(output_size_width, (int, tuple))
        self.output_size_height = output_size_height
        self.output_size_width = output_size_width
        self.obfuscation = obfuscation

    def __call__(self, sample):
        
            
              
        new_h, new_w = self.output_size_height,self.output_size_width
        new_h, new_w = int(new_h), int(new_w)
        if self.obfuscation:
            image, obfus_image,original_obfus_image, labels = sample['image'],sample['obfuscated'],sample['original'], sample['labels']
#             original_obfus_image = transform.resize(obfus_image, (new_h, new_w))#########
            obfus_image = transform.resize(obfus_image, (new_h, new_w))
            img = transform.resize(image, (new_h, new_w))
            return {'image': img,'obfuscated': obfus_image,'original':original_obfus_image, 'labels': labels}
        else:
            image, labels = sample['image'], sample['labels']
            img = transform.resize(image, (new_h, new_w))
            return {'image': img, 'labels': labels}



class ObfuscaionToTensor_test(object):
    """Convert ndarrays in sample to Tensors."""


    def __call__(self, sample):
        image, obfus_image,original_obfus_image, labels = sample['image'],sample['obfuscated'],sample['original'], sample['labels']
        # gray scale image
#         obfus_image = np.expand_dims(obfus_image, axis=2)
#         image = np.expand_dims(image, axis=2)
#         original_obfus_image = np.expand_dims(original_obfus_image, axis=2)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        obfus_image = obfus_image.transpose((2, 0, 1))
        original_obfus_image = original_obfus_image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'obfuscated': torch.from_numpy(obfus_image),
                'original': torch.from_numpy(original_obfus_image),
                'labels': torch.from_numpy(labels)}
    
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image