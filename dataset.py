import numbers
import os
import queue as Queue
import threading
from typing import Iterable

import mxnet as mx
import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
transform_image = transforms.Compose([
    transforms.Resize(112,112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
transform_mask =  transforms.Compose([
    transforms.Resize(112,112),
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
    ])
def get_files(base_dir):
    png_files = []
    # Traverse the directory tree
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            # Check if the file has a .png extension
            if file.endswith('.png') or file.endswith('.jpg'):
                # Join the directory path with the file name
                file_path = os.path.join(root, file)
                png_files.append(file_path)
    return png_files
def get_label(file_path,labels_name:list):
    label = file_path.split('/')[-2]
    idx = labels_name.index(label)
    return idx
    
    
class Dataset(data.Dataset):
    def __init__(self,images_list,masks_list,labels_name):
        self.images_list = images_list
        self.masks_list = masks_list
        self.labels_name = labels_name
    def __len__(self):
        return len(self.masks_list)
    def __getitem__(self, index):
        
        image_path = self.images_list[index]
        mask_path = self.masks_list[index]
        
        image = Image.open(image_path).convert('L').convert('RGB')
        mask = Image.open(mask_path)
        
        label = get_label(image_path,self.labels_name)
        
        image_trans = transform_image(image)
        mask_trans = transform_mask(mask)
        
        return image_trans,mask_trans,label
        


def get_dataloader(
    batch_size,
    num_workers = 2,
    root_dir='dataset',
    ) -> Iterable:
    
    list_images = get_files(f'{root_dir}/images')
    list_masks = get_files(f'{root_dir}/masks')
    labels_name = os.listdir(f'{root_dir}/images')
    
    dataset = Dataset(list_images,list_masks,labels_name)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=num_workers)
    return data_loader

