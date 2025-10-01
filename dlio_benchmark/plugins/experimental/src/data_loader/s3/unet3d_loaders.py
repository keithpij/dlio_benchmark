'''
This module contains classes and functions to load MNIST data.
The following classes are defined:
    MNISTMemory: A map-style dataset that loads all images into memory during initialization.
    MNISTMap: A map-style dataset that loads images from MinIO when the get function is called.
    MNISTIterableTar: An iterable dataset that loads images from tar files in MinIO.
    MNISTIterable: An iterable dataset that loads images from MinIO.
    MNISTList: A map-style dataset that loads images from MinIO when the get function is called.
    S3IterTransform: A callable class that transforms S3Reader objects into image tensors.
    IterTransform: A callable class that transforms object paths into image tensors.
    IterTarTransform: A callable class that transforms image tuples into image tensors.
    S3MapTransform: A callable class that transforms S3Reader objects into image tensors.
    create_mnist_loader: A function that creates a DataLoader object for MNIST data.
'''
from io import BytesIO
import math
import os
import tarfile
import time
from typing import List, Tuple

import PIL
from PIL import Image
from s3torchconnector import S3MapDataset, S3IterableDataset, S3Reader
from torchvision import datasets
import torch
from torch.utils.data import Dataset, DataLoader

import data_utilities as du


class UNET3DMap(Dataset):
    '''
    This map-style dataset is initialized with a list of image paths. 
    Each individual image is retrieved when the get function is called.
    '''
    def __init__(self, bucket_name: str, image_list: List[str]):
        self.bucket_name = bucket_name
        self.X = image_list

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img = du.get_object_from_minio(self.bucket_name, self.X[index])
        return img


def create_unet3d_loader(bucket_name: str, split: str, loader_type:str, batch_size:int, num_workers: int=1, prefetch_factor: int=1,
                           smoke_test_count: int=0) -> Tuple[DataLoader, float]:
    '''
    This function creates a DataLoader object for MNIST data.
    Args:
        bucket_name: The name of the MinIO bucket.
        split: A string that indicates whether the data is for training or testing.
        loader_type: A string that indicates the type of loader to use.
        batch_size: The number of images to load in each batch.
        num_workers: The number of worker processes to use.
        smoke_test_count: The number of images to use for a smoke test.
    Returns:
        loader: A DataLoader object that loads the MNIST data.
        load_time: The time it took to load the data.
    '''
    # Start of load time.
    start_time = time.perf_counter()


    if loader_type == 's3map':
        uri = f's3://{bucket_name}/{split}'
        aws_region = os.environ['AWS_REGION']
        dataset = S3MapDataset.from_prefix(uri, region=aws_region)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor, drop_last=True, shuffle=True)
        return loader, (time.perf_counter()-start_time)

    # The remaining loader types load from S3.
    # Get a list of objects for either train or val.
    X = du.get_unet3d_list(bucket_name, split, smoke_test_count)

    loader = None
    if loader_type == 'map':
        dataset = UNET3DMap(bucket_name, X)
    elif loader_type == 'iter':
        pass
        #dataset = UNET3DIterable(bucket_name, X, y, transform=data_transforms)
        #loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    else:
        raise ValueError('loader_type must be either file, list, full, map or s3map.')

    if loader is None:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor, drop_last=True, shuffle=True)

    return loader, (time.perf_counter()-start_time)
