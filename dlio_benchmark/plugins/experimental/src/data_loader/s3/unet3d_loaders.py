'''
This module contains classes and functions to load UNDET3D data.
The following classes are defined:
    UNET3DMap: A map-style dataset that loads images from MinIO when the get function is called.
    create_mnist_loader: A function that creates a DataLoader object for MNIST data.
'''
from io import BytesIO
import math
import os
import tarfile
import time
from typing import List, Tuple

import numpy as np
import PIL
from PIL import Image
from s3torchconnector import S3MapDataset, S3IterableDataset, S3Reader
from torchvision import datasets
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

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
        data_bytes = du.get_object_from_minio(self.bucket_name, self.X[index])
        bytes_io = BytesIO(data_bytes)
        with np.load(bytes_io) as data:
            sample = data['x']
            label = data['y']
            sample_tensor = torch.tensor(sample[0:2000, 0:2000,:], dtype=torch.uint8)
            label_tensor = torch.tensor(label, dtype=torch.int64)

        return sample_tensor, label_tensor
        #return torch.tensor([1,2,3], dtype=torch.float16)


class UNET3DIter(IterableDataset):
    '''
    Iterable dataset that returns samples in a streaming fashion.
    '''

    def __init__(self, bucket_name: str, object_list, num_samples: int, comm_size: int, rank: int, batch_size: int):
        logging.info('MyIterableDataset.__init__() called.')
        self.batch_size = batch_size
        self.bucket_name = bucket_name
        self.comm_size = comm_size
        self.num_samples = num_samples
        self.object_list = object_list
        self.rank = rank
        self.rank_start = 0
        self.rank_end = 0

    def _get_process_indices(self):
        '''
        Get the indicies for this worker within the process.
        '''
        # Indecies for the entire process. This is the process created by mpirun.
        samples_per_rank = int(math.ceil(self.num_samples/self.comm_size)) 
        self.rank_start = self.rank * samples_per_rank
        self.rank_end = (self.rank + 1) * samples_per_rank - 1
        if self.rank_end > self.num_samples - 1:
            self.rank_end = self.num_samples - 1
        self.rank_indices = list(range(self.rank_start, self.rank_end + 1))

        # Indecies for this dataloader worker within the process.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self.worker_start = self.rank_start
            self.worker_end = self.rank_end
            worker_id = -1

        else:  # in a worker process
            # split workload
            worker_id = worker_info.id
            per_worker = int(math.ceil((self.rank_end - self.rank_start) / float(worker_info.num_workers)))
            self.worker_start = self.rank_start + (worker_id * per_worker)
            self.worker_end = min(self.worker_start + per_worker, self.rank_end)

        logging.info(f'Rank: {self.rank} process start: {self.process_start} process end: {self.process_end}.')
        logging.info(f'Worker ID: {worker_id} worker_start: {self.worker_start} worker_end: {self.worker_end}.')


    def __iter__(self):
        samples = []
        for index in range(self.worker_start, self.worker_end):
            if index == 0:
                logging.info(f'Rank {self.rank} reading {index} sample.')
            data_bytes = du.get_object_from_minio(self.bucket_name, self.object_list[index], self.minio_client)
            bytes_io = BytesIO(data_bytes)
            with np.load(bytes_io) as data:
                sample = data['x']
                label = data['y']
                #sample_tensor = torch.tensor(sample, dtype=torch.uint8)
                sample_tensor = torch.tensor(sample[0:2000, 0:2000,:], dtype=torch.uint8)
                label_tensor = torch.tensor(label, dtype=torch.int64)
                yield sample_tensor, label_tensor

                #samples.append((sample_tensor, label_tensor))
                #if len(samples) == self.batch_size:
                #    yield samples
                #    samples = []
        
        #if len(samples):
            #return samples
        
        #return iter(range(iter_start, iter_end))


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
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=True, prefetch_factor=prefetch_factor,
                            drop_last=True, shuffle=False)
        return loader, (time.perf_counter()-start_time)

    # The remaining loader types load from S3.
    # Get a list of objects for either train or val.
    X = du.get_unet3d_list(bucket_name, split, smoke_test_count)

    loader = None
    if loader_type == 'map':
        dataset = UNET3DMap(bucket_name, X)
    elif loader_type == 'iter':
        dataset = UNET3DIter(bucket_name, X)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    else:
        raise ValueError('loader_type must be either file, list, full, map or s3map.')

    if loader is None:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=True, prefetch_factor=prefetch_factor,
                            drop_last=True, shuffle=False)

    return loader, (time.perf_counter()-start_time)
