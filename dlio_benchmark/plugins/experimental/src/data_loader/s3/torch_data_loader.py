from io import BytesIO
import logging
import math
import numpy as np
import os
import pathlib
from time import time
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset, RandomSampler, SequentialSampler
from torch.utils.data.sampler import Sampler
from typing import Any, Dict, List, Optional, Tuple, Union

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import Shuffle, DatasetType, DataLoaderType, DataLoaderSampler
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, DLIOMPI
from dlio_benchmark.utils.utility import Profile
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error

#load_dotenv(os.path.join(pathlib.Path(__file__).resolve().parent, 'dlio.env'))
load_dotenv('/home/keithpij/dlio_benchmark/dlio_benchmark/plugins/experimental/src/data_loader/s3/s3.env')
BUCKET_NAME = os.environ['BUCKET_NAME']
TRAINING_PATH = os.environ['TRAINING_PATH']
EVALUATIUON_PATH = os.environ['EVALUATION_PATH']
dlp = Profile(MODULE_DATA_LOADER)


def get_minio_credentials() -> Tuple[str, str, str, bool]:
    endpoint = os.environ['MINIO_ENDPOINT']
    access_key = os.environ['MINIO_ACCESS_KEY']
    secret_key = os.environ['MINIO_SECRET_KEY']
    if os.environ['MINIO_SECURE']=='true': secure = True 
    else: secure = False 
    return (endpoint, access_key, secret_key, secure)


def load_balancer(url: str) -> str:
    '''Super fancy client side load balancer.'''

    match DLIOMPI.get_instance().rank():
        case 0:
            url = url.replace('coehp01', 'coehp01')
        case 1:
            url = url.replace('coehp01', 'coehp02')
        case 2:
            url = url.replace('coehp01', 'coehp03')
        case 3:
            url = url.replace('coehp01', 'coehp04')
        case 4:
            url = url.replace('coehp01', 'coehp01')
        case 5:
            url = url.replace('coehp01', 'coehp02')
        case 6:
            url = url.replace('coehp01', 'coehp03')
        case 7:
            url = url.replace('coehp01', 'coehp04')
        case _:
            logging.error(f'{utcnow()} Unsupported rank {DLIOMPI.get_instance().rank}')

    return url


def get_object_from_minio(bucket_name: str, object_name: str, client: Optional[Union[Minio, None]]=None) -> bytes:

    url, access_key, secret_key, secure = get_minio_credentials()

    # Get data of an object.
    try:
        if client is None:
            # Create client with access and secret key
            client = Minio(url,  # host.docker.internal
                        access_key,  
                        secret_key, 
                        secure=secure)

        response = client.get_object(bucket_name, object_name)
        
        logging.debug(f'Object: {object_name} has been retrieved from bucket: {bucket_name} in MinIO object storage.')

    except S3Error as s3_err:
        logging.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logging.error(f'Error occurred: {err}.')
        raise err

    return response.data


def get_object_list(bucket_name: str, prefix: Optional[Union[str, None]]=None) -> List[str]:
    '''
    Gets a list of objects from a bucket.
    '''

    url, access_key, secret_key, secure = get_minio_credentials()

    try:
        # Create client with access and secret key
        client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=secure)

        object_list = []
        objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)
        for obj in objects:
            object_list.append(obj.object_name)
    except S3Error as s3_err:
        logging.error(f'S3 Error occurred: {s3_err}.')
        raise s3_err
    except Exception as err:
        logging.error(f'Error occurred: {err}.')
        raise err

    return object_list


class dlio_sampler(Sampler):
    '''
    This sampler divides the samples across the ranks. It is a function that is called by the DataLoader to get the next sample.
    It cannot be used with an iterable dataset.
    '''
    def __init__(self, rank, comm_size, num_samples, epochs):
        self.comm_size = comm_size
        self.rank = rank
        self.num_samples = num_samples
        #self.epochs = epochs
        samples_per_proc = int(math.ceil(num_samples/comm_size)) 
        start_sample = self.rank * samples_per_proc
        end_sample = (self.rank + 1) * samples_per_proc - 1
        if end_sample > num_samples - 1:
            end_sample = num_samples - 1
        self.indices = list(range(start_sample, end_sample + 1))

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for sample in self.indices:
            yield sample


class S3MapDataset(Dataset):
    '''
    This dataset expects one sample per object.
    '''
    @dlp.log_init
    def __init__(self, bucket_name, object_list, num_samples):
        self.bucket_name = bucket_name
        self.object_list = object_list
        self.num_samples = num_samples
        self.num_images_read = 0
    
        url, access_key, secret_key, secure = get_minio_credentials()

        url = load_balancer(url)

        logging.info(f'Rank {DLIOMPI.get_instance().rank()} creating Minio client to {url}')

        self.minio_client = Minio(url,
                    access_key,  
                    secret_key, 
                    secure=secure)

    @dlp.log
    def worker_init(self, worker_id):
        logging.info(f'{utcnow()} worker initialized {worker_id}.')

    @dlp.log
    def __len__(self):
        return self.num_samples

    @dlp.log
    def __getitem__(self, index):
        self.num_images_read += 1
        #step = int(math.ceil(self.num_images_read / self.batch_size))
        if index == 0:
            logging.info(f"{utcnow()} Rank {DLIOMPI.get_instance().rank()} reading {index} sample")
        data_bytes = get_object_from_minio(self.bucket_name, self.object_list[index], self.minio_client)
        bytes_io = BytesIO(data_bytes)
        with np.load(bytes_io) as data:
            sample = data['x']
            label = data['y']
            #sample_tensor = torch.tensor(sample, dtype=torch.uint8)
            sample_tensor = torch.tensor(sample[0:2000, 0:2000,:], dtype=torch.uint8)
            label_tensor = torch.tensor(label, dtype=torch.int64)

        return sample_tensor, label_tensor


class S3IterDataset(IterableDataset):
    '''
    Iterable dataset that returns samples in a streaming fashion.
    '''
    def __init__(self, bucket_name: str, object_list, rank: int=0, comm_size: int=1):
        logging.info('UNET3DIter.__init__() called.')
        self.bucket_name = bucket_name
        self.comm_size = comm_size
        self.num_samples = len(object_list)
        self.object_list = object_list
        self.rank = rank
        self.rank_start = 0
        self.rank_end = 0

    def _get_worker_indices(self):
        '''
        Get the indicies for this worker within the process.
        '''
        # Calculate the indecies for each worker within a rank.
        if self.rank: # If renak is non-zero then distributed training is being used or emulated.
            samples_per_rank = int(math.ceil(self.num_samples/self.comm_size)) 
            self.rank_start = self.rank * samples_per_rank
            self.rank_end = (self.rank + 1) * samples_per_rank - 1
            if self.rank_end > self.num_samples - 1:
                self.rank_end = self.num_samples - 1

        else: # rank 0 meaning that distributed training is not being used.
            self.rank_start = 0
            self.rank_end = self.num_samples - 1

        # Indecies for this dataloader worker.
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

        logging.info(f'Rank: {self.rank} Worker ID: {worker_id} start: {self.worker_start} end: {self.worker_end}.')

    def __iter__(self):
        self._get_worker_indices()

        yield_list = []
        for index in range(self.worker_start, self.worker_end):
            if index == 0:
                logging.info(f'Rank {self.rank} reading {index} sample.')
            #start = time.perf_counter()
            data_bytes = get_object_from_minio(self.bucket_name, self.object_list[index])
            bytes_io = BytesIO(data_bytes)
            with np.load(bytes_io) as data:
                samples = data['x']
                labels = data['y']
                for i in range(labels.shape[0]):
                    sample = samples[0:2000, 0:2000, i]
                    label = labels[i]
                    sample_tensor = torch.tensor(sample[0:2000, 0:2000], dtype=torch.uint8)
                    label_tensor = torch.tensor(label, dtype=torch.int64)
                    #yield_list.append((sample_tensor, label_tensor))
                    yield sample_tensor, label_tensor

            #logging.info(f'{self.object_list[index]} retrieved in {time.perf_counter()-start}.')
            #yield iter(yield_list)


class S3TorchDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch_number):
        super().__init__(format_type, dataset_type, epoch_number, DataLoaderType.PYTORCH)

    @dlp.log
    def read(self):
        num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval

        if DatasetType.TRAIN == self.dataset_type:
            object_list = get_object_list(BUCKET_NAME, 'train')
            self._args.total_samples_train = len(object_list)
            logging.info(f'{utcnow()} Rank {self._args.my_rank} reading training data list from bucket {BUCKET_NAME} with {self._args.total_samples_train} samples')
        else:
            object_list = get_object_list(BUCKET_NAME, 'valid')
            self._args.total_samples_eval = len(object_list)
            logging.info(f'{utcnow()} Rank {self._args.my_rank} reading validation data list from bucket {BUCKET_NAME} with {self._args.total_samples_eval} samples')

        if self._args.data_loader_sampler == DataLoaderSampler.INDEX:
            logging.info(f"{utcnow()} Setting up rank {self._args.my_rank} dataloader with {self._args.read_threads} workers, prefetch: {self._args.prefetch_size}.")
            sampler = dlio_sampler(self._args.my_rank, self._args.comm_size, self.num_samples, self._args.epochs)
            dataset = S3MapDataset(BUCKET_NAME, object_list, num_samples)
            self._dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    sampler=sampler,
                                    num_workers=self._args.read_threads,
                                    pin_memory=True,
                                    drop_last=True,
                                    worker_init_fn=dataset.worker_init,
                                    persistent_workers=True,
                                    prefetch_factor=self._args.prefetch_size) #prefetch_factor if prefetch_factor > 0 else 2)  # 2 is the default value

            logging.info(f"{utcnow()} Rank {self._args.my_rank} will read {len(self._dataloader) * batch_size} files")

        else:  #Iterative
            dataset = S3IterDataset(BUCKET_NAME, object_list, self._args.my_rank, self._args.comm_size)
            self._dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    num_workers=self._args.read_threads,
                                    pin_memory=True,
                                    drop_last=True,
                                    persistent_workers=False,
                                    prefetch_factor=self._args.prefetch_size)

    @dlp.log
    def next(self):
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        logging.info(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        for batch in self._dataloader:
            yield batch

    @dlp.log
    def finalize(self):
        pass
