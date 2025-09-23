from time import time
import logging
import math
import os
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.sampler import Sampler
from typing import Any, Dict, List, Optional, Tuple, Union

from dlio_benchmark.common.constants import MODULE_DATA_LOADER
from dlio_benchmark.common.enumerations import Shuffle, DatasetType, DataLoaderType
from dlio_benchmark.data_loader.base_data_loader import BaseDataLoader
from dlio_benchmark.reader.reader_factory import ReaderFactory
from dlio_benchmark.utils.utility import utcnow, DLIOMPI
from dlio_benchmark.utils.utility import Profile
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error


load_dotenv('/home/minio/dlio_benchmark/dlio_benchmark/plugins/experimental/src/data_loader/s3/dlio.env')
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
    def __init__(self, rank, size, num_samples, epochs):
        self.size = size
        self.rank = rank
        self.num_samples = num_samples
        self.epochs = epochs
        samples_per_proc = int(math.ceil(num_samples/size)) 
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


class S3TorchDataset(Dataset):
    '''
    This dataset expects one sample per object.
    '''
    @dlp.log_init
    def __init__(self, bucket_name, object_list, format_type, dataset_type, epoch, num_samples, num_workers, batch_size):
        self.bucket_name = bucket_name
        self.object_list = object_list
        self.format_type = format_type
        self.dataset_type = dataset_type
        self.epoch = epoch
        self.num_samples = num_samples
        self.num_images_read = 0
        self.batch_size = batch_size
        if num_workers == 0:
            self.worker_init(-1)
    
        url, access_key, secret_key, secure = get_minio_credentials()

        # Super fancy client side load balancer.
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

        logging.info(f'Rank {DLIOMPI.get_instance().rank()} creating Minio client to {url}')

        self.minio_client = Minio(url,  # host.docker.internal
                    access_key,  
                    secret_key, 
                    secure=secure)

    @dlp.log
    def worker_init(self, worker_id):
        logging.debug(f"{utcnow()} worker initialized {worker_id} with format {self.format_type}")

    @dlp.log
    def __len__(self):
        return self.num_samples

    @dlp.log
    def __getitem__(self, index):
        self.num_images_read += 1
        step = int(math.ceil(self.num_images_read / self.batch_size))
        if index == 0:
            logging.info(f"{utcnow()} Rank {DLIOMPI.get_instance().rank()} reading {index} sample")
        obj = get_object_from_minio(self.bucket_name, self.object_list[index], self.minio_client)
        return obj


class S3TorchDataLoader(BaseDataLoader):
    @dlp.log_init
    def __init__(self, format_type, dataset_type, epoch_number):
        super().__init__(format_type, dataset_type, epoch_number, DataLoaderType.PYTORCH)

    @dlp.log
    def read(self):
        num_samples = self._args.total_samples_train if self.dataset_type is DatasetType.TRAIN else self._args.total_samples_eval
        batch_size = self._args.batch_size if self.dataset_type is DatasetType.TRAIN else self._args.batch_size_eval

        if DatasetType.TRAIN == self.dataset_type:
            logging.info(f'{utcnow()} Rank {self._args.my_rank} reading training data list from bucket {BUCKET_NAME} with {num_samples} samples')
            object_list = get_object_list(BUCKET_NAME, 'train')
        else:
            logging.info(f'{utcnow()} Rank {self._args.my_rank} reading validation data list from bucket {BUCKET_NAME} with {num_samples} samples')
            object_list = get_object_list(BUCKET_NAME, 'valid')
        dataset = S3TorchDataset(BUCKET_NAME, object_list, self.format_type, self.dataset_type, self.epoch_number, num_samples, self._args.read_threads, batch_size)

        sampler = dlio_sampler(self._args.my_rank, self._args.comm_size, self.num_samples, self._args.epochs)

        #if self._args.read_threads > 1:
        #    prefetch_factor = math.ceil(self._args.prefetch_size / self._args.read_threads)
        #else:
        #    prefetch_factor = self._args.prefetch_size
        #if prefetch_factor > 0:
        #    if self._args.my_rank == 0:
        #        logging.info(f"{utcnow()} Prefetch size is {self._args.prefetch_size}; prefetch factor of {prefetch_factor} will be set to Torch DataLoader.")
        #else:
        #    if self._args.my_rank == 0:
        #        logging.info(f"{utcnow()} Prefetch size is 0; a default prefetch factor of 2 will be set to Torch DataLoader.")

        logging.info(f"{utcnow()} Setting up rank {self._args.my_rank} dataloader with {self._args.read_threads} workers, prefetch: {self._args.prefetch_size}.")

        self._dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                sampler=sampler,
                                num_workers=self._args.read_threads,
                                pin_memory=True,
                                drop_last=True,
                                worker_init_fn=dataset.worker_init,
                                prefetch_factor=self._args.prefetch_size) #prefetch_factor if prefetch_factor > 0 else 2)  # 2 is the default value

        logging.info(f"{utcnow()} Rank {self._args.my_rank} will read {len(self._dataloader) * batch_size} files")

    @dlp.log
    def next(self):
        super().next()
        total = self._args.training_steps if self.dataset_type is DatasetType.TRAIN else self._args.eval_steps
        logging.debug(f"{utcnow()} Rank {self._args.my_rank} should read {total} batches")
        for batch in self._dataloader:
            yield batch

    @dlp.log
    def finalize(self):
        pass
