'''
This module drives the MNIST training options.
'''
import argparse
from io import BytesIO
from multiprocessing import get_context
import os
import time
from typing import Any, Dict, List, Sequence, Tuple

from dotenv import load_dotenv
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data.dataloader import DataLoader

import data_utilities as du
import environment as env
from training_base import TrainingBase
import unet3d_loaders as ul


# Load the credentials and connection information.
load_dotenv('s3.env')
BUCKET_NAME = os.environ['BUCKET_NAME']
CHECKPOINT_BUCKET = os.environ['CHECKPOINT_BUCKET']
COMPUTATION_TIME = os.environ['COMPUTATION_TIME']


class TrainUNET3D(TrainingBase):
    '''
    Class that contains training and testing methods for both local processing and distributed
    processing.
    '''

    def train_model_local(self, model: nn.Module, loader: DataLoader, parameters: Dict[str, Any]) -> None:
        '''
        Trains a model with the specified data loader.
        '''
        logger = du.get_logger()

        training_start_time = time.perf_counter()

        #model.to(self.device)
        #self.logger.info(f'Model moved to device: {self.device}')

        # Define the loss function and optimizer..
        #loss_func = nn.NLLLoss()
        #optimizer = optim.SGD(model.parameters(), lr=parameters['lr'], momentum=parameters['momentum'])

        # Epoch loop
        for epoch in range(parameters['epochs']):

            # Initialize epic based metrics.
            total_loss = 0
            epoch_compute_time: float = 0.0
            epoch_io_time: float = 0.0
            epoch_device_transfer_time: float = 0.0
            io_start: float = time.perf_counter()
            batch_count:int = 0
            epoch_byte_size:int = 0

            # Batch loop
            logger.info(f'Starting data retrieval for epoch: {epoch+1} batch: {batch_count+1}.')
            for samples, labels in loader:
                logger.info(f'Data retrieval complete for epoch: {epoch+1} batch: {batch_count+1}.')
                # IO time for the batch.
                batch_io_time = time.perf_counter() - io_start
                # Iterable datasets do not have a __len__ method since batches are determined dynamically.
                batch_count += 1

                batch_byte_size: int = 0
                #for sample in samples:
                #    batch_byte_size += len(sample)

                # samples
                number_of_elements = samples.numel()
                element_byte_size = samples.element_size()
                batch_byte_size = number_of_elements * element_byte_size
                # labels
                number_of_elements = labels.numel()
                element_byte_size = labels.element_size()
                batch_byte_size += number_of_elements * element_byte_size

                epoch_byte_size += batch_byte_size

                #device_start = time.perf_counter()
                #images, labels = images.to(self.device), labels.to(self.device)
                #batch_device_transfer_time = time.perf_counter() - device_start

                # Start of compute time for the batch.
                compute_start = time.perf_counter()
                time.sleep(float(parameters['computation_time']))  # Simulate GPU compute time.

                # Training pass
                #optimizer.zero_grad()
                #output = model(images)
                #loss = loss_func(output, labels)
                #loss.backward()
                #optimizer.step()

                # Loss calculations
                #total_loss += loss.item()

                # Track timing metrics.
                batch_compute_time = time.perf_counter() - compute_start
                epoch_compute_time += batch_compute_time
                #epoch_device_transfer_time += batch_device_transfer_time
                epoch_io_time += batch_io_time

                utilization = (batch_compute_time / (batch_compute_time+batch_io_time)) * 100
                self.logger.info(f'Epoch {epoch+1} - ' \
                            f'Batch {batch_count} - ' \
                            f'Batch Size (bytes): {(batch_byte_size/1e9):.4f}Gb - ' \
                            f'Compute time: {batch_compute_time:.4f} - ' \
                            f'IO time: {batch_io_time:.4f} - ' \
                            #f'Device Transfer time: {batch_device_transfer_time:.4f} - ' \
                            f'Utilization: {utilization:.2f}%.')

                # Need to set this here for the next loop.
                io_start = time.perf_counter()
                logger.info(f'Starting data retrieval for epoch: {epoch+1} batch: {batch_count+1}.')

            self.logger.info(f'Epoch {epoch+1} - ' \
                             f'Compute time: {epoch_compute_time:.4f} - ' \
                             f'IO time: {epoch_io_time:.4f}.')

            io_bandwidth = ((epoch_byte_size/epoch_io_time) * 8) / 1e9
            compute_bandwidth = ((epoch_byte_size/epoch_compute_time) * 8) / 1e9
            self.logger.info(f'Epoch {epoch+1} - ' \
                             f'Compute bandwidth: {compute_bandwidth:.4f} Gbps - ' \
                             f'IO bandwidth: {io_bandwidth:.4f} Gbps.')

            self.training_metrics.batches = batch_count
            self.training_metrics.compute_time += epoch_compute_time
            self.training_metrics.io_time += epoch_io_time
            #self.training_metrics.device_transfer_time += epoch_device_transfer_time

            # Save checkpoint to S3 if enabled.
            if parameters['checkpoint']:
                self.checkpoint_model(model)

        self.training_metrics.training_time = time.perf_counter() - training_start_time

    def run_local_training(self, parameters: Dict[str, Any], loader_type: str):
        '''
        Set up the model and the data loader for local training.
        '''
        self.logger.info(f'Loader type: {loader_type}')
        run_start_time = time.perf_counter()

        # No model for now.
        model = None

        # Create the training loader.
        split = 'train'
        train_loader, load_time = ul.create_unet3d_loader(parameters['bucket_name'], split,
                                                            loader_type, parameters['batch_size'],
                                                            num_workers=parameters['num_workers'],
                                                            prefetch_factor=parameters['prefetch_factor'],
                                                            smoke_test_count=parameters['smoke_test_count'])
        self.logger.info(f'Training Loader Creation Time (in seconds) = {load_time:.4f}')
        self.training_metrics.load_time = load_time

        # Iterable datasets do not have a __len__ method since batches are determined dynamically.
        if not 'iter' in loader_type:
            self.training_metrics.dataset_size = len(train_loader.dataset)

        # Train the model.
        self.train_model_local(model, train_loader, parameters)
        self.training_metrics.run_time = time.perf_counter() - run_start_time
        self.log_training_metrics()


def single_process(bucket_name: str, split: str) -> float:
    '''
    Simple single threaded network test.
    '''
    # Get a list of objects for either train or val.
    X = du.get_unet3d_list(bucket_name, split, smoke_test_count=0)

    object_bandwidths = []
    total_bytes = 0
    total_io_time = 0
    for object_path in X:
        start = time.perf_counter()
        data_bytes = du.get_object_from_minio(bucket_name, object_path)
        io_time = time.perf_counter() - start
        total_io_time += io_time
        byte_size = len(data_bytes)
        total_bytes += byte_size
        object_bandwidth = ((byte_size/io_time) * 8) / 1e9 # Gbps
        object_bandwidths.append(object_bandwidth)
        bytes_io = BytesIO(data_bytes)
        with np.load(bytes_io) as data:
            sample = data['x']
            label = data['y']
            sample_tensor = torch.tensor(sample, dtype=torch.uint8)
            label_tensor = torch.tensor(label, dtype=torch.int64)

    return total_bytes, total_io_time, object_bandwidths


def _worker(worker_id: int, samples: Sequence[str], out_q) -> None:
    '''
    Worker run in a subprocess: loops through each character of each string
    and reports elapsed time back via out_q as a tuple (idx, elapsed_seconds).
    '''
    # Create the logger.
    logger = du.create_logger(use_file=False)

    object_bandwidths = []
    total_bytes = 0
    total_io_time = 0
    for object_path in samples:
        start = time.perf_counter()
        logger.info(f'Process {worker_id} retrieving {object_path}.')
        try:
            data_bytes = du.get_object_from_minio(BUCKET_NAME, object_path)
            io_time = time.perf_counter() - start
            total_io_time += io_time
            byte_size = len(data_bytes)
            total_bytes += byte_size
            object_bandwidth = byte_size/io_time  #((byte_size/io_time) * 8) / 1e9 # Gbps
            object_bandwidths.append(object_bandwidth)
            #bytes_io = BytesIO(data_bytes)
            #with np.load(bytes_io) as data:
            #    sample = data['x']
            #    label = data['y']
            #    sample_tensor = torch.tensor(sample, dtype=torch.uint8)
            #    label_tensor = torch.tensor(label, dtype=torch.int64)

        except Exception as err:
            logger.error(f'Error occurred retrieving {object_path}. Error: {err}.')
            #raise err

    out_q.put((worker_id, total_bytes, total_io_time, object_bandwidths))


def multi_process(object_list: List[str], num_workers: int) -> Dict[int, float]:
    '''
    Create 8 subprocesses and send each subprocess a list of strings.
    Each subprocess loops through the characters of each string and reports
    back how long its loop took.

    Args:
      lists_of_strings: list of lists of strings. If fewer than 8 lists are
                        provided the remaining processes receive empty lists.
                        If more than 8 are provided only the first 8 are used.

    Returns:
      Dict mapping process index (0..7) -> elapsed seconds.
    '''
    logger = du.get_logger()

    ctx = get_context('spawn')  # safe on macOS without requiring __main__ guard
    Queue = ctx.Queue
    Process = ctx.Process

    # Ensure exactly 8 tasks
    #tasks = lists_of_strings[:8] + [[] for _ in range(max(0, 8 - len(lists_of_strings)))]
    tasks = []
    total_length = len(object_list)

    # Need to manage the remainder so objects are not ommitted.
    objects_per_worker = (total_length // num_workers)
    remainder = total_length % num_workers

    for i in range(num_workers):
        start_index = i * objects_per_worker
        end_index = min(start_index + objects_per_worker, total_length)
        if remainder:
            end_index += 1
            remainder -= 1

        tasks.append(object_list[start_index:end_index])

    logger.debug(f'Tasks by worker: {tasks}.')

    q = Queue()
    procs = []
    start = time.perf_counter()
    for i, task in enumerate(tasks):
        p = Process(target=_worker, args=(i, task, q))
        p.start()
        procs.append(p)

    results: Dict[int, Tuple[float, float, List[float]]] = {}
    # collect one result per started process
    for _ in procs:
        worker_id, total_bytes, total_io_time, object_bandwidths = q.get()
        results[worker_id] = (total_bytes, total_io_time, object_bandwidths)

    for p in procs:
        p.join()
    
    final_results = {i: results.get(i, 0.0) for i in range(num_workers)}
    wall_clock_time = time.perf_counter() - start

    return wall_clock_time, objects_per_worker, final_results


def batch_test(parameters: Dict[str, Any], loader_type: str):
    '''
    Set up the model and the data loader for local training.
    '''
    # Create the training loader.
    split = 'train'
    train_loader, load_time = ul.create_unet3d_loader(parameters['bucket_name'], split,
                                                        loader_type, parameters['batch_size'],
                                                        num_workers=parameters['num_workers'],
                                                        prefetch_factor=parameters['prefetch_factor'],
                                                        smoke_test_count=parameters['smoke_test_count'])

    def next():
        for batch in train_loader:
            yield batch

    batch_count:int = 0
    for batch in train_loader:
        batch_count += 1
        print(f'Batch {batch_count} retrieved.')


def main():
    '''
    Main function that processes the arguments sent to this module via the command line.
    '''
    # Create the logger.
    du.create_logger(use_file=False)

    # Setup the command line options.
    parser = argparse.ArgumentParser(description='Simulation Command line interface.')
    parser.add_argument('-train', '--train', help='Train the UNET3D model.', action='store_true')
    parser.add_argument('-lt', '--loader_type', help='Type of loader to use for loading training and test sets ' \
                        '(map, iter, s3map or s3iter).')

    parser.add_argument('-sp', '--single_process', help='Single process test.')
    parser.add_argument('-mp', '--multi_process', help='Multi-process test.')
    parser.add_argument('-bt', '--batch_test', help='Batch test.', action='store_true')
    parser.add_argument('-mu', '--memory_usage', help='Memory usage.', action='store_true')

    args = parser.parse_args()

    if args.memory_usage:
        memory_usage = env.get_memory_usage()
        print(f'Total: {memory_usage["total"]} bytes')
        print(f'Available: {memory_usage["available"]} bytes')
        print(f'Used: {memory_usage["used"]} bytes')
        print(f'Free: {memory_usage["free"]} bytes')
        print(f'Percent: {memory_usage["percent"]}%')
        print(f'Cached: {memory_usage["cached"]} bytes')
        print(f'Buffers: {memory_usage["buffers"]} bytes')
        print(f'Swap Total: {memory_usage["swap_total"]} bytes')
        print(f'Swap Used: {memory_usage["swap_used"]} bytes')  

    if args.single_process:
        total_bytes, total_io_time, object_bandwidths = single_process(args.single_process, 'train')
        print(f'Total IO Time (in seconds) = {total_io_time:.4f}')
        print(f'Total dataset size (in bytes) = {total_bytes / 1e9:.4f}')
        print(f'Bandwidth: {((total_bytes/total_io_time) * 8) / 1e9:.4f} Gbps') # Gbps
        print(f'Number of objects = {len(object_bandwidths)}')
        print(f'Max object bandwidth (in Gbps) = {max(object_bandwidths):.4f} Gbps')
        print(f'Avg object bandwidth (in Gbps) = {sum(object_bandwidths)/len(object_bandwidths):.4f} Gbps')
        print(f'Min object bandwidth (in Gbps) = {min(object_bandwidths):.4f} Gbps')

    if args.multi_process:
        object_list = du.get_unet3d_list(BUCKET_NAME, 'train', smoke_test_count=0)

        wall_clock_time, objects_per_worker, results = multi_process(object_list, int(args.multi_process))
        print(f'Total Wall-clock Time: {wall_clock_time:.6f}s')
        print(f'Objects per worker: {objects_per_worker}')

        gb = 1e9 # 1024*1024*1024  # 1e9
        total_bytes = 0
        for worker_id in sorted(results):
            r = results[worker_id]
            total_bytes += r[0]
            print(f'Process {worker_id}: {r[0]/gb:.2f} GB - IO time: {r[1]:.2f} s')
        print(f'Total dataset size: {total_bytes / gb:.2f} GB')
        print(f'Bandwidth (Gbps): {((total_bytes/wall_clock_time) * 8) / gb:.2f} Gbps') # Gbps
        print(f'Bandwidth (GB/s): {(total_bytes/wall_clock_time) / gb:.2f} GB/s') # GB/s
        print(f'Number of objects: {len(object_list)}')

    if args.batch_test:
        smoke_test_count = 0
        parameters = {
            'batch_size': 7,
            'bucket_name': BUCKET_NAME,
            'checkpoint': False,
            'checkpoint_bucket': CHECKPOINT_BUCKET,
            'computation_time': 0.636,
            'epochs': 2,
            'num_workers': 4, #os.cpu_count(),   # Use the number of CPUs available on the system.
            'prefetch_factor': 2,
            'smoke_test_count': smoke_test_count,
            }
        batch_test(parameters, args.loader_type)

    if args.train:
        # Hyperparameters
        model_name = 'unet3D'
        smoke_test_count = 0
        parameters = {
            'batch_size': 7,
            'bucket_name': BUCKET_NAME,
            'checkpoint': False,
            'checkpoint_bucket': CHECKPOINT_BUCKET,
            'computation_time': 0.323,
            'epochs': 2,
            'model_name': model_name,
            'num_workers': 4, #os.cpu_count(),   # Use the number of CPUs available on the system.
            'prefetch_factor': None,
            'smoke_test_count': smoke_test_count,
            'use_gpu': True,
            }
        tm = TrainUNET3D(parameters)
        tm.run_local_training(parameters, args.loader_type)


if __name__ == '__main__':
    main()
